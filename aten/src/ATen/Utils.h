#pragma once

#include "ATen/ATenGeneral.h"
#include "ATen/ArrayRef.h"
#include "ATen/Error.h"
#include "ATen/UndefinedTensor.h"

#include <algorithm>
#include <sstream>
#include <typeinfo>

namespace at {

//===----------------------------------------------------------------------===//
//                std::index_sequence shim for C++11
//===----------------------------------------------------------------------===//

// A container of type-template parameter indices.
template<size_t... Is>
struct Indices {};

// Decrements the index N, adds N-1 to the list of indices and forwards
// whatever we arleady have.
template<size_t N, size_t... Is>
struct MakeIndices : MakeIndices<N-1, N-1, Is...> {};

// Partial specialization that forms our base case. When N is zero, we stop
// and define a typedef that will be visible to earlier classes due to
// inheritance. The typedef we define is an index list containing the numbers
// 0 through N-1.
template<size_t... Is>
struct MakeIndices<0, Is...> { using indices = Indices<Is...>; };

template <typename T, typename Base>
static inline T* checked_cast_storage(Base* expr, const char * name, int pos) {
  if (typeid(*expr) != typeid(T))
    AT_ERROR("Expected object of type %s but found type %s for argument #%d '%s'",
      T::typeString(),expr->type().toString(),pos,name);
  return static_cast<T*>(expr);
}

template <typename T, typename Base>
inline T* checked_cast_tensor(Base* expr, const char * name, int pos, bool allowNull) {
  if(allowNull && expr == UndefinedTensor::singleton()) {
    return nullptr;
  }
  if (typeid(*expr) != typeid(T))
    AT_ERROR("Expected object of type %s but found type %s for argument #%d '%s'",
      T::typeString(),expr->type().toString(),pos,name);
  return static_cast<T*>(expr);
}

// Converts a TensorList (i.e. ArrayRef<Tensor> to the underlying TH* Tensor Pointer)
template <typename T, typename TBase, typename TH>
static inline std::vector<TH*> tensor_list_checked_cast(ArrayRef<TBase> tensors, const char * name, int pos) {
  std::vector<TH*> casted(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    auto *expr = tensors[i].pImpl;
    auto result = dynamic_cast<T*>(expr);
    if (result) {
      casted[i] = result->tensor;
    } else {
      AT_ERROR("Expected a Tensor of type %s but found a type %s for sequence element %u "
                    " in sequence argument at position #%d '%s'",
                    T::typeString(),expr->type().toString(),i,pos,name);

    }
  }
  return casted;
}

template <size_t N>
std::array<int64_t, N> check_intlist(ArrayRef<int64_t> list, const char * name, int pos, ArrayRef<int64_t> def={}) {
  if (list.empty()) {
    list = def;
  }
  auto res = std::array<int64_t, N>();
  if (list.size() == 1 && N > 1) {
    res.fill(list[0]);
    return res;
  }
  if (list.size() != N) {
    AT_ERROR("Expected a list of %zd ints but got %zd for argument #%d '%s'",
        N, list.size(), pos, name);
  }
  std::copy_n(list.begin(), N, res.begin());
  return res;
}

} // at
