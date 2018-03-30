#pragma once

#include <sstream>
#include <tuple>
#include "ATen/TensorUtils.h"

namespace at {

/*
 * The basic strategy for apply is as follows:
 *
 * 1. Starting with the outermost index, loop until we reach a dimension where
 * the data is no longer contiguous, i.e. the stride at that dimension is not
 * equal to the size of the tensor defined by the outer dimensions. Let's call
 * this outer (contiguous) tensor A. Note that if the Tensor is contiguous, then
 * A is equal to the entire Tensor. Let's call the inner tensor B.
 *
 * 2. We loop through the indices in B, starting at its outermost dimension. For
 * example, if B is a 2x2 matrix, then we do:
 *
 * B[0][0]
 * B[0][1]
 * B[1][0]
 * B[1][1]
 *
 * We set the offset into the underlying storage as (storageOffset + stride_B *
 * index_B), i.e. basically we compute the offset into the storage as we would
 * normally for a Tensor. But because we are guaranteed the subsequent data is
 * contiguous in memory, we can simply loop for sizeof(A) iterations and perform
 * the operation, without having to follow the order described by the strides of
 * A.
 *
 * 3. As an optimization, we merge dimensions of A that are contiguous in
 * memory. For example, if A is a 3x3x3x3 tensor narrowed from a 3x3x4x3 tensor,
 * then the first two dimensions can be merged for the purposes of APPLY,
 * reducing the number of nested loops.
 */


//TODO:
// Write rigorous test for different non-contiguous tensors!
// I.e. generate a bunch of non-contiguous tensors and feed it into some tests

template <typename T>
struct util_tensor {
  T* data = NULL;
  int64_t *counter = NULL, *sizes = NULL, *strides = NULL, *dimOffset = NULL;
  int64_t stride = 0, size = 0, dim = 0, i, numel;
  int contiguous;
  util_tensor(Tensor& tensor, int64_t dim, bool ALLOW_CONTIGUOUS) {
    dim = dim;
    int64_t TH_TENSOR_dim_index = 0;
    int contiguous = ALLOW_CONTIGUOUS && dim < 0;
    numel = tensor.numel();
    data = tensor.data<T>();
    size = 1;
    stride = 1;
    for (i = tensor.dim() - 1; i >= 0; i--) {
      if (tensor.sizes()[i] != 1) {
        if (tensor.strides()[i] == size && i != dim)
          size *= tensor.sizes()[i];
        else {
          contiguous = 0;
          break;
        }
      }
    }
    if (!contiguous) {
      dim = 1;
      for (i = tensor.dim() - 2; i >= 0; i--) {
        if (tensor.strides()[i] !=
                tensor.strides()[i + 1] * tensor.sizes()[i + 1] ||
            i == dim || i + 1 == dim)
          dim++;
      }
      counter = new int64_t[3 * dim];
      sizes = counter + dim;
      strides = counter + 2 * dim;
      TH_TENSOR_dim_index = dim - 1;
      dimOffset = (dim == tensor.dim() - 1) ? &i : &counter[dim];
      sizes[TH_TENSOR_dim_index] = tensor.sizes()[tensor.dim() - 1];
      strides[TH_TENSOR_dim_index] = tensor.strides()[tensor.dim() - 1];
      for (i = dim - 1; i >= 0; --i) {
        counter[i] = 0;
      }
      for (i = tensor.dim() - 2; i >= 0; --i) {
        if (tensor.strides()[i] ==
                tensor.strides()[i + 1] * tensor.sizes()[i + 1] &&
            i != dim && i + 1 != dim) {
          sizes[TH_TENSOR_dim_index] =
              tensor.sizes()[i] * sizes[TH_TENSOR_dim_index];
          if (dim != tensor.dim() - 1 && i < dim)
            dimOffset--;
        } else {
          --TH_TENSOR_dim_index;
          sizes[TH_TENSOR_dim_index] = tensor.sizes()[i];
          strides[TH_TENSOR_dim_index] = tensor.strides()[i];
        }
      } /* Size of the inner most section */
      size = sizes[dim - 1]; /* Stride of the inner most section */
      stride = strides[dim - 1];
    }
    i = 0;
  }
  ~util_tensor() {
    if (counter != NULL)
      delete[] counter;
  }
};

// Return true if finished
template <typename T>
static bool update(util_tensor<T>& ut) {
  if (ut.i == ut.size) {
    if (ut.contiguous)
      return true;

    if (ut.dim == 1)
      return true;

    /* Reset pointer to beginning of loop */
    ut.data -= ut.size * ut.stride;
    for (ut.i = ut.dim - 2; ut.i >= 0; ut.i--) {
      ut.counter[ut.i]++;
      /* Jump ahread by the stride of this dimension */
      ut.data += ut.strides[ut.i];

      if (ut.counter[ut.i] == ut.sizes[ut.i]) {
        if (ut.i == 0) {
          return true;
        } else {
          /* Reset the pointer to the beginning of the chunk defined by this
           * dimension */
          ut.data -= ut.counter[ut.i] * ut.strides[ut.i];
          ut.counter[ut.i] = 0;
        }
      } else {
        return false;
      }
    }
    ut.i = 0;
  }
  return false;
}

inline bool _all_equal_numel(at::ArrayRef<Tensor> tensors) {
  if (tensors.size() == 0)
    return true;
  int64_t all_numel = tensors[0].numel();
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].numel() != all_numel)
      return false;
  }
  return true;
}

inline std::string _all_equal_numel_error(at::ArrayRef<Tensor> tensors) {
  std::ostringstream oss;
  oss << "inconsistent tensor size, expected ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].sizes() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1]
      << " to have the same number of elements, but got ";
  for (size_t i = 0; i < tensors.size() - 1; i++) {
    oss << tensors[i].numel() << ", ";
  }
  oss << "and " << tensors[tensors.size() - 1].numel()
    << " elements respectively";
  return oss.str();
}

template <typename... scalars>
std::tuple<util_tensor<scalars>...> build_util_tensors(
    int64_t dim,
    bool allow_cont,
    Tensor& tensors...) {
  return 
      std::make_tuple(util_tensor<scalars>(tensors, dim, allow_cont)...);
}

template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type update_all(
    std::tuple<Tp...>& t) {
  return false;
}

template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type update_all(std::tuple<Tp...>& t) {
  return update(std::get<I>(t)) || update_all<I + 1, Tp...>(t);
}

template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type iterate_all(
    std::tuple<Tp...>& t) {}

template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type iterate_all(std::tuple<Tp...>& t) {
  auto ut = std::get<I>(t);
  ut.data += ut.stride;
  iterate_all<I + 1, Tp...>(t);
}

template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type check_all(
    std::tuple<Tp...>& t) {
  return true;
}

template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type check_all(std::tuple<Tp...>& t) {
  auto ut = std::get<I>(t);
  return ut.i < ut.size && check_all<I + 1, Tp...>(t);
}

inline bool _apply_preamble(ArrayRef<Tensor> tensors) {
  checkBackend(
      "CPU_tensor_apply",
      tensors,
      Backend::CPU);
  if (!_all_equal_numel(tensors))
    throw std::runtime_error(_all_equal_numel_error(tensors));
  for (auto& t : tensors)
    if (t.sizes().equals({0}))
      return false;
  return true;
}


template <typename Op, typename... scalars, size_t... Is> 
void _apply_op(Op op, std::tuple<util_tensor<scalars>...> t, Indices<Is...>) {
    op(*std::get<Is>(t).data...);
    }   

template <typename Op, typename... scalars>
void apply_op(Op op, std::tuple<util_tensor<scalars>...> t) {
    _apply_op(op, t, typename MakeIndices<sizeof... (scalars)>::indices());
}

template <typename Op, typename... scalars>
void CPU_tensor_apply_dim(
    int64_t dim,
    Op op,
    Tensor& tensors...) {
  if (!_apply_preamble({tensors}))
    return;
  auto uts = build_util_tensors<scalars...>(dim, true, tensors);
  while (true) {
    while (check_all(uts)) {
      apply_op(op, uts);
      iterate_all(uts);
    }
    if (update_all(uts))
      break;
  }
}

template <typename Op, typename... scalars>
void CPU_tensor_apply(Op op, Tensor& tensors...) {
  CPU_tensor_apply_dim<Op, scalars...>(-1, op, tensors);
}

/*
  Apply a pointwise operator to sequence of tensors

  The calling convention for op is a function/functor that takes takes two
  references to type scalar; at least one of these references should be
  non-const in order to write the output. For example, to compute a = b^2, op
  would be of the form:
  [](scalar &a_val, const scalar &b_val) { a_val = b_val * b_val; };
*/

template <typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, Op op) {
  CPU_tensor_apply<Op, scalar1, scalar2>(op, tensor1, tensor2);
}

/*
  Apply a pointwise operator to three tensors.

  The calling convention for op is a function/functor that takes takes three
  references to type scalar; at least one of these references should be
  non-const in order to write the output. For example, to compute a = b + c, op
  would be of the form:
  [](scalar &a_val, const scalar &b_val, const scalar &c_val) { a_val = b_val +
  c_val; };
*/
template <typename scalar1, typename scalar2, typename scalar3, typename Op>
void CPU_tensor_apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Op op) {
  CPU_tensor_apply<Op, scalar1, scalar2, scalar3>(
      op, tensor1, tensor2, tensor3);
}

/*
  Apply a pointwise operator to four tensors.

  The calling convention for op is a function/functor that takes takes four
  references to type scalar; at least one of these references should be
  non-const in order to write the output. For example, to compute a = b + c * d,
  op would be of the form:
  [](scalar &a_val, const scalar &b_val, const scalar &c_val, const scalar
  &d_val) { a_val = b_val + c_val * d_val;
  };
*/
template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename Op>
void CPU_tensor_apply4(
    Tensor tensor1,
    Tensor tensor2,
    Tensor tensor3,
    Tensor tensor4,
    Op op) {
  CPU_tensor_apply<Op, scalar1, scalar2, scalar3, scalar4>(
      op, tensor1, tensor2, tensor3, tensor4);
}

} // namespace at
