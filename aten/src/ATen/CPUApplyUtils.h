#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <tuple>
#include "ATen/TensorUtils.h"
#include "SmallVector.h"

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

template <typename T>
struct strided_tensor_iter {
 public:
  T* data_ = NULL;
  int64_t dim_;

  SmallVector<int64_t, 8> counter_;
  SmallVector<int64_t, 8> sizes_;
  SmallVector<int64_t, 8> strides_;


  strided_tensor_iter(strided_tensor_iter const&) = delete;
  void operator=(strided_tensor_iter const& x) = delete;
  strided_tensor_iter(strided_tensor_iter&&) = default;
  strided_tensor_iter(Tensor& tensor)
      : data_(tensor.data<T>()),
        dim_(0),
        counter_(8, 0),
        sizes_(8, 0),
        strides_(8, 0) {
    int64_t max_dim = tensor.ndimension();
    if (max_dim > 8) {
      counter_.resize(max_dim);
      sizes_.resize(max_dim);
      strides_.resize(max_dim);
    }
    dim_ = 0;
    for (int64_t i = 0; i < max_dim; i++) {
      int64_t size = tensor.size(i);
      int64_t stride = tensor.stride(i);
      while (i + 1 < max_dim &&
             tensor.stride(i) == tensor.size(i + 1) * tensor.stride(i + 1)) {
        size = size * tensor.size(i + 1);
        stride = tensor.stride(i + 1);
        i++;
      }
      sizes_[dim_] = size;
      strides_[dim_] = stride;
      counter_[dim_] = 0;
      dim_++;
    }
  }
};

template <typename Arg>
inline void iterate(Arg& ut) {
  if (ut.counter_[ut.dim_ - 1] == ut.sizes_[ut.dim_ - 1]) {
    for (int64_t i = ut.dim_ - 1; i > 0; i--) {
      if (ut.counter_[i] == ut.sizes_[i]) {
        ut.counter_[i] = 0;
        ut.counter_[i - 1]++;
        ut.data_ =
            ut.data_ - (ut.sizes_[i] * ut.strides_[i]) + ut.strides_[i - 1];
      }
    }
  }
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

inline bool _apply_preamble(ArrayRef<Tensor> tensors) {
  checkBackend("CPU_tensor_apply", tensors, Backend::CPU);
  if (!_all_equal_numel(tensors))
    throw std::runtime_error(_all_equal_numel_error(tensors));
  // An empty tensor has no elements
  for (auto& t : tensors)
    if (t.sizes().equals({0}))
      return false;
  return true;
}

inline void iterate_all(){};

template <typename Arg, typename... Args>
inline void iterate_all(Arg& iter, Args&... iter_tail) {
  iterate(iter);
  iterate_all(iter_tail...);
}

inline bool continue_loop() {
  return true;
}

template <typename Arg, typename... Args>
inline bool continue_loop(Arg& iter, Args&... iter_tail) {
  bool result = false;
  if (iter.dim_ != 0) {
    result = (iter.counter_[iter.dim_ - 1] < iter.sizes_[iter.dim_ - 1]);
  }
  return result && continue_loop(iter_tail...);
}

inline void increase_counters(){};

template <typename Arg, typename... Args>
inline void increase_counters(Arg& iter, Args&... iter_tail) {
  iter.counter_[iter.dim_ - 1]++;
  iter.data_ += iter.strides_[iter.dim_ - 1];
  increase_counters(iter_tail...);
}

inline void apply_op(){};

template <typename Op, typename... Args>
inline void apply_op(int64_t numel, const Op& op, Args... iters) {
  // For 0-dim tensors
  if (numel == 1 && !continue_loop(iters...)) {
    op(*iters.data_...);
    return;
  }
  for (int64_t i = 0; i < numel;) {
    for (; continue_loop(iters...); increase_counters(iters...)) {
      op(*iters.data_...);
      i++;
    }
    iterate_all(iters...);
  }
}

/*
  Apply a pointwise operator to sequence of tensors

  The calling convention for op is a function/functor that takes takes the same
  number of pointers of type scalar as the number of given tensors. For example,
  to compute a = b * c, op would be of the form:
  [](scalar* a_val, const scalar* b_val, const scalar* c_val) { a_val[0] =
  b_val[0] * c_val[0]; };
*/

template <typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, Op op) {
  if (!_apply_preamble({tensor1, tensor2}))
    return;
  apply_op(
      tensor1.numel(),
      op,
      strided_tensor_iter<scalar1>(tensor1),
      strided_tensor_iter<scalar2>(tensor2));
}

template <typename scalar1, typename scalar2, typename scalar3, typename Op>
void CPU_tensor_apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Op op) {
  if (!_apply_preamble({tensor1, tensor2, tensor3}))
    return;
  apply_op(
      tensor1.numel(),
      op,
      strided_tensor_iter<scalar1>(tensor1),
      strided_tensor_iter<scalar2>(tensor2),
      strided_tensor_iter<scalar3>(tensor3));
}

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
  if (!_apply_preamble({tensor1, tensor2, tensor3, tensor4}))
    return;
  apply_op(
      tensor1.numel(),
      op,
      strided_tensor_iter<scalar1>(tensor1),
      strided_tensor_iter<scalar2>(tensor2),
      strided_tensor_iter<scalar3>(tensor3),
      strided_tensor_iter<scalar4>(tensor4));
}

} // namespace at
