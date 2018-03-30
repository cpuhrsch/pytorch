#pragma once

#include <sstream>
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

template <typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2_dim(
    Tensor& tensor1,
    Tensor& tensor2,
    int64_t dim,
    Op op) {
  checkBackend("CPU_tensor_apply2", {tensor1, tensor2}, Backend::CPU);
  if (tensor1.numel() != tensor2.numel()) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " 
        << tensor1.sizes() << ", and "
        << tensor2.sizes() << " to have the same number of elements, but got "
        << tensor1.numel() << ", and "
        << tensor2.numel() << " elements respectively";
    throw std::runtime_error(oss.str());
  }
  if (tensor1.sizes().equals({0})) return;
  if (tensor2.sizes().equals({0})) return;
  util_tensor<scalar1> ut1(tensor1, dim, true);
  util_tensor<scalar2> ut2(tensor2, dim, true);
  while (true) {
    /* Loop through the inner most region of the Tensor */
    for (; ut1.i < ut1.size && ut2.i < ut2.size;
         ut1.i++, ut2.i++, ut1.data += ut1.stride, ut2.data += ut2.stride) {
      op(*ut1.data, *ut2.data);
    }
    if (update(ut1)) break;
    if (update(ut2)) break;
  }
}

/*
  Apply a pointwise operator to two tensors.

  The calling convention for op is a function/functor that takes takes two
  references to type scalar; at least one of these references should be
  non-const in order to write the output. For example, to compute a = b^2, op
  would be of the form:
  [](scalar &a_val, const scalar &b_val) { a_val = b_val * b_val; };
*/
template <typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, Op op) {
  CPU_tensor_apply2_dim<scalar1, scalar2, Op>(tensor1, tensor2, -1, op);
}

template <typename scalar1, typename scalar2, typename scalar3, typename Op>
void CPU_tensor_apply3_dim(
    Tensor& tensor1,
    Tensor& tensor2,
    Tensor& tensor3,
    int64_t dim,
    Op op) {
  checkBackend("CPU_tensor_apply3", {tensor1, tensor2, tensor3}, Backend::CPU);

  if (!(tensor1.numel() == tensor2.numel() &&
        tensor2.numel() == tensor3.numel())) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " 
        << tensor1.sizes() << ", "
        << tensor2.sizes() << ", and "
        << tensor3.sizes() << " to have the same number of elements, but got "
        << tensor1.numel() << ", " 
        << tensor2.numel() << ", and "
        << tensor3.numel() << " elements respectively";
    throw std::runtime_error(oss.str());
  }
  if (tensor1.sizes().equals({0})) return;
  if (tensor2.sizes().equals({0})) return;
  if (tensor3.sizes().equals({0})) return;

  util_tensor<scalar1> ut1(tensor1, dim, true);
  util_tensor<scalar2> ut2(tensor2, dim, true);
  util_tensor<scalar3> ut3(tensor3, dim, true);
  while (true) {
    /* Loop through the inner most region of the Tensor */
    for (; ut1.i < ut1.size && ut2.i < ut2.size && ut3.i < ut3.size;
         ut1.i++,
         ut2.i++,
         ut3.i++,
         ut1.data += ut1.stride,
         ut2.data += ut2.stride,
         ut3.data += ut3.stride) {
      op(*ut1.data, *ut2.data, *ut3.data);
    }
    if (update(ut1)) break;
    if (update(ut2)) break;
    if (update(ut3)) break;
  }
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
  CPU_tensor_apply3_dim<scalar1, scalar2, scalar3, Op>(
      tensor1, tensor2, tensor3, -1, op);
}

template <
    typename scalar1,
    typename scalar2,
    typename scalar3,
    typename scalar4,
    typename Op>
void CPU_tensor_apply4_dim(
    Tensor& tensor1,
    Tensor& tensor2,
    Tensor& tensor3,
    Tensor& tensor4,
    int64_t dim,
    Op op) {
  checkBackend(
      "CPU_tensor_apply4", {tensor1, tensor2, tensor3, tensor4}, Backend::CPU);

  if (!(tensor1.numel() == tensor2.numel() &&
        tensor2.numel() == tensor3.numel() &&
        tensor3.numel() == tensor4.numel())) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " 
        << tensor1.sizes() << ", "
        << tensor2.sizes() << ", " 
        << tensor3.sizes() << ", and "
        << tensor4.sizes() << " to have the same number of elements, but got "
        << tensor1.numel() << ", " 
        << tensor2.numel() << ", " 
        << tensor3.numel() << ", and "
        << tensor4.numel() << " elements respectively";
    throw std::runtime_error(oss.str());
  }
  if (tensor1.sizes().equals({0})) return;
  if (tensor2.sizes().equals({0})) return;
  if (tensor3.sizes().equals({0})) return;
  if (tensor4.sizes().equals({0})) return;

  util_tensor<scalar1> ut1(tensor1, dim, true);
  util_tensor<scalar2> ut2(tensor2, dim, true);
  util_tensor<scalar3> ut3(tensor3, dim, true);
  util_tensor<scalar4> ut4(tensor4, dim, true);
  while (true) {
    /* Loop through the inner most region of the Tensor */
    for (; ut1.i < ut1.size && ut2.i < ut2.size && ut3.i < ut3.size &&
         ut4.i < ut4.size;
         ut1.i++,
         ut2.i++,
         ut3.i++,
         ut4.i++,
         ut1.data += ut1.stride,
         ut2.data += ut2.stride,
         ut3.data += ut3.stride,
         ut4.data += ut4.stride) {
      op(*ut1.data, *ut2.data, *ut3.data, *ut4.data);
    }
    if (update(ut1)) break;
    if (update(ut2)) break;
    if (update(ut3)) break;
    if (update(ut4)) break;
  }
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
  CPU_tensor_apply4_dim<scalar1, scalar2, scalar3, scalar4, Op>(
      tensor1, tensor2, tensor3, tensor4, -1, op);
}

} // namespace at
