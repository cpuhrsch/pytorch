#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THTensor.h"
#include "THStorageFunctions.hpp"

#include <atomic>
#include <ATen/ATen.h>
#include <ATen/TensorImpl.h>

#define THTensor at::TensorImpl

inline int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->sizes_.data();
}

inline int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->strides_.data();
}

// NB: Non-retaining
inline THStorage* THTensor_getStoragePtr(const THTensor* tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  AT_CHECK(tensor->storage_, "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor->storage_;
}


inline bool THTensor_isZeroDim(const THTensor *tensor) {
  return tensor->is_zero_dim_;
}

inline void THTensor_setIsZeroDim(THTensor *tensor, bool is_zero_dim) {
  tensor->is_zero_dim_ = is_zero_dim;
}

// [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
// nDimension                 corresponds to the "true" ATen dimension. TODO: implement.
// nDimensionLegacyNoScalars  correpsonds to the ATen dimension, except scalars are viewed as 1-dimensional tensors.
// nDimensionLegacyAll        corresponds to the ATen dimension, except scalars are viewed as 1-dimensional tensors
//                            and tensors with a dimension of size zero are collapsed to 0-dimensional tensors.
//
// Eventually, everything should go through nDimension or tensor->dim().
inline int THTensor_nDimensionLegacyNoScalars(const THTensor* tensor) {
  if (THTensor_isZeroDim(tensor)) {
    return 1;
  } else {
    return tensor->dim();  
  }
}

inline int THTensor_nDimensionLegacyAll(const THTensor* tensor) {
  if (tensor->is_empty()) {
    return 0;  
  } else if (THTensor_isZeroDim(tensor)) {
    return 1;
  } else {
    return tensor->dim();  
  }
}

inline int64_t THTensor_strideLegacyNoScalars(const THTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_nDimensionLegacyNoScalars(self));
  return THTensor_isZeroDim(self) ? 1 : self->stride(dim);
}

inline int64_t THTensor_sizeLegacyNoScalars(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_nDimensionLegacyNoScalars(self));
  return THTensor_isZeroDim(self) ? 1 : self->size(dim);
}

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  // NB: This is *truly* a resize; calling code (e.g., squeeze)
  // assumes that old values are preserved
  tensor->is_zero_dim_ = bool(ndim == 0);
  tensor->sizes_.resize(ndim);
  tensor->strides_.resize(ndim);
}

inline void THTensor_setSizesAndStrides(THTensor* tensor, std::vector<int64_t>&& new_size, std::vector<int64_t>&& new_stride) {
  tensor->sizes_ = std::move(new_size);
  tensor->strides_ = std::move(new_stride);
}

inline void THTensor_setSizeAtDim(THTensor* tensor, int dim, int64_t new_size) {
  tensor->sizes_[dim] = new_size;
}

inline void THTensor_setStrideAtDim(THTensor* tensor, int dim, int64_t new_stride) {
  tensor->strides_[dim] = new_stride;
}

inline void THTensor_setStorageOffset(THTensor* tensor, ptrdiff_t storage_offset) {
  tensor->storage_offset_ = storage_offset;
}

// NB: Steals ownership of storage
inline void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage) {
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  AT_ASSERT(storage);
  tensor->storage_ = storage;
}

TH_API void THTensor_free(THTensor *self);
TH_CPP_API at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                                      at::IntList newshape);
