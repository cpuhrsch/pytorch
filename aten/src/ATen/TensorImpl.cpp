#include <ATen/TensorImpl.h>

#include "ATen/Context.h"
#include <ATen/Tensor.h>
#include <ATen/core/optional.h>

#include <TH/THTensor.hpp>

namespace at {
Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

Tensor TensorImpl::detach() const {
  AT_ERROR("detach is not implemented for Tensor");
}

const char* TensorImpl::toString() const {
  // This matches behavior with VariableImpl
  return type().toString();
}

void TensorImpl::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  AT_ERROR("backward is not implemented for Tensor");
}

void TensorImpl::set_data(Tensor new_data) {
  AT_ERROR("set_type is not implemented for Tensor");
}

void Tensor::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  pImpl->backward(std::move(gradient), keep_graph, create_graph);
}

TensorImpl::TensorImpl(Backend backend, ScalarType scalar_type) {
  type_ = &globalContext().getType(backend, scalar_type);
  Storage* storage = type_->storage().release();
  StorageImpl* storage_impl = storage->pImpl();
  // storage_impl->retain();
  storage_impl->set_resizable(true);
  tensor = new THTensor(storage_impl);
}

TensorImpl::TensorImpl(
    Backend backend,
    ScalarType scalar_type,
    THTensor* tensor_) {
  type_ = &globalContext().getType(backend, scalar_type);
  tensor = tensor_;
}

TensorImpl::~TensorImpl() {
  if (tensor) tensor->release();
}

IntList TensorImpl::sizes() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(THTensor_getSizePtr(tensor), dim());
}

IntList TensorImpl::strides() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(THTensor_getStridePtr(tensor), dim());
}

void TensorImpl::release_resources() {
  if (tensor) {
      tensor->release();
      tensor = nullptr;
  }
}

int64_t TensorImpl::dim() const {
  if(THTensor_isZeroDim(tensor)) {
    return 0;
  }
  return tensor->dim();
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(tensor, "TensorImpl without THTensor in maybe_zero_dim");
  bool is_zero_dim = condition_when_zero_dim && tensor->sizes().size() == 1 && tensor->size(0) == 1;
  THTensor_setIsZeroDim(tensor, is_zero_dim);
  return this;
}

void * TensorImpl::unsafeGetTH(bool retain) {
  if (retain) {
    tensor->retain();
  }
  return tensor;
}

std::unique_ptr<Storage> TensorImpl::storage() {
  StorageImpl* storage = tensor->storage_;
  storage->retain();
  return std::unique_ptr<Storage>(new Storage(storage));
}

} // namespace at
