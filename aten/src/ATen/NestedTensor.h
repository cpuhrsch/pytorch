#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/nestedtensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      at::Tensor rep,
      TensorList tensors)
    : TensorImpl(
        c10::DispatchKeySet(NestedTensorKey),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep))
    , tensors_(tensors)
    , level_(level)
    {
      TORCH_INTERNAL_ASSERT(level >= 0);
    }

  int64_t level() const {
    return level_;
  }

  at::Tensor rep_;
  TensorList tensors_;
  int64_t level_;

};

inline Tensor makeNested(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Require at least one Tensor");
  return at::detail::make_tensor<NestedTensorImpl>(tensors[0], tensors);
}

}
