#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/nestedtensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      at::Tensor rep)
    : TensorImpl(
        c10::DispatchKeySet(NestedTensorKey),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep))
    {
    }

  at::Tensor rep_;

};

inline Tensor makeNested(Tensor tensors) {
  TORCH_CHECK(tensors.size() > 0, "Require at least one Tensor");
  return at::detail::make_tensor<NestedTensorImpl>(tensors);
}

}
