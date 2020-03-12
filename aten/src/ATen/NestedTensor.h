#pragma once

#include <ATen/ATen.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      at::Tensor rep,
      int64_t level)
    : TensorImpl(
        c10::DispatchKeySet(NestedTensorKey),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep))
    , level_(level)
    {
      TORCH_INTERNAL_ASSERT(level >= 0);
    }

  int64_t level() const {
    return level_;
  }

  at::Tensor rep_;
  int64_t level_;

};

}
