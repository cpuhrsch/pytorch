#pragma once

#include <ATen/ATen.h>
#include <nested_tensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(
      torch::nested_tensor::NestedTensor rep)
    : TensorImpl(
        c10::DispatchKeySet(NestedTensorKey),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep))
    {
  std::cout << "DOING THIS" << std::endl;
    }

  torch::nested_tensor::NestedTensor rep_;

  IntArrayRef sizes() const {
  std::cout << "DOING THAT" << std::endl;
    return IntArrayRef(std::vector<int64_t>({1, 2, 3}));
  }

};

}
