#pragma once

#include <ATen/ATen.h>
#include <nested_tensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(torch::nested_tensor::NestedTensor rep)
      : TensorImpl(
            c10::DispatchKeySet(NestedTensorKey),
            rep.dtype(),
            rep.device()),
        rep_(std::move(rep)) {}

  int64_t dim() const {
    return rep_.dim();
  }

  IntArrayRef sizes() const {
    std::vector<c10::optional<int64_t>> size = rep_.size();
    std::vector<int64_t> sizes;
    for (auto opt_int : size) {
      if (opt_int) {
        sizes.push_back(*opt_int);
      } else {
        sizes.push_back(-1);
      }
    }
    return IntArrayRef(sizes);
  }

  torch::nested_tensor::NestedTensor rep_;

};

}
