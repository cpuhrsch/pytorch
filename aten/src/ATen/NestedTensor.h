#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensorId;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(torch::nested_tensor::NestedTensor&& rep)
      : TensorImpl(
            c10::DispatchKeySet(NestedTensorKey),
            rep.dtype(),
            rep.device()),
        rep_(std::move(rep)) {}

  int64_t dim() const {
    return rep_.dim();
  }

  IntArrayRef sizes() const {
    std::vector<c10::optional<int64_t>> size = rep_.sizes();
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

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor.rep_.get_structure();
  out << "NESTED_TENSOR";
  out << std::endl;
  // out << NestedNode___str__(
  //     node, "nested_tensor", [](c10::IValue payload, const std::string& tabs) {
  //       std::stringstream ss;
  //       ss << payload.toTensor();
  //       ss << std::endl;
  //       return ss.str();
  //     });
  return out;
}

}
