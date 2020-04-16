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
        rep_(std::move(rep)) {
    std::vector<c10::optional<int64_t>> size = rep_.sizes();
    for (auto opt_int : size) {
      if (opt_int) {
        sizes_.push_back(*opt_int);
      } else {
        sizes_.push_back(-1);
      }
    }
  }

  int64_t dim() const {
    return rep_.dim();
  }
  int64_t nested_dim() const {
    return rep_.nested_dim();
  }

  IntArrayRef sizes() const {
    return IntArrayRef(sizes_);
  }
  Tensor operator[](int64_t index) const { 
    std::cout << "1" << std::endl;
    auto rep_structure = rep_.get_structure();
    std::cout << "2" << std::endl;
    auto unbound = rep_structure.unbind();
    std::cout << "3" << std::endl;
    if (nested_dim() == 1) {
    std::cout << "4" << std::endl;
      return unbound[0].payload();
    }
    std::cout << "5" << std::endl;
    return at::detail::make_tensor<NestedTensorImpl>(
        std::move(torch::nested_tensor::NestedTensor(std::move(unbound[0]))));
  }
  int64_t numel() const {
    return rep_.numel();
  }

  torch::nested_tensor::NestedTensor rep_;
  std::vector<int64_t> sizes_;

};

inline bool isNested(Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(NestedTensorKey);
}

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor.rep_.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

}
