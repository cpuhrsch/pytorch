#pragma once

#include <ATen/ATen.h>
#include <nested_tensor.h>

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
  std::cout << "DOING THIS" << std::endl;
    }

  at::Tensor rep_;

  IntArrayRef sizes() const {
  std::cout << "DOING THAT" << std::endl;
    return IntArrayRef(std::vector<int64_t>({1, 2, 3}));
  }

};

inline Tensor makeNested(TensorList tensors) {
  std::cout << "IM HERE" << std::endl;
  TORCH_CHECK(tensors.size() > 0, "Require at least one Tensor");
  return at::detail::make_tensor<NestedTensorImpl>(tensors[0]);
}

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  out << "NestedTensor[lvl" << batch_tensor.rep_ << "/bdim";
  return out;
}

}
