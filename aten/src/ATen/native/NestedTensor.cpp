#include <ATen/NestedTensor.h>

namespace at { namespace native {

using namespace torch::nested_tensor;

Tensor _make_nested(TensorList self) {
  std::vector<at::Tensor> tensors = self.vec();
  std::vector<TensorNode> tensor_nodes;
  for (size_t i = 0; i < tensors.size(); i++) {
    tensor_nodes.push_back(TensorNode(std::move(tensors[i])));
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      NestedTensor(TensorNode(std::move(tensor_nodes))));
}

}}
