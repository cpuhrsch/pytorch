#include <ATen/NestedTensor.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>

namespace at {

bool isNestedTensor(const Tensor& tensor) {
  return tensor.defined() &&
      tensor.unsafeGetTensorImpl()->key_set().has(NestedTensorKey);
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_nested(TensorList self) -> Tensor")
      .kernel(NestedTensorKey, &at::native::_make_nested))
  ;

}
