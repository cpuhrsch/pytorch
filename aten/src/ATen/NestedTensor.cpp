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
  // Some operations need to be transformed to their batched versions
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_nested(Tensor[] self) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_make_nested))

}
