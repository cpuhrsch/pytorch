#include <ATen/NestedTensor.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>

namespace at {

using namespace torch::nested_tensor;

// TODO: The dispatcher has trouble with this so we register an unboxed kernel.
Tensor NestedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  return at::ones({1, 2, 3, 1});
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &NestedTensor_conv2d>(NestedTensorKey))
  ;

}
