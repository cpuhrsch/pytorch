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
  // The following lines need to happen in each kernel
  // auto cur_level = maxLevel({input, weight, bias});
  // auto input_and_bdim = unwrapAtLevel(input, cur_level);
  // auto weight_and_bdim = unwrapAtLevel(weight, cur_level);
  // auto bias_and_bdim = unwrapAtLevel(bias, cur_level);

  // auto result_and_bdim = conv2d_batching_rule(
  //     input_and_bdim.first,
  //     input_and_bdim.second,
  //     weight_and_bdim.first,
  //     weight_and_bdim.second,
  //     bias_and_bdim.first,
  //     bias_and_bdim.second,
  //     stride, padding, dilation, groups);
  std::cout << "REEEEEEEEEEEEEE" << std::endl;
  return at::detail::make_tensor<NestedTensorImpl>(
      NestedTensor(TensorNode(at::ones({}))));
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &NestedTensor_conv2d>(NestedTensorKey))
  ;

}
