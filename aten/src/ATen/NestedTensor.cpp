#include <ATen/NestedTensor.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <nestedtensor/csrc/functions.h>

namespace at {

using namespace torch::nested_tensor;

// TODO: The dispatcher has trouble with this so we register an unboxed kernel.
Tensor NestedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  // auto nt = NestedTensor(at::ones({2, 3, 2, 1}));
  auto input_impl = static_cast<NestedTensorImpl*>(input.unsafeGetTensorImpl());
  std::cout << "HERE" << std::endl;
  auto nt = torch::nested_tensor::conv2d(
      input_impl->rep_, weight, bias, stride, padding, dilation, groups);
  std::cout << "MADE" << std::endl;
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

Tensor NestedTensor_select(const Tensor& self, int64_t dim, int64_t index) {
  std::cout << "HEEEEEEEEEEE1 dim: " << dim << " - index: " << index << std::endl;
  auto input_impl = static_cast<NestedTensorImpl*>(self.unsafeGetTensorImpl());
  auto nt = input_impl->rep_;
  auto structure = nt.get_structure();
  auto unbound = structure.unbind();
  auto unbound_i = unbound[0];
  std::cout << "HEEEEEEEEEEE2" << std::endl;
  if (nt.nested_dim() == 1) {
    std::cout << "HEEEEEEEEEEE21" << std::endl;
    return unbound_i.payload();
  }
  std::cout << "HEEEEEEEEEEE22" << std::endl;
  auto nt_i = NestedTensor(std::move(unbound_i));
  std::cout << "HEEEEEEEEEEE3" << std::endl;
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &NestedTensor_conv2d>(NestedTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
      .impl_unboxedOnlyKernel<Tensor (const Tensor& self, int64_t dim, int64_t index), &NestedTensor_select>(NestedTensorKey))
  ;

}
