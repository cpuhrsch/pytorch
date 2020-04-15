#include <ATen/Batching.h>
#include <ATen/BatchingUtils.h>
#include <ATen/WrapDimUtils.h>

namespace at {

/*
 * Batching rules for regular tensors.
 * To override any of these, register it as an operator.
 * NB: BatchDimsRef isn't supported for operators syntax. Might need to break it
 * into two vector<int64_t>.
 */ 


std::pair<Tensor,BatchDims>
add_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    const Tensor& other, BatchDimsRef other_bdims,
    Scalar alpha) {
  Tensor self_, other_;
  BatchDims result_bdims;
  std::tie(self_, other_, result_bdims) = alignBdimsAtFront(self, self_bdims, other, other_bdims);
  return { at::add(self_, other_, alpha), result_bdims };
}

std::pair<Tensor,BatchDims>
mul_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    const Tensor& other, BatchDimsRef other_bdims) {
  Tensor self_, other_;
  BatchDims result_bdims;
  std::tie(self_, other_, result_bdims) = alignBdimsAtFront(self, self_bdims, other, other_bdims);
  return { at::mul(self_, other_), result_bdims };
}

template <Tensor (*Op)(const Tensor&)>
std::pair<Tensor,BatchDims> unary_pw_batching_rule(const Tensor& self, BatchDimsRef self_bdims) {
  return { Op(self), self_bdims.vec() };
}

template <Tensor& (*Op)(Tensor&)>
void unary_pw_inplace_batching_rule(Tensor& self, BatchDimsRef self_bdims) {
  Op(self);
}

std::pair<Tensor,BatchDims>
dropout_batching_rule(const Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout(self, p, train), self_bdims.vec() };
}

std::pair<Tensor,BatchDims>
dropout__batching_rule(Tensor& self, BatchDimsRef self_bdims, double p, bool train) {
  return { at::dropout_(self, p, train), self_bdims.vec() };
}

std::pair<Tensor,BatchDims> conv2d_batching_rule(
    const Tensor& input, BatchDimsRef input_bdims,
    const Tensor& weight, BatchDimsRef weight_bdims,
    const Tensor& bias, BatchDimsRef bias_bdims,
    IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, int64_t groups) {
  if (weight_bdims.size() > 0) {
    // TODO: call fallback
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched weight");
  }
  if (bias_bdims.size() > 0) {
    // TODO: call fallback
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched bias");
  }

  auto input_ = moveBdimsToFront(input, input_bdims);
  auto result_bdims = moveBdimsToFront(input_bdims);

  if (input_.dim() <= 4) {
    // No need to flatten
    auto result = at::conv2d(input_, weight, bias, stride, padding, dilation, groups);
    return { result, result_bdims };
  }

  auto num_dims_to_flatten = input_bdims.size() + 1;
  auto flat_input_ = input_.flatten(0, num_dims_to_flatten - 1);
  auto flat_result = at::conv2d(flat_input_, weight, bias, stride, padding, dilation, groups);
  auto result = flat_result.unflatten(
      0, IntArrayRef(input.sizes().begin(), input.sizes().begin() + num_dims_to_flatten));

  return { result, result_bdims };
}

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype) {
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto result_bdims = moveBdimsToFront(self_bdims);

  // Real dims to reduce over
  std::vector<int64_t> actual_dims;
  for (int64_t dim : dims) {
    actual_dims.push_back(dim + self_bdims.size());
  }

  auto result = at::sum(self_, actual_dims, keepdim, dtype);
  return { result, result_bdims };
}

} // namespace at
