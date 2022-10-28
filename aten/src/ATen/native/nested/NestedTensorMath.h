#pragma once

#include <ATen/core/ATen_fwd.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

TORCH_API Tensor NestedTensor_softmax_generic(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float);

} // namespace native
} // namespace at
