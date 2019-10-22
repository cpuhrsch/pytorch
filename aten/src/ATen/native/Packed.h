#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <stdexcept>

namespace at { struct TensorIterator; }

namespace at { namespace native {

PackedTensor add_packed(PackedTensor input, Tensor input2);

}} // namespace at::native
