#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeId.h>

namespace torch {
namespace nested_tensor {

void initialize_python_bindings();

static void py_bind_nestedtensor();

} // namespace nestedtensor
} // namespace torch
