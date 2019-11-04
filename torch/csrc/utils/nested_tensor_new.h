#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

namespace torch { namespace utils {

at::NestedTensor nested_tensor_ctor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::NestedTensor as_nested_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::NestedTensor new_nested_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);
at::NestedTensor new_nested_ones(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs);

}} // namespace torch::utils
