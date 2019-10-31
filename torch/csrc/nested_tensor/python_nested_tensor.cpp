#include <torch/csrc/nested_tensor/python_nested_tensor.h>

#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <pybind11/pybind11.h>
#include <torch/script.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>

#include <pybind11/pybind11.h>
#include <structmember.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace torch {

namespace nested_tensor {

using namespace at;
using namespace torch::autograd;

void initialize_python_bindings() { py_bind_nestedtensor(); }

static void py_bind_nestedtensor() {
  C10_LOG_API_USAGE_ONCE("tensor_list.python.import");
  auto nestedtensor_module =
      THPObjectPtr(PyImport_ImportModule("torch.nestedtensor"));
  if (!nestedtensor_module) {
    throw python_error();
  }

  auto m = py::handle(nestedtensor_module).cast<py::module>();
  py::class_<at::NestedTensor>(m, "NestedTensor");
}
}
}
