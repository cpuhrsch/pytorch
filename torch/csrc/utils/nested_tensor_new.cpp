#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/tensor_new.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/core/EnableNamedTensor.h>

#include <stdexcept>
#include <vector>

using at::Backend;
using at::Device;
using at::DeviceType;
using at::IntArrayRef;
using at::kCPU;
using at::kCUDA;
using at::kLong;
using at::Scalar;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::NestedTensor;
using at::TensorOptions;
using at::Type;
using c10::optional;

namespace torch { namespace utils {
namespace {

NestedTensor nested_tensor_ctor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
#ifdef BUILD_NAMEDTENSOR
    "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
#else
    "nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
#endif
  });
  return NestedTensor();
}

NestedTensor as_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  // TODO: add requires_grad once we decide on semantics for sharing data.
  static PythonArgParser parser({
    "as_nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });
  return NestedTensor();
}

NestedTensor new_nested_tensor(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_nested_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });
  return NestedTensor();
}

NestedTensor new_nested_ones(c10::TensorTypeId type_id, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);
  return NestedTensor();
}

}} // namespace torch::utils
