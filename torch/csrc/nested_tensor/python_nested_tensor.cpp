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
#include <torch/csrc/autograd/python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/nested_tensor_new.h>
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

struct PyNestedTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  char name[64];
  int backend;
  int scalar_type;

  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  TensorTypeId get_type_id() const {
    return backendToTensorTypeId(static_cast<Backend>(backend));
  }

  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

static_assert(std::is_standard_layout<PyNestedTensorType>::value, "PyNestedTensorType must be standard layout");

// This is always an instance of VariableType
// TODO: Applies to NestedTensor?
static PyNestedTensorType* default_tensor_type;

static void py_bind_nested_tensor_types(const std::vector<PyNestedTensorType>& nested_tensor_types);

static TypeError unavailable_type(const PyNestedTensorType& type) {
  return TypeError("type %s not available. Torch not compiled with CUDA enabled.", type.name);
}

static PyObject* NestedTensor_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto& nested_tensor_type = *((PyNestedTensorType*)type);
  if (nested_tensor_type.is_cuda && !torch::utils::cuda_enabled()) {
    throw unavailable_type(nested_tensor_type);
  }
  return THPNestedTensor_Wrap(torch::utils::nested_tensor_ctor(nested_tensor_type.get_type_id(), nested_tensor_type.get_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

PyObject *NestedTensor_dtype(PyTensorType* self, void *unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject *NestedTensor_layout(PyTensorType* self, void *unused) {
  return torch::autograd::utils::wrap(self->layout);
}

PyObject *NestedTensor_is_cuda(PyTensorType* self, void *unused) {
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *NestedTensor_is_sparse(PyTensorType *self, void *unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

static struct PyMethodDef metaclass_methods[] = {
  {"__instancecheck__", (PyCFunction)Tensor_instancecheck, METH_O, nullptr},
  {nullptr}
};

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef metaclass_properties[] = {
  {"dtype",        (getter)NestedTensor_dtype, nullptr, nullptr, nullptr},
  {"layout",       (getter)NestedTensor_layout, nullptr, nullptr, nullptr},
  {"is_cuda",      (getter)NestedTensor_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse",    (getter)NestedTensor_is_sparse, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyTypeObject metaclass;

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  ((PyObject*)&metaclass)->ob_refcnt = 1;
  metaclass.tp_basicsize = sizeof(PyTypeObject);
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_name = "torch.tensortype";
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static void py_initialize_tensor_type(PyTypeObject& type, const char* name, PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // The typical PyVarObject_HEAD_INIT(nullptr, 0) is described in the Python
  // documentation: it initializes the refcnt to 1 and the other object header
  // fields to zero.
  memset(&type, 0, sizeof(PyTypeObject));
  ((PyObject*)&type)->ob_refcnt = 1;
  ((PyObject*)&type)->ob_type = &metaclass;
  type.tp_basicsize = sizeof(PyNestedTensorType);
  // Subclassing from torch.<ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = NestedTensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static const char* get_module(Backend backend) {
  switch (backend) {
    case Backend::CPU: return "torch";
    case Backend::CUDA: return "torch.cuda";
    case Backend::SparseCPU: return "torch.sparse";
    case Backend::SparseCUDA: return "torch.cuda.sparse";
    default: AT_ERROR("invalid backend: ", toString(backend));
  }
}

static void set_type(PyNestedTensorType& type_obj, Backend backend, ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout = torch::getLayout(backend);
  type_obj.dtype = torch::getDtype(scalarType);
  type_obj.is_cuda = (backend == at::Backend::CUDA || backend == at::Backend::SparseCUDA);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch) throw python_error();

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) throw python_error();

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) throw python_error();

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

static std::vector<PyTensorType> tensor_types;

void initialize_python_bindings() {
  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor, etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(tensor_type.py_type, tensor_type.name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_nested_tensor_types(const std::vector<PyNestedTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  auto tensor_classes = THPObjectPtr(PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) throw python_error();

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

}
}
