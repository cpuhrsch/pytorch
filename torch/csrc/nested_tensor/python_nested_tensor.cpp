#include <torch/csrc/nested_tensor/python_nested_tensor.h>
#include <torch/csrc/tensor/python_tensor.h>

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

namespace torch {
namespace nested_tensor {

using namespace at;
using namespace torch::autograd;
namespace py = pybind11;

// PyObject *_ListNestedTensorVariableClass = nullptr;

// template <class F> std::vector<py::object> map_fn(_ListNestedTensor nt, F fn)
// {
//  std::vector<py::object> result;
//  if (nt.nested_dim() == 1) {
//    for (_NestedNode node : nt.get_structure()._children) {
//      result.push_back(fn(node._tensor_node._tensor));
//    }
//  } else {
//    for (_ListNestedTensor nti : nt.get_structure()._children) {
//      result.push_back(py::cast(map_fn(nti, fn)));
//    }
//  }
//  return result;
// }

// std::vector<py::object> _ListNestedTensor::nested_size() {
//   return map_fn(*this, [](at::Tensor tensor) -> py::object {
//     return py::reinterpret_borrow<py::object>(
//         torch::autograd::utils::wrap(tensor.sizes()));
//   });
// }

// std::vector<py::object> _ListNestedTensor::nested_stride() {
//   return map_fn(*this, [](at::Tensor tensor) -> py::object {
//     return py::reinterpret_borrow<py::object>(
//         torch::autograd::utils::wrap(tensor.strides()));
//   });
// }

// std::vector<py::object> _ListNestedTensor::unbind() {
//   std::vector<py::object> result;
//   if (nested_dim() == 1) {
//     for (_NestedNode node : _structure._children) {
//       result.push_back(py::reinterpret_borrow<py::object>(
//           torch::autograd::utils::wrap(node._tensor_node._tensor)));
//     }
//   } else {
//     for (_NestedNode node : _structure._children) {
//       result.push_back(py::cast(_ListNestedTensor(node)));
//     }
//   }
//   return result;
// }

// std::string _ListNestedTensor::__str__() {
//   std::stringstream result;
//   if (nested_dim() == 1) {
//     for (_NestedNode node : _structure._children) {
//       result << "  ";
//       result << node._tensor_node._tensor;
//       result << ",";
//       result << std::endl;
//     }
//   } else {
//     for (_NestedNode node : _structure._children) {
//       _ListNestedTensor nt(node);
//       result << "  ";
//       result << nt.__str__();
//       result << ",";
//       result << std::endl;
//     }
//   }
//   result << "])";
//   return result.str();
// }

// std::string _ListNestedTensor::__repr__() {
//   std::stringstream result;
//   if (nested_dim() == 1) {
//     for (_NestedNode node : _structure._children) {
//       result << "  ";
//       // TODO: There appears to be no difference between
//       // __str__ and __repr__ for torch.Tensor.
//       result << node._tensor_node._tensor;
//       result << ",";
//       result << std::endl;
//     }
//   } else {
//     for (_NestedNode node : _structure._children) {
//       _ListNestedTensor nt(node);
//       result << "  ";
//       result << nt.__repr__();
//       result << ",";
//       result << std::endl;
//     }
//   }
//   result << "])";
//   return result.str();
// }

// PyTypeObject *_init_ListNestedTensorVariableTypeObject(PyTypeObject &type) {
//   // TODO: Necessary for NestedTensor as well?
//   // NOTE: we don't use the typical static declaration of PyTypeObject
//   because
//   // we need to initialize as many types as there are VariableType instances.
//   // The typical PyVarObject_HEAD_INIT(nullptr, 0) is described in the Python
//   // documentation: it initializes the refcnt to 1 and the other object
//   header
//   // fields to zero.
//   memset(&type, 0, sizeof(PyTypeObject));
//   ((PyObject *)&type)->ob_refcnt = 1;
//   ((PyObject *)&type)->ob_type = &_ListNestedTensorVariableType;
//   type.tp_basicsize = sizeof(_ListNestedTensorVariableType);
//   type.tp_flags = Py_TPFLAGS_DEFAULT;
//   type.tp_name = "torch._ListNestedTensorVariable";
//   type.tp_new = PyType_GenericNew;
//
//   // type.tp_doc = "Custom objects";
//   // type.tp_itemsize = 0;
//
//   // type.tp_basicsize = sizeof(_ListNestedTensorVariableType);
//   // type.tp_flags = Py_TPFLAGS_DEFAULT;
//   // type.tp_name = "_ListNestedTensorVariable";
//   // type.tp_new = PyType_GenericNew;
//   if (PyType_Ready(&type) < 0) {
//     throw python_error();
//   }
//   return &type;
// }
//
static void _ListNestedTensorVariable_dealloc(_ListNestedTensorVariable *self) {
  // PyObject_GC_UnTrack(self);
  // THPVariable_clear(self);
  self->cdata.~_ListNestedTensor();
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static PyObject* _ListNestedTensorVariable_NewWithVar(PyTypeObject* type, _ListNestedTensor nested_tensor) 
{
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (_ListNestedTensorVariable *)obj;
    new (&v->cdata) _ListNestedTensor(std::move(nested_tensor));
    // v->cdata.set_pyobj(obj);
    return obj;
  } else {
    throw python_error();
  }
}

static PyObject *_ListNestedTensorVariable_pynew(PyTypeObject *type,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  PyObject *listObj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj)) {
    throw std::runtime_error("invalid arguments");
  }
  _NestedNode structure = _get_structure(listObj);
  auto nested_tensor = _ListNestedTensor(structure);

  return _ListNestedTensorVariable_NewWithVar(type, nested_tensor);

  // PyObject *obj = type->tp_alloc(type, 0);
  // if (obj) {
  //   auto v = (_ListNestedTensorVariable *)obj;
  //   new (&v->cdata) _ListNestedTensor(std::move(nested_tensor));
  //   // v->cdata.set_pyobj(obj);
  //   return obj;
  // } else {
  //   throw python_error();
  // }
  // return THPVariable_NewWithVar(type, std::move(tensor));
  // throw std::runtime_error("new(): invalid arguments");
}

// inline Tensor dispatch_sigmoid(const Tensor & self, Tensor out) {
//
//   AutoNoGIL no_gil;
//   return at::sigmoid_out(out, self);
// }
//
// static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args)
// {
//   HANDLE_TH_ERRORS
//
//   auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
//   return wrap(dispatch_sigmoid(self));
//   END_HANDLE_TH_ERRORS
// }
//
PyObject *_ListNestedTensorVariable_Wrap(_ListNestedTensor var)
{
  return _ListNestedTensorVariable_NewWithVar((PyTypeObject *)_ListNestedTensorVariableClass, std::move(var));
}

static PyObject *_ListNestedTensorVariable_element_size(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.element_size());
}

static PyObject *_ListNestedTensorVariable_pin_memory(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.pin_memory());
}

static PyObject *_ListNestedTensorVariable_grad(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.grad());
}

static PyObject *_ListNestedTensorVariable_detach(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.detach());
}

static PyMethodDef _ListNestedTensorVariable_methods[] = {
    {"element_size", (PyCFunction)_ListNestedTensorVariable_element_size,
     METH_NOARGS, "Return element size."},
    {NULL} /* Sentinel */
};

// TODO: "Py_TPFLAGS_DEFAULT enables all memebers defined until Python 3.3"
// https://docs.python.org/3/extending/newtypes_tutorial.html
// Does that mean it won't work before Python 3.3?
PyTypeObject _ListNestedTensorVariableType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._ListNestedTensor", /* tp_name */
    sizeof(_ListNestedTensorVariable),             /* tp_basicsize */
    0,                                             /* tp_itemsize */
    (destructor)_ListNestedTensorVariable_dealloc, /* tp_dealloc */
    nullptr,                                       /* tp_print */
    nullptr,                                       /* tp_getattr */
    nullptr,                                       /* tp_setattr */
    nullptr,                                       /* tp_reserved */
    nullptr,                                       /* tp_repr */
    nullptr,                                       /* tp_as_number */
    nullptr,                                       /* tp_as_sequence */
    nullptr,                                       /* tp_as_mapping */
    nullptr,                                       /* tp_hash  */
    nullptr,                                       /* tp_call */
    nullptr,                                       /* tp_str */
    nullptr,                                       /* tp_getattro */
    nullptr,                                       /* tp_setattro */
    nullptr,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                            /* tp_flags */
    nullptr,                                       /* tp_doc */
    nullptr,                                       /* tp_traverse */
    nullptr,                                       /* tp_clear */
    nullptr,                                       /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    nullptr,                                       /* tp_iter */
    nullptr,                                       /* tp_iternext */
    _ListNestedTensorVariable_methods,             /* tp_methods */
    nullptr,                                       /* tp_members */
    nullptr,                                       /* tp_getset */
    nullptr,                                       /* tp_base */
    nullptr,                                       /* tp_dict */
    nullptr,                                       /* tp_descr_get */
    nullptr,                                       /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    nullptr,                                       /* tp_init */
    nullptr,                                       /* tp_alloc */
    _ListNestedTensorVariable_pynew,               /* tp_new */
};

void initialize_python_bindings() {
  PyObject *m;
  if (PyType_Ready(&_ListNestedTensorVariableType) < 0) {
    throw python_error();
  }

  m = PyImport_ImportModule("torch");
  if (!m) {
    throw python_error();
  }

  Py_INCREF(&_ListNestedTensorVariableType);
  if (PyModule_AddObject(m, "_ListNestedTensor",
                         (PyObject *)&_ListNestedTensorVariableType) < 0) {
    Py_DECREF(&_ListNestedTensorVariableType);
    Py_DECREF(m);
    throw python_error();
  }

  // auto obj = py::module::import("torch");
  // auto m = py::handle(obj).cast<py::module>();

  // py::class_<_ListNestedTensor>(m, "_ListNestedTensor")
  //     .def(py::init<std::vector<py::object>>())
  //     .def_property_readonly("dtype", &_ListNestedTensor::get_dtype)
  //     .def_property_readonly("layout", &_ListNestedTensor::get_layout)
  //     .def_property_readonly("device", &_ListNestedTensor::get_device)
  //     .def_property_readonly("requires_grad",
  //     &_ListNestedTensor::requires_grad)
  //     .def_property_readonly("grad", &_ListNestedTensor::grad)
  //     .def("detach", &_ListNestedTensor::detach)
  //     .def("pin_memory", &_ListNestedTensor::pin_memory)
  //     .def("backward", &_ListNestedTensor::backward)
  //     .def("requires_grad_", &_ListNestedTensor::requires_grad_)
  //     .def("element_size", &_ListNestedTensor::element_size)
  //     .def("size", &_ListNestedTensor::size)
  //     .def("unbind", &_ListNestedTensor::unbind)
  //     .def("nested_size", &_ListNestedTensor::nested_size)
  //     .def("nested_stride", &_ListNestedTensor::nested_stride)
  //     .def("is_pinned", &_ListNestedTensor::is_contiguous)
  //     .def("is_contiguous", &_ListNestedTensor::is_contiguous)
  //     .def("__len__", &_ListNestedTensor::__len__)
  //     .def("__str__", &_ListNestedTensor::__str__)
  //     .def("dim", &_ListNestedTensor::dim)
  //     .def("numel", &_ListNestedTensor::numel)
  //     .def("nested_dim", &_ListNestedTensor::nested_dim);
}
}
}
