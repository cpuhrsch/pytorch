#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <ATen/ATen.h>

#include <torch/csrc/autograd/nested_tensor.h>
#include <torch/csrc/THP_export.h>

// Python object that backs torch.autograd.NestedTensor
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPNestedTensor {
    PyObject_HEAD
    // Payload
    torch::autograd::NestedTensor cdata;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;
};

THP_API PyObject *THPNestedTensorClass;

bool THPNestedTensor_initModule(PyObject *module);
THP_API PyObject * THPNestedTensor_Wrap(torch::autograd::NestedTensor var);

static inline bool THPNestedTensor_CheckExact(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject*)THPNestedTensorClass;
}

inline bool THPNestedTensor_Check(PyObject *obj)
{
  return THPNestedTensorClass && PyObject_IsInstance(obj, THPNestedTensorClass);
}

inline torch::autograd::NestedTensor& THPNestedTensor_Unpack(PyObject* obj) {
  auto var = (THPNestedTensor*)obj;
  return var->cdata;
}
