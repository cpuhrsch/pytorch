#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <pybind11/pybind11.h>

using namespace std;

namespace py = pybind11;

struct NestedTensor : torch::jit::CustomClassHolder {
  int x, y;
  NestedTensor(): x(0), y(0){}
  NestedTensor(int x_, int y_) : x(x_), y(y_) {}
  int64_t info() {
    return this->x * this->y;
  }
  int64_t add(int64_t z) {
    return (x+y)*z;
  }
  void increment(int64_t z) {
    this->x+=z;
    this->y+=z;
  }
  int64_t combine(c10::intrusive_ptr<NestedTensor> b) {
    return this->info() + b->info();
  }
  ~NestedTensor() {
    // std::cout<<"Destroying object with values: "<<x<<' '<<y<<std::endl;
  }
};

static auto test = torch::jit::class_<NestedTensor>("NestedTensor")
                    .def(torch::jit::init<int64_t, int64_t>())
                    // .def(torch::jit::init<>())
                    .def("info", &NestedTensor::info)
                    .def("increment", &NestedTensor::increment)
                    // .def("add", &NestedTensor::add);
                    .def("combine", &NestedTensor::combine)
                    ;
