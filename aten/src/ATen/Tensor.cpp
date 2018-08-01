#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Storage.h>
#include <ATen/Scalar.h>
#include <ATen/Half.h>

// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <iostream>

namespace at {

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::unique_ptr<Storage> Tensor::storage() {
  auto storage = THTensor_getStoragePtr(tensor);
  THStorage_retain(storage);
  return std::unique_ptr<Storage>(new Storage(storage));
}

} // namespace at
