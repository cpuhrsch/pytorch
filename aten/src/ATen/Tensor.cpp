#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#include <iostream>

namespace at {

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

// std::unique_ptr<Storage> Tensor::storage() const {
//   auto storage = THTensor_getStoragePtr(tensor);
// StorageImpl* storage = tensor->
// storage->retain();
// //  THStorage_retain(storage);
//   return std::unique_ptr<Storage>(new Storage(storage));
// }

} // namespace at
