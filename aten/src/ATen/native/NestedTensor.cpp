#include <ATen/NestedTensor.h>

namespace at { namespace native {

Tensor _make_nested(const TensorList& self) {
  return makeNested(self);
}

}}
