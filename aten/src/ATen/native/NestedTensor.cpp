#include <ATen/NestedTensor.h>

namespace at { namespace native {

Tensor _make_nested(const Tensor& self) {
  return makeNested(self);
}

}}
