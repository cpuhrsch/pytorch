#include <ATen/NestedTensor.h>

namespace at { namespace native {

Tensor _make_nested(Tensor self) {
  return makeNested(self);
}

}}
