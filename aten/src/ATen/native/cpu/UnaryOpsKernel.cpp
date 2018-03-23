#include "ATen/native/cpu/UnaryOpsKernel.h"
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include <iostream>
#include <cmath>
#include "ATen/native/cpu/Vec256.h"

namespace at {
namespace native {

using namespace vec256;

// This modifies arr in place with given OP
template <class scalar_t, template <class, CPUCapability> class VOP, template <class, CPUCapability> class SOP, CPUCapability C>
inline void kernel_(scalar_t *arr_out, const scalar_t *arr_in, size_t start, size_t end) {
  // Use all 16 registers.
  Vec256<scalar_t> a[8];
  Vec256<scalar_t> b[8];
  size_t width =
      256 / sizeof(scalar_t); // primitives per 256 bytes (four cache lines)
  size_t epr = 32 / sizeof(scalar_t); // primitives per Vec256
  size_t k = 0;
  for (; k < (end - start) / width; k++) {
    for (size_t i = 0; i < 8; i++) {
      a[i].load(arr_in + (k * width) + i * epr + start);
    }
    for (size_t i = 0; i < 8; i++) {
      b[i] = VOP<scalar_t, C>()(a[i]);
    }
    for (size_t i = 0; i < 8; i++) {
      b[i].store(arr_out + (k * width) + i * epr + start);
    }
  }
  k = k * width + start;
  for (; k < end; k++) {
    arr_out[k] = SOP<scalar_t, C>()(arr_in[k]);
  }
}

template <template <class, CPUCapability> class VOP, template <class, CPUCapability> class SOP, CPUCapability C>
inline void allImpl(Tensor & result, const Tensor &self, const char* name) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), name, [&] {
    at::parallel_for_1d<scalar_t>(&kernel_<scalar_t, VOP, SOP, CURRENT_CAPABILITY>, result, self);
  });
}

template <typename T, CPUCapability C> struct atanVOP {
  Vec256<T> operator()(const Vec256<T> &x) const { return atan(x); }
};

template <typename T, CPUCapability C> struct atanSOP {
  T operator()(const T x) const { return std::atan(x); }
};

template <>
void atanImplC<CURRENT_CAPABILITY>::function(Tensor &result, const Tensor &self) {
  allImpl<atanVOP, atanSOP, CURRENT_CAPABILITY>(result, self, "atan");
}

}
}
