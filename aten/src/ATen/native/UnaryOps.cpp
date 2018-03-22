#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "cpu/UnaryOpsKernel.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {
namespace native {

using unary_type = void(Tensor &, const Tensor &);
//TODO: Turn this into a macro
// unary_type *acosImpl           = DispatchStub<unary_type>::init<acosImplC, &acosImpl>;
// unary_type *asinImpl           = DispatchStub<unary_type>::init<asinImplC, &asinImpl>;
unary_type *atanImpl              = DispatchStub<unary_type>::init<atanImplC, &atanImpl>;
// unary_type *ceilImpl           = DispatchStub<unary_type>::init<ceilImplC, &ceilImpl>;
// unary_type *cosImpl            = DispatchStub<unary_type>::init<cosImplC, &cosImpl>;
// unary_type *coshImpl           = DispatchStub<unary_type>::init<coshImplC, &coshImpl>;
// unary_type *digammaImpl        = DispatchStub<unary_type>::init<digammaImplC, &digammaImpl>;
// unary_type *erfImpl            = DispatchStub<unary_type>::init<erfImplC, &erfImpl>;
// unary_type *erfinvImpl         = DispatchStub<unary_type>::init<erfinvImplC, &erfinvImpl>;
// unary_type *expImpl            = DispatchStub<unary_type>::init<expImplC, &expImpl>;
// unary_type *expm1Impl          = DispatchStub<unary_type>::init<expm1ImplC, &expm1Impl>;
// unary_type *floorImpl          = DispatchStub<unary_type>::init<floorImplC, &floorImpl>;
// unary_type *fracImpl           = DispatchStub<unary_type>::init<fracImplC, &fracImpl>;
// unary_type *lgammaImpl         = DispatchStub<unary_type>::init<lgammaImplC, &lgammaImpl>;
// unary_type *log1pImpl          = DispatchStub<unary_type>::init<log1pImplC, &log1pImpl>;
// unary_type *logImpl            = DispatchStub<unary_type>::init<logImplC, &logImpl>;
// unary_type *roundImpl          = DispatchStub<unary_type>::init<roundImplC, &roundImpl>;
// unary_type *rsqrtImpl          = DispatchStub<unary_type>::init<rsqrtImplC, &rsqrtImpl>;
// unary_type *sigmoidImp         = DispatchStub<unary_type>::init<sigmoidImplC, &sigmoidImpl>;
// unary_type *sinImpl            = DispatchStub<unary_type>::init<sinImplC, &sinImpl>;
// unary_type *sinhImpl           = DispatchStub<unary_type>::init<sinhImplC, &sinhImpl>;
// unary_type *sqrtImpl           = DispatchStub<unary_type>::init<sqrtImplC, &sqrtImpl>;
// unary_type *tanImpl            = DispatchStub<unary_type>::init<tanImplC, &tanImpl>;
// unary_type *tanhImpl           = DispatchStub<unary_type>::init<tanhImplC, &tanhImpl>;
// unary_type *truncImpl          = DispatchStub<unary_type>::init<truncImplC, &truncImpl>;

// WRAP OPS #################################################################

Tensor atan(const Tensor &self) {
  Tensor result = self.type().tensor();
  return at::atan_out(result, self);
}

Tensor & atan_(Tensor &self) {
  return at::atan_out(self, self);
}

Tensor erf(const Tensor &self) {
  Tensor result = self.type().tensor();
  return at::erf_out(result, self);
}

Tensor & erf_(Tensor &self) {
  return at::erf_out(self, self);
}

// \WRAP OPS #################################################################

// TODO: Probably need to specialize in copy and in-place

// CPU OPS #################################################################

template <typename T>
static void _basic_apply(T (*f)(T), T* out, const T * in, size_t numel) {
  std::cerr << "HERE!" << std::endl;
  for (size_t i = 0; i < numel; i ++) {
    out[i] = f(in[i]);
  }
}

Tensor & _atan_out_cpu(Tensor & result, const Tensor & self) {
  if (result.is_contiguous() && self.is_contiguous()) {
    result.resize_(self.sizes());
    atanImpl(result, self);
    return result;
  }
  return at::_atan_out(result, self);
}

Tensor &_erf_out_cpu(Tensor &result, const Tensor &self) {
  if (result.is_contiguous() && self.is_contiguous()) {
    result.resize_(self.sizes());
    result.zero_();
    AT_DISPATCH_FLOATING_TYPES(self.type(), "erf", [&] {
      _basic_apply<scalar_t>(std::erf, result.data<scalar_t>(),
                             self.data<scalar_t>(), self.numel());
    });
    return result;
  }
  return at::_erf_out(result, self);
}

// \CPU OPS ################################################################

// CUDA OPS #################################################################

Tensor & _atan_out_cuda(Tensor & result, const Tensor & self) {
  return at::_atan_out(result, self);
}

Tensor & _erf_out_cuda(Tensor & result, const Tensor & self) {
  return at::_erf_out(result, self);
}

// \CUDA OPS ################################################################
}
}
