#pragma once
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

template <CPUCapability C>
struct ceilImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct floorImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct roundImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct truncImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct sqrtImplC {
  static void function(Tensor& result, const Tensor& self);
};

// These functions will be deleted once vectorized (if useful)

#define UNARY_OPS_MACRO(MACRO) \
  MACRO (acos, std::acos) \
  MACRO (asin, std::asin) \
  MACRO (atan, std::atan) \
  MACRO (cos, std::cos) \
  MACRO (cosh, std::cosh) \
  MACRO (erf, std::erf) \
  MACRO (exp, std::exp) \
  MACRO (expm1, std::expm1) \
  MACRO (lgamma, std::lgamma) \
  MACRO (log1p, std::log1p) \
  MACRO (log, std::log) \
  MACRO (sin, std::sin) \
  MACRO (sinh, std::sinh) \
  MACRO (tan, std::tan) \
  MACRO (tanh, std::tanh) \

#define HEADER(NAME, FUNC) \
template <CPUCapability C> \
struct NAME ## ImplC { \
  static void function(Tensor& result, const Tensor& self); \
}; \

UNARY_OPS_MACRO(HEADER)

// Missing unary functions (not part of cmath and not vectorized)
// MACRO (frac, std::frac)
// MACRO (rsqrt, std::rsqrt)
// MACRO (digamma, std::digamma)
// MACRO (erfinv, std::erfinv)
// MACRO (sigmoid, std::sigmoid)

}} // namespace at::native
