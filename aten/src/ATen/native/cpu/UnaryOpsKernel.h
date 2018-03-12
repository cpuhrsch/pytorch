#pragma once
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include "CapabilityDispatch.h"
#include <stdexcept>


namespace at {
namespace native {

// template <CPUCapability C> struct acos_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct asin_ImplC { static void function(Tensor &self); };
template <CPUCapability C> struct atanImplC { static void function(Tensor & result, const Tensor &self); };
// template <CPUCapability C> struct ceil_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct cos_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct cosh_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct digamma_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct erf_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct erfinv_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct exp_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct expm1_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct floor_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct frac_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct lgamma_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct log1p_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct log_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct round_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct rsqrt_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct sigmoid_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct sin_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct sinh_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct sqrt_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct tan_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct tanh_ImplC { static void function(Tensor &self); };
// template <CPUCapability C> struct trunc_ImplC { static void function(Tensor &self); };

}
}
