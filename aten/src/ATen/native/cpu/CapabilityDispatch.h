#include "ATen/cpu/cpuinfo/include/cpuinfo.h"
#include <type_traits>
#include <iostream>

namespace at {
namespace native {

enum class CPUCapability { DEFAULT, AVX, AVX2 };

#if defined(CPUCAPABILITYDEFAULT)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::DEFAULT;
#elif defined(CPUCAPABILITYAVX)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX;
#elif defined(CPUCAPABILITYAVX2)
constexpr CPUCapability CURRENT_CAPABILITY = CPUCapability::AVX2;
#endif

template <typename FnType> struct DispatchStub {};

template <typename... ArgTypes>
struct DispatchStub<void(ArgTypes...)> {
  using FnType = void(ArgTypes...);

  template <template <CPUCapability, class...Args> class allImpl,
            FnType **dispatch_ptr>
  static void init(ArgTypes... args) {
    *dispatch_ptr = allImpl<CPUCapability::DEFAULT, Args...>::function;
    // Check if platform is supported
    if (cpuinfo_initialize()) {
      // Set function pointer to best implementation last
#if defined(HAVE_AVX_CPU_DEFINITION)
      if (cpuinfo_has_x86_avx()) {
        *dispatch_ptr = allImpl<CPUCapability::AVX, Args...>::function;
      }
#endif
#if defined(HAVE_AVX2_CPU_DEFINITION)
      if (cpuinfo_has_x86_avx2()) {
        *dispatch_ptr = allImpl<CPUCapability::AVX2, Args...>::function;
      }
#endif
    }
    (*dispatch_ptr)(args...);
  }
};

}
}
