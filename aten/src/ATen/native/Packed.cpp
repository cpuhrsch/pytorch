// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/CPUApplyUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {
namespace native {

PackedTensor add_packed(PackedTensor input, Tensor input2){
  std::cout << "HEEE" << std::endl;
  return input;
}

}
} // namespace at
