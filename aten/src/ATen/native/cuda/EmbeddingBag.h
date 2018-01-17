
#include "ATen/ATen.h"
#include "ATen/Check.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"

#include "ATen/cuda/AccumulateType.h"

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
namespace at {
namespace native {
Tensor embedding_bag_backward_cuda(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offsets,
                                   const Tensor &offset2bag,
                                   int64_t num_weights, bool scale_grad_by_freq,
                                   int64_t mode);
}}
