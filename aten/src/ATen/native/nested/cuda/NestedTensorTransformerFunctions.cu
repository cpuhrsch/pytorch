#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef USE_ROCM
#ifndef _WIN32
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#endif
#endif

#include <ATen/NestedTensorImpl.h>

#include <limits>
#include <ATen/cuda/cub.cuh>
#include <ATen/native/nested/NestedTensorMath.h>
#include <iostream>

#define BLOCK_DIM 256
#define GRID_DIM_Y 16

namespace at {
namespace native {

template <typename T>
__global__ void remove_padding_transform0213_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);

    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i2 = i / sizes_i[1];
    const int i13 = i % sizes_i[1];
    const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
    const int i3 = i13 % (sizes_i[1] / input_sizes[1]);
    output[offset + i] = input
        [input_offset + i1 * input_sizes[2] * input_sizes[3] +
         i2 * input_sizes[3] + i3];
  }
}

template <typename T>
__global__ void remove_padding_2(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  int input_offset = batch_id * input_sizes[1] * input_sizes[2];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / sizes_i[1];
    const int i1 = i % sizes_i[1];
    const int i0_offset = i0 * input_sizes[2];
    output[offset + i] = input[input_offset + i0_offset + i1];
  }
}

template <typename T>
__global__ void remove_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  int input_offset =
      batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
  for (int ii = 0; ii < (numel_i / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
  const int i = (numel_i / grainsize) * grainsize + tid;
  if (i < numel_i) {
    const int i0 = i / (sizes_i[1] * sizes_i[2]);
    const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
    const int i2 = i % sizes_i[2];
    const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
    const int i1_offset = i1 * input_sizes[3];
    output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
  }
}

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  if (output_dim == 2) {
    remove_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  } else {
    remove_padding<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
  }
}

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  dim3 grid;
  grid.x = batch_size;
  grid.y = GRID_DIM_Y;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  remove_padding_transform0213_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

template void remove_padding_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_transform0213_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template void remove_padding_transform0213_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template <typename T>
__global__ void add_padding_1(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int batch_output_offset = batch_id * output_sizes_1;
  for (int ii = 0; ii < (output_sizes_1 / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && i < sizes_i[0]) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
  const int i = (output_sizes_1 / grainsize) * grainsize + tid;
  if (i < output_sizes_1) {
    const int output_offset = batch_output_offset + i;
    if (batch_id < batch_size && (i < sizes_i[0])) {
      const int batch_input_offset = offsets[batch_id];
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_2(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset = batch_id * output_sizes_1 * output_sizes_2;
  const int output_numel = output_sizes_1 * output_sizes_2;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2);
    const int i1 = i - i0 * output_sizes_2;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int offset = offsets[batch_id];
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
__global__ void add_padding_3(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    int output_sizes_1,
    int output_sizes_2,
    int output_sizes_3,
    const int batch_size) {
  const int batch_id = blockIdx.x;
  const int grid_id = blockIdx.y;
  const int tid = threadIdx.x + grid_id * BLOCK_DIM;
  const int grainsize = GRID_DIM_Y * BLOCK_DIM;
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int output_offset =
      batch_id * output_sizes_1 * output_sizes_2 * output_sizes_3;
  const int output_numel = output_sizes_1 * output_sizes_2 * output_sizes_3;
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes_2 * output_sizes_3);
    const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
    const int i2 = i % output_sizes_3;
    if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
        i2 < sizes_i[2]) {
      const int offset = offsets[batch_id];
      const int input_offset =
          offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template <typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  dim3 grid;
  grid.x = output_batch_size;
  grid.y = GRID_DIM_Y;
  if (input_dim == 1) {
    add_padding_1<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        batch_size);
  }
  if (input_dim == 2) {
    add_padding_2<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        batch_size);
  }
  if (input_dim == 3) {
    add_padding_3<T><<<grid, BLOCK_DIM, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        output_sizes[3],
        batch_size);
  }
}

template void add_padding_kernelLauncher<double>(
    double* input,
    double* output,
    double padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    float padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    c10::Half padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

namespace {

#ifndef USE_ROCM
#ifndef _WIN32
template <
    typename scalar_t,
    unsigned int kPad,
    typename LayoutA,
    typename LayoutB,
    typename OpClass,
    typename Arch,
    typename ThreadBlockShape,
    typename WarpShape,
    typename InstructionShape>
void gemm_grouped_cuda_internal(
    const std::vector<int64_t>& lda,
    const std::vector<int64_t>& ldb,
    const std::vector<int64_t>& ldd,
    const std::vector<scalar_t*>& aptr,
    const std::vector<scalar_t*>& bptr,
    const std::vector<scalar_t*>& dptr,
    const std::vector<cutlass::gemm::GemmCoord>& gemm_sizes,
    const int problem_count,
    at::Device& device) {
  using Element = scalar_t;
  using ElementAcc = float;

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          Arch,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      Arch,
      ThreadBlockShape,
      WarpShape,
      InstructionShape,
      typename GemmConfiguration::EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      GemmConfiguration::kStages>::GemmKernel;

  using GemmGrouped = typename cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(/*alpha*/ 1, /*beta*/ 0);

  const int64_t gemm_coord_size =
      problem_count * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
  // Number of gmm args not including *problem_sizes
  at::Tensor gmm_args = at::empty(
      {problem_count * 6 + gemm_coord_size},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  // Obtain pointers for each argument (on host)
  int64_t* lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  int64_t* ldb_data = lda_data + problem_count;
  int64_t* ldd_data = lda_data + 2 * problem_count;
  int64_t* ptr_a_data = lda_data + 3 * problem_count;
  int64_t* ptr_b_data = lda_data + 4 * problem_count;
  int64_t* ptr_d_data = lda_data + 5 * problem_count;
  cutlass::gemm::GemmCoord* problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Set arguments into gmm_args from input args
  for (int i = 0; i < problem_count; ++i) {
    problem_sizes_data[i] = gemm_sizes[i];
    lda_data[i] = lda[i];
    ldb_data[i] = ldb[i];
    ldd_data[i] = ldd[i];
    ptr_a_data[i] = reinterpret_cast<int64_t>(aptr[i]);
    ptr_b_data[i] = reinterpret_cast<int64_t>(bptr[i]);
    ptr_d_data[i] = reinterpret_cast<int64_t>(dptr[i]);
  }
  const int threadblock_count =
      GemmGrouped::sufficient(problem_sizes_data, problem_count);

  // Transfer arguments to GPU
  gmm_args = gmm_args.to(device, true);

  // Obtain pointers for each of arguments (on GPU)
  lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  ldb_data = lda_data + problem_count;
  ldd_data = lda_data + 2 * problem_count;
  ptr_a_data = lda_data + 3 * problem_count;
  ptr_b_data = lda_data + 4 * problem_count;
  ptr_d_data = lda_data + 5 * problem_count;
  problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Create GemmGrouped::Arguments using the arguments prepared above
  typename GemmGrouped::Arguments args(
      problem_sizes_data,
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<Element**>(ptr_a_data),
      reinterpret_cast<Element**>(ptr_b_data),
      reinterpret_cast<Element**>(ptr_d_data),
      reinterpret_cast<Element**>(ptr_d_data),
      lda_data,
      ldb_data,
      ldd_data,
      ldd_data);

  GemmGrouped gemm;
  cutlass::Status status =
      gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

  // Run CUTLASS group GEMM
  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool group_gemm_dispatch(
    at::ScalarType scalar_type,
    at::Device device,
    Tensor a_buffer,
    std::vector<int64_t> a_offsets,
    Tensor b_buffer,
    std::vector<int64_t> b_offsets,
    Tensor d_buffer,
    std::vector<int64_t> d_offsets,
    std::vector<int64_t> lda,
    std::vector<int64_t> ldb,
    std::vector<int64_t> ldd,
    std::vector<cutlass::gemm::GemmCoord> gemm_sizes,
    int64_t ntensors,
    bool all_row_major) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;

  if (!all_row_major) {
    return false;
  }

  // Check alignment
  bool all_pad_8 = true;
  for (int i = 0; i < ntensors; i++) {
    all_pad_8 = all_pad_8 && (gemm_sizes[i].n() % 8 == 0);
    all_pad_8 = all_pad_8 && (gemm_sizes[i].k() % 8 == 0);

    // Not sure if this is a requirement, on the safe side
    all_pad_8 = all_pad_8 && (lda[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldb[i] % 8 == 0);
    all_pad_8 = all_pad_8 && (ldd[i] % 8 == 0);
  }

  /* Float case */
  if (scalar_type == at::kFloat) {
    std::vector<float*> aptr;
    std::vector<float*> bptr;
    std::vector<float*> dptr;
    for (int64_t i = 0; i < ntensors; i++) {
      aptr.push_back(a_buffer.data_ptr<float>() + a_offsets[i]);
      bptr.push_back(b_buffer.data_ptr<float>() + b_offsets[i]);
      dptr.push_back(d_buffer.data_ptr<float>() + d_offsets[i]);
    }
    if (is_sm8x) {
      gemm_grouped_cuda_internal<
          float,
          1,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor,
          cutlass::arch::OpClassSimt,
          cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<128, 128, 8>,
          cutlass::gemm::GemmShape<64, 32, 8>,
          cutlass::gemm::GemmShape<1, 1, 1>>(
          lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
      return true;
    }
  }
  /* Half case */
  else if (scalar_type == at::kHalf) {
    std::vector<cutlass::half_t*> aptr;
    std::vector<cutlass::half_t*> bptr;
    std::vector<cutlass::half_t*> dptr;
    for (int64_t i = 0; i < ntensors; i++) {
      aptr.push_back(reinterpret_cast<cutlass::half_t*>(
          a_buffer.data_ptr<c10::Half>() + a_offsets[i]));
      bptr.push_back(reinterpret_cast<cutlass::half_t*>(
          b_buffer.data_ptr<c10::Half>() + b_offsets[i]));
      dptr.push_back(reinterpret_cast<cutlass::half_t*>(
          d_buffer.data_ptr<c10::Half>() + d_offsets[i]));
    }
    if (is_sm8x) {
      if (all_pad_8) {
        gemm_grouped_cuda_internal<
            cutlass::half_t,
            8,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>>(
            lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
        return true;
      } else {
        gemm_grouped_cuda_internal<
            cutlass::half_t,
            1,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<64, 32, 8>,
            cutlass::gemm::GemmShape<1, 1, 1>>(
            lda, ldb, ldd, aptr, bptr, dptr, gemm_sizes, ntensors, device);
        return true;
      }
    }
  }

  // Did not perform GEMM
  return false;
}
#endif
#endif

} // namespace

Tensor bmm_nested_cuda(const Tensor& self, const Tensor& mat2) {
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = get_nested_tensor_impl(self);
  auto mat2_ptr = get_nested_tensor_impl(mat2);
  TORCH_CHECK(self_ptr->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_ptr->dim() == 3, "batch2 must be a 3D tensor");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");
  const Tensor &self_buffer = self_ptr->get_unsafe_storage_as_tensor();
  const Tensor &mat2_buffer = mat2_ptr->get_unsafe_storage_as_tensor();
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> mat2_sizes = NestedTensor_get_sizes(mat2_ptr);
  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>& self_offsets = self_ptr->get_storage_offsets();
  const std::vector<int64_t>& mat2_offsets = mat2_ptr->get_storage_offsets();

  // create a contiguous output
  int64_t out_numel = 0;
  int64_t a_numel = 0;
  int64_t b_numel = 0;
  const Tensor& self_sizemat = self_ptr->get_nested_size_tensor();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  std::vector<int64_t> output_offsets;
  std::vector<int64_t> a_offsets;
  std::vector<int64_t> b_offsets;
  std::vector<int64_t> lda;
  std::vector<int64_t> ldb;
  std::vector<int64_t> ldd;
#ifndef USE_ROCM
#ifndef _WIN32
  std::vector<cutlass::gemm::GemmCoord> gemm_sizes;
#endif
#endif
  bool all_row_major = true;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = self_sizes[i], &mat2_shape = mat2_sizes[i];
    const int64_t &self_size0 = self_shape[0], &self_size1 = self_shape[1],
                  &mat2_size0 = mat2_shape[0], &mat2_size1 = mat2_shape[1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        i,
        "-th nested matrices in batch cannot be multiplied (",
        self_size0,
        "x",
        self_size1,
        " and ",
        mat2_size0,
        "x",
        mat2_size1,
        ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    output_offsets.push_back(out_numel);
    out_numel += self_size0 * mat2_size1;
#ifndef USE_ROCM
#ifndef _WIN32
    gemm_sizes.push_back(
        cutlass::gemm::GemmCoord(self_size0, mat2_size1, self_size1));
#endif
#endif
    lda.push_back(self_strides[i][0]);
    ldb.push_back(mat2_strides[i][0]);
    ldd.push_back(mat2_size1);
    a_offsets.push_back(self_offsets[i]);
    b_offsets.push_back(mat2_offsets[i]);
    a_numel += self_size0 * self_strides[i][0];
    b_numel += mat2_size0 * mat2_strides[i][0];
    all_row_major = all_row_major && (self_strides[i][1] == 1);
    all_row_major = all_row_major && (mat2_strides[i][1] == 1);
  }
  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);

#ifndef USE_ROCM
#ifndef _WIN32
  bool success = group_gemm_dispatch(
      self.scalar_type(),
      output.device(),
      self_buffer,
      a_offsets,
      mat2_buffer,
      b_offsets,
      out_buffer,
      output_offsets,
      lda,
      ldb,
      ldd,
      gemm_sizes,
      ntensors,
      all_row_major);
  if (success) {
    return output;
  }
#endif
#endif
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(
        output_unbind[i],
        self_buffer.as_strided(self_sizes[i], self_strides[i], self_offsets[i]),
        mat2_buffer.as_strided(
            mat2_sizes[i], mat2_strides[i], mat2_offsets[i]));
  }
  return output;
}

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1) {
    T tmp = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    val = val > tmp ? val : tmp;
  }
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}


template<typename T>
struct ComputeType {
  using type = T;
};

template<>
struct ComputeType<c10::Half> {
  using type = float;
};


#define MAX_SEQ_LEN 256
// Assumes that it's given a variably shapes
// list of batched small square matrices for last dim softmax.
// Maximum size for square matrices is 256 by 256.
template <typename T, typename compute_type>
__global__ void softmax_squares(
    const T* input,
    T* output,
    const int64_t batch_size,
    const int64_t num_heads,
    const int32_t* batch_offsets,
    const int64_t* seq_lens) {
  const int64_t seq_len = seq_lens[blockIdx.x];
  int64_t j = blockIdx.z;

  compute_type thread_val;
  compute_type thread_max = -std::numeric_limits<compute_type>::max();
  compute_type thread_sum = 0;

  const int64_t seq_len_squared = seq_len * seq_len;
  if (j < seq_len && threadIdx.x < seq_len) {
    j = j * seq_len;
    int64_t idx = threadIdx.x + j;
    int32_t batch_offset = batch_offsets[blockIdx.x];
    const T* input_ptr = input + batch_offset;
    T* output_ptr = output + batch_offset;
    // Size of entire square matrix
    // One thread per sequence entry. At most 256 sequence length.
    // If a thread is out of bounds, we don't execute.

    int64_t i = blockIdx.y;

    const T* head_offset = i * seq_len_squared + input_ptr;
    T* head_offset_out = i * seq_len_squared + output_ptr;



    // The thread's value.
    thread_val = head_offset[idx];
    // Find the max across the threads and share it.
    thread_max = thread_val;
    thread_max = blockReduceMax(thread_max);
    // Now every thread has access to the max.
    // Subtract the max from the data in shared mem and apply exp
    compute_type tmp = std::exp(thread_val - thread_max);
    thread_sum = tmp;
    // Now we have to sum all of this together
    thread_sum = blockReduceSum(thread_sum);
    // Now we divide the value in shared memory by this sum
    // and write out the result into shared memory
    head_offset_out[idx] = tmp / thread_sum;
  }
}

template <typename T>
void softmax_kernelLauncher(
    const T* input,
    T* output,
    const int64_t batch_size,
    const int64_t num_heads,
    int32_t* sample_size_ptr,
    int64_t* seq_lens) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  // TODO: Optimize settings by getting device properties
  dim3 grid_dim;
  grid_dim.x = batch_size;
  grid_dim.y = num_heads;
  grid_dim.z = 256; // Max seq length
  using compute_type = typename ComputeType<T>::type;
  softmax_squares<T, compute_type><<<grid_dim, 256, 0, stream>>>(
      input, output, batch_size, num_heads, sample_size_ptr, seq_lens);
}

std::tuple<Tensor, int64_t> cumulative_and_max_seq_len2(int64_t num_heads,
    const Tensor& sizes) {
  auto size_tensor_stride = sizes.stride(0);

  const int64_t batch_size = sizes.size(0);
  auto cumulative_seqlen = at::zeros(
      {batch_size + 1}, TensorOptions().device(at::kCPU).dtype(at::kInt));

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  auto* cumulative_seqlen_ptr = cumulative_seqlen.data_ptr<int32_t>();

  int32_t sum = 0;
  int64_t max_seqlen = -1;
  cumulative_seqlen_ptr[0] = sum;
  for (const auto i : c10::irange(batch_size)) {
    // Calculate the cumulative sum of the sequence lengths
    auto current_seq_len = sizes_ptr[(i * size_tensor_stride) + 1];
    // Processing batches of squares matrices
    current_seq_len = current_seq_len * current_seq_len * num_heads;
    sum += current_seq_len;
    cumulative_seqlen_ptr[i + 1] = sum;

    // Find the max element while we traverse
    max_seqlen = std::max(max_seqlen, current_seq_len);
  }
  // Send to GPU, this is pretty light weight calc for normal batch size
  // but maybe this needs to be on gpu
  cumulative_seqlen = cumulative_seqlen.to(TensorOptions().device(at::kCUDA));
  return std::tuple<Tensor, int64_t>{cumulative_seqlen, max_seqlen};
}

Tensor NestedTensor_softmax_cuda(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
  const auto* query_nt = get_nested_tensor_impl_or_null(input);
  if (!(
        input.dim() == 4 &&
        query_nt->opt_size(1) &&
        dim == -1
        )
      ) {
    return NestedTensor_softmax_generic(input, dim, half_to_float);
  }
  auto sizes = query_nt->get_nested_size_tensor();
  const int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes.size(0); i++) {
    // If the sizes aren't compromised of small squares, we can't run our special kernel.
    if (!(sizes_ptr[i + 3 + 1] < 256 &&
          sizes_ptr[i * 3 + 1] == sizes_ptr[i * 3 + 2])) {
      return NestedTensor_softmax_generic(input, dim, half_to_float);
    }
  }
  auto input_buffer = get_buffer(input);
  auto sizes_dim2 = at::native::narrow_symint(sizes, 1, 1, 1);
  auto sizes_dim2_cuda = sizes_dim2.contiguous().to(TensorOptions().device(at::kCUDA));

  // TORCH_CHECK(input.contiguous(), "Need input to be contiguous.");
  // TODO: "Extra checks needed:
  // size(1) has to be regular
  // size(2) needs to equal size(3)
  // sequence length needs to be below 256.

  const int64_t num_tensors = input.size(0);
  const int64_t num_heads = input.size(1);
  auto cumulative_and_max_q = cumulative_and_max_seq_len2(num_heads, sizes);
  Tensor cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q);

  auto output = input.clone();
  auto output_buffer = get_buffer(output);

  if (input.dtype() == kDouble) {
    softmax_kernelLauncher(
        input_buffer.data_ptr<double>(),
        output_buffer.data_ptr<double>(),
        num_tensors,
        num_heads,
        cumulative_sequence_length_q.data_ptr<int32_t>(),
        sizes_dim2_cuda.data_ptr<int64_t>());
  } else if (input.dtype() == kFloat) {
    softmax_kernelLauncher(
        input_buffer.data_ptr<float>(),
        output_buffer.data_ptr<float>(),
        num_tensors,
        num_heads,
        cumulative_sequence_length_q.data_ptr<int32_t>(),
        sizes_dim2_cuda.data_ptr<int64_t>());
  } else if (input.dtype() == kHalf) {
    softmax_kernelLauncher(
        input_buffer.data_ptr<c10::Half>(),
        output_buffer.data_ptr<c10::Half>(),
        num_tensors,
        num_heads,
        cumulative_sequence_length_q.data_ptr<int32_t>(),
        sizes_dim2_cuda.data_ptr<int64_t>());
  } else {
    AT_ERROR("Only support fp64/fp32/fp16 for softmax_cuda");
  }
  return output;
}

} // namespace native
} // namespace at
