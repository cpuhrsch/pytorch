/*
The following source file implements a sparse linear operator using cusparseLt
*/

#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>

#include <iostream>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include <cuda_runtime.h>

#include <typeinfo>
#include <limits>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

// // The code section below describes datatype for input, output matrices and computation between
// // elements in input matrices, which will all be used as template parameters for cutlass::gemm::device::SparseGemm
// using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
// using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
// using ElementOutput = cutlass::half_t;                      // <- data type of elements in output matrix D
// 
// // The code section below describes matrix layout of input and output matrices. Row Major for
// // Matrix A, Column Major for Matrix B and Row Major for Matrix C
// using LayoutInputA = cutlass::layout::RowMajor;
// using LayoutInputB = cutlass::layout::RowMajor;
// using LayoutOutput = cutlass::layout::RowMajor;
// 
// using Gemm = cutlass::gemm::device::SparseGemm<
//     cutlass::half_t,
//     cutlass::layout::RowMajor,
//     cutlass::half_t,
//     cutlass::layout::RowMajor,
//     cutlass::half_t,
//     cutlass::layout::RowMajor,
//     float,
//     cutlass::arch::OpClassTensorOp,
//     cutlass::arch::Sm80,
//     cutlass::gemm::GemmShape<128, 256, 64>,
//     cutlass::gemm::GemmShape<64, 64, 64>,
//     cutlass::gemm::GemmShape<16, 8, 32>,
//     cutlass::epilogue::thread::LinearCombination<
//         cutlass::half_t,
//         128 / cutlass::sizeof_bits<ElementOutput>::value,
//         float,
//         float>,
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//     3>;
// 
// // Data type and layout of meta data matrix E can be inferred from template Gemm.
// using ElementInputE = typename Gemm::ElementE;
// using LayoutInputE = cutlass::layout::RowMajor;
// using ReorderedLayoutInputE = typename Gemm::LayoutE;
// 
// // Below property is defined in include/cutlass/arch/sp_mma_sm80.h
// // 50% Sparsity on Ampere
// constexpr int kSparse = Gemm::kSparse;
// // How many elements of A are covered per ElementE
// constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// // The size of individual meta data
// constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct SparseTestbed {

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  static int const kSparse = Gemm::GemmKernel::kSparse;
  static int const kMetaSizeInBits = Gemm::GemmKernel::kMetaSizeInBits;
  static int const kMaxID2 = Gemm::GemmKernel::kMaxID2;
  static int const kElementsPerElementE = Gemm::GemmKernel::kElementsPerElementE;

  using ElementE = typename Gemm::GemmKernel::ElementE;
  using LayoutE = cutlass::layout::RowMajor;
  using ReorderedLayoutE = typename Gemm::GemmKernel::LayoutE;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_E;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A_uncompressed;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;
  cutlass::HostTensor<ElementE, LayoutE> tensor_E;
  cutlass::HostTensor<ElementE, ReorderedLayoutE> tensor_E_reordered;

  //
  // Methods
  //

  SparseTestbed(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_E_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = 2080)
      : init_A(init_A_),
        init_B(init_B_),
        init_C(init_C_),
        init_E(init_E_),
        seed(seed_) {}

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {
      // TODO: Implement the rest
      // EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //
    tensor_A.resize(cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));
    tensor_A_uncompressed.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);
    tensor_E.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    tensor_E_reordered.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);

    if (init_E == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomSparseMeta(
          tensor_E.host_view(), seed, kMetaSizeInBits);
    } else if (init_E == cutlass::Distribution::Identity) {
      uint32_t content = (kMaxID2 == 1) ? 0x44444444 : 0x4444;
      cutlass::reference::host::TensorFill(tensor_E.host_view(),
                                           (ElementE)(content));
    } else {
      // TODO: Implement the rest
//      EXPECT_TRUE(false);
    }

    cutlass::reorder_meta(tensor_E_reordered.host_ref(), tensor_E.host_ref(),
                          {problem_size.m(), problem_size.n(),
                           problem_size.k() / kSparse / kElementsPerElementE});

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    tensor_C.host_view().at({0, 0}) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_E_reordered.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    tensor_D.sync_host();

    // EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    // EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    // EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    // if (tensor_D.size() > 1)
    //   EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);

    // if (reference_D.size() > 1)
    //   EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    std::cout << "00 passed: " << passed << std::endl;

    // EXPECT_TRUE(passed);

    // if (!passed) {

      std::stringstream fname;

      fname << "error_Gemm_device_" 
        << problem_size.m() << "x"
        << problem_size.n() << "x"
        << problem_size.k() << "_"
        << Gemm::ThreadblockShape::kM << "x"  
        << Gemm::ThreadblockShape::kN << "x"  
        << Gemm::ThreadblockShape::kK << "_"
        << Gemm::WarpShape::kM << "x"  
        << Gemm::WarpShape::kN << "x"  
        << Gemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());

      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nE =\n" << tensor_E.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
//    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    //
    // Verify
    //

    cutlass::uncompress(tensor_A_uncompressed.host_ref(), tensor_A.host_ref(),
                        tensor_E.host_ref(), problem_size.m(), problem_size.k());

    cutlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, 
        ElementCompute,
        ElementAccumulator, typename Gemm::Operator>
        reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A_uncompressed.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
      reference_D.host_ref(),
      ElementAccumulator(0)
    );

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    at::Tensor tensor_a,
    at::Tensor tensor_b,
    at::Tensor tensor_c,
    at::Tensor tensor_d,
    c10::cuda::CUDAStream stream,
    cutlass::gemm::GemmCoord problem_size, 
    int split_k_slices = 1,
    ElementCompute alpha = ElementCompute(1), 
    ElementCompute beta = ElementCompute(0)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      std::cout << "NOT SUFFICIENT" << std::endl;
//      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
//        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
//      }
      return true;
    }
      std::cout << "VERY SUFFICIENT" << std::endl;

//    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //


   cutlass::layout::RowMajor layout_a;
   cutlass::layout::RowMajor layout_b;
   cutlass::layout::RowMajor layout_c;
   cutlass::layout::RowMajor layout_d;
   auto tensor_a_device_ref = cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t*)tensor_a.data_ptr<at::Half>(), layout_a);
   auto tensor_b_device_ref = cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t*)tensor_b.data_ptr<at::Half>(), layout_b);
   auto tensor_c_device_ref = cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t*)tensor_c.data_ptr<at::Half>(), layout_c);
   auto tensor_d_device_ref = cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>((cutlass::half_t*)tensor_d.data_ptr<at::Half>(), layout_d);

    tensor_E.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    tensor_E_reordered.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    cutlass::reorder_meta(tensor_E_reordered.host_ref(), tensor_E.host_ref(),
                          {problem_size.m(), problem_size.n(),
                           problem_size.k() / kSparse / kElementsPerElementE});

   // tensor_E_reordered.sync_device();

    typename Gemm::Arguments arguments{
      problem_size,
      // tensor_A.device_ref(),
      tensor_a_device_ref,
      // tensor_B.device_ref(),
      tensor_b_device_ref,
      // tensor_C.device_ref(),
      tensor_c_device_ref,
      // tensor_D.device_ref(),
      tensor_d_device_ref,

//      tensor_E_reordered.device_ref(),
      {alpha, beta},
      split_k_slices
    };

    Gemm gemm_op;

    // size_t workspace_size = Gemm::get_workspace_size(arguments);

    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // cutlass::Status status = gemm_op.initialize(arguments, workspace.get(), stream);
    cutlass::Status status = gemm_op.initialize(arguments, nullptr, stream);

  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

//  cutlass::Status status =
//      gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());

//        // This failure is likely due to insufficient device capabilities. Waive the test.
//    if (status != cutlass::Status::kSuccess) {
//      return true;
//    }
//
//    //
//    // Run the GEMM
//    //
//
    // status = gemm_op();
    status = gemm_op.run(stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
//
//    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
//
//    //
//    // Verify
//    //
//
//    cudaDeviceSynchronize();
    // tensor_D.sync_host();
    // std::cout << "\nComputed =\n" << tensor_D.host_view();
//
//    bool passed = this->verify(problem_size, alpha, beta);

//
//    if (!passed) {
//      std::cout << "Error with split_k_slices = " << split_k_slices << ", alpha: " << alpha << std::endl;
//    }
//
//    return passed;
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// template <typename Gemm>
// bool TestAllSparseGemm() {
//   bool passed = true;
// 
//   int const kMinimumOperandElementSize = 
//     std::min(
//       int(cutlass::sizeof_bits<typename Gemm::ElementA>::value), 
//       int(cutlass::sizeof_bits<typename Gemm::ElementB>::value));
// 
//   // M dimension has to be multiple of 32 (sparse float) or 16 (sparse int)
//   // because of the reordering of operand E
//   int const kAlignmentM = std::max(((sizeof(typename Gemm::ElementE) == 2) ? 32 : 16),
//                                    kMinimumOperandElementSize);
// 
//   int const kAlignmentN = 128 / kMinimumOperandElementSize;
// 
//   int problem_size_m[] = {kAlignmentM, 512 - 3 * kAlignmentM};
// 
//   int problem_size_n[] = {kAlignmentN, 512 - 2 * kAlignmentN};
// 
//   int problem_size_k[] = {Gemm::ThreadblockShape::kK,
//                           Gemm::ThreadblockShape::kK * (Gemm::kStages + 1)};
// 
//   int split_k_slices[] = {
//     1, 2, 3
//   };
// 
//   double problem_alpha[] = {
//     1
//   };
// 
//   double problem_beta[] = {
//     2.0
//   };
// 
//   SparseTestbed<Gemm> testbed;
// 
//   using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
// 
//   for (int m : problem_size_m) {
//     for (int n : problem_size_n) {
//       for (int k : problem_size_k) {
//         for (int split_k : split_k_slices) {
// 
//           if (!Gemm::kSplitKSerial && split_k > 1) {
//             continue;
//           }
// 
//           if (split_k > 1 && k / Gemm::ThreadblockShape::kK < split_k) {
//             continue;
//           }
// 
//           for (auto alpha : problem_alpha) {
//             for (auto beta : problem_beta) {
// 
//               cutlass::gemm::GemmCoord problem_size(m, n, k);
// 
//               passed = testbed.run(
//                 problem_size, 
//                 split_k,
//                 cutlass::from_real<ElementCompute>(alpha), 
//                 cutlass::from_real<ElementCompute>(beta)
//               );
// 
//               if (!passed) {
//                 return false;
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// 
//   return passed;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

int run(
    at::Tensor tensor_a,
    at::Tensor tensor_b,
    at::Tensor tensor_c,
    at::Tensor tensor_d,
    c10::cuda::CUDAStream stream) {

  // tensor a is m x (k // kSparse); tensor b is k x n
  const int length_m = tensor_a.size(0);
  const int length_k = tensor_b.size(0);
  const int length_n = tensor_b.size(1);


  TORCH_CHECK(tensor_c.size(0) == tensor_a.size(0));
  TORCH_CHECK(tensor_d.size(0) == tensor_a.size(0));

  TORCH_CHECK(tensor_c.size(1) == tensor_b.size(1));
  TORCH_CHECK(tensor_d.size(1) == tensor_b.size(1));

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

//  // TODO:
//  // Feed tensor_e as CPU int16 Tensor.
//  // Try various valid 2:4 meta configurations
//  // 0x4 0100
//  // 0x8 1000
//  // 0x9 1001
//  // 0xc 1100
//  // 0xd 1101
//  // 0xe 1110
//
//  // Create matrix E with dimensions M x (K / 2 / kElementsPerElementE). This one is used by reference computing.
//  cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
//      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
//  // Same size as the above.  The above one needs to be reordered and stored in this one.
//  cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
//      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
//
//  cutlass::reference::host::TensorFillRandomSparseMeta(
//      tensor_e.host_view(),
//      1,
//      kMetaSizeInBits);   // <- Fill matrix E on host with uniform-distribution random meta data
//
//  // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
//  // instructions.
//  cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),
//                        {problem_size.m(), problem_size.n(),
//                         problem_size.k() / kSparse / kElementsPerElementE});
//
//  tensor_e_reordered.sync_device();
//
  // Initialize alpha and beta for dot product computation
  float alpha = 1;
  float beta  = 0;
//
//  LayoutInputA layout_a;
//  LayoutInputB layout_b;
//  LayoutOutput layout_c;
//  LayoutOutput layout_d;
//
//  // Split K dimension into 1 partitions
//  int split_k_slices = 1;
//  auto tensor_a_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputA>((cutlass::half_t*)tensor_a.data_ptr<at::Half>(), layout_a);
//  auto tensor_b_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputB>((cutlass::half_t*)tensor_b.data_ptr<at::Half>(), layout_b);
//  auto tensor_c_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_c.data_ptr<at::Half>(), layout_c);
//  auto tensor_d_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_d.data_ptr<at::Half>(), layout_d);
//
//  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
//  // instantiated CUTLASS kernel
//  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
//                                     tensor_a_device_ref,  // <- reference to matrix A on device
//                                     tensor_b_device_ref,  // <- reference to matrix B on device
//                                     tensor_c_device_ref,  // <- reference to matrix C on device
//                                     tensor_d_device_ref,  // <- reference to matrix D on device
//                                     tensor_e_reordered.device_ref(),  // <- reference to matrix E on device
//                                     {alpha, beta},          // <- tuple of alpha and beta
//                                     split_k_slices};        // <- k-dimension split factor
//
//  // Using the arguments, query for extra workspace required for matrix multiplication computation
//  size_t workspace_size = Gemm::get_workspace_size(arguments);
//
//  // Allocate workspace memory
//  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
//
//  // Instantiate CUTLASS kernel depending on templates
//  Gemm gemm_op;
//
//  // Check the problem size is supported or not
//  cutlass::Status status = gemm_op.can_implement(arguments);
//  CUTLASS_CHECK(status);
//
//  // Initialize CUTLASS kernel with arguments and workspace pointer
//  status = gemm_op.initialize(arguments, workspace.get());
//  CUTLASS_CHECK(status);
//
//  // Launch initialized CUTLASS kernel
//  status = gemm_op();
//  CUTLASS_CHECK(status);
//  return 0;

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::SparseGemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::RowMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  test::gemm::device::SparseTestbed<Gemm> testbed;
//  cutlass::gemm::GemmCoord problem_size(m, n, k);

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  static int const kSparse = Gemm::GemmKernel::kSparse;

  TORCH_CHECK(
      tensor_b.size(0) % kSparse == 0,
      "Expected tensor_b.size(0) of value ",
      tensor_b.size(0),
      " to be evenly divisible by ",
      kSparse,
      " but got.");
  TORCH_CHECK(
      tensor_a.size(1) * kSparse == tensor_b.size(0),
      "Expected tensor_a.size(1) of value ",
      tensor_a.size(1),
      " to match tensor_b.size(0) of value ",
      tensor_b.size(0),
      " to match after being multiplied by ",
      kSparse);

  int split_k = 1;

  bool passed = testbed.run(
    tensor_a,
    tensor_b,
    tensor_c,
    tensor_d,
    stream,
    problem_size, 
    split_k,
    cutlass::from_real<ElementCompute>(alpha), 
    cutlass::from_real<ElementCompute>(beta)
  );

  std::cout << "passed: " << passed << std::endl;

//  std::cout << 
//    "test::gemm::device::TestAllSparseGemm<Gemm>(): " <<
//     test::gemm::device::TestAllSparseGemm<Gemm>() << std::endl;

  return 0;
}

namespace at {
namespace native {

// TODO: Pull back in device and cuda version constraints.
Tensor _cusparselt_linear(const Tensor& sparse, const Tensor& dense) {
  auto result = sparse.new_empty({sparse.size(0), dense.size(1)}); //.fill_(1);
//  auto init = sparse.new_empty({sparse.size(0), dense.size(1)}).fill_(2);
  auto stream = at::cuda::getDefaultCUDAStream();
  // at::cuda::stream_synchronize(stream);
  run(sparse, dense, result, result, stream);
  // at::cuda::stream_synchronize(stream);
  return result;
}

}
}
