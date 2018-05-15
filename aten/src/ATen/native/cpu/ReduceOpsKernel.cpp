#include "ATen/native/cpu/ReduceOpsKernel.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/optional.h"

// Vectorized reduction defined by reduce operation `Op` with identity `ident`.
// The reduction is built on top of reduce128, which reduces down a column
// 128 bytes wide (WIDTH scalar elements). The width of 128 bytes is chosen
// because of the "adjacent cache line prefetch" behavior on x86 CPUs.

namespace at { namespace native {
namespace {

using namespace vec256;

static inline int64_t round_down(int64_t a, int64_t m) {
  return a - (a % m);
}

template <typename F>
static void parallel_for(int64_t end, int64_t step, bool parallelize, F func) {
  if (parallelize) {
    tbb::parallel_for<int64_t>(0, end, step, func);
  } else {
    for (int64_t i = 0; i != end; i += step) {
      func(i);
    }
  }
}

static tbb::affinity_partitioner ap;

// Reduce down a column of WIDTH elements (128 bytes) with the given number
// of rows. Stores the results in out[0 ... WIDTH-1].
template <typename scalar_t, typename VecOp>
static void reduce128(
    const scalar_t* data,
    scalar_t* out,
    int64_t rows,
    int64_t stride,
    scalar_t ident,
    VecOp Reduce) {
  using Vec = Vec256<scalar_t>;
  Vec acc[4] = {ident, ident, ident, ident}; // 128 bytes (two cache lines)
  static_assert(sizeof(acc) == 128, "accumulator should be 128 bytes");
  for (int64_t row = 0; row != rows; row++) {
    for (int j = 0; j != 4; j++) {
      auto val = Vec::loadu(&data[row * stride + j * Vec::size]);
      acc[j] = Reduce(acc[j], val);
    }
  }
  for (int j = 0; j != 4; j++) {
    acc[j].store(&out[j * Vec::size]);
  }
}

template <
    typename scalar_t,
    typename VecOp,
    typename ScalarOp>
static scalar_t reduce_all(
    const scalar_t* data,
    int64_t size,
    scalar_t ident,
    VecOp Reduce,
    ScalarOp ReduceScalar) {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);
  int64_t k = size / WIDTH;

  scalar_t acc;
  if (size > internal::TBB_GRAIN_SIZE) {
    acc = tbb::parallel_reduce(
        tbb::blocked_range<int64_t>(0, k, internal::TBB_GRAIN_SIZE / WIDTH),
        ident,
        [=](const tbb::blocked_range<int64_t>& r, scalar_t init) {
          scalar_t buf[WIDTH];
          reduce128(
              &data[r.begin() * WIDTH],
              buf,
              r.end() - r.begin(),
              WIDTH,
              ident,
              Reduce);
          return std::accumulate(buf, buf + WIDTH, init, ReduceScalar);
        },
        ReduceScalar,
        ap);
  } else {
    scalar_t buf[WIDTH];
    reduce128(data, buf, k, WIDTH, ident, Reduce);
    acc = std::accumulate(buf, buf + WIDTH, ident, ReduceScalar);
  }

  for (int i = k * WIDTH; i != size; i++) {
    acc = ReduceScalar(acc, data[i]);
  }
  return acc;
}

// Reduce a 2d matrix down each column. Stores the results in out[0 ... cols-1]
template <typename scalar_t, typename VecOp, typename ScalarOp>
static void reduce2d(
    const scalar_t* data,
    scalar_t* out,
    int64_t rows,
    int64_t cols,
    int64_t stride,
    scalar_t ident,
    VecOp Reduce,
    ScalarOp ReduceScalar) {
  // reduction width in number of scalar elements
  static constexpr int WIDTH = 128 / sizeof(scalar_t);
  int64_t cols_rounded = round_down(cols, WIDTH);
  bool paralellize = cols * rows > internal::TBB_GRAIN_SIZE;
  parallel_for(cols_rounded, WIDTH, paralellize, [=](int64_t col) {
    reduce128(&data[col], &out[col], rows, stride, ident, Reduce);
  });

  if (cols_rounded != cols) {
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    scalar_t buf[WIDTH];

    // Initializes the entire (tiny) array to avoid uninitialized warnings
    std::fill(std::begin(buf), std::end(buf), ident);

    for (int64_t row = 0; row != rows; row++) {
      for (int64_t j = 0; j != cols - cols_rounded; j++) {
        auto val = data[row * stride + j + cols_rounded];
        buf[j] = ReduceScalar(buf[j], val);
      }
    }
    for (int64_t j = 0; j != cols - cols_rounded; j++) {
      out[j + cols_rounded] = buf[j];
    }
  }
}

template <
    typename scalar_t,
    typename VecOp,
    typename ScalarOp>
static void reduce(
    Tensor& res,
    const Tensor& self,
    at::optional<int64_t> dim,
    scalar_t ident,
    VecOp Reduce,
    ScalarOp ReduceScalar) {
  internal::init_tbb_num_threads();

  auto out = res.data<scalar_t>();
  auto data = self.data<scalar_t>();
  auto numel = self.numel();
  if (!dim.has_value()) {
    *out = reduce_all(data, numel, ident, Reduce, ReduceScalar);
    return;
  }

  int64_t n = self.size(*dim);
  int64_t stride = self.stride(*dim);
  // A contiguous tensor does not need to hold a meaningful stride
  // if the corresponding size is 1
  if (n == 1) {
    stride = 1;
    for (int64_t i = self.ndimension() - 1; i > *dim; i--) {
      stride *= self.size(i);
    }
  }
  int64_t batch = numel / (n * stride);
  bool paralellize = batch * n > internal::TBB_GRAIN_SIZE;
  parallel_for(batch, 1, paralellize, [=](int64_t b) {
    if (stride == 1) {
      out[b] = reduce_all(&data[b * n], n, ident, Reduce, ReduceScalar);
    } else {
      reduce2d(
          &data[b * n * stride],
          &out[b * stride],
          n,
          stride,
          stride,
          ident,
          Reduce,
          ReduceScalar);
    }
  });
}

static void
max_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim) {
  if (self.type().scalarType() == at::ScalarType::Float) {
    reduce<float>(
        result,
        self,
        dim,
        -std::numeric_limits<float>::infinity(),
        [](Vec256<float>& x, Vec256<float>& y) { return vec256::max(x, y); },
        [](float x, float y) { return std::max(x, y); });
  } else if (self.type().scalarType() == at::ScalarType::Double) {
    reduce<double>(
        result,
        self,
        dim,
        -std::numeric_limits<double>::infinity(),
        [](Vec256<double>& x, Vec256<double>& y) { return vec256::max(x, y); },
        [](double x, double y) { return std::max(x, y); });
  } else {
    AT_DISPATCH_INTEGER_TYPES(self.type(), "max", [&] {
      reduce<scalar_t>(
          result,
          self,
          dim,
          std::numeric_limits<scalar_t>::lowest(),
          [](Vec256<scalar_t>& x, Vec256<scalar_t>& y) {
            return vec256::max(x, y);
          },
          [](scalar_t x, scalar_t y) { return std::max(x, y); });
    });
  }
}

static void
min_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim) {
  if (self.type().scalarType() == at::ScalarType::Float) {
    reduce<float>(
        result,
        self,
        dim,
        -std::numeric_limits<float>::infinity(),
        [](Vec256<float>& x, Vec256<float>& y) { return vec256::min(x, y); },
        [](float x, float y) { return std::min(x, y); });
  } else if (self.type().scalarType() == at::ScalarType::Double) {
    reduce<double>(
        result,
        self,
        dim,
        -std::numeric_limits<double>::infinity(),
        [](Vec256<double>& x, Vec256<double>& y) { return vec256::min(x, y); },
        [](double x, double y) { return std::min(x, y); });
  } else {
    AT_DISPATCH_INTEGER_TYPES(self.type(), "min", [&] {
      reduce<scalar_t>(
          result,
          self,
          dim,
          std::numeric_limits<scalar_t>::lowest(),
          [](Vec256<scalar_t>& x, Vec256<scalar_t>& y) {
            return vec256::min(x, y);
          },
          [](scalar_t x, scalar_t y) { return std::min(x, y); });
    });
  }
}

static void
sum_kernel_impl(Tensor& result, const Tensor& self, at::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES(self.type(), "sum", [&] {
    reduce<scalar_t>(
        result,
        self,
        dim,
        (scalar_t)0,
        [](Vec256<scalar_t>& x, Vec256<scalar_t>& y) { return x + y; },
        [](scalar_t x, scalar_t y) { return x + y; });
  });
}

static void prod_kernel_impl(
    Tensor& result,
    const Tensor& self,
    at::optional<int64_t> dim) {
  AT_DISPATCH_ALL_TYPES(self.type(), "prod", [&] {
    reduce<scalar_t>(
        result,
        self,
        dim,
        (scalar_t)1,
        [](Vec256<scalar_t>& x, Vec256<scalar_t>& y) { return x * y; },
        [](scalar_t x, scalar_t y) { return x * y; });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(max_kernel, &max_kernel_impl);
REGISTER_DISPATCH(min_kernel, &min_kernel_impl);
REGISTER_DISPATCH(sum_kernel, &sum_kernel_impl);
REGISTER_DISPATCH(prod_kernel, &prod_kernel_impl);

}} // namespace at::native
