#include <cmath>
#include <limits>
#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

Tensor relu(const Tensor& self) {
  return self.clamp_min(0.0);
}

Tensor& relu_(Tensor& self) {
  return self.clamp_min_(0.0);
}

Tensor selu(const Tensor& self) {
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor& selu_(Tensor& self) {
  return at::elu_(self, SELU_ALPHA, SELU_SCALE);
}

Tensor rrelu(
    const Tensor& self,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  return at::rrelu_with_noise(
      self, self.type().tensor(), lower, upper, training, generator);
}

Tensor& rrelu_(
    Tensor& self,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  return at::rrelu_with_noise_(
      self, self.type().tensor(), lower, upper, training, generator);
}

Tensor softmax(const Tensor& self_, int64_t dim_) {
  int64_t dim = 0;
  if (self_.ndimension() != 0)
    dim = maybe_wrap_dim(dim_, self_.ndimension());

  auto self = self_.contiguous();
  auto output = at::zeros(self.type(), self.sizes());

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  if (self_.ndimension() != 0)
    dim_size = self.size(dim);

  for (int64_t i = 0; i < dim; ++i)
    outer_size *= self.size(i);
  for (int64_t i = dim + 1; i < self.ndimension(); ++i)
    inner_size *= self.size(i);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "softmax", [&]() {
    scalar_t* self_data_base = self.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();

    int64_t dim_stride = inner_size;
    int64_t outer_stride = dim_size * dim_stride;

    int64_t i, d;

    for (i = 0; i < outer_size * inner_size; i++) {
      int64_t outer_idx = i / inner_size;
      int64_t inner_idx = i % inner_size;
      scalar_t* self_data =
          self_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* output_data =
          output_data_base + outer_idx * outer_stride + inner_idx;

      scalar_t self_max = std::numeric_limits<scalar_t>::min();
      for (d = 0; d < dim_size; d++)
        self_max = std::max(self_max, self_data[d * dim_stride]);

      scalar_t sum = 0;
      for (d = 0; d < dim_size; d++) {
        scalar_t z = std::exp(self_data[d * dim_stride] - self_max);
        output_data[d * dim_stride] = z;
        sum += z;
      }

      scalar_t invsum = 1 / sum; // NOTE: truncate sum to scalar_t once
      for (d = 0; d < dim_size; d++) {
        output_data[d * dim_stride] *= invsum;
      }
    }
  });
  return output;
}

Tensor softmax_backward(
    const Tensor& grad_output_,
    const Tensor& self,
    int64_t dim_,
    const Tensor& output_) {
  int64_t dim = 0;
  if (output_.ndimension() != 0)
    dim = maybe_wrap_dim(dim_, output_.ndimension());

  auto output = output_.contiguous();
  auto grad_output = grad_output_.contiguous();
  auto grad_input = at::zeros(output.type(), output.sizes());

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  if (output_.ndimension() != 0)
    dim_size = output.size(dim);

  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.ndimension(); ++i)
    inner_size *= output.size(i);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "softmax backward", [&]() {
    scalar_t* grad_input_data_base = grad_input.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();
    scalar_t* grad_output_data_base = grad_output.data<scalar_t>();

    int64_t dim_stride = inner_size;
    int64_t outer_stride = dim_size * dim_stride;

    int64_t i, d;

    for (i = 0; i < (outer_size * inner_size); i++) {
      int64_t outer_idx = i / inner_size;
      int64_t inner_idx = i % inner_size;
      scalar_t* grad_input_data =
          grad_input_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* output_data =
          output_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* grad_output_data =
          grad_output_data_base + outer_idx * outer_stride + inner_idx;

      scalar_t sum = 0;
      for (d = 0; d < dim_size; d++)
        sum += ((scalar_t)grad_output_data[d * dim_stride]) *
            ((scalar_t)output_data[d * dim_stride]);

      for (d = 0; d < dim_size; d++)
        grad_input_data[d * dim_stride] = output_data[d * dim_stride] *
            (grad_output_data[d * dim_stride] - sum);
    }
  });
  return grad_input;
}

Tensor log_softmax(const Tensor& self_, int64_t dim_) {
  int64_t dim = 0;
  if (self_.ndimension() != 0)
    dim = maybe_wrap_dim(dim_, self_.ndimension());

  auto self = self_.contiguous();
  auto output = at::zeros(self_.type(), self.sizes());

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  if (self_.ndimension() != 0)
    dim_size = self.size(dim);

  for (int64_t i = 0; i < dim; ++i)
    outer_size *= self.size(i);
  for (int64_t i = dim + 1; i < self.ndimension(); ++i)
    inner_size *= self.size(i);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "log_softmax", [&]() {
    scalar_t* self_data_base = self.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();

    uint64_t dim_stride = inner_size;
    uint64_t outer_stride = dim_size * dim_stride;

    int64_t i, d;

    for (i = 0; i < (outer_size * inner_size); i++) {
      uint64_t outer_idx = i / inner_size;
      uint64_t inner_idx = i % inner_size;
      scalar_t* self_data =
          self_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* output_data =
          output_data_base + outer_idx * outer_stride + inner_idx;

      scalar_t max_self = std::numeric_limits<scalar_t>::min();
      for (d = 0; d < dim_size; d++)
        max_self = std::max(max_self, self_data[d * dim_stride]);

      scalar_t logsum = 0;
      for (d = 0; d < dim_size; d++)
        logsum += std::exp(self_data[d * dim_stride] - max_self);
      logsum = max_self + std::log(logsum);

      for (d = 0; d < dim_size; d++)
        output_data[d * dim_stride] = self_data[d * dim_stride] - logsum;
    }
  });

  return output;
}

Tensor log_softmax_backward(
    const Tensor& grad_output_,
    const Tensor& self,
    int64_t dim_,
    const Tensor& output_) {
  int64_t dim = 0;
  if (output_.ndimension() != 0)
    dim = maybe_wrap_dim(dim_, output_.ndimension());

  auto output = output_.contiguous();
  auto grad_output = grad_output_.contiguous();
  auto grad_input = at::zeros(output.type(), output.sizes());

  int64_t outer_size = 1;
  int64_t dim_size = 1;
  int64_t inner_size = 1;
  if (output_.ndimension() != 0)
    dim_size = output.size(dim);

  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.ndimension(); ++i)
    inner_size *= output.size(i);

  grad_output = grad_output.contiguous();
  output = output.contiguous();
  grad_input.resize_as_(output);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "log_softmax_backward", [&]() {

    scalar_t* grad_input_data_base = grad_input.data<scalar_t>();
    scalar_t* output_data_base = output.data<scalar_t>();
    scalar_t* grad_output_data_base = grad_output.data<scalar_t>();

    uint64_t dim_stride = inner_size;
    uint64_t outer_stride = dim_size * dim_stride;

    int64_t i, d;

    for (i = 0; i < (outer_size * inner_size); i++) {
      uint64_t outer_idx = i / inner_size;
      uint64_t inner_idx = i % inner_size;
      scalar_t* grad_input_data =
          grad_input_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* output_data =
          output_data_base + outer_idx * outer_stride + inner_idx;
      scalar_t* grad_output_data =
          grad_output_data_base + outer_idx * outer_stride + inner_idx;

      scalar_t sum = 0;
      for (d = 0; d < dim_size; d++)
        sum += grad_output_data[d * dim_stride];

      for (d = 0; d < dim_size; d++)
        grad_input_data[d * dim_stride] = grad_output_data[d * dim_stride] -
            std::exp(output_data[d * dim_stride]) * sum;
    }
  });
  return grad_input;
}

}} // namespace at::native
