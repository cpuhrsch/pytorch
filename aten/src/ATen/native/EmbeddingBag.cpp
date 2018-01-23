#include "ATen/ATen.h"
#include "ATen/Check.h"
#include "ATen/NativeFunctions.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {

// TODO: Is it ok if offsets is [0, 0, 0]? - Orig versions and this one appears
// to disagree
// TODO: This is CPU only. Separrately link the GPU code.
Tensor embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                         const Tensor &offsets, const Tensor &offset2bag,
                         const bool scale_grad_by_freq, const int64_t mode, bool sparse) {
  auto output = weight.type().zeros({offsets.sizes()[0], weight.sizes()[1]});
  auto index_output = weight.index_select(0, indices);
  output.index_add_(0, offset2bag, index_output);
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] == 1) {
      auto bag_size = indices.sizes()[0];
      output /= bag_size;
    } else {
      auto bag_size = weight.type().tensor(offsets.sizes());
      bag_size.slice(0, 0, bag_size.sizes()[0] - 1, 1) =
          offsets.slice(0, 1, offsets.sizes()[0], 1) -
          offsets.slice(0, 0, offsets.sizes()[0] - 1, 1);
      bag_size[-1] = indices.sizes()[0] - offsets[-1];
      bag_size = bag_size.unsqueeze(1).expand_as(output);
      output /= bag_size;
    }
  }
  return output;
}

Tensor embedding_bag_backward(const Tensor &grad_, const Tensor &indices,
                                  const Tensor &offsets,
                                  const Tensor &offset2bag, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode, bool sparse) {
  if (sparse) {
    return at::embedding_bag_sparse_backward(grad_, indices,
                                  offsets,
                                  offset2bag, num_weights,
                                  scale_grad_by_freq, mode);
  } else {
    return at::embedding_bag_dense_backward(grad_, indices,
                                  offsets,
                                  offset2bag, num_weights,
                                  scale_grad_by_freq, mode);
  }
}

Tensor embedding_bag_backward_cpu(const Tensor &grad_, const Tensor &indices,
                                  const Tensor &offsets,
                                  const Tensor &offset2bag, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode) {

  Tensor grad = grad_;
  // std::cerr << "STARTING BACKWARD CPU" << std::endl;
  // std::cerr << "grad_: " << grad_ << std::endl;
  // std::cerr << "mode: " << mode << std::endl;
  // std::cerr << "indices: " << indices << std::endl;
  // std::cerr << "offsets: " << offsets << std::endl;
  // std::cerr << "offset2bag: " << offset2bag << std::endl;

  // Tensor index_grad = grad_.type().tensor({offset2bag.sizes()[0], grad_.sizes()[1]});
  Tensor index_grad = grad_.index_select(0, offset2bag);
  // std::cerr << "index_grad: " << index_grad << std::endl;
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] == 1) {
      auto bag_size = indices.sizes()[0];
      // std::cerr << "scalar bag size: " << bag_size << std::endl;
      index_grad /= bag_size;
    } else {
      auto bag_size = index_grad.type().tensor(offsets.sizes());
      bag_size.slice(0, 0, bag_size.sizes()[0] - 1, 1) =
          offsets.slice(0, 1, offsets.sizes()[0], 1) -
          offsets.slice(0, 0, offsets.sizes()[0] - 1, 1);
      bag_size[-1] = indices.sizes()[0] - offsets[-1];
      // std::cerr << "bag_size1: " << bag_size << std::endl;
      bag_size = bag_size.unsqueeze(1).index_select(0, offset2bag);
      // std::cerr << "bag_size2: " << bag_size << std::endl;
      index_grad /= bag_size;
    }
  }
  // std::cerr << "post index_grad: " << index_grad << std::endl;
  // std::cerr << "sparse: " << sparse << std::endl;

  bool sparse = false;
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, sparse);
}
Tensor embedding_bag_sparse_backward(const Tensor &grad_, const Tensor &indices,
                                  const Tensor &offsets,
                                  const Tensor &offset2bag, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode) {

  Tensor grad = grad_;
  // std::cerr << "STARTING BACKWARD CPU" << std::endl;
  // std::cerr << "grad_: " << grad_ << std::endl;
  // std::cerr << "mode: " << mode << std::endl;
  // std::cerr << "indices: " << indices << std::endl;
  // std::cerr << "offsets: " << offsets << std::endl;
  // std::cerr << "offset2bag: " << offset2bag << std::endl;

  // Tensor index_grad = grad_.type().tensor({offset2bag.sizes()[0], grad_.sizes()[1]});
  Tensor index_grad = grad_.index_select(0, offset2bag);
  // std::cerr << "index_grad: " << index_grad << std::endl;
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] == 1) {
      auto bag_size = indices.sizes()[0];
      // std::cerr << "scalar bag size: " << bag_size << std::endl;
      index_grad /= bag_size;
    } else {
      auto bag_size = index_grad.type().tensor(offsets.sizes());
      bag_size.slice(0, 0, bag_size.sizes()[0] - 1, 1) =
          offsets.slice(0, 1, offsets.sizes()[0], 1) -
          offsets.slice(0, 0, offsets.sizes()[0] - 1, 1);
      bag_size[-1] = indices.sizes()[0] - offsets[-1];
      // std::cerr << "bag_size1: " << bag_size << std::endl;
      bag_size = bag_size.unsqueeze(1).index_select(0, offset2bag);
      // std::cerr << "bag_size2: " << bag_size << std::endl;
      index_grad /= bag_size;
    }
  }
  // std::cerr << "post index_grad: " << index_grad << std::endl;
  // std::cerr << "sparse: " << sparse << std::endl;

  bool sparse = true;
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, sparse);
}
}
} // namespace at::native
