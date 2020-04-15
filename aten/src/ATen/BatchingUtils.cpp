#include <ATen/BatchingUtils.h>

namespace at {

bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    if (bdims[idx].index() != idx) {
      return false;
    }
  }
  return true;
}

std::bitset<64> createLevelsBitset(BatchDimsRef bdims) {
  std::bitset<64> result; 
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

BatchDims moveBdimsToFront(BatchDimsRef bdims) {
  BatchDims result;
  result.reserve(bdims.size());
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    result.push_back(BatchDim(idx, bdims[idx].level()));
  }
  return result;
}

Tensor moveBdimsToFront(const Tensor& self, BatchDimsRef bdims) {
  auto self_sizes = self.sizes();
  std::vector<int64_t> permutation(self_sizes.size(), 0);
  auto is_bdim = createIsBdimBitset(bdims);
  int64_t idx = 0;
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.index();
  }
  for (int64_t ptr = 0; idx < self_sizes.size(); ptr++) {
    if (is_bdim[ptr]) {
      continue;
    } 
    permutation[idx++] = ptr;
  }
  return self.permute(permutation);
}

std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self) {
  if (isBatched(self)) {
    const auto* batched = getBatched(self);
    TORCH_INTERNAL_ASSERT(!isBatched(batched->value()));
    return { batched->value(), batched->bdims() };
  }
  TORCH_INTERNAL_ASSERT(!isBatched(self));
  return { self, {} };
}

Tensor unsqueezeMultiple(const Tensor& self, int64_t before_dim, int64_t ndims) {
  auto result = self;
  for (int64_t i = 0; i < ndims; i++) {
    result = result.unsqueeze(before_dim);
  }
  return result;
}

Tensor alignTensorTo(
    const Tensor& tensor,
    BatchDimsRef tensor_bdims,
    std::bitset<64> result_levels,
    int64_t max_result_level,
    int64_t num_result_regular_dims) {
  // NB: Two prerequisites:
  // 1. tensor_bdims are all at the front of tensor
  // 2. all of tensor_bdims are accounted for in result_levels
  auto tensor_sizes = tensor.sizes();
  int64_t num_result_bdims = result_levels.count();
  int64_t num_tensor_regular_dims = tensor_sizes.size() - tensor_bdims.size();
  if (num_tensor_regular_dims == num_result_regular_dims &&
      tensor_bdims.size() == num_result_bdims) {
    return tensor;
  }

  std::vector<int64_t> aligned_sizes(num_result_bdims + num_result_regular_dims, 1);

  // align the regular (non-bdims) first
  std::copy(
      tensor_sizes.rbegin(),
      tensor_sizes.rbegin() + num_tensor_regular_dims,
      aligned_sizes.rbegin());

  // align the bdims
  int64_t level = 0;
  int64_t dim = 0;
  auto tensor_bdims_iter = tensor_bdims.begin();
  while (level <= max_result_level && tensor_bdims_iter != tensor_bdims.end()) {
    if (!result_levels[level]) {
      level++;
      continue;
    }
    if (tensor_bdims_iter->level() == level) {
      aligned_sizes[dim] = tensor_sizes[tensor_bdims_iter - tensor_bdims.begin()]; 
      level++;
      tensor_bdims_iter++;
    } else if (tensor_bdims_iter->level() < level) {
      tensor_bdims_iter++; 
    } else if (tensor_bdims_iter->level() > level) {
      level++;
    }
    dim++;
  }

  return tensor.view(aligned_sizes);
}

BatchDims computeFrontBatchDims(std::bitset<64> levels_bitset) {
  BatchDims bdims;
  int64_t dim = 0;
  for (int64_t level = 0; level < 64; level++) {
    if (!levels_bitset[level]) {
      continue;
    }
    bdims.push_back({dim++, level});
  }
  return bdims;
}

std::tuple<Tensor,Tensor,BatchDims> alignBdimsAtFront(
    const Tensor& self,
    BatchDimsRef self_bdims,
    const Tensor& other,
    BatchDimsRef other_bdims) {

  // Step 1: Permute the bdims to the front of the tensors
  auto self_ = moveBdimsToFront(self, self_bdims);
  auto other_ = moveBdimsToFront(other, other_bdims);

  auto self_sizes = self.sizes();
  auto other_sizes = other.sizes();

  // Step 2: Align the bdims
  auto self_levels = createLevelsBitset(self_bdims);
  auto other_levels = createLevelsBitset(other_bdims);
  auto result_levels = self_levels | other_levels;
  auto max_result_level = 0;
  if (self_bdims.size() == 0) {
    max_result_level = other_bdims.back().level();
  } else if (other_bdims.size() == 0) {
    max_result_level = self_bdims.back().level();
  } else {
    max_result_level = std::max(self_bdims.back().level(), other_bdims.back().level());
  }
  auto num_result_regular_dims = std::max(
      self_sizes.size() - self_bdims.size(),
      other_sizes.size() - other_bdims.size());
  self_ = alignTensorTo(self_, self_bdims, result_levels, max_result_level, num_result_regular_dims);
  other_ = alignTensorTo(other_, other_bdims, result_levels, max_result_level, num_result_regular_dims);

  // Step 3: construct the result bdims
  BatchDims result_bdims = computeFrontBatchDims(result_levels);

  return { self_, other_, result_bdims };
}


}
