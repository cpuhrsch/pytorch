#include <ATen/Batching.h>
#include <ATen/Batching.h>

namespace at {

/*
 * Utility functions used to implement batching rules
 */ 

bool areBdimsAtFrontInOrder(BatchDimsRef bdims);
std::bitset<64> createLevelsBitset(BatchDimsRef bdims);
BatchDims moveBdimsToFront(BatchDimsRef bdims);
Tensor moveBdimsToFront(const Tensor& self, BatchDimsRef bdims);
std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self);
Tensor unsqueezeMultiple(const Tensor& self, int64_t before_dim, int64_t ndims);

Tensor alignTensorTo(
    const Tensor& tensor,
    BatchDimsRef tensor_bdims,
    std::bitset<64> result_levels,
    int64_t max_result_level,
    int64_t num_result_regular_dims);

BatchDims computeFrontBatchDims(std::bitset<64> levels_bitset);

std::tuple<Tensor,Tensor,BatchDims> alignBdimsAtFront(
    const Tensor& self,
    BatchDimsRef self_bdims,
    const Tensor& other,
    BatchDimsRef other_bdims);

}
