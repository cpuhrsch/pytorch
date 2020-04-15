#pragma once

#include <ATen/ATen.h>
#include <bitset>

namespace at {

// TODO: use a real key
constexpr auto BatchTensorKey = DispatchKey::TESTING_ONLY_GenericWrapperTensorId;

struct BatchDim {
  BatchDim(int64_t index, int64_t level) : index_(index), level_(level) {}
  int64_t index() const {
    return index_;
  }
  int64_t level() const {
    return level_;
  }
 private:
  int64_t index_;
  int64_t level_;
};

using BatchDims = std::vector<BatchDim>;
using BatchDimsRef = ArrayRef<BatchDim>;

inline std::bitset<64> createIsBdimBitset(BatchDimsRef bdims) {
  std::bitset<64> is_bdim;
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.index());
  }
  return is_bdim;
}


struct BatchTensorImpl : public c10::TensorImpl {
  explicit BatchTensorImpl(Tensor value, BatchDims bdims)
    : TensorImpl(
        c10::DispatchKeySet(BatchTensorKey),
        value.dtype(),
        value.device()
      )
    , value_(std::move(value))
    , bdims_(std::move(bdims))
  {
    TORCH_INTERNAL_ASSERT(value_.defined());
    initSizes();
  }

  int64_t actualDim(int64_t dim, bool wrap_dim = true) const {
    if (wrap_dim) {
      const auto ndim = sizes_.size();
      dim = maybe_wrap_dim(dim, ndim);
    }

    auto is_bdim = createIsBdimBitset(bdims_);
    int64_t non_bdim_count = 0;
    for (int64_t result = 0; result < 64; result++) {
      if (is_bdim[result]) {
        continue;
      }
      if (non_bdim_count == dim) {
        return result;
      }
      non_bdim_count++;
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  const BatchDims& bdims() const { return bdims_; }
  const Tensor& value() const { return value_; };

 private:
  void initSizes() {
    const auto public_dims = value_.dim() - bdims_.size();
    const auto value_sizes = value_.sizes();
    sizes_.clear();
    sizes_.reserve(public_dims);
    for (int64_t dim = 0; dim < public_dims; dim++) {
      auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
      sizes_.push_back(value_sizes.at(actual_dim));
    }
  }

  at::Tensor value_;
  BatchDims bdims_;
};

// store a vector of BatchDim
// BatchDim: optional<index>, level

// should I call them "mapped_dims"?
// move_bdims_to_front(Tensor a)
// align_and_move_bdims_to_front(Tensor a, Tensor b) -> Tensor a, Tensor b
// move bdims to front, align them, and then return the unwrapped tensors (!!)
// why are we doing so much all at once?
//
// make_batched(Tensor a, List[Tuple[Optional[int], int]])
// add_bdim(Tensor a, dim, level)
// remove_bdim(Tensor a, level, size)
//
// flatten_bdims(Tensor a) -> Tensor a', (idx, sizes, levels)
// unflatten_bdims(Tensor a, idx, sizes, levels)
// has_bdims(Tensor a)
// actual_dim(Tensor a, dim(s))
//
// JIT: Takes operations on BatchedTensor, translates it to operations on Tensors
// add(Tensor, bdims, Tensor, bdims)

inline bool isBatched(Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(BatchTensorKey);
}

inline BatchTensorImpl* getBatched(Tensor tensor) {
  return static_cast<BatchTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline Tensor makeBatched(const Tensor& tensor, optional<int64_t> batch_dim, int64_t level) {
  if (!batch_dim.has_value()) {
    return tensor;
  }
  if (!isBatched(tensor)) {
    BatchDims bdims;
    bdims.push_back({batch_dim.value(), level});
    return at::detail::make_tensor<BatchTensorImpl>(tensor, std::move(bdims));
  }
  const auto* batched = getBatched(tensor);
  auto new_bdims = batched->bdims();
  new_bdims.push_back({batch_dim.value(), level});
  return at::detail::make_tensor<BatchTensorImpl>(batched->value(), std::move(new_bdims));
}

inline Tensor unwrapBatched(Tensor tensor, int64_t ntimes=1) {
  for (auto time = 0; time < ntimes; time++) {
    TORCH_INTERNAL_ASSERT(isBatched(tensor));
    const auto* batched = getBatched(tensor);
    auto new_bdims = batched->bdims();
    new_bdims.pop_back();
    if (new_bdims.size() > 0) {
      return at::detail::make_tensor<BatchTensorImpl>(batched->value(), std::move(new_bdims));
    } else {
      return batched->value();
    }
  }
  return tensor;
}

inline std::ostream& operator<<(std::ostream& out, const BatchDim& bdim) {
  out << "(idx=" << bdim.index() << ", lvl=" << bdim.level() << ")";
  return out;
}


// 
// inline std::ostream& operator<<(std::ostream& out, const BatchTensorImpl& batch_tensor) {
//   int64_t batch_dim = batch_tensor.batch_dim_;
//   // TODO: this prints out really bizarrely
//   out << "BatchTensor[lvl" << batch_tensor.level_ << "/bdim" << batch_dim << "]"
//       << batch_tensor.sizes();
//   return out;
// }

}
