#include <ATen/Batching.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/BatchingRules.h>

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>
#include <bitset>
#include <iostream>

namespace at {

/*
 * Operator Registrations for BatchedTensorKey.
 * Contains some glue to hook up the batching rules to BatchedTensorImpl.
 */

Tensor BatchedTensor_mul(const Tensor& self, const Tensor& other) {
  Tensor self_, other_, result_;
  BatchDimsRef self_bdims, other_bdims;
  BatchDims result_bdims;
  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(other_, other_bdims) = unpackBatched(other);
  std::tie(result_, result_bdims) = mul_batching_rule(self_, self_bdims, other_, other_bdims);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor self_, other_, result_;
  BatchDimsRef self_bdims, other_bdims;
  BatchDims result_bdims;
  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(other_, other_bdims) = unpackBatched(other);
  std::tie(result_, result_bdims) = add_batching_rule(self_, self_bdims, other_, other_bdims, alpha);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  Tensor input_, weight_, bias_;
  BatchDimsRef input_bdims, weight_bdims, bias_bdims;
  IntArrayRef unflatten_sizes;

  std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(weight_, weight_bdims) = unpackBatched(weight);
  std::tie(bias_, bias_bdims) = unpackBatched(bias);

  Tensor result_;
  BatchDims result_bdims;
  std::cout << "BBB1" << std::endl;
  std::tie(result_, result_bdims) = conv2d_batching_rule(
      input_, input_bdims,
      weight_, weight_bdims,
      bias_, bias_bdims,
      stride, padding, dilation, groups);
  std::cout << "BBB2" << std::endl;
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor BatchedTensor_dropout(const Tensor& input, double p, bool train) {
	Tensor input_, result_;
	BatchDimsRef input_bdims;
	BatchDims result_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(result_, result_bdims) = dropout_batching_rule(input_, input_bdims, p, train);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor& BatchedTensor_dropout_(Tensor& self, double p, bool train) {
	Tensor self_;
	BatchDimsRef self_bdims;
	std::tie(self_, self_bdims) = unpackBatched(self);
  dropout__batching_rule(self_, self_bdims, p, train);
  return self;
}

Tensor BatchedTensor_relu(const Tensor& input) {
	Tensor input_, result_;
	BatchDimsRef input_bdims;
	BatchDims result_bdims;

	std::tie(input_, input_bdims) = unpackBatched(input);
  std::tie(result_, result_bdims) = unary_pw_batching_rule<at::relu>(input_, input_bdims);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}

Tensor& BatchedTensor_relu_(Tensor& self) {
	Tensor self_;
	BatchDimsRef self_bdims;
	std::tie(self_, self_bdims) = unpackBatched(self);
  unary_pw_inplace_batching_rule<at::relu_>(self_, self_bdims);
  return self;
}

Tensor BatchedTensor_sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = sum_batching_rule(self_, self_bdims, dim, keepdim, dtype);
  return detail::make_tensor<BatchTensorImpl>(result_, result_bdims);
}


// int64_t BatchTensorImpl::batch_size() const {
//   return rep_.sizes()[batch_dim_];
// }
// 
// 
// int64_t maxLevel(const std::vector<Tensor>& maybeBatchTensors) {
//   int64_t max = -1;
//   auto it = maybeBatchTensors.begin();
//   auto end_it = maybeBatchTensors.end();
//   while (it != end_it) {
//     it = std::find_if(it, end_it, isBatched);
//     if (it != end_it) {
//       const auto* batchTensor = static_cast<const BatchTensorImpl*>(it->unsafeGetTensorImpl());
//       if (batchTensor->level_ > max) {
//         max = batchTensor->level_;
//       }
//       it++;
//     }
//   }
//   return max;
// }
// 
// std::pair<Tensor,optional<int64_t>> unwrapAtLevel(const Tensor& tensor, int64_t level) {
//   if (!isBatched(tensor)) {
//     return { tensor, nullopt };
//   }
//   auto* batch_tensor = getBatched(tensor);
//   if (batch_tensor->level_ != level) {
//     TORCH_INTERNAL_ASSERT(batch_tensor->level_ < level);
//     return { tensor, nullopt };
//   }
//   return { batch_tensor->rep_, batch_tensor->batch_dim_ };
// }
// 
// Tensor broadcastTo(const Tensor& tensor, int64_t ndim) {
//   auto old_sizes = tensor.sizes();
//   if (old_sizes.size() == ndim) {
//     return tensor;
//   }
//   TORCH_INTERNAL_ASSERT(old_sizes.size() <= ndim);
//   // TODO: This is really slow, we should probably write a new operator for
//   // this. Note that we can't call view because it is not "semantic" enough.
//   // It might be possible to just call reshape here.
//   int64_t diff = ndim - old_sizes.size();
//   Tensor result = tensor;
//   for (int64_t i = 0; i < diff; ++i) {
//     result = result.unsqueeze(0);
//   }
//   return result;
// }
// 
// Tensor moveBatchDimToFront(
//     const Tensor& tensor,
//     optional<int64_t> batch_dim,
//     int64_t result_dim) {
//   if (!batch_dim) {
//     return broadcastTo(tensor, result_dim);
//   }
//   auto bdim = *batch_dim;
//   auto extra_dims = result_dim - tensor.dim();
//   auto result = broadcastTo(tensor, result_dim);
//   auto actual_bdim = bdim + extra_dims;
//   if (actual_bdim == 0) {
//     return result;
//   }
//   // should be an op...
//   std::vector<int64_t> permutation(result_dim);
//   permutation[0] = actual_bdim;
//   for (int64_t i = 1; i < result_dim; i++) {
//     if (i <= actual_bdim) {
//       permutation[i] = i - 1;
//     } else {
//       permutation[i] = i;
//     }
//   }
//   result = result.permute(permutation);
//   return result;
// }
// 
// int64_t actualDim(int64_t dim, optional<int64_t> maybe_batch_dim) {
//   if (maybe_batch_dim && dim >= *maybe_batch_dim) {
//     return dim + 1;
//   }
//   return dim;
// }
// 
// std::tuple<int64_t,int64_t>
// discoverBatchSizeAndLevel(torch::jit::Stack* stack) {
//   int64_t max_level = -1;
//   int64_t batch_size = -1;
//   for (auto& ivalue : *stack) {
//     if (!ivalue.isTensor()) continue;
//     auto tensor = ivalue.toTensor();
//     if (!isBatched(tensor)) continue;
//     auto* batched = getBatched(tensor);
//     if (batched->level_ > max_level) {
//       max_level = batched->level_;
//       // TODO: should probably validate somewhere that the batch sizes are the same
//       batch_size = batched->batch_size();
//     }
//   }
//   TORCH_INTERNAL_ASSERT(batch_size != -1);
//   TORCH_INTERNAL_ASSERT(max_level != -1);
//   return { batch_size, max_level };
// }
// 
// int64_t minDim(const Tensor& tensor, optional<int64_t> batch_dim) {
//   auto result = tensor.dim(); 
//   if (!batch_dim) {
//     result += 1;
//   }
//   return result;
// }
// 
// Copy pasta'ed from backed_fallback_test.cpp
void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // This should just be op.callBoxed(stack), but that doesn't work for all ops yet.
  // Note: If op.callBoxed(stack) works for you, then that is preferrable because
  // it's much faster and doesn't come with a dependency on JIT code.
  // Instead, we take a path through the JIT operator registry, which has a boxed
  // calling mechanism that works for all ops from native_functions.yaml.

  auto s = Symbol::fromQualString(op.schema().name());
  auto operators = torch::jit::getAllOperatorsFor(s);
  // Find the exact match
  std::shared_ptr<torch::jit::Operator> jit_op;
  for (const auto& candidate_op : operators) {
    auto candidate_schema = candidate_op->schema();
    // NB: this is a VERY slow equality test
    if (candidate_schema == op.schema()) {
      jit_op = candidate_op;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(jit_op);

  auto offset = jit_op->getOperation()(*stack);
  TORCH_INTERNAL_ASSERT(offset == 0);
}

// Modifies `stack` in-place.
// Returns: levels_bitset, batch_sizes
std::pair<std::bitset<64>,std::vector<int64_t>>
broadcastBdimsAtFront(torch::jit::Stack& stack) {
  std::unordered_map<int64_t,int64_t> level_to_size;
  std::bitset<64> levels_bitset;
  int64_t max_level = -1;
  BatchDimsRef bdims;
  Tensor arg;

  // Compute levels_bitset, level_to_size mapping
  for (const auto& ivalue : stack) {
    if (!ivalue.isTensor()) continue;
    std::tie(arg, bdims) = unpackBatched(ivalue.toTensor());
    auto arg_levels = createLevelsBitset(bdims);
    levels_bitset |= arg_levels;
    if (!arg.defined()) continue;
    auto arg_sizes = arg.sizes();
    for (const auto& bdim : bdims) {
      if (level_to_size.count(bdim.level())) {
        TORCH_CHECK(level_to_size[bdim.level()] == arg_sizes[bdim.index()]);
      } else {
        level_to_size[bdim.level()] = arg_sizes[bdim.index()];
      }
    }
  }

  // Get max_level
  for (int64_t idx = 0; idx < 64; idx++) {
    if (levels_bitset[idx]) {
      max_level = idx; 
    }
  }

  // Get batch_sizes
  std::vector<int64_t> batch_sizes;
  for (int64_t level = 0; level <= max_level; level++) {
    if (!levels_bitset[level]) {
      continue;
    }
    batch_sizes.push_back(level_to_size[level]);
  }

  // Move all bdims to front, align them, and expand them.
  for (int64_t idx = 0; idx < stack.size(); idx++) {
    const auto& ivalue = stack[idx];
    if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
      continue;
    }
    std::tie(arg, bdims) = unpackBatched(ivalue.toTensor());
    arg = moveBdimsToFront(arg, bdims);
    arg = alignTensorTo(arg, bdims, levels_bitset, max_level, arg.dim() - bdims.size());
    std::vector<int64_t> expanded_sizes(batch_sizes);
    expanded_sizes.insert(
        expanded_sizes.end(),
        arg.sizes().begin() + batch_sizes.size(),
        arg.sizes().end());
    stack[idx] = arg.expand(expanded_sizes);
  }
  return { levels_bitset, batch_sizes };
}

std::vector<int64_t> computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  std::vector<int64_t> result;
  result.reserve(sizes.size());
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {
    result.push_back(linear_idx % *it);
    linear_idx -= *it;
    linear_idx /= *it;
  }
  std::reverse(std::begin(result), std::end(result));
  return result;
}

Tensor selectAll(const Tensor& tensor, IntArrayRef indices) {
  auto tensor_ = tensor;
  for (int64_t dim = 0; dim < indices.size(); dim++) {
    tensor_ = tensor_.select(dim, indices[dim]); 
  }
  return tensor_;
}

void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  TORCH_CHECK(
      !schema.is_mutable() && !schema.hasAnyAliasInfo(),
      "Batching rule not implemented for ", schema, "; ",
      "the fallback path doesn't work on in-place or view ops.");
  
  TORCH_WARN("Batching rule not implemented for ", op.schema(), " falling back "
             "to slow (for-loop) implementation");
  TORCH_CHECK(std::all_of(op.schema().returns().begin(),
                          op.schema().returns().end(),
                          [] (const Argument& arg) { return arg.type() == TensorType::get(); }),
              "Batching rule not implemented for ", op.schema(), ". ",
              "We could not generate a fallback.");


  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  std::bitset<64> levels_bitset;
  std::vector<int64_t> batch_sizes;

  auto args = torch::jit::pop(*stack, num_arguments);
  std::tie(levels_bitset, batch_sizes) = broadcastBdimsAtFront(args);
  auto total_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(),
      1, std::multiplies<int64_t>());


  // Set up batch_size stacks, one for each unbatched computation.
  std::vector<torch::jit::Stack> unbatched_stacks(total_batches);
  auto pushToEachStack = [&](const auto& ivalue) {
    for (auto& stack : unbatched_stacks) {
      torch::jit::push(stack, ivalue);
    }
  };
  for (int64_t linear_idx = 0; linear_idx < total_batches; linear_idx++) {
    std::vector<int64_t> idx = computeIndex(linear_idx, batch_sizes);
    for (const auto& ivalue : args) {
      if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
        torch::jit::push(unbatched_stacks[linear_idx], ivalue);
        continue;
      }
      auto tensor = ivalue.toTensor();
      if (!tensor.defined()) {
        torch::jit::push(unbatched_stacks[linear_idx], ivalue);
        continue;
      }
      torch::jit::push(
          unbatched_stacks[linear_idx],
          torch::jit::IValue(selectAll(tensor, idx)));
    }
  }

  // Call the op on each stack.
  for (auto& stack : unbatched_stacks) {
    callBoxedWorkaround(op, &stack);
  }

  // Only support num_returns == 1 for now. Also assume Tensor returns
  TORCH_INTERNAL_ASSERT(num_returns == 1);
  for (int64_t return_idx = 0; return_idx < num_returns; return_idx++) {
    std::vector<Tensor> output_shards;
    for (const auto& stack : unbatched_stacks) {
      output_shards.push_back(stack[return_idx].toTensor());
    }
    auto flat_output = at::stack(output_shards);
    std::vector<int64_t> output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    auto output = detail::make_tensor<BatchTensorImpl>(
        flat_output.view(output_sizes),
        computeFrontBatchDims(levels_bitset));
    torch::jit::push(*stack, std::move(output));
  }
}
// 
// typedef std::pair<Tensor,optional<int64_t>> TensorAndBdim;
// 
// std::pair<TensorAndBdim,int64_t> unwrap(const Tensor& tensor) {
//   auto* batch_tensor = getBatched(tensor);
//   return { { batch_tensor->rep_, batch_tensor->batch_dim_ }, batch_tensor->level_ };
// }
// 
// TensorAndBdim unsqueeze_batching_rule(const TensorAndBdim& self, int64_t dim) {
//   const auto& tensor = self.first;
//   const auto& maybe_batch_dim = self.second;
// 
//   dim = maybe_wrap_dim(dim, tensor.dim());
//   auto actual_dim = actualDim(dim, maybe_batch_dim);
//   optional<int64_t> new_batch_dim = nullopt;
//   if (maybe_batch_dim) {
//     auto bdim = *maybe_batch_dim;
//     new_batch_dim = bdim < actual_dim ? bdim : bdim + 1;
//   }
//   return { tensor.unsqueeze(actual_dim), new_batch_dim };
// }
// 
// std::pair<Tensor,optional<int64_t>> mul_batching_rule(
//     const Tensor& self, optional<int64_t> self_bdim,
//     const Tensor& other, optional<int64_t> other_bdim) {
//   auto self_dim = minDim(self, self_bdim);
//   auto other_dim = minDim(other, other_bdim);
//   auto result_dim = std::max({self_dim, other_dim});
// 
//   auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
//   auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);
//   return { at::mul(self_value, other_value), 0 };
// }
// #else
// std::pair<Tensor,optional<int64_t>> mul_batching_rule(
//     const Tensor& self, optional<int64_t> self_bdim,
//     const Tensor& other, optional<int64_t> other_bdim) {
//   auto result = get_module()->run_method(
//       "mul_batching_rule",
//       self, self_bdim, other, other_bdim).toTuple();
//   auto res = result->elements()[0].toTensor();
//   auto bdim = result->elements()[1].toOptional<int64_t>();
//   return { res, bdim };
// }
// #endif
// 
// // TODO: it's not fine that we moved the batch dim,
// // but that should be easy to fix.
// std::pair<Tensor&,optional<int64_t>> mul__batching_rule(
//     Tensor& self, optional<int64_t> self_bdim,
//     const Tensor& other, optional<int64_t> other_bdim) {
//   auto self_dim = minDim(self, self_bdim);
//   auto other_dim = minDim(other, other_bdim);
//   auto result_dim = std::max({self_dim, other_dim});
// 
//   // NB: Produces view
//   auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
//   auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);
// 
//   // Probably want a nice error message here.
//   self_value.mul_(other_value);
//   return { self, 0 } ;
// }
// 
// Tensor& BatchedTensor_mul_(Tensor& self, const Tensor& other) {
//   // The following lines need to happen in each kernel
//   auto cur_level = maxLevel({self, other});
//   auto self_and_bdim = unwrapAtLevel(self, cur_level);
//   auto other_and_bdim = unwrapAtLevel(other, cur_level);
// 
//   mul__batching_rule(
//       self_and_bdim.first, self_and_bdim.second,
//       other_and_bdim.first, other_and_bdim.second);
//   return self;
// }
// 

// void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
//   TORCH_CHECK(false, "Batching rule not implemented for ", op.schema());
// }

// TODO: the fallback runs the un-batched kernel in a for loop.
// However, in many cases, operators are composed of other operators.
// If those operators have batched versions, then we don't need to
// run our for-loop-fallback. There should probably be some way to specify that.
auto batched_registry = c10::Dispatcher::singleton().registerBackendFallbackKernel(
    BatchTensorKey,
    KernelFunction::makeFromBoxedFunction<&batchTensorFallback>()
);

static auto batched_registry2 = torch::RegisterOperators()
  // Some operations need to be transformed to their batched versions
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_batched(Tensor self, int? batch_dim, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_make_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::_unwrap_batched(Tensor self, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_unwrap_batched))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> Tensor {
  //       TensorAndBdim unwrapped;
  //       int64_t cur_level;
  //       std::tie(unwrapped, cur_level) = unwrap(self);
  //       auto result_with_batch = unsqueeze_batching_rule(unwrapped, dim);
  //       return makeBatched(
  //           result_with_batch.first,
  //           result_with_batch.second,
  //           cur_level);
  //     }))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim0, int64_t dim1) -> Tensor {
  //       // TODO: don't forget to wrap dim0 & dim1
  //       auto* self_batched = getBatched(self);
  //       auto batch_dim = self_batched->batch_dim_;
  //       return makeBatched(
  //         self_batched->rep_.transpose(
  //           actualDim(dim0, batch_dim),
  //           actualDim(dim1, batch_dim)),
  //         batch_dim,
  //         self_batched->level_);
  //     }))
  .op(torch::RegisterOperators::options()
      .schema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
      .kernel(BatchTensorKey, &BatchedTensor_mul))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
  //     .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), &BatchedTensor_mul_>(BatchTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .kernel(BatchTensorKey, &BatchedTensor_add))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::detach(Tensor self) -> (Tensor)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self) -> Tensor {
  //       auto* batched = getBatched(self);
  //       return makeBatched(
  //           batched->rep_.detach(),
  //           batched->batch_dim_,
  //           batched->level_);
  //     }))
  .op(torch::RegisterOperators::options()
      .schema("aten::dropout(Tensor input, float p, bool train) -> Tensor")
      .kernel(BatchTensorKey, &BatchedTensor_dropout))
  .op(torch::RegisterOperators::options()
      .schema("aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<Tensor & (Tensor &, double, bool), &BatchedTensor_dropout_>(BatchTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor &, IntArrayRef, bool, optional<ScalarType>), &BatchedTensor_sum>(BatchTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::relu(Tensor self) -> Tensor")
      .kernel(BatchTensorKey, &BatchedTensor_relu))
  .op(torch::RegisterOperators::options()
      .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
      .impl_unboxedOnlyKernel<Tensor & (Tensor &), &BatchedTensor_relu_>(BatchTensorKey))
  .op(torch::RegisterOperators::options()
      .schema("aten::_is_batched(Tensor self) -> bool")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> bool {
        return true;
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::size.int(Tensor self, int dim) -> int")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> int64_t {
        dim = maybe_wrap_dim(dim, self.dim());
        return self.sizes()[dim];
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &BatchedTensor_conv2d>(BatchTensorKey))
  // .op(torch::RegisterOperators::options()
  //     .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
  //     .kernel(BatchTensorKey, [] (const Tensor& self, IntArrayRef size) -> Tensor {
  //       SmallVector<int64_t,5> new_sizes;
  //       new_sizes.reserve(size.size());
  //       auto* batched = getBatched(self);
  //       return makeBatched(
  //           batched->rep_.detach(),
  //           batched->batch_dim_,
  //           batched->level_);
  //     }))
  ;

}
