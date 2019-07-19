import os
import torch

from . import nested
from . import codegen

USE_NESTEDTENSOR = os.getenv('USE_NESTEDTENSOR', 'OFF') == 'ON'
if not USE_NESTEDTENSOR:
    raise RuntimeError("Attempting to use NestedTensor code "
                       "without the environment flag USE_NESTEDTENSOR "
                       "set to ON")

NestedTensor = nested.NestedTensor

def _nary_gen(out_dtype=None):
    # Follows signature of torch nary functions
    def _nary(*args, **kwargs):
        func_name = args[0]
        func = args[1]
        inputs = args[2:]
        out = kwargs.get('out', None)
        # NOTE: We are disabling broadcasting for now.
        for i in range(1, len(inputs)):
            for j in range(len(inputs[i])):
                assert inputs[0].tensors[j].size() == inputs[i].tensors[j].size()
        if out is None:
            out_tensors = []
            for i in range(len(inputs[0])):
                out_tensor = func(*list(map(lambda x: x.tensors[i], inputs)))
                if out_dtype is not None:
                    out_tensor = out_tensor.to(out_dtype)
                out_tensors.append(out_tensor)
            return NestedTensor(out_tensors)
        else:
            # NOTE: We are disabling broadcasting for now.
            for i in range(len(out)):
                assert out.tensors[i].size() == inputs[0].tensors[i].size()
            if out_dtype is not None:
                out = out.to(out_dtype)
            for i in range(len(inputs[0])):
                func(*list(map(lambda x: x.tensors[i], inputs)), out=out.tensors[i])
            return out
    return _nary


# NOTE: This is inefficient! The functions that are being overwritten in torch
# are being replaced by functions with very inefficient dispatch mechanisms to add
# support for NestedTensor to torch.
torch, NestedTensor = codegen.add_pointwise_unary_functions(torch, NestedTensor, _nary_gen())
torch, NestedTensor = codegen.add_pointwise_binary_functions(torch, NestedTensor, _nary_gen())
torch, NestedTensor = codegen.add_pointwise_comparison_functions(torch, NestedTensor, _nary_gen(torch.uint8))
torch.nestedtensor = nested.make_nested_tensor
torch.as_nestedtensor = nested.as_nestedtensor
