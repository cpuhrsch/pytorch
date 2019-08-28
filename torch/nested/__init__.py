import os
import torch

from . import nested
from . import codegen


# NOTE: This is inefficient! The functions that are being overwritten in torch
# are being replaced by functions with very inefficient dispatch mechanisms to add
# support for NestedTensor to torch.

def monkey_patch(module):
    module = nested.monkey_patch(module)
    return module


__all__ = ["monkey_patch"]
