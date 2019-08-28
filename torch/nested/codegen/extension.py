import torch
from functools import wraps
from collections import namedtuple

# This entire file is half hack, half useful information.
#
# The function classifications (unary, binary, comparison)
# are useful to generate generic code based on certaion assumptions
# such as arity. For example, someone might implement a single function
# to efficiently implement a pointwise unary function such as cos
# and then generalize it using the list of unary functions.
#
# The hacky part of this file overwrites a module function via set_function
# and adds a dispatch mechanism via isinstance. The user can specify
# a function to be called if the overwritten function is called
# with a new object of type cls (based on the first argument).
# This dispatch mechanism is inherently inefficient and should be replaced.
# In fact, no release should include this mechanism and it is solely
# to support incremental development.

# Stores the relationship between a torch module function
# and a torch.Tensor method. For example torch.add
# maps to torch.Tensor.add and torch.Tensor.add can either
# be inplace or not.
Signature = namedtuple('Signature', ['module', 'method', 'inplace'])

# These functions only take two inputs. An input and a output.
# No scalars etc.


# TODO: Call this tensor-wise functions and add a path
# A tensorwise function is defined by a regular
# torch function that accepts Tensors as arguments
# (including maybe some kwargs) and Tensors only


def get_unary_functions():
    return [
        'abs',
        'acos',
        'asin',
        'atan',
        'ceil',
        # 'clamp', # TODO: Requires extra kwargs
        'cos',
        'cosh',
        'digamma',
        'erf',
        'erfc',
        'erfinv',
        'exp',
        'expm1',
        # 'exponential_', # TODO: Method only
        'floor',
        # 'fmod',
        'frac',
        # 'hardshrink', # TODO: Not part of aten
        'lgamma',
        'log',
        'log10',
        'log1p',
        'log2',
        # 'mvlgamma',
        'neg',
        # 'nonzero', #TODO: Special case because it modifies dtype
        # 'polygamma',
        # 'prelu', # TODO: no prelu_out in aten
        'reciprocal',
        # 'relu', # TODO: no relu_out in aten
        # 'renorm', # TODO: Requires extra kwargs
        'round',
        'rsqrt',
        'sigmoid',
        'sign',
        'sin',
        'sinh',
        'sqrt',
        'tan',
        'tanh',
        'trunc']


# These functions take exactly two Tensor arguments.
# It might be that they support scalar arguments as well,
# but we assume that the user will not use it in that way.
def get_binary_functions():
    return [
        'add',
        'mul',
        'sub',
        'div',
        'pow',
        'atan2',
        'remainder',
    ]


def get_comparison_functions():
    return [
        'eq',
        'ge',
        'gt',
        'le',
        'ne',
        'lt'
    ]


# Stores path (e.g. torch.nn.lstm) -> new function (e.g. nested_cat)
# Path is represented as a list of strings
def get_tensorwise_functions():
    funcs = []
    funcs += get_unary_functions()
    funcs += get_binary_functions()
    funcs += get_comparison_functions()
    return funcs
