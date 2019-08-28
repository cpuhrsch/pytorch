import torch
from functools import wraps
from collections import namedtuple

# The function classifications (unary, binary, comparison)
# are useful to generate generic code based on certaion assumptions
# such as arity. For example, someone might implement a single function
# to efficiently implement a pointwise unary function such as cos
# and then generalize it using the list of unary functions.


#TODO: Store path to function instead
# e.g.: ['NestedTensor', add_] -> NestedTensor.add_
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


# A tensorwise function is defined by a regular
# torch function that accepts Tensors as arguments
# (including maybe some kwargs) and Tensors only
def get_tensorwise_functions():
    funcs = []
    funcs += get_unary_functions()
    funcs += get_binary_functions()
    funcs += get_comparison_functions()
    return funcs
