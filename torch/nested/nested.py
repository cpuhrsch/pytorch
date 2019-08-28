import torch
import torch.nn.functional as F
import numbers
from functools import wraps
# from .codegen import get_tensorwise_functions
from . import codegen

from . import masking

# Set this flag to true, if you want to enable additional verifications.
DEBUG = False


# Stores path (e.g. torch.nn.lstm) -> new function (e.g. nested_cat)
# Path is represented as a list of strings
REGISTER_FUNCTIONS = {}


def _nested_apply(f):
    @wraps(f)
    def decorator(self, *args, **kwargs):
        if not(torch.is_tensor(self) or is_nested_tensor(self)):
            raise ValueError("First argument must be Tensor or NestedTensor")
        if self.nested_dim == 1:
            f(self, *args, **kwargs)
        else:
            for component in self.unbind():
                f(component, *args, **kwargs)
    return decorator


def _gen_unbound_args_kwargs(args, kwargs):
    unbound_args = [arg.unbind() for arg in args]
    unbound_kwargs = {k: v.unbind() for (k, v) in kwargs.items()}
    for i in range(len(args[0])):
        new_args = []
        for ua in unbound_args:
            new_args.append(ua[i])
        new_kwargs = {}
        for k in kwargs.keys():
            new_kwargs[k] = unbound_kwargs[k][i]
        yield (new_args, new_kwargs)


# The assumption is that f can handle a list of tensors
# This is used to write tensor-wise functions
# The resulting function accepts a multiple NestedTensors as arguments
# and calls f tensor-wise
def _nested_apply_return(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        def _func(*args, **kwargs):
            if torch.is_tensor(args[0]):
                return f(*args, **kwargs)
            else:
                assert is_nested_tensor(args[0])
                results = []
                for local_args, local_kwargs in _gen_unbound_args_kwargs(args, kwargs):
                    result = _func(*local_args, **local_kwargs)
                    if result is None:
                        continue
                    results.append(result)
                if len(results):
                    return as_nested_tensor(results)
        return _func(*args, **kwargs)
    return decorator


def _gen_nary_tensors(func):
    @wraps(func)
    def _nary_tensors(inputs):
        out_tensor = func(*inputs)
        if out_dtype is not None:
            out_tensor = out_tensor.to(out_dtype)
        return out_tensor
    return _nary_tensors


def _dispatch(cls):
    def decorator(new_fn):

        orig_fn = getattr(torch, new_fn.__name__)

        @wraps(orig_fn)
        def monkeyd(self, *args, **kwargs):
            if isinstance(self, cls):
                return new_fn(self, *args, **kwargs)
            else:
                return orig_fn(self, *args, **kwargs)

        return monkeyd
    return decorator


def monkey_patch(module):
    module.is_nested_tensor = is_nested_tensor
    module.as_nested_tensor = as_nested_tensor
    module.nested_tensor = nested_tensor
    module.tensor_mask_to_nested_tensor = tensor_mask_to_nested_tensor

    for function_name in codegen.get_tensorwise_functions():
        setattr(module, function_name, _dispatch(NestedTensor)(_nested_apply_return(getattr(module, function_name))))

    for function_name in codegen.get_tensorwise_functions():
        setattr(NestedTensor, function_name,
                _nested_apply_return(getattr(torch.Tensor, function_name)))
        setattr(NestedTensor, function_name + '_',
                _nested_apply_return(getattr(torch.Tensor, function_name + '_')))

    for function_name in ['add', 'mul', 'sub', 'div']:
        setattr(NestedTensor, "__" + function_name + '__',
                _nested_apply_return(getattr(torch.Tensor, "__" + function_name + '__')))

    for function_name in codegen.get_comparison_functions():
        setattr(NestedTensor, "__" + function_name + '__',
                _nested_apply_return(getattr(torch.Tensor, "__" + function_name + '__')))

    module.NestedTensor = NestedTensor

    # module.mv = mv
    # module.cat = cat

    return module


def _check_meaningful_overwrite(cls, method_name):
    class DefaultClass(object):
        pass

    if getattr(cls, method_name, False) and not getattr(DefaultClass, method_name, False):
        raise Exception("WARNING: " + method_name + " already exists "
                        "and not part of default class")


orig_cat = torch.cat

# TODO: Needs manual nesting semantics


@_dispatch(lambda a0: is_nested_tensor(a0[0]))
def cat(orig_cat, nested_tensors, dim=None):
    # Assuming 1 level of nesting
    if dim is not None:
        dim = dim - 1
    ret = []
    all_tensors = list(nested_tensors[0][i]._tensors for i in range(len(nested_tensors[0])))
    for tensors in zip(*all_tensors):
        ret.append(orig_cat(tensors, dim=dim))
    return NestedTensor(ret)


@_dispatch(lambda a0: is_nested_tensor(a0))
def mv(orig_mv, matrices, vectors):
    if matrices.nested_dim > 1:
        ntdim = matrices.nested_dim
        assert vectors.size()[:ntdim] == matrices.size()[:ntdim]
        return as_nested_tensor([mv(m, v) for (m, v) in zip(matrices.unbind(), vectors.unbind())])
    else:
        return as_nested_tensor([orig_mv(m, v) for (m, v) in zip(matrices.unbind(), vectors.unbind())])


orig_stack = torch.stack


def stack(*args, **kwargs):
    if is_nested_tensor(args[0]):
        import pdb
        pdb.set_trace()
    else:
        return orig_stack(*args, **kwargs)


def is_nested_tensor(obj):
    return isinstance(obj, NestedTensor)


# Given a tensor of size N x T x D and mask of size N x T
# return a NestedTensor of nested size ((t_1 x D), ..., (t_N x D))
# If mask[i][j] is 1, tensor[i][j] is an included Vector
def tensor_mask_to_nested_tensor(tensor, mask):
    if tensor.dim() != 3:
        raise NotImplementedError("Only support tensor arguments of dimension 3")
    if mask.dim() == 1:
        raise NotImplementedError("Not implemented for masks of dimension 1 yet.")
    if mask.dim() == 2:
        matrices = tensor.unbind()
        lengths = list(map(sum, mask.unbind()))
        return nested_tensor([matrices[i][:lengths[i]] for i in range(len(lengths))])
    raise NotImplementedError("Not implemented for masks of dimension 3 or more yet.")

# Arguments match torch.tensor


def nested_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if is_nested_tensor(data):
        # This is consistent with torch.tensor(torch.Tensor)
        # but errors out.
        raise ValueError("To copy construct from a NestedTensor, "
                         "use sourceTensor.clone().detach() or "
                         "sourceTensor.clone().detach().requires_grad_(True), "
                         "rather than torch.tensor(sourceTensor).")
    elif torch.is_tensor(data):
        # The user has the right to expect a NestedTensor from this
        # function, but we can't meaningfully provide one if passed a Tensor
        raise ValueError("Can't construct a NestedTensor from a Tensor")
    else:
        if not (isinstance(data, list) or isinstance(data, tuple)):
            raise ValueError("Pass a list or tuple to construct a NestedTensor. Got {} instead.".format(type(data)))

        nested_tensors = []
        for data_ in data:
            if is_nested_tensor(data_):
                nested_tensors.append(data_.clone().detach())

        if len(nested_tensors) == 0:
            for data_ in data:
                if isinstance(data_, list) or isinstance(data_, tuple):
                    nested_tensors.append(nested_tensor(data_))

        if len(nested_tensors) > 0:
            if len(nested_tensors) != len(data):
                raise ValueError("All entries of the passed list must either be Tensors or NestedTensors")
            return NestedTensor(nested_tensors)

        data = tuple(map(lambda x: torch.tensor(x) if isinstance(x, numbers.Number) else x, data))
        if any(map(lambda x: not torch.is_tensor(x), data)):
            raise ValueError("Each element of the tuple or list must "
                             "be a torch.Tensor or number")

        tensors = []
        for data_ in data:
            # torch.tensor copies on construction
            new_data = data_.clone().detach()
            new_data = new_data.to(dtype=dtype, device=device)
            new_data = new_data.requires_grad_(requires_grad)
            if pin_memory:
                new_data = new_data.pin_memory()
            tensors.append(new_data)

        return NestedTensor(tensors)


def as_nested_tensor(data, dtype=None, device=None):
    ret = NestedTensor(data)
    if dtype is not None:
        ret = ret.to(dtype)
    if device is not None:
        ret = ret.to(device)
    return ret


def _nested_property(f):
    @wraps(f)
    def decorator(self):
        if self.nested_dim == 1:
            return f(self)
        else:
            return f(self.unbind()[0])


@_nested_apply
def _verify_tensors(obj):
    tensors = obj.unbind()
    for tensor in tensors:
        assert torch.is_tensor(tensor)
    if len(tensors):
        dim = tensors[0].dim()
        layout = tensors[0].layout
        device = tensors[0].device
        dtype = tensors[0].dtype
        requires_grad = tensors[0].requires_grad
        is_pinned = tensors[0].is_pinned()
        for tensor in tensors:
            if not (dim == tensor.dim() and
                    layout == tensor.layout
                    and device == tensor.device
                    and dtype == tensor.dtype
                    and requires_grad == tensor.requires_grad
                    and is_pinned == tensor.is_pinned()):
                raise ValueError("Each passed Tensor "
                                 "must match in dim, layout, "
                                 "device, dtype and requires_grad")
    else:
        # Carrying around information as member variables vs.
        # checking one entry of the owned Tensors is annoying
        # and error-prone. Carrying around an is_empty attribute
        # to hide the fact that we carry around a list with a
        # single empty Tensor is also annoying and error-prone.
        # Both are not worth it for a minor feature.
        raise ValueError("We do not support empty lists for now.")


class NestedTensor(object):
    # The attributes must match across all constiuents
    #
    # The NestedTensor's attributes then become that of its
    # constiuents.
    #
    # The passed lists of tensors must be non-empty for now.
    #
    # Attributes:
    #     dim
    #     layout
    #     device
    #     dtype
    #     requires_grad
    #     is_pinned
    def __init__(self, tensors):
        self._tensors = tensors
        if DEBUG:
            _verify_tensors(self)

    # Cannot be decorated as _nested_property since
    # it's used for dispatch within the function
    @property
    def nested_dim(self):
        if torch.is_tensor(self._tensors[0]):
            return 1
        else:
            return (self._tensors[0]).nested_dim + 1

    @property
    @_nested_property
    def dim(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].dim

    @property
    @_nested_property
    def dtype(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].dtype

    @property
    @_nested_property
    def layout(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].layout

    @property
    @_nested_property
    def device(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].device

    @property
    @_nested_property
    def requires_grad(self):
        if DEBUG:
            _verify_tensors(self)
        return self._tensors[0].requires_grad

    def __len__(self):
        return len(self._tensors)

    def __bool__(self):
        raise NotImplementedError(
            "This has not been covered by NestedTensor 0.0.1")

    def __str__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__str__() + ",\n"
        result += "])"
        return result

    def __repr__(self):
        result = "nestedtensor([\n"
        for tensor in self._tensors:
            result += "  " + tensor.__repr__() + ",\n"
        result += "])"
        return result

    def __iadd__(self, other):
        for i in range(len(self)):
            self._tensors[i].add_(other._tensors[i])
        return self

    def __apply(self, fn):
        return [fn(tensor) for tensor in self._tensors]

    def nested_size(self, dim=None):
        if dim is not None:
            if dim == 0:
                return len(self)
            if self.nested_dim == 1:
                return tuple(t.size()[dim - 1] for t in self._tensors)
            return tuple(t.nested_size(dim - 1) for t in self.unbind())
        if self.nested_dim == 1:
            return tuple(t.size() for t in self._tensors)
        else:
            return tuple(t.nested_size() for t in self.unbind())

    def size(self):
        all_sizes = tuple(t.size() for t in self.unbind())

        def compare_sizes(size, other_size):
            result_size = list(size)
            for i in range(len(size)):
                result_size[i] = size[i] if size[i] == other_size[i] else None
            return tuple(result_size)

        result_size = list(all_sizes[0])
        for size in all_sizes:
            result_size = compare_sizes(result_size, size)
        return (len(self),) + result_size

    def all(self):
        return all(t.all() for t in self.unbind())

    def any(self):
        return any(t.any() for t in self.unbind())

    def sum(self, dim=None):
        # We currently assume len(self._tensors) is always non-zero
        if dim is None:
            return torch.stack(tuple(t.sum() for t in self._tensors)).sum()
        else:
            if dim > self.nested_dim - 1:
                return torch.as_nested_tensor(tuple(t.sum(dim - 1) for t in self._tensors))
            else:
                raise NotImplementedError("Reductions over NestedTensor dimension not defined")

    # TODO: This needs indicies!!! - not clear
    def argmax(self, dim=None):
        # We currently asmaxe len(self._tensors) is always non-zero
        if dim is None:
            raise NotImplementedError("Full reduction currently not supported")
        else:
            if dim > self.nested_dim - 1:
                return torch.as_nested_tensor(tuple(t.argmax(dim - 1) for t in self._tensors))
            else:
                raise NotImplementedError("Reductions over NestedTensor dimension not defined")

    # Tensor ops
    def detach(self):
        return NestedTensor(self.__apply(lambda x: x.detach()))

    def clone(self):
        return NestedTensor(self.__apply(lambda x: x.clone()))

    @_nested_property
    def numel(self):
        all_numel = 0
        for tensor in self._tensors:
            all_numel += tensor.numel()
        return all_numel

    def unbind(self):
        return tuple(self._tensors)

    def to_tensor(self):
        if None in self.size():
            raise ValueError("Cannot convert irreguarly shaped NestedTensor into a Tensor")
        else:
            if self.nested_dim == 1:
                return torch.stack(self.unbind())
            else:
                return torch.stack(list(map(lambda x: x.to_tensor(), self.unbind())))

    def to_list(self):
        if self.nested_dim == 1:
            return self._tensors
        else:
            return list(map(lambda x: x.to_list(), self.unbind()))

    def to_tensor_mask(self):
        tensor, mask = masking.make_tensor_mask(self.to_list())
        mask = mask.sum(-1)
        mask = (mask > 0)
        return tensor, mask

    def to(self, *args, **kwargs):
        return NestedTensor(self.__apply(lambda x: x.to(*args, **kwargs)))
