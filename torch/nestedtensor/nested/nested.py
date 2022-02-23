import torch
import numbers

def _filter_impl(args, kwargs):
    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        if isinstance(a, torch.NestedTensor):
            impl_args.append(a._impl)
        elif torch.is_tensor(a):
            impl_args.append(a)
        elif isinstance(a, list):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(a_impl)
        elif isinstance(a, tuple):
            a_impl, _ = _filter_impl(a, {})
            impl_args.append(tuple(a_impl))
        else:
            impl_args.append(a)
    impl_kwargs = {
        k: v._impl if isinstance(v, torch.NestedTensor) else v for (k, v) in kwargs.items()
    }
    return impl_args, impl_kwargs

def _wrap_result(result):
    if isinstance(result, list):
        return list(_wrap_result(r) for r in result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    return (
        torch.NestedTensor(result)
        if torch.is_tensor(result) and torch.is_nt_impl(result)
        else result
    )

class NestedTensorMeta(type):
    def __getattr__(cls, name):
        if getattr(torch.Tensor, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(impl_args[0], name)(
                    *(impl_args[1:]), **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return cls.__dict__[name]



class NestedTensor(metaclass=NestedTensorMeta):
    # data is a torch.Tensor backed by a NestedTensorImpl

    def __init__(self, impl):
        self._impl = impl

    def __getattr__(self, name):
        if hasattr(self._impl, name):
            def _wrapped_fn(*args, **kwargs):
                impl_args, impl_kwargs = _filter_impl(args, kwargs)
                result = getattr(self._impl, name)(*impl_args, **impl_kwargs)
                return _wrap_result(result)
            return _wrapped_fn
        return self.__dict__[name]

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._impl.requires_grad

    def nested_dim(self):
        """
        The nested dimension of ```self``` NestedTensor.
        The nested dimension is defined as the level of indexing required
        to reach a Tensor constiuent.
        """
        # This NT only supports nesting of 1.
        return 1

    def size(self, dim):
        return torch.nested_tensor_size_int(self._impl, dim)

    def __str__(self):
        def _str(x, indent=0, tab="  "):
            if x.nested_dim() == 0:
                return ""
            s = indent*tab + "[\n"
            if x.nested_dim() == 1:
                strs = list(map(str, x.unbind()))
                strs = list(map(lambda xi: "\n".join(
                    map(lambda xij: (indent + 1)*tab + xij, xi.split("\n"))), strs))
                s += ",\n".join(strs)
            else:
                s += ",\n".join(list(map(
                    lambda xi: _str(xi, indent + 1), x.unbind())))
            s += "\n" + indent * tab + "]"
            return s
        return "nested_tensor(" + _str(self) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.nn.functional.multi_head_attention_forward:
            return _wrap_result(nt_multi_head_attention_forward(*args, **kwargs))
        impl_args, impl_kwargs = _filter_impl(args, kwargs)
        return _wrap_result(func(*impl_args, **impl_kwargs))
