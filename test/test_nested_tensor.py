import traceback
import functools
import pdb
import sys
import torch
import unittest
from common_utils import TestCase

torch = torch.nested.monkey_patch(torch)


def debug_on(*exceptions):
    if not exceptions:
        exceptions = (BaseException, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])
        return wrapper
    return decorator


def _shape_prod(shape_):
    shape = tuple(shape_)
    start = 1
    for s in shape:
        start = start * s
    return start


def random_float_tensor(seed, size, a=22695477, c=1, m=2 ** 32,
                        requires_grad=False):
    """ Generates random tensors given a seed and size
    https://en.wikipedia.org/wiki/Linear_congruential_generator
    X_{n + 1} = (a * X_n + c) % m
    Using Borland C/C++ values
     The tensor will have values between [0,1)
    Inputs:
        seed (int): an int
        size (Tuple[int]): the size of the output tensor
        a (int): the multiplier constant to the generator
        c (int): the additive constant to the generator
        m (int): the modulus constant to the generator
    """
    num_elements = 1
    for s in size:
        num_elements *= s

    arr = [(a * seed + c) % m]
    for i in range(num_elements - 1):
        arr.append((a * arr[i] + c) % m)

    return torch.tensor(arr, requires_grad=requires_grad).float().view(size) / m


def random_int_tensor(seed, size, low=0, high=2 ** 32, a=22695477, c=1, m=2 ** 32):
    """ Same as random_float_tensor but integers between [low, high)
    """
    return (torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low).to(torch.int64)


def gen_float_tensor(seed, shape, requires_grad=False):
    return random_float_tensor(seed, shape, requires_grad=requires_grad)


def gen_random_int(seed, low=0, high=2 ** 32):
    """ Returns random integer in [low, high)
    """
    return int(random_int_tensor(seed, (), low=low, high=high))


def gen_nested_list(seed, nested_dim):
    tensors = []
    num_tensors = gen_random_int((seed * nested_dim + seed) * 1024, low=1, high=10)
    assert nested_dim > 0
    if nested_dim == 1:
        for i in range(num_tensors):
            ran = gen_random_int((seed * nested_dim + seed) * (1024 * i), low=1, high=10)
            tensors.append(gen_float_tensor(ran, (ran + 1, 128, 128)))
    else:
        tensors.append(gen_nested_list(num_tensors * seed, nested_dim - 1))
    return tensors


def nested_map(fn, data):
    if isinstance(data, list):
        for d in data:
            return nested_map(fn, d)
    else:
        return [fn(d) for d in data]


def gen_nested_tensor(seed, nested_dim):

    return torch.nested_tensor(gen_nested_list(seed, nested_dim))


class TestNestedTensor(TestCase):

    def test_nested_constructor(self):
        num_nested_tensor = 3
        # TODO: Shouldn't be constructable
        nested_tensors = [gen_nested_tensor(i, i) for i in range(1, num_nested_tensor)]
        nested_tensor = torch.nested_tensor(nested_tensors)
        nested_tensor.cos_()

    def test_constructor(self):
        tensors = []
        num_tensors = 16
        for i in range(num_tensors):
            tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
        nested_tensor = torch.nested_tensor(tensors)
        for i in range(num_tensors):
            tensors[i].mul_(i + 2)
        for i in range(num_tensors):
            self.assertTrue((tensors[i] != nested_tensor._tensors[i]).all())
        self.assertRaises(ValueError, lambda: torch.nested_tensor([]))
        self.assertRaises(ValueError, lambda: torch.nested_tensor(torch.tensor([3.0])))

    def test_nested_size(self):
        a = torch.nested_tensor([torch.rand(1, 2), torch.rand(2, 3), torch.rand(4, 5)])
        na = (torch.Size([1, 2]), torch.Size([2, 3]), torch.Size([4, 5]))
        self.assertEqual(a.nested_size(), na)

    def test_len(self):
        a = torch.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([3, 4]),
                                 torch.tensor([5, 6]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 4)
        a = torch.nested_tensor([torch.tensor([1, 2]),
                                 torch.tensor([7, 8])])
        self.assertEqual(len(a), 2)
        a = torch.nested_tensor([torch.tensor([1, 2])])
        self.assertEqual(len(a), 1)

    def test_equal(self):
        a1 = torch.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a2 = torch.nested_tensor([torch.tensor([1, 2]),
                                  torch.tensor([7, 8])])
        a3 = torch.nested_tensor([torch.tensor([3, 4]),
                                  torch.tensor([5, 6])])
        # Just exercising them until we have __bool__, all() etc.
        self.assertTrue((a1 == a2).all())
        self.assertTrue((a1 != a3).all())
        self.assertTrue(not (a1 != a2).any())
        self.assertTrue(not (a1 == a3).any())

    @debug_on()
    def test_nested_dim(self):
        nt = torch.nested_tensor([torch.tensor(3)])
        self.assertTrue(nt.nested_dim == 1)
        for i in range(2, 5):
            nt = gen_nested_tensor(i, i)
            self.assertTrue(nt.nested_dim == i)

    # TODO: Make nested test
    @debug_on()
    def test_unary(self):
        for func__ in torch.nested.codegen.extension.get_unary_functions():
            for nested_dim in range(1, 5):
                data = gen_nested_list(1, nested_dim)

                if func__ in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
                    data = nested_map(lambda x: x.abs(), data)
                if func__ in ['acos', 'asin', 'erfinv', 'log1p']:
                    data = nested_map(lambda x: x.clamp(min=0, max=1), data)
                if func__ in ['mvlgamma']:
                    data = nested_map(lambda x: x.clamp(min=1), data)

                a1 = torch.nested_tensor(data)
                a3 = torch.nested_tensor(data)
                func_ = getattr(torch, func__)
                method_ = getattr(torch.NestedTensor, func__)
                method_inplace_ = getattr(torch.NestedTensor, func__ + "_")
                if func__ in ['clamp']:
                    def func(x, out=None):
                        return func_(x, min=-1, max=1, out=out)

                    def method(x): return method_(x, min=-1, max=1)

                    def method_inplace(x): return method_inplace_(x, min=-1, max=1)
                elif func__ in ['mvlgamma']:

                    def func(x):
                        return func_(x, p=2)

                    def method(x): return method_(x, p=2)

                    def method_inplace(x): return method_inplace_(x, p=2)
                elif func__ in ['renorm']:

                    def func(x, out=None):
                        return func_(x, 2, 0, 1.0, out=out)

                    def method(x):
                        return method_(x, 2, 0, 1.0)

                    def method_inplace(x): return method_inplace_(x, 2, 0, 1.0)
                elif func__ in ['fmod']:

                    def func(x, out=None):
                        return func_(x, 0.3, out=out)

                    def method(x): return method_(x, 0.3)

                    def method_inplace(x): return method_inplace_(x, 0.3)
                else:
                    func = func_
                    method = method_
                    method_inplace = method_inplace_

                a2 = torch.nested_tensor(nested_map(func, data))

                if func__ not in ['mvlgamma']:
                    func(a1, out=a3)
                    self.assertTrue((func(a1) == a3).all())
                self.assertTrue((func(a1) == a2).all())
                self.assertTrue((method(a1) == a2).all())
                self.assertTrue((method_inplace(a1) == a2).all())
                self.assertTrue((a1 == a2).all())

    def test_binary(self):
        for func in torch.nested.codegen.extension.get_binary_functions():
            a = gen_float_tensor(1, (2, 3))
            b = gen_float_tensor(2, (2, 3))
            c = gen_float_tensor(3, (2, 3))
            # The constructor is supposed to copy!
            a1 = torch.nested_tensor([a, b])
            a2 = torch.nested_tensor([b, c])
            a3 = torch.nested_tensor([getattr(torch, func)(a, b),
                                      getattr(torch, func)(b, c)])
            self.assertTrue((a3 == getattr(torch, func)(a1, a2)).all())
            self.assertTrue((a3 == getattr(a1, func)(a2)).all())
            self.assertTrue((a3 == getattr(a1, func + "_")(a2)).all())
            self.assertTrue((a3 == a1).all())


if __name__ == "__main__":
    unittest.main()
