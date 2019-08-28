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
        exceptions = (AssertionError, )

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
    return torch.floor(random_float_tensor(seed, size, a, c, m) * (high - low)) + low


def gen_float_tensor(seed, shape, requires_grad=False):
    return random_float_tensor(seed, shape, requires_grad=requires_grad)


class TestNestedTensor(TestCase):

    def test_nested_constructor(self):
        def _gen_nested_tensor():
            tensors = []
            num_tensors = 4
            for i in range(num_tensors):
                tensors.append(gen_float_tensor(i, (i + 1, 128, 128)))
            return torch.nested_tensor(tensors)
        num_nested_tensor = 3
        nested_tensors = [_gen_nested_tensor() for _ in range(num_nested_tensor)]
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

    @debug_on
    def test_unary(self):
        for func in torch.nested.codegen.extension.get_unary_functions():
            data = [gen_float_tensor(1, (2, 3)) - 0.5,
                    gen_float_tensor(2, (2, 3)) - 0.5]
            if func in ['log', 'log10', 'log2', 'rsqrt', 'sqrt']:
                data = list(map(lambda x: x.abs(), data))
            a1 = torch.nested_tensor(data)
            a2 = torch.nested_tensor(list(map(lambda x: getattr(torch, func)(x), data)))
            self.assertTrue(getattr(torch, func)(a1) == a2)
            self.assertTrue(getattr(a1, func)() == a2)
            self.assertTrue(getattr(a1, func + "_")() == a2)
            self.assertTrue(a1 == a2)

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
