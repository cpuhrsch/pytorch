from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import vmap, Tensor
import torch.nn.functional as F


# ------------------- Tests specific to our prototype -----------------------
class TestPrototype(TestCase):

    # --------------- CONV2D ------------------
    def test_conv2d_accepts_3d_tensors(self):
        img = torch.randn(3, 5, 5)
        weight = torch.randn(3, 3, 2, 2)
        output = F.conv2d(img, weight)
        self.assertEqual(output, F.conv2d(img.unsqueeze(0), weight).squeeze(0))

    def test_vmap_conv2d(self):
        imgs = torch.randn(7, 3, 5, 5)
        weight = torch.randn(3, 3, 2, 2)
        expected = F.conv2d(imgs, weight)
        output = vmap(F.conv2d, (0, None))(imgs, weight)
        self.assertEqual(output, expected)

        imgs = torch.randn(3, 7, 5, 5)
        weight = torch.randn(3, 3, 2, 2)
        expected = F.conv2d(imgs.transpose(0, 1), weight)
        output = vmap(F.conv2d, (1, None))(imgs, weight)
        self.assertEqual(output, expected)

    def test_nested_tensor_conv2d(self):
        imgs = torch._make_nested([torch.randn(3, 5, 5) for _ in range(7)])
        weight = torch.randn(3, 3, 2, 2)
        # expected = F.conv2d(imgs, weight)
        print("AAA")
        output = vmap(F.conv2d, (0, None))(imgs, weight)
        print("BBB")
        # self.assertEqual(output, expected)

    def test_vmap_conv2d_two_batch_dims(self):
        y25739 = torch.randn(2, 5, 7, 3, 9)
        weight = torch.randn(13, 7, 2, 2, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)

        output = vmap(F.conv2d, (0, None, None))(y25739, weight, bias)
        expected = F.conv2d(y25739.view(10, 7, 3, 9), weight, bias).view(2, 5, 13, 2, 8)
        self.assertEqual(output, expected)

    def test_vmap_conv2d_autograd(self):
        pass

    def test_vmap_batch_norm(self):
        N, C, H, W = (7, 3, 5, 5)
        imgs = torch.randn(N, C, H, W)
        running_mean = torch.randn(C)
        running_var = torch.randn(C)
        # NB: Using "None" because we're not vectorizing over a dimension.
        output = vmap(F.batch_norm, (None, None, None))(imgs, running_mean, running_var)
        self.assertEqual(output, F.batch_norm(imgs, running_mean, running_var))

    def test_vmap_batch_norm_autograd(self):
        pass

    def test_vmap_add(self):
        x23 = torch.randn(2, 3)
        y23 = torch.randn(2, 3)
        output = vmap(torch.add, (0, 0))(x23, y23)
        self.assertEqual(output, x23 + y23)

    def test_vmap_add_autograd(self):
        x23 = torch.randn(2, 3, requires_grad=True)
        y23 = torch.randn(2, 3)
        output = vmap(torch.add, (0, 0))(x23, y23)
        output.sum().backward()
        self.assertEqual(x23.grad, torch.ones_like(x23))

    def test_vmap_sum(self):
        pass

    def test_vmap_sum_autograd(self):
        pass



if __name__ == '__main__':
    run_tests()

