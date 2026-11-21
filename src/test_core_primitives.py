import unittest

import torch

from losses import TritonBCEWithLogitsLoss, TritonCrossEntropyLoss, TritonMSELoss
from optimizers import TritonAdam, TritonAdamW, TritonSGD, get_optimizer


class LossTests(unittest.TestCase):
    def test_mse_cuda_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        input = torch.randn(32, 16, device="cuda", requires_grad=True)
        target = torch.randn(32, 16, device="cuda")
        custom = TritonMSELoss("mean")
        native = torch.nn.MSELoss("mean")
        actual = custom(input, target)
        expected = native(input, target)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        actual.backward()
        grad = input.grad.clone()
        input.grad = None
        expected.backward()
        self.assertTrue(torch.allclose(grad, input.grad, atol=1e-5, rtol=1e-5))

    def test_cross_entropy_cuda_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        logits = torch.randn(16, 8, device="cuda", requires_grad=True)
        target = torch.randint(0, 8, (16,), device="cuda")
        actual = TritonCrossEntropyLoss()(logits, target)
        expected = torch.nn.CrossEntropyLoss()(logits, target)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        actual.backward()
        grad = logits.grad.clone()
        logits.grad = None
        expected.backward()
        self.assertTrue(torch.allclose(grad, logits.grad, atol=1e-5, rtol=1e-5))

    def test_bce_cuda_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        logits = torch.randn(32, 4, device="cuda", requires_grad=True)
        target = torch.rand(32, 4, device="cuda")
        actual = TritonBCEWithLogitsLoss()(logits, target)
        expected = torch.nn.BCEWithLogitsLoss()(logits, target)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        actual.backward()
        grad = logits.grad.clone()
        logits.grad = None
        expected.backward()
        self.assertTrue(torch.allclose(grad, logits.grad, atol=1e-5, rtol=1e-5))


class OptimizerTests(unittest.TestCase):
    def test_three_optimizers_reduce_loss(self):
        factories = (TritonSGD, TritonAdam, TritonAdamW)
        for factory in factories:
            parameter = torch.nn.Parameter(torch.tensor([2.0]))
            optimizer = factory([parameter], lr=1e-2)
            initial = parameter.item()
            for _ in range(100):
                optimizer.zero_grad()
                parameter.square().sum().backward()
                optimizer.step()
            self.assertLess(parameter.item(), initial)

    def test_factory_and_cuda_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        adamw = get_optimizer("adamw", [parameter])
        self.assertIsInstance(adamw, TritonAdamW)
        with self.assertRaises(ValueError):
            get_optimizer("rmsprop", [parameter])

    def test_cuda_adamw_step_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        custom_param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
        native_param = torch.nn.Parameter(custom_param.detach().clone())
        custom = TritonAdamW([custom_param], lr=1e-2, weight_decay=1e-2)
        native = torch.optim.AdamW([native_param], lr=1e-2, weight_decay=1e-2)
        for _ in range(10):
            custom.zero_grad()
            native.zero_grad()
            grad = torch.tensor([0.25, -0.5], device="cuda")
            custom_param.grad = grad
            native_param.grad = grad
            custom.step()
            native.step()
        self.assertTrue(torch.allclose(custom_param, native_param, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
