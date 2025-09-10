import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.dataset.utils import balanced_resize, BalancedRandomResize


class TestBalancedResize(TestCase):
    def setUp(self):
        super().setUp()
        self.test_image = Image.new('RGB', (100, 200), color='red')
        self.test_target = {
            "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            "area": torch.tensor([200, 400]),
            "size": torch.tensor([200, 100])
        }

    def test_balanced_resize_no_target(self):
        resized_image, target = balanced_resize(self.test_image, None, 50)
        self.assertEqual(min(resized_image.size), 50)
        self.assertEqual(max(resized_image.size), 100)
        self.assertIsNone(target)

    def test_balanced_resize_with_target(self):
        resized_image, target = balanced_resize(self.test_image, self.test_target, 50)
        self.assertEqual(min(resized_image.size), 50)
        self.assertEqual(max(resized_image.size), 100)

        self.assertIsNotNone(target)
        self.assertIn("boxes", target)
        self.assertIn("area", target)
        self.assertIn("size", target)

    def test_balanced_resize_with_max_size(self):
        resized_image, target = balanced_resize(self.test_image, self.test_target, 50, max_size=60)
        self.assertEqual(max(resized_image.size), 60)

    def test_balanced_resize_with_masks_raises_error(self):
        target_with_masks = self.test_target.copy()
        target_with_masks["masks"] = torch.ones(1, 200, 100)
        with self.assertRaises(RuntimeError):
            balanced_resize(self.test_image, target_with_masks, 50)


class TestBalancedRandomResize(TestCase):
    def setUp(self):
        super().setUp()
        self.sizes = [50, 100, 150]
        self.transform = BalancedRandomResize(self.sizes)

    def test_init_with_invalid_sizes(self):
        with self.assertRaises(TypeError):
            BalancedRandomResize("invalid")

    def test_call_without_target(self):
        test_image = Image.new('RGB', (100, 200), color='red')
        for _ in range(10):
            resized_image, target = self.transform(test_image, None)
            self.assertIn(resized_image.size[0], self.sizes)
            self.assertIsNone(target)

    def test_call_with_target(self):
        test_image = Image.new('RGB', (100, 200), color='red')
        test_target = {
            "boxes": torch.tensor([[10, 20, 30, 40]]),
            "area": torch.tensor([200]),
            "size": torch.tensor([200, 100])
        }

        resized_image, target = self.transform(test_image, test_target)
        self.assertIn(min(resized_image.size), self.sizes)
        self.assertIsNotNone(target)

if __name__ == '__main__':
    run_tests()