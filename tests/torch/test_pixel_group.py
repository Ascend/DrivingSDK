import unittest
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.detection


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@dataclass
class KernelParams:
    score: torch.Tensor
    mask: torch.Tensor
    embedding: torch.Tensor
    kernel_label: torch.Tensor
    kernel_contour: torch.Tensor
    kernel_region_num: int
    distance_threshold: float


@golden_data_cache(__file__)
def pixel_group_cpu_golden(params: KernelParams):
    score = params.score
    mask = params.mask
    embedding = params.embedding
    kernel_label = params.kernel_label
    kernel_region_num = params.kernel_region_num
    distance_threshold = params.distance_threshold
    embedding_dim = embedding.shape[2]
    kernel_vector = torch.zeros((kernel_region_num, embedding_dim), dtype=torch.float32)

    for label in range(1, kernel_region_num):
        label_mask = kernel_label == label
        label_embeddings = embedding[label_mask]
        kernel_vector[label, :] = label_embeddings.sum(dim=0)
        vector_sum = label_mask.sum()
        kernel_vector[label, :] /= vector_sum

        kernel_cv = kernel_vector[label, :]
        valid_mask = (mask == 1) & (kernel_label == 0)
        valid_embeddings = embedding[valid_mask]
        distances = torch.sum((valid_embeddings - kernel_cv) ** 2, dim=1)
        within_threshold = distances < distance_threshold**2
        kernel_label[valid_mask] = torch.where(within_threshold, label, kernel_label[valid_mask])

    point_vector = torch.zeros((kernel_region_num, 2), dtype=torch.float32)

    label_flat = kernel_label.flatten()
    score_flat = score.flatten()

    mask = label_flat > 0
    valid_labels = label_flat[mask]
    valid_scores = score_flat[mask]

    point_vector.index_add_(
        0, valid_labels, torch.stack((valid_scores, torch.ones_like(valid_scores)), dim=1),
    )

    valid_mask = point_vector[:, 1] > 0
    point_vector[valid_mask, 0] /= point_vector[valid_mask, 1]

    point_vector_list = point_vector.tolist()
    for index in range(1, kernel_region_num):
        coords = (kernel_label == index).nonzero(as_tuple=False).float()
        coords = coords[:, [1, 0]]
        point_vector_list[index].extend(coords.flatten().tolist())

    return point_vector_list


def pixel_group_npu_golden(params: KernelParams):
    output1 = mx_driving.pixel_group(
        params.score.npu(),
        params.mask.npu(),
        params.embedding.npu(),
        params.kernel_label.npu(),
        params.kernel_contour.npu(),
        params.kernel_region_num,
        params.distance_threshold,
    )

    output2 = mx_driving.detection.pixel_group(
        params.score.npu(),
        params.mask.npu(),
        params.embedding.npu(),
        params.kernel_label.npu(),
        params.kernel_contour.npu(),
        params.kernel_region_num,
        params.distance_threshold,
    )
    return output1, output2


@golden_data_cache(__file__)
def generate_data(H, W, dim, num):
    score = np.random.uniform(0, 1, [H, W]).astype(np.float32)
    score = torch.from_numpy(score)
    mask = (score) > 0.5
    embedding = np.random.uniform(0, 10, [H, W, dim]).astype(np.float32)
    embedding = torch.from_numpy(embedding)
    kernel_label = np.random.uniform(0, num, [H, W]).astype(np.int32)
    kernel_label = torch.from_numpy(kernel_label)
    kernel_contour = np.random.uniform(0, 1, [H, W]).astype(np.uint8)
    kernel_contour = torch.from_numpy(kernel_contour)
    kernel_region_num = num
    distance_threshold = float(0.8)
    input_data = [
        score,
        mask,
        embedding,
        kernel_label,
        kernel_contour,
        kernel_region_num,
        distance_threshold,
    ]

    return input_data


class TestNpuPixelGroup(TestCase):
    seed = 1024
    np.random.seed(seed)

    def test_pixel_group(self, device="npu"):
        shapes = [
            [10, 10, 8, 3],
            [100, 100, 10, 5],
            [200, 100, 15, 6],
            [256, 256, 10, 8],
            [500, 1000, 15, 10],
        ]
        for shape in shapes:
            H, W, dim, num = shape

            data_input = generate_data(H, W, dim, num)
            params = KernelParams(*data_input)

            cpu_output = pixel_group_cpu_golden(params)
            npu_output1, npu_output2 = pixel_group_npu_golden(params)

            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output2)


if __name__ == "__main__":
    run_tests()
