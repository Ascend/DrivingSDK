# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import math
from abc import ABC, abstractmethod
from typing import Any, List

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


# Abstract Base Class: concrete subclasses must implement `bucket_arange` method
class DynamicSampler(ABC):
    @abstractmethod
    def bucket_arange(self, *args: Any, **kwargs: Any) -> List:
        pass


class DynamicDistributedSampler(Sampler, DynamicSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if not hasattr(dataset, 'buckets') or not isinstance(dataset.buckets, list):
            raise ValueError("Dataset must have a 'buckets' attribute of type list")
        if not all(isinstance(bucket, list) for bucket in dataset.buckets):
            raise ValueError("All buckets must be lists")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.bucket_arange()
        else:
            indices = []
            for bct in self.dataset.buckets:
                indices.extend(bct)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if not len(indices) == self.total_size:
            raise ValueError(f"indices length should be equal to total_size, but got indices length: {len(indices)}, expected: {self.total_size}")

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        if not len(indices) == self.num_samples:
            raise ValueError(f"indices length should be equal to num_samples, but got indices length: {len(indices)}, expected: {self.num_samples}")

        return iter(indices)

    def bucket_arange(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        bucket_index = torch.randperm(len(self.dataset.buckets), generator=g).tolist()
        indices = []
        for bct_idx in bucket_index:
            bucket = self.dataset.buckets[bct_idx]
            indices.extend([bucket[i] for i in torch.randperm(len(bucket), generator=g).tolist()])

        return indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ReplicasDistributedSampler(Sampler, DynamicSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if not hasattr(dataset, 'buckets') or not isinstance(dataset.buckets, list):
            raise ValueError("Dataset must have a 'buckets' attribute of type list")
        if not all(isinstance(bucket, list) for bucket in dataset.buckets):
            raise ValueError("All buckets must be lists")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.bucket_arange()
        else:
            indices = []
            for bct in self.dataset.buckets:
                indices.extend(bct)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if not len(indices) == self.total_size:
            raise ValueError(f"indices length should be equal to total_size, but got indices length: {len(indices)}, expected: {self.total_size}")

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        if not len(indices) == self.num_samples:
            raise ValueError(f"indices length should be equal to num_samples, but got indices length: {len(indices)}, expected: {self.num_samples}")

        return iter(indices)

    def bucket_arange(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        replicas_buckets = [[] for _ in range(self.num_replicas)]
        bucket_index = torch.randperm(len(self.dataset.buckets), generator=g).tolist()
        for bct_idx in bucket_index:
            bct = [self.dataset.buckets[bct_idx][i] for i in torch.randperm(len(self.dataset.buckets[bct_idx]), generator=g).tolist()]
            for i in range(len(bct)):
                replicas_buckets[i].append(bct[i])

        replicas_bucket_index = torch.randperm(self.num_replicas, generator=g).tolist()
        for bct_idx in replicas_bucket_index:
            indices.extend(replicas_buckets[bct_idx])

        return indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch