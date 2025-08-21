# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, List


# Abstract Base Class: concrete subclasses must implement both `sorting` and `bucketing` methods
class DynamicDataset(ABC):
    def __init__(self) -> None:
        self.sorted_ids = None
        self.buckets = None

    @abstractmethod
    def sorting(self, *args: Any, **kwargs: Any) -> List:
        """
        Return:
            self.sorted_ids(List).
        """
        pass

    @abstractmethod
    def bucketing(self, *args: Any, **kwargs: Any) -> List:
        """
        Return:
            self.buckets(List).
        """
        pass


class UniformBucketingDynamicDataset(DynamicDataset):
    @abstractmethod
    def sorting(self, *args: Any, **kwargs: Any) -> List:
        pass

    def bucketing(self, *args: Any, **kwargs: Any) -> List:
        """`Performing a fixed number of bucketings according to the sorted list based on 'num_bucket' parameter.`
        Args:
            num_buckets(int): Number of Buckets.
        Return:
            buckets(List):  Store the bucketing results
        """
        if "num_buckets" not in kwargs:
            raise KeyError("Missing required argument: 'num_buckets'.")

        num_buckets = kwargs['num_buckets']
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise ValueError("'num_buckets' must be a positive integer.")

        buckets = []
        bucket_size = len(self.sorted_ids) // num_buckets
        remainder_size = len(self.sorted_ids) % num_buckets
        bucket_start = 0

        for bkt_id in range(num_buckets):
            bucket_end = bucket_start + bucket_size + (1 if bkt_id < remainder_size else 0)
            buckets.append(self.sorted_ids[bucket_start: bucket_end])
            bucket_start = bucket_end

        return buckets


class CapacityBucketingDynamicDataset(DynamicDataset):
    @abstractmethod
    def sorting(self, *args: Any, **kwargs: Any) -> List:
        pass

    def bucketing(self, *args: Any, **kwargs: Any) -> List:
        """`Dynamically bucketize according to the sorted list based on 'bucket_capacity' parameter.`
        Args:
            bucket_capacity(int)
        Return:
            buckets(List):  Store the bucketing results
        """
        if "bucket_capacity" not in kwargs:
            raise KeyError("Missing required argument: 'bucket_capacity'.")

        bucket_capacity = kwargs['bucket_capacity']
        if not isinstance(bucket_capacity, int) or bucket_capacity <= 0:
            raise ValueError("'bucket_capacity' must be a positive integer.")

        buckets = []

        for idx in range(0, len(self.sorted_ids), bucket_capacity):
            buckets.append(self.sorted_ids[idx: idx + bucket_capacity])
        return buckets