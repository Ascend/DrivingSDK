# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# Copyright (c) 2023 PyG Team. All rights reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2018 Matthias Fey. All rights reserved.
# ------------------------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]

import os
import math
import json
from collections.abc import Mapping
from typing import TypeVar, Optional, Iterator, List, Optional, Sequence, Union
from functools import partial

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, BatchSampler, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

from ..dataset.utils.dynamic_dataset import UniformBucketingDynamicDataset
from ..dataset.utils.dynamic_sampler import DynamicDistributedSampler


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):
        return self(batch)


class AgentDynamicDataset(UniformBucketingDynamicDataset):
    def get_agents_num(self):
        '''
        Get agents number from json file.
        Used to sort dataset by cluster based on agents number.
        '''
        if self.split != 'train':
            return
        agents_num_file_name = f"{self.split}_agents_num.json"
        agents_num_file_path = os.path.join(self.processed_dir, agents_num_file_name)
        if os.path.exists(agents_num_file_path):
            with open(agents_num_file_path, "r") as handle:
                self.agents_num = json.load(handle)
            return

        for idx in tqdm(range(self._num_samples)):
            data = self.get(idx)
            self.agents_num[idx] = data["agent"]['num_nodes']

        with open(agents_num_file_path, 'w') as handle:
            json.dump(self.agents_num, handle)

    def sorting(self):
        self.indice = torch.arange(self._num_samples)
        self.agents_indices = torch.tensor(self.agents_num)[self.indice]

        # Clustering samples in dataset.
        # Samples will be clustered into 10 classes, each class has samples with similar agents number.
        self.clusters = self.agents_indices // 10
        self.max_cluster = int(self.clusters.max().item()) + 1

        self.sorted_ids = self.indice[self.clusters.argsort()]
        self.sorted_clusters = self.clusters.sort()[0]

    def bucketing(self):
        self.buckets = []
        start_idx = 0
        
        # Bucketing samples after cluster and sort.
        # Each cluster class will be bucketed into a bucket.
        # `start_idx` and `end_idx` is used to find cluster start index and end index.
        for clst_idx in range(self.max_cluster):
            end_idx = torch.searchsorted(self.sorted_clusters, clst_idx + 1, side='left')
            
            if end_idx > start_idx:
                current_cluster = self.sorted_ids[start_idx:end_idx]
                self.buckets.append(current_cluster.tolist())
                start_idx = end_idx


class DynamicBatchSampler(BatchSampler):
    def __init__(self, 
                 dataset: Dataset, 
                 sampler, 
                 batch_size: int, 
                 drop_last: bool = False) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.lengths = dataset.agents_num
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch


class AgentDynamicBatchSampler(DynamicDistributedSampler):
    def __init__(self, 
                 dataset: Dataset, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 shuffle: bool = True, 
                 seed: int = 0, 
                 drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            indices = self.bucket_arange()
        else:
            indices = []
            for bct in self.dataset.buckets:
                indices.extend(bct)

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            indices = torch.tensor(indices)
            indices = torch.cat([indices, indices[:padding_size]])
        else:
            indices = torch.tensor(indices)
            indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        if not len(indices) == self.num_samples:
            raise ValueError(f"indices length should be equal to num_samples, but got indices length: {len(indices)}, expected: {self.num_samples}")

        return iter(indices)


class AgentDynamicBatchDataLoader(DataLoader):
    def __init__(self, 
                 dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
                 batch_size: int,
                 train_batch_size: int,
                 shuffle: bool = True,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None,
                 **kwargs) -> None:
        kwargs.pop('collate_fn', None)
        kwargs.pop('batch_sampler', None)

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        sampler = AgentDynamicBatchSampler(dataset, shuffle=True)

        super().__init__(
            dataset, collate_fn=Collater(follow_batch, exclude_keys),
            batch_sampler=DynamicBatchSampler(dataset, sampler, train_batch_size), **kwargs)