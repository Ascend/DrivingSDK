import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.dataset.utils import DynamicSampler, DynamicDistributedSampler, ReplicasDistributedSampler
from mx_driving.dataset.agent_dataset import DynamicBatchSampler, AgentDynamicBatchSampler, AgentDynamicBatchDataLoader


class TestDynamicSampler(TestCase):
    def test_abstract_methods(self):
        with self.assertRaises(TypeError):
            DynamicSampler()


class TestDynamicDistributedSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.buckets = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        self.mock_dataset.__len__ = MagicMock(return_value=10)

    @patch('torch.distributed.is_available')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_init_with_distributed(self, mock_get_rank, mock_get_world_size, mock_is_available):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        sampler = DynamicDistributedSampler(self.mock_dataset)

        self.assertEqual(sampler.num_replicas, 2)
        self.assertEqual(sampler.rank, 0)
        self.assertEqual(sampler.num_samples, 5)
        self.assertEqual(sampler.total_size, 10)

    def test_init_without_distributed(self):
        with patch('torch.distributed.is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                DynamicDistributedSampler(self.mock_dataset)

    def test_init_with_invalid_dataset(self):
        invalid_dataset = MagicMock()
        invalid_dataset.buckets = "not a list"

        with self.assertRaises(ValueError):
            DynamicDistributedSampler(invalid_dataset)

        invalid_dataset.buckets = [[1, 2], "not a list", [3, 4]]
        with self.assertRaises(ValueError):
            DynamicDistributedSampler(invalid_dataset)

    def test_init_with_custom_replicas_and_rank(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = DynamicDistributedSampler(self.mock_dataset, num_replicas=4, rank=1)

        self.assertEqual(sampler.num_replicas, 4)
        self.assertEqual(sampler.rank, 1)
        self.assertEqual(sampler.num_samples, 3)
        self.assertEqual(sampler.total_size, 12)

    def test_bucket_arange(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = DynamicDistributedSampler(self.mock_dataset, num_replicas=2, rank=0)
        sampler.epoch = 1
        result = sampler.bucket_arange()
        expected_length = 10
        self.assertEqual(len(result), expected_length)

        all_elements = []
        for bucket in self.mock_dataset.buckets:
            all_elements.extend(bucket)
        self.assertEqual(sorted(result), sorted(all_elements))

    def test_iter_with_shuffle(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = DynamicDistributedSampler(self.mock_dataset, num_replicas=2, rank=0, shuffle=True)
        with patch.object(sampler, 'bucket_arange', return_value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            indices = list(iter(sampler))
            self.assertEqual(indices, [1, 2, 3, 4, 5])
            self.assertEqual(len(indices), sampler.num_samples)

    def test_iter_without_shuffle(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = DynamicDistributedSampler(self.mock_dataset, num_replicas=2, rank=0, shuffle=False)
        indices = list(iter(sampler))

        all_elements = []
        for bucket in self.mock_dataset.buckets:
            all_elements.extend(bucket)
        expected = all_elements[:5]
        self.assertEqual(indices, expected)
        self.assertEqual(len(indices), sampler.num_samples)

    def test_set_epoch(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = DynamicDistributedSampler(self.mock_dataset, num_replicas=2, rank=0)
        sampler.set_epoch(5)
        self.assertEqual(sampler.epoch, 5)


class TestReplicasDistributedSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.buckets = [[1, 2], [3, 4], [5, 6]]
        self.mock_dataset.__len__ = MagicMock(return_value=6)

    def test_init_with_valid_dataset(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = ReplicasDistributedSampler(self.mock_dataset, num_replicas=2, rank=0)
        self.assertEqual(sampler.num_replicas, 2)
        self.assertEqual(sampler.rank, 0)
        self.assertEqual(sampler.num_samples, 3)
        self.assertEqual(sampler.total_size, 6)

    def test_bucket_arange(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = ReplicasDistributedSampler(self.mock_dataset, num_replicas=2, rank=0)
        sampler.epoch = 1
        result = sampler.bucket_arange()
        expected_length = 6
        self.assertEqual(len(result), expected_length)

        all_elements = []
        for bucket in self.mock_dataset.buckets:
            all_elements.extend(bucket)
        self.assertEqual(sorted(result), sorted(all_elements))

    def test_iter_with_shuffle(self):
        with patch('torch.distributed.is_available', return_value=False):
            sampler = ReplicasDistributedSampler(self.mock_dataset, num_replicas=2, rank=0, shuffle=True)
        with patch.object(sampler, 'bucket_arange', return_value=[1, 2, 3, 4, 5, 6]):
            indices = list(iter(sampler))
            self.assertEqual(indices, [1, 2, 3])
            self.assertEqual(len(indices), sampler.num_samples)


class TestDynamicBatchSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.agents_num = {0: 15, 1: 25, 2: 5, 3: 35, 4: 45}

        self.mock_sampler = MagicMock()
        self.mock_sampler.__iter__ = MagicMock(return_value=iter([0, 1, 2, 3, 4]))

    def test_iter_without_drop_last(self):
        batch_sampler = DynamicBatchSampler(
            self.mock_dataset, self.mock_sampler, batch_size=2, drop_last=False)
        batches = list(batch_sampler)
        expected_batches = [[0, 1], [2, 3], [4]]
        self.assertEqual(batches, expected_batches)

    def test_iter_with_drop_last(self):
        batch_sampler = DynamicBatchSampler(
            self.mock_dataset, self.mock_sampler, batch_size=2, drop_last=True)
        batches = list(batch_sampler)
        expected_batches = [[0, 1], [2, 3]]
        self.assertEqual(batches, expected_batches)


class TestAgentDynamicBatchSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.buckets = [[0, 1], [2, 3], [4]]
        self.mock_dataset.__len__ = MagicMock(return_value=5)

    @patch('torch.distributed.is_available')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_init_with_drop_last(self, mock_get_rank, mock_get_world_size, mock_is_available):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        sampler = AgentDynamicBatchSampler(
            self.mock_dataset, drop_last=True
        )

        self.assertEqual(sampler.num_samples, 2)
        self.assertEqual(sampler.total_size, 4)

    @patch('torch.distributed.is_available')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_iter_with_drop_last(self, mock_get_rank, mock_get_world_size, mock_is_available):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        sampler = AgentDynamicBatchSampler(
            self.mock_dataset, drop_last=True
        )

        with patch.object(sampler, 'bucket_arange', return_value=[0, 1, 2, 3, 4]):
            indices = list(sampler)
            self.assertEqual(indices, [0, 2])
            self.assertEqual(len(indices), sampler.num_samples)

    @patch('torch.distributed.is_available')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_iter_without_drop_last(self, mock_get_rank, mock_get_world_size, mock_is_available):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        sampler = AgentDynamicBatchSampler(self.mock_dataset, drop_last=False)

        with patch.object(sampler, 'bucket_arange', return_value=[0, 1, 2, 3, 4]):
            indices = list(sampler)
            self.assertEqual(indices, [0, 2, 4])
            self.assertEqual(len(indices), sampler.num_samples)


class TestAgentDynamicBatchDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.agents_num = {0: 15, 1: 25, 2: 5, 3: 35, 4: 45}

    @patch('mx_driving.dataset.agent_dataset.AgentDynamicBatchSampler')
    @patch('mx_driving.dataset.agent_dataset.DynamicBatchSampler')
    @patch('mx_driving.dataset.agent_dataset.Collater')
    def test_init(self, mock_collater, mock_batch_sampler, mock_sampler):
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance

        mock_batch_sampler_instance = MagicMock()
        mock_batch_sampler.return_value = mock_batch_sampler_instance

        mock_collater_instance = MagicMock()
        mock_collater.return_value = mock_collater_instance

        dataloader = AgentDynamicBatchDataLoader(
            self.mock_dataset,
            batch_size=2,
            train_batch_size=2,
            shuffle=True,
            follow_batch=['agent'],
            exclude_keys=['exclude_key']
        )

        mock_sampler.assert_called_once_with(self.mock_dataset, shuffle=True)
        mock_batch_sampler.assert_called_once_with(
            self.mock_dataset, mock_sampler_instance, 2
        )
        mock_collater.assert_called_once_with(['agent'], ['exclude_key'])

        self.assertEqual(dataloader.dataset, self.mock_dataset)
        self.assertEqual(dataloader.collate_fn, mock_collater_instance)
        self.assertEqual(dataloader.batch_sampler, mock_batch_sampler_instance)

if __name__ == '__main__':
    run_tests()