import unittest
from unittest.mock import MagicMock, patch

from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.dataset.utils import DynamicDataset, UniformBucketingDynamicDataset, CapacityBucketingDynamicDataset
from mx_driving.dataset.point_dataset import PointCloudDynamicDataset
from mx_driving.dataset.agent_dataset import AgentDynamicDataset


class TestDynamicDataset(TestCase):
    def test_abstract_methods(self):
        with self.assertRaises(TypeError):
            DynamicDataset()

    def test_abstract_methods_implementation(self):
        class IncompleteDataset(DynamicDataset):
            pass

        with self.assertRaises(TypeError):
            IncompleteDataset()


class TestUniformBucketingDynamicDataset(TestCase):
    def setUp(self):
        super().setUp()

        class TestDataset(UniformBucketingDynamicDataset):
            def sorting(self, *args, **kwargs):
                self.sorted_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return self.sorted_ids

        self.dataset = TestDataset()
        self.dataset.sorting()

    def test_bucketing_missing_num_buckets(self):
        with self.assertRaises(KeyError):
            self.dataset.bucketing()

    def test_bucketing_invalid_num_buckets(self):
        with self.assertRaises(ValueError):
            self.dataset.bucketing(num_buckets=0)
        with self.assertRaises(ValueError):
            self.dataset.bucketing(num_buckets=-1)
        with self.assertRaises(ValueError):
            self.dataset.bucketing(num_buckets="invalid")

    def test_bucketing_division(self):
        even_buckets = self.dataset.bucketing(num_buckets=5)
        even_expected = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        uneven_buckets = self.dataset.bucketing(num_buckets=3)
        uneven_expected = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
        self.assertEqual(even_buckets, even_expected)
        self.assertEqual(uneven_buckets, uneven_expected)

    def test_bucketing_single_bucket(self):
        buckets = self.dataset.bucketing(num_buckets=1)
        expected = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        self.assertEqual(buckets, expected)

    def test_bucketing_more_buckets_than_items(self):
        buckets = self.dataset.bucketing(num_buckets=15)
        self.assertEqual(len(buckets), 15)
        self.assertEqual(sum(len(bucket) for bucket in buckets), 10)


class TestCapacityBucketingDynamicDataset(TestCase):
    def setUp(self):
        super().setUp()

        class TestDataset(CapacityBucketingDynamicDataset):
            def sorting(self, *args, **kwargs):
                self.sorted_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return self.sorted_ids

        self.dataset = TestDataset()
        self.dataset.sorting()

    def test_bucketing_missing_bucket_capacity(self):
        with self.assertRaises(KeyError):
            self.dataset.bucketing()

    def test_bucketing_invalid_bucket_capacity(self):
        with self.assertRaises(ValueError):
            self.dataset.bucketing(bucket_capacity=0)
        with self.assertRaises(ValueError):
            self.dataset.bucketing(bucket_capacity=-1)
        with self.assertRaises(ValueError):
            self.dataset.bucketing(bucket_capacity="invalid")

    def test_bucketing_capacity(self):
        even_buckets = self.dataset.bucketing(bucket_capacity=5)
        even_expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        uneven_buckets = self.dataset.bucketing(bucket_capacity=4)
        uneven_expected = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
        self.assertEqual(even_buckets, even_expected)
        self.assertEqual(uneven_buckets, uneven_expected)

    def test_bucketing_larger_capacity(self):
        buckets = self.dataset.bucketing(bucket_capacity=20)
        expected = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        self.assertEqual(buckets, expected)

    def test_bucketing_single_item_buckets(self):
        buckets = self.dataset.bucketing(bucket_capacity=1)
        expected = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        self.assertEqual(buckets, expected)


class TestPointCloudDynamicDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_dataset = MagicMock()
        self.mock_dataset.infos = [
            {'voxel_num': 100},
            {'voxel_num': 200},
            {'voxel_num': 50},
            {'voxel_num': 300},
            {'voxel_num': 150}
        ]

    def test_sorting(self):
        dataset = PointCloudDynamicDataset()
        dataset.infos = self.mock_dataset.infos
        dataset.sorting()

        sorted_voxel_nums = [dataset.infos[i]['voxel_num'] for i in dataset.sorted_ids]
        expected_voxel_nums = [300, 200, 150, 100, 50]  # 按voxel_num降序排列
        self.assertEqual(sorted_voxel_nums, expected_voxel_nums)

        sorted_voxels = [(i, info['voxel_num']) for i, info in enumerate(self.mock_dataset.infos)]
        sorted_indexs = sorted(sorted_voxels, key=lambda x: x[1], reverse=True)
        expected_sorted_ids = [i for i, _ in sorted_indexs]
        self.assertEqual(dataset.sorted_ids, expected_sorted_ids)


class TestAgentDynamicDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = AgentDynamicDataset()
        self.dataset.split = 'train'
        self.dataset.processed_dir = '/mock/processed/dir'

    def test_sorting(self):
        self.dataset.agents_num = [15, 25, 5, 35, 45]
        self.dataset._num_samples = 5
        self.dataset.sorting()
        self.assertEqual(self.dataset.max_cluster, 5)
        self.assertTrue(all(
            self.dataset.sorted_clusters[i] <= self.dataset.sorted_clusters[i + 1]
            for i in range(len(self.dataset.sorted_clusters) - 1)
        ))

    def test_bucketing(self):
        self.dataset.agents_num = [15, 25, 5, 35, 45]
        self.dataset._num_samples = 5
        self.dataset.sorting()
        self.dataset.bucketing()

        self.assertEqual(len(self.dataset.buckets), self.dataset.max_cluster)
        for bucket in self.dataset.buckets:
            if bucket:
                cluster_id = self.dataset.clusters[bucket[0]]
                self.assertTrue(all(
                    self.dataset.clusters[sample_id] == cluster_id
                    for sample_id in bucket
                ))

if __name__ == '__main__':
    run_tests()