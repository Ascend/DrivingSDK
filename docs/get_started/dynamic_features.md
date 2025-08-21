# DrivingSDK负载均衡特性
自动驾驶场景中，涉及多种传感器数据和不同的道路交通环境，导致自动驾驶模型易出现计算负载不均衡的现象，造成性能瓶颈。

DrivingSDK实现了自动驾驶模型负载均衡特性，通过负载均衡策略，缓解计算负载不均衡瓶颈，提升模型性能。

## 背景
自动驾驶模型的数据集通常为多传感器数据，包含点云、雷达数据、照相机图像等。

根据模型任务的差异，不同道路场景下的数据量差别较大，例如，`QCNet`模型是一种用于轨迹预测的神经网络架构，使用`Argoverse2`数据集学习轨迹特征。数据集采用`Agents`表达道路场景中的交通参与目标，不同道路上的`Agents`数量差异较大，在分布式训练中造成了节点负载不均衡的现象，为了等待负载最大的节点，其他节点常常进入通信等待阶段，增加了额外的训练开销。

具有类似负载不均衡的模型还有Deformable DETR、CenterPoint3D等，因此，通过对模型进行动态的负载均衡处理，能够让分布式训练的各节点具有相近的负载，避免过多通信等待开销，提升模型性能。

## 特性介绍

### 1. `Dynamic Dataset`
DrivingSDK负载均衡特性提供`DynamicDataset`抽象类，该类包含`sorting`和`bucketing`两个抽象方法。
 - `sorting`抽象类方法
 根据模型中造成负载不均衡瓶颈的主要元素，对元素进行排序。

 - `bucketing`抽象类方法
 根据`sorting`方法结果，对数据集样本进行分桶。

用户可以自行继承该抽象类，实现抽象类方法，同时，DrivingSDK提供两个继承`DynamicDataset`的子类，均已实现`bucketing`类方法。
 - `UniformBucketingDynamicDataset`
 该类需传入`num_bucket`参数，作为分桶总量，数据集样本将按照`sorting`方法结果，均匀分布到所有桶内。

 - `CapacityBucketingDynamicDataset`
 该类需传入`bucket_capacity`参数，作为桶容量，数据集样本将按照`sorting`方法结果，每个桶中分配`bucket_capacity`个样本。

### 2. `Dynamic Sampler`
DrivingSDK负载均衡特性提供`DynamicSampler`抽象类，该类包含`bucket_arange`抽象方法。
 - `bucket_arange`抽象类方法
 根据`DynamicDataset`类`bucketing`方法结果，对分桶结果进行随机化。

用户可以自行继承该抽象类，实现抽象类方法，同时，DrivingSDK提供两个继承`DynamicSampler`的子类，均已实现`bucket_arange`类方法。
 - `DynamicDistributedSampler`
 该类进行两次随机化处理：
 1）对样本桶进行随机化
 2）在样本桶内，对桶内数据进行随机化
 适用于`UniformBucketingDynamicDataset`的分桶结果随机化。

 - `ReplicasDistributedSampler`
 该类进行三次随机化处理：
 1）对样本桶进行随机化
 2）在样本桶内，对桶内数据进行随机化
 3）根据分布式训练的节点数，将桶内数据均匀分布到每个节点上，并再次进行随机化
 适用于`CapacityBucketingDynamicDataset`的分桶结果随机化。

### 3. `Dynamic Transforms`
包含负载均衡预处理的函数。

## 使用方法
以CenterPoint3D模型为例，介绍DrivingSDK负载均衡特性的使用方法。

### 1. 确认引入负载不均衡瓶颈的元素
CenterPoint3D的数据集为点云，在模型训练过程中，首先会将点云转换为`voxel`张量，模型对`voxel`张量的尺寸敏感，当`voxel`较大时，模型计算耗时长，反之则耗时短。因此，确认引入负载不均衡瓶颈的元素为点云转换获得的`voxel`张量尺寸。

### 2. 按照`voxel`尺寸进行排序（`sorting`）
按照`model_examples/CenterPoint/README.md`中`准备数据集`一节生成数据集所需的序列化文件。在生成代码中，以下代码用于获取数据集样本的`voxel`张量尺寸。
```
from mx_driving.dataset.utils import get_voxel_number_from_mean_vfe
voxel_num = get_voxel_number_from_mean_vfe(data_path, ref_sd_rec['filename'], sweeps, max_sweeps)
info["voxel_num"] = voxel_num
```
其中，`get_voxel_number_from_mean_vfe`函数位于`dataset/utils/dynamic_transform.py`中，用户无需再对模型进行过多侵入式修改，即可获取`voxel`张量尺寸。

对模型文件`pcdet/datasets/nuscenes/nuscenes_dataset.py`进行修改，原代码为：
```
class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
            root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        ...
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
```
修改后为：
```
from mx_driving.dataset import PointCloudDynamicDataset
class NuScenesDataset(PointCloudDynamicDataset, DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        DatasetTemplate.__init__(
            self, dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        ...
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        if training:
            self.sorting()
            self.buckets = self.bucketing(bucket_capacity=8)
        else:
            self.buckets = None
```
`NuScenesDataset`在继承原有父类的基础上，还继承了`PointCloudDynamicDataset`类，因此可以直接调用类中的`sorting`和`bucketing`函数。

### 3. 对`bucketing`结果进行随机化
对模型文件`pcdet/datasets/__init__.py`进行修改，原代码为：
```
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```
修改后为：
```
from mx_driving.dataset import ReplicasDistributedSampler
sampler = ReplicasDistributedSampler(dataset)
```
完成修改后，模型即可正常训练，基于负载均衡特性的训练性能比未使用负载均衡特性的性能提升18%。