# 自驾模型迁移优化指导
## 1. 一键Patcher快速迁移

为方便用户快速迁移原本基于GPU/CUDA生态实现的模型至昇腾生态，一键Patcher提供了一个预定义的PatcherBuilder实例，帮助用户仅需添加几行代码即可使模型在昇腾NPU上进行训练。

### 定位模型训练脚本入口

找到模型的训练脚本，通常命名为`train.py`，定位到它的入口函数，通常为：

```Python
if __name__ == '__main__':
	main()
```

### 应用Context

```Python
from mx_driving.patcher import default_patcher_builder
#......
if __name__ == '__main__':
	with default_patcher_builder.build() as patcher:
		main()
```

然后照常运行模型即可。具体可以参考[一键patcher](patcher.md)

## 2. 模型优化
### 2.1 框架特性
torch_npu框架可以使能多种特性, 包括环境变量配置，框架编译优化以及高性能内存库等。
#### 2.1.1 通用环境变量配置及介绍
| 环境变量配置  |  含义 |   
|--------------|-------|
|export TASK_QUEUE_ENABLE=2 | 设置是否开启task queue，0-关闭/1-Level 1优化/2-Level 2优化 |
|export CPU_AFFINITY_CONF=1 | 设置任务绑核，减少调度开销，0-关闭/1-粗粒度绑核/2-细粒度绑核|
|export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" | 使能内存池扩展段功能，此设置将指示缓存分配器创建特定的内存块分配|
|export COMBINED_ENABLE=1 | 用于优化非连续两个算子组合类场景，0-关闭/1-开启|
#### 2.1.2 编译优化
框架支持Python, PyTorch, torch_npu的编译优化，能够缩短链接耗时和内存占用，具体使用方式可参考：
| 编译优化方式  |  参考链接 |   
|--------------|-------|
| Python | https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0064.html|
| PyTorch | https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0065.html |
| torch_npu | https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0066.html |

#### 2.1.3 高性能内存库替换
tcmalloc（即Thread-Caching Malloc）是一个通用的内存分配器，通过引入多层次缓存结构、减少互斥锁竞争、优化大对象处理流程等手段，在保证低延迟的同时也提升了整体性能表现。这对于需要频繁进行内存操作的应用来说尤为重要，尤其是在高并发场景下能够显著改善系统响应速度和服务质量。 使用可参考：https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0068.html

### 2.2 高性能算子替换
通过使用融合算子可以减少小算子下发，提升模型性能。
首先可以替换融合优化器，详细可参考：https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0036.html
自驾模型通常用adamw优化器，替换如下：
| 优化器  |  样例源码 |  修改后代码 | 
|--------------|-------|----------|
| adamw | optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) | optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=args.lr) |

进一步可替换计算融合算子，DrivingSDK仓高性能算子介绍可参考：https://gitcode.com/Ascend/DrivingSDK/blob/branch_v7.3.0/docs/api/README.md
常用融合算子举例：
| 算子名 | 使用方式 |
| ------| ---------|
| deform_conv2d | from mx_driving import deform_conv2d, DeformConv2dFunction |
| npu_add_relu | from mx_driving import npu_add_relu 适用于resnet作为backbone的模型优化，替换 out+=identity out=self.relu(out)|
| npu_max_pool2d | from mx_driving import npu_max_pool2d 替换 nn.MaxPool2d操作 |
| multi_scale_deformable_attn | from mx_driving import multi_scale_deformable_attn |
| bev_pool_v3 | from mx_driving import bev_pool_v3 |
| SparseConv3d | from mx_driving import SparseConv3d |
| SubMConv3d | from mx_driving import SubMConv3d |
#### 【算子替换示例】

当采集完profiling后，查看算子耗时统计，分析耗时占比大的算子，进行替换，以BEVFormer模型为例，multi_scale_deformable_attn算子正反向在模型中占比很高，需要替换成mxDriving中的亲和算子：
<img src="op_statistic.png" alt="算子耗时占比" width="800" align="center">


替换过程如下，算子详细参数参考[算子清单](../api/README.md)，算子使用位置为projects/mmdet3d_plugin/bevformer/modules/decoder.py：
- 头文件引入mx_driving
```python
import mx_driving
```
- 找到算子使用的位置，位置在325行，原始代码为：
```python
    if torch.cuda.is_available() and value.is_cuda:

        # using fp16 deformable attention is unstable because it performs many sum operations
        if value.dtype == torch.float16:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        else:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)
    else:
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
```
替换为mx_driving仓中算子：
```python
    output = mx_driving.multi_scale_deformable_attn(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
```

### 2.3 自驾模型host bound问题优化
自动驾驶算法有很多slice、gather、sort等小算子，2D图像特征转换到BEV空间，涉及大量投影、插值、采样等操作，点云数据的体素化处理、稀疏卷积等。自动驾驶算法中涉及很多逻辑处理，如检测算法的target assign，规控类算法lovaz loss，以及一些对groundtruth处理的操作。Host bound问题：小算子下发多，CPU计算处理逻辑多、负载大，算子没有NPU高性能替换等造成host bound问题严重。
#### 2.3.1 减少算子下发次数
在车道线检测、多边形实例分割等任务中，目标通常由连续的点序列构成，通过插值监督，模型可以更好地学习这种连续几何结构。例如Pivotnet中,interpolate函数使用循环每次计算一个插值点，此函数单Step调用90次，引入大量的小算子及Free time。减少算子的下发次数的主要解决方案是通过等价的api进行脚本替换，规控模型决策模块以及感知模型的数据处理等模块由于开源模型代码一般没有经过优化，经常会出现大量的for循环切片操作，可通过一些mask相关的api替换这块大量的小算子逻辑。在目标检测等任务中，目标通常由连续的点序列构成，常会出现插值相关的逻辑，若模型中为循环单点计算，可尝试修改为大粒度运算减少小算子下发。
例如在模型源代码中，存在循环单点计算插值逻辑：
```python
@staticmethod
def interpolate(start_pt, end_pt, inter_num):
    res = torch.zeros((inter_num, 2), dtype=start_pt.dtype, device=start_pt.device)
    num_len = inter_num + 1 # segment num.
    for i in range(1, num_len):
        ratio = i / num_len
        res[i-1] = (1 - ratio) * start_pt + ratio * end_pt
    return res
```
可向量化增大数据处理粒度避免循环单点插值，优化后代码：
```python
@staticmethod
def interpolate(start_pt, end_pt, inter_num):
    ratios = torch.arange(1, inter_num + 1, dtype=start_pt.dtype, device=start_pt.device) / (inter_num + 1)
    ratios = ratios.view(-1, 1)
    return (1 - ratios) * start_pt + ratios * end_pt
```
#### 2.3.2 Cpu计算转为Npu小算子拼接
规控模型规模很小，整体端到端耗时较短，因此如果模型中还存在cpu计算会成为模型严重的性能瓶颈，下图是MultiPath++模型中调用了torch.linalg.slogdet这个接口用来评估预测轨迹与真实轨迹之间的差异，但当前torch.nn.linalg线性代数库在昇腾上没有对应的npu算子，因此走的原生torch cpu的计算，会引入较长的cpu计算时间，可通过同等小算子拼接脚本替换。模型源代码如下：
```python
def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
```
可替换为：
```python
def custom_logdet(covariance_matrices):
    U, S, V = torch.svd(A)
    abs_det = torch.prod(S)
    return torch.log(abs_det)

def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * custom_logdet(covariance_matrices).unsqueeze(-1))
```
#### 2.3.3 减少同步引起的流水等待
匈牙利匹配算法在目标检测、实例分割等模型中有非常广泛的应用，用于匹配预测和真实对象，pivotnet当中匈牙利算法的forward方法中引入了大量的aclrtSynchronizeStream，打断模型流水，常见原因包括调用aten::item，.npu()，.cpu()操作等。
例如其中的这段代码根据 valid_lens（每个样本的有效长度）生成一个掩码 gt_pts_mask，其中有效长度内的位置为 1，其余位置为 0，其中获取ll会隐式调用.item()打断流水：
```python
for i, ll in enumerate(targets[0]["valid_len"][cid]):
    gt_pts_mask[i][:ll] = 1
```
通过修改脚本规避：
```python
valid_lens = torch.tensor(targets[0]["valid_len"][cid], device=gt_pts.device, dtype=torch.long)
row_indices = torch.arange(n_pt, device=gt_pts.device).expand(len(valid_lens), -1)
gt_pts_mask = (row_indices < valid_lens.unsqueeze(1)).to(dtype=torch.float32)
```

## 3 迁移优化示例-BEVDet模型
### 3.1 模型介绍
BEVDet是一种用于3D目标检测的深度学习模型，可以从一个俯视图像中检测出三维空间中的物体，并预测他们的位置、大小和朝向。在自动驾驶、智能交通等领域中有广泛应用。其基于深度学习技术，使用卷积神经网络和残差网络，在训练过程中使用了大量的3D边界框数据，以优化模型的性能和准确性。
通过使用如下的优化方法，提升模型性能：
| 优化手段 | 修改文件 |
| -------  | --------|
|export TASK_QUEUE_ENABLE=2 开启L2流水优化 | 环境变量配置 |
| export CPU_AFFINITY=1 开启粗粒度绑核 | 环境变量配置  |
| export COMBINED_ENABLE=1 优化非连续两个算子组合 | 环境变量配置 |
| 替换NpuFusedAdamW融合优化器 | configs/bevdet/bevdet-r50.py |
| 替换bev_pool_v3融合算子 | mmdet3d/models/necks/view_transformer.py |
| 替换npu_gaussian融合算子 | mmdet3d/models/dense_heads/centerpoint_head.py |

### 3.2 模型迁移优化
#### 3.2.1 代码实现
参考实现：
```python
url=https://github.com/HuangJunJie2017/BEVDet.git
commit_id=58c2587a8f89a1927926f0bdb6cde2917c91a9a5
```
适配昇腾 AI 处理器的实现：
```python
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/BEVDet
```
#### 3.2.2 安装依赖
1. 自驾模型当前主要以PyTorch2.1进行配置， 以PyTorch2.1为例进行介绍，首先进行模型相关依赖的安装，可以创建一个requirements.txt，添加如下依赖及版本：
```python
setuptools==65.7.0
torchvision==0.16.0
nuscenes-devkit==1.1.11
numba==0.58.1
numpy==1.23.1
lyft_dataset_sdk
nuscenes-devkit==1.1.11
scikit-image
trimesh==2.35.39
tensorboard
networkx
attrs
decorator
sympy
cffi
pyyaml
pathlib2
psutil
protobuf==4.25.0
scipy
requests
absl-py
yapf
mmdet==2.28.2
mmsegmentation==0.30.0
ninja
```
通过pip install -r requirements.txt进行安装。

2. mmcv三方库安装，目前1.7.2版本支持npu，需要手动下载代码，进行源码编译安装：
```python
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
```
适配pytorch2.x版本，修改mmcv/mmcv/parallel/distributed.py文件159行，原始代码为：
```python
module_to_run = self._replicated_tensor_module if
self._use_replicated_tensor_module else self.module
```
替换为：
```python
module_to_run = self.module
```
进入mmcv根目录，使用如下指令编译安装mmcv：
```python
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
```
#### 3.2.3 迁移适配
1. 首先下载模型官方源码并指定commit id：
```python
git clone https://github.com/HuangJunJie2017/BEVDet.git
cd BEVDet
git checkout 58c2587a8f89a1927926f0bdb6cde2917c91a9a5
```
2. 在tools/train.py和tools/test.py中添加自动迁移的代码：
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
3. ddp相关调用适配成NPUDataParallel, NPUDistributedDataParallel，修改tools/test.py文件：
```python
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
```
改为：
```python
from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
```
并将原始代码：
```python
model = MMDataParallel(model, device_ids=cfg.gpu_ids)
…
model = MMDistributedDataParallel(
model.cuda(),
```
改为：
```python
model = NPUDataParallel(model.npu(), device_ids=cfg.gpu_ids)
…
model = NPUDistributedDataParallel(
model.npu(),
```
修改mmdet3d/apis/train.py文件，将原始代码：
```python
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
```
改为：
```python
from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
```
将代码中相应的代码：
```python
model = MMDistributedDataParallel( 和 model = MMDataParallel(
```
改为：
```python
model = NPUDistributedDataParallel( 和 model = NPUDataParallel(
```
4. 适配Pytorch2.1，添加--local-rank参数，在tools/train.py和tools/test.py中原始代码：
```python
parser.add_argument('--local_rank', type=int, default=0)
```
前添加：
```python
parser.add_argument('--local-rank', type=int, default=0)
```
5. 将cuda相关代码去掉， 在mmdet3d/models/detectors/init.py文件中，将原始代码：
```python
from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVDetTRT, BEVStereo4D
```
改为：
```python
from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVStereo4D
```
注释掉原始代码：
```python
from .dal import DAL
```
将原始代码：
```python
all = [
'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
'BEVDetTRT', 'BEVStereo4D', 'BEVStereo4DOCC'
]
```
改为：
```python
all = [
'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
'BEVStereo4D', 'BEVStereo4DOCC'
]
```
修改mmdet3d/models/detectors/bevdet.py文件，删除或注释如下代码：
```python
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
…
@DETECTORS.register_module()
class BEVDetTRT(BEVDet):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                            ranks_depth, ranks_feat, ranks_bev,
                            interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)
```
6. mmdet_3d版本兼容适配，修改mmdet3d/\_\_init\_\_.py文件，将原始代码：
```python
mmcv_maximum_version = '1.7.0'
```
改为：
```python
mmcv_maximum_version = '1.7.2'
```
注释或删除如下原始代码:
```python
import mmseg
…
mmseg_version = digit_version(mmseg.__version__)
assert (mmseg_version >= digit_version(mmseg_minimum_version)
    and mmseg_version <= digit_version(mmseg_maximum_version)),
    f'MMSEG=={mmseg.__version__} is used but incompatible. '
    f'Please install mmseg>={mmseg_minimum_version}, '
    f'<={mmseg_maximum_version}.'
```
7. 原始代码中bev_pool_v2算子为cuda代码，需要替换DrivingSDK仓的bev_pool_v3，更加亲和高效，安装DrivingSDK，参考DrivingSDK: https://gitcode.com/Ascend/DrivingSDK/blob/branch_v7.3.0/README.md 
在mmdet3d/models/necks/view_transformer.py文件中，删除原始代码：
```python
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
```
并加入引用
```python
from mx_driving.point import bev_pool_v3
```
在代码中将LSSViewTransformer类中voxel_pooling_v2函数中原始代码：
```python
bev_feat = bev_pool_v2(depth, feat, ranks_depth,
        ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts,
        interval_lengths)
```
替换为：
```python
bev_feat = bev_pool_v3(depth, feat, ranks_depth,
        ranks_feat, ranks_bev,
        bev_feat_shape)
```
将LSSViewTransformer类中view_transform_core函数中原始代码：
```python
bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
        self.ranks_feat, self.ranks_bev,
        bev_feat_shape, self.interval_starts,
        self.interval_lengths)
```
替换为：
```python
bev_feat = bev_pool_v3(depth, feat, self.ranks_depth,
        self.ranks_feat, self.ranks_bev,
        bev_feat_shape)
```
8. npu上matmul不支持6维以上，因此需要进行修改适配，替换mmdet3d/models/necks/view_transformer.py文件中LSSViewTransformer类中get_lidar_coor函数：
```python
def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
    """Calculate the locations of the frustum points in the lidar
    coordinate system.

    Args:
        rots (torch.Tensor): Rotation from camera coordinate system to
            lidar coordinate system in shape (B, N_cams, 3, 3).
        trans (torch.Tensor): Translation from camera coordinate system to
            lidar coordinate system in shape (B, N_cams, 3).
        cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
            (B, N_cams, 3, 3).
        post_rots (torch.Tensor): Rotation in camera coordinate system in
            shape (B, N_cams, 3, 3). It is derived from the image view
            augmentation.
        post_trans (torch.Tensor): Translation in camera coordinate system
            derived from image view augmentation in shape (B, N_cams, 3).

    Returns:
        torch.tensor: Point coordinates in shape
            (B, N_cams, D, ownsample, 3)
    """
    B, N, _, _ = sensor2ego.shape

    # post-transformation
    # B x N x D x H x W x 3
    points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
    B, N, D, H, W, _ = points.shape
    points = points.view(B, N, D*H*W, 3, 1)
    points = torch.inverse(post_rots).view(B, N, 1, 3, 3).matmul(points)

    # cam_to_ego
    points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 3)
    combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
    points = combine.view(B, N, 1, 3, 3).matmul(points).squeeze(-1)
    points += sensor2ego[:,:,:3, 3].view(B, N, 1, 3)
    points = bda[:, :3, :3].view(B, 1, 1, 3, 3).matmul(
        points.unsqueeze(-1)).squeeze(-1)
    points += bda[:, :3, 3].view(B, 1, 1, 3)
    return points.view(B, N, D, H, W, 3)
```
#### 3.2.4 模型优化
1. 替换融合优化器，在configs/bevdet/bevdet-r50.py中，原始代码为：
```python
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
```
改为：
```python
optimizer = dict(type='NpuFusedAdamW', lr=2e-4, weight_decay=1e-07)
```

2. 替换npu_gaussian融合算子，在mmdet3d/models/dense_heads/centerpoint_head.py中添加引用:
```python
from mx_driving import npu_gaussian
```
替换get_targets_single函数，减少很多小算子下发，提高计算效率：
```python
def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
    """Generate training targets for a single sample.

    Args:
        gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
        gt_labels_3d (torch.Tensor): Labels of boxes.

    Returns:
        tuple[list[torch.Tensor]]: Tuple of target including
            the following results in order.

            - list[torch.Tensor]: Heatmap scores.
            - list[torch.Tensor]: Ground truth boxes.
            - list[torch.Tensor]: Indexes indicating the position
                of the valid boxes.
            - list[torch.Tensor]: Masks indicating which boxes
                are valid.
    """
    device = gt_labels_3d.device
    gt_bboxes_3d = torch.cat(
        (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
        dim=1).to(device)
    max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
    grid_size = torch.tensor(self.train_cfg['grid_size'])
    pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
    voxel_size = torch.tensor(self.train_cfg['voxel_size'])

    feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

    # reorganize the gt_dict by tasks
    task_masks = []
    flag = 0
    for class_name in self.class_names:
        task_masks.append([
            torch.where(gt_labels_3d == class_name.index(i) + flag)
            for i in class_name
        ])
        flag += len(class_name)

    task_boxes = []
    task_classes = []
    flag2 = 0
    for idx, mask in enumerate(task_masks):
        task_box = []
        task_class = []
        for m in mask:
            task_box.append(gt_bboxes_3d[m])
            # 0 is background for each task, so we need to add 1 here.
            task_class.append(gt_labels_3d[m] + 1 - flag2)
        task_boxes.append(torch.cat(task_box, axis=0).to(device))
        task_classes.append(torch.cat(task_class).long().to(device))
        flag2 += len(mask)
    draw_gaussian = draw_heatmap_gaussian
    heatmaps, anno_boxes, inds, masks = [], [], [], []

    for idx, task_head in enumerate(self.task_heads):
        heatmap = gt_bboxes_3d.new_zeros(
            (len(self.class_names[idx]), feature_map_size[1],
            feature_map_size[0]))

        if self.with_velocity:
            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                            dtype=torch.float32)
        else:
            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                            dtype=torch.float32)

        ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
        mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

        num_objs = min(task_boxes[idx].shape[0], max_objs)

        temp_classes = task_classes[idx]
        temp_boxes = task_boxes[idx]
        center_int, radius, mask, ind, anno_box = npu_gaussian(temp_boxes,
                                                            self.train_cfg['out_size_factor'],
                                                            self.train_cfg['gaussian_overlap'],
                                                            self.train_cfg['min_radius'],
                                                            voxel_size[0],
                                                            voxel_size[1],
                                                            pc_range[0],
                                                            pc_range[1],
                                                            feature_map_size[0],
                                                            feature_map_size[1],
                                                            self.norm_bbox,
                                                            self.with_velocity)
        for k in range(num_objs):
            cls_id = temp_classes[k] - 1

            width = temp_boxes[k][3]
            length = temp_boxes[k][4]
            width = width / voxel_size[0] / self.train_cfg[
                'out_size_factor']
            length = length / voxel_size[1] / self.train_cfg[
                'out_size_factor']

            if width > 0 and length > 0:
                draw_gaussian(heatmap[cls_id], center_int[k], radius[k].item())
        heatmaps.append(heatmap)
        anno_boxes.append(anno_box)
        masks.append(mask)
        inds.append(ind)

    return heatmaps, anno_boxes, inds, masks
```

