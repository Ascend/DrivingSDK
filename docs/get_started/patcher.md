# 快速迁移模型
mx_driving 提供了 `Patcher` monkey_patch 类来帮助用户快速迁移模型。

## 1. 解决mmcv系列版本冲突问题
假设你安装了mmcv 1.7.2，但是mmdet3d 需要 mmcv <= 1.7.0, 而mmdet 需要 mmcv <= 1.6.0, 这时候你就可以使用 mx_driving 来解决这个问题。

```python
from mx_driving.patcher import patch_mmcv_version
patch_mmcv_version("1.6.0")
```
注意，你可能需要在import mmcv 之前调用 `patch_mmcv_version` 函数，否则仍然可能会出现版本冲突问题。

## 2. 使用默认patcher
mx_driving 提供了一个默认的patcher，可以帮助用户快速迁移模型。

```python
from mx_driving.patcher import default_patcher_builder
with default_patcher_builder.build() as patcher:
    # train model here
```

## 3. 使用自定义patcher
你也可以使用 `PatcherBuilder` 类来创建一个自定义patcher。

```python
from mx_driving.patcher import PatcherBuilder, Patch
patcher_builder = PatcherBuilder()
patcher_builder.add_module_patch("torch", Patch(index))
with patcher_builder.build() as patcher:
    # train model here
```

## 4. 使用profiler
mx_driving 提供了一个profiler，可以帮助开发者快速定位性能瓶颈。

```python
with default_patcher_builder.with_profiling(path="/path/to/save/profiler/result", level=0).build() as patcher:
    # train model here
```
level 0: 最小膨胀，只记录NPU活动
level 1: 记录NPU和CPU活动
level 2: 记录NPU和CPU活动，并打印调用栈

## 5. 禁用某个patch

```python
with default_patcher_builder.disable_patches("msda", "index").build() as patcher:
    # train model here
```
## 6. 支持特性
- [x] 支持一键迁移npu（默认关闭私有格式）
- [x] 支持mmcv系列版本冲突问题
- [x] 支持自定义patcher
- [x] 支持profiler
- [x] 支持禁用patch
- [x] 支持DeformConv2d
- [x] 支持ModulatedDeformConv2d
- [x] 支持MultiScaleDeformableAttnFunction
- [x] 支持bool index改写masked_select
- [x] 支持Resnet优化
- [x] 支持提前终止训练

