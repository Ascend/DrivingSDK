# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>DrivingSDK</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers=><p>26.0.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年4月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]  
> 有关DrivingSDK的版本维护，具体请参见[分支维护策略](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.0.0/README.md#driving-sdk-%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

### 相关产品版本配套说明

**表 1**  Driving SDK配套表

|Driving SDK代码分支名称|CANN版本|Ascend Extension for PyTorch版本|Python版本|PyTorch版本|
|--|--|--|--|--|
|branch_v26.0.0|9.0.0|26.0.0|3.9, 3.10, 3.11|2.7.1, 2.8.0|
|branch_v7.3.0|8.5.0|7.3.0|3.9, 3.10, 3.11|2.7.1, 2.8.0|
|branch_v7.2.0|8.3.RC1|7.2.0|3.8, 3.9, 3.10, 3.11|2.1.0, 2.7.1, 2.8.0|

>[!NOTE]  
>用户可根据需要选择Driving SDK代码分支下载源码并进行安装。

## 版本兼容性说明

|Driving SDK版本|CANN版本|Ascend Extension for PyTorch版本|
|--|--|--|
|26.0.0|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>|26.0.0|
|7.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.3.0|
|7.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>|7.2.0|

## 版本使用注意事项

无

## 更新说明

### 新增特性

- GraphSoftmax算子
- SparseInverseConv3D算子
- SparseInverseConv3DGrad算子
- ScatterAdd算子性能优化
- SparseConv3D算子支持FP16，支持channel大于128
- SubMSparseConv3D算子性能优化
- 新增FBOCC模型
- 新增Cosmos-Predict2-14B-Video2World模型
- 新增Cosmos-Transfer1-7B模型
- 新增Cosmos-Transfer1-7B-Sample-AV-Single2MultiView模型
- 新增Cosmos-Transfer1-7B-Sample-AV模型
- 新增Cosmos-Transfer1-7B-4KUpscaler推理适配
- 新增DriverAgent模型
- 新增Sparse4D模型支持混精训练
- 新增BEVFusion模型支持混精训练

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[自驾模型迁移优化指导](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.0.0/docs/zh/migration_tuning/model_optimization.md)》|指导具有一定自动驾驶模型训练基础的用户将原本在其他硬件平台（例如GPU）上训练的模型迁移到昇腾平台（NPU），并在合理精度误差范围内高性能运行。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

无

### 漏洞修补列表

无
