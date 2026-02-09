# Driving SDK

##  简介

Driving SDK是基于昇腾NPU平台开发的适用于自动驾驶场景、具身智能VLA及世界模型的算子和模型加速库，提供了一系列高性能算子和模型迁移示例，支持PyTorch框架。

## 加入我们

为了交流开发经验、分享使用心得、及时获取项目更新，我们创建了DrivingSDK官方微信群。

无论你是正在使用这个项目，还是有奇思妙想，都欢迎加入👋

<p align="center"> <img src="./docs/zh/figures//DrivingSDK_wechat_qrcode.jpg" width=150> </p>

## 未来规划

📅未来规划会动态刷新在 [DrivingSDK RoadMap](https://gitcode.com/Ascend/DrivingSDK/issues/132) 中，欢迎大家通过此链接进行互动并提出诉求。

## 最新消息

* [Dec. 09, 2025]: 🚀 DrivingSDK仓中Spconv3d算子支持channel大于等于128并优化显存。
* [Dec. 05, 2025]: 🚀 DrivingSDK仓中submSparseConv3d性能优化。
* [Nov. 20, 2025]: 🚀 DrivingSDK仓支持Pi0.5模型。
* [Nov. 20, 2025]: 🚀 DrivingSDK仓支持FBOcc模型。
* [Nov. 13, 2025]: 🚀 DrivingSDK仓支持Cosmos-drive-dreams模型。
* [Nov. 10, 2025]: 🚀 DrivingSDK仓中scatter_add算子性能优化。
* [Nov. 07, 2025]: 🚀 DrivingSDK仓支持cosmos-predict2模型。
* [Oct. 28, 2025]: 🚀 DrivingSDK仓支持VGGT模型。
* [Oct. 28, 2025]: 🚀 DrivingSDK仓支持GROOT-N1.5模型。

## 版本说明

请参见 [版本说明](docs/zh/version.md)。

## 硬件配套
| 产品系列               | 产品型号                         |
|-----------------------|----------------------------------|
| Atlas A2 训练系列产品  | Atlas 800T A2 训练服务器          |
|                       | Atlas 900 A2 PoD 集群基础单元     |

##  安装部署
请参见 [安装部署文档](docs/zh/installation/installation.md)。

## 目录结构

```
DrivingSDK
├── kernels                     # 算子实现
│  ├── op_host
│  ├── op_kernel
│  └── CMakeLists.txt
├── kernels_arch35              
├── onnx_plugin                 # onnx框架适配层
├── mx_driving
│  ├── __init__.py
│  ├── csrc                     # 加速库API适配层
│  ├── ops                      # API
│  ├── patcher                  # 一键patcher特性
│  ├── dataset                  # 负载均衡特性
│  └── ...   
├── model_examples              # 自动驾驶模型示例
│  └── BEVFormer                # BEVFormer模型示例
├── ci                          # ci脚本
├── cmake                       # cmake脚本
├── CMakeLists.txt              # cmake配置文件
├── CMakePresets.json           # cmake配置文件
├── docs                        # 文档
|  ├── api                      # 算子api调用文档
|  └── ...
├── include                     # 头文件
├── LICENSE                     # 开源协议
├── OWNERS                      # 代码审查
├── README.md                   # 项目说明
├── requirements.txt            # 依赖
├── scripts                     # 工程脚本
├── setup.py                    # whl打包配置
└── tests                       # 测试文件

```

## 快速上手
如何调用 Driving SDK 高性能 API，以及基于 Driving SDK 进行模型迁移优化，可以参考[快速上手文档](./docs/zh/get_started/quick_start_guide.md)。


## 支持特性
- [一键 Patcher](docs/zh/features/patcher.md)
- [负载均衡](docs/zh/features/dataload_balance.md)
- [训推一体](docs/zh/features/onnx_example.md)

## API 清单
请参见 [算子清单](/docs/zh/api/README.md)。

## 模型清单
请参见 [模型清单](docs/zh/models/support_list.md)。Driving SDK仓提供了包括感知、规划、端到端、VLA等自动驾驶模型在昇腾服务器上的迁移优化案例。每个模型都有详细的使用指导，后续将持续增加和优化典型模型。使用过程中，若遇到报错问题，可查看 [自动驾驶模型FAQ](docs/zh/faq/model_faq.md) 自助解决，或在 [Issues](https://gitcode.com/Ascend/DrivingSDK/issues) 中留言。

## Driving SDK 分支维护策略

Driving SDK版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1-3 个月 | 计划特性                                         |
| 开发                | 3 个月   | 开发特性                                         |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的Driving SDK版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0-3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |


## Driving SDK 版本维护策略

| **Driving SDK版本**     | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期** |
|---------------------|-----------|---------|------------|--------------------|-----------|
| v7.3.0  |  常规版本  | 维护      | 2025/12/30 | 预计2026/06/30起无维护	   |        |
| v7.2.RC1  |  常规版本  | 维护      | 2025/09/30 | 预计2026/03/30起无维护	   |         |
| v7.1.RC1  |  常规版本  | 无维护      | 2025/06/30 | 2025/12/30起无维护	   |         |
| v7.0.RC1  |  常规版本  | 无维护      | 2025/03/30 | 2025/9/30起无维护	   |           |
| v6.0.0   |  常规版本  | 生命周期终止      | 2024/12/30 | 2025/6/30起无维护	   |    2025/09/30  |  
| v6.0.0-RC3 |  常规版本  | 生命周期终止      | 2024/09/30 | 2025/3/30起无维护	   |   2025/06/30 |
| v6.0.0-RC2 |  常规版本  | 生命周期终止      | 2024/06/30 | 2024/12/30起无维护	   |    2025/03/30 |
| v6.0.0-RC1 |  常规版本  | 生命周期终止  | 2024/03/30 | 2024/9/30起无维护           |    2024/12/30 |

## 安全声明
[DrivingSDK 安全声明](./docs/zh/SECURITYNOTE.md)。

## 许可证
DrivingSDK 使用BSD许可证。详见[LICENSE](./docs/zh/LICENSE)文件。

## 免责声明
- Driving SDK提供的模型仅供您用于非商业目的。
- 对于各模型，Driving SDK平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
- 如您在使用Driving SDK模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。
- 如果您不希望您的数据集在Driving SDK中的模型被提及，或希望更新Driving SDK中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对Driving SDK的理解和贡献。
