# Driving SDK

## 最新消息

* [Mar. 30, 2026]: 🚀 BEVFusion、Sparse4D模型支持混合精度。
* [Mar. 30, 2026]: 🚀 SparseConv3D算子支持fp16，MSDA算子支持dim大于64。
* [Mar. 30, 2026]: 🚀 支持GraphSoftmax、SparseInverseConv3D算子。
* [Mar. 30, 2026]: 🚀 ScatterAdd、SubMSparseConv3D算子性能优化。

## 软件介绍

Driving SDK是基于昇腾NPU平台开发的适用于自动驾驶场景、具身智能VLA及世界模型的算子和模型加速库，提供了一系列高性能算子和模型迁移示例，支持PyTorch框架，可参考[简介](docs/zh/introduction/Introduction.md)。

## 目录结构

```shell
DrivingSDK
├── ci                          # CI脚本
├── cmake                       # CMake脚本
├── docker                      # 自定义镜像构建
├── docs                        # 资料文档
├── include                     # 头文件
├── kernels                     # 算子实现
│  ├── add_relu
│  ├── ├── op_host              # host侧实现
│  ├── ├── op_kernel            # device侧实现
│  ├── ...
│  └── CMakeLists.txt
├── model_examples              # 模型示例
│   ├── BEVFormer               # BEVFormer模型示例
│   ├── BEVFusion               # BEVFusion模型示例
│   └── ...
├── mx_driving                  # 加速库核心
│   ├── csrc                    # 加速库API适配层
│   ├── dataset                 # 负载均衡特性
│   ├── modules                 # 稀疏模块
│   ├── ops                     # 高性能API
│   ├── patcher                 # 一键Patcher
│   └── ...
├── onnx_plugin                 # ONNX框架适配层
├── pre-commit                  # pre-commit配置
├── scripts                     # 工程脚本
├── tests                       # 测试文件
├── CMakeLists.txt              # CMake配置文件
├── CMakePresets.json           # CMake配置文件
├── LICENSE                     # 开源协议
├── OWNERS                      # 代码审查
├── README.md                   # 项目说明
├── requirements.txt            # 环境依赖
└── setup.py                    # whl打包配置
├── Third_Party_Open_Source_Software_Notice  # 第三方开源软件声明
```

## 版本说明

Driving SDK算子支持的CPU架构、Python，PyTorch和torch_npu版本对应关系请参考[版本说明](docs/zh/version/versions.md)。

## 安装指导

社区提供了容器部署、自定义镜像构建和源码编译安装三种方式。推荐使用基于容器的方式部署Driving SDK环境，以实现快速上手。具体部署过程，请参考[部署Driving SDK环境](docs/zh/installation/installation.md)。

## 快速上手

提供了如何调用 Driving SDK 高性能 API，以及基于 Driving SDK 进行模型迁移优化，可以参考[快速上手文档](docs/zh/get_started/quick_start_guide.md)。

## 特性说明

* 通过Python自身的补丁机制，可将原本基于GPU平台的代码实现，简洁快速地迁移到适配昇腾的NPU亲和优化实现上。具体信息，请参考[一键Patcher](docs/zh/features/patcher.md)。

* Driving SDK 提供了多种类型的高性能API，包括通用、采样、体素化、检测、稀疏和融合类。具体信息，请参考[API](docs/zh/api/README.md)。

* 支持通过转换工具（ATC）将ONNX转换为OM模型。具体信息，请参考[onnx转换om](docs/zh/features/onnx_example.md)。

* Driving SDK实现了自动驾驶模型负载均衡特性，通过负载均衡策略，缓解计算负载不均衡瓶颈，提升模型性能。具体信息，请参考[Driving SDK负载均衡特性](docs/zh/features/dataload_balance.md)。

## 模型使用指导

Driving SDK仓提供了包括感知、规划、端到端、VLA等自动驾驶模型在昇腾服务器上的迁移优化案例。每个模型都有详细的使用指导，后续将持续增加和优化典型模型。具体请参见 [模型清单](docs/zh/models/support_list.md)。

## 模型迁移优化指导

Driving SDK 通过一键patcher提供了从 GPU 模型到昇腾 NPU 的一键迁移能力，以及包括框架特性配置、高性能算子替换、Host Bound问题优化等在内的系统性性能调优方案。具体请参考[模型迁移优化指导](docs/zh/migration_tuning/model_optimization.md)。

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
| v26.0.0  |  常规版本  | 维护      | 2026/03/30 | 预计2026/09/30起无维护    |        |
| v7.3.0  |  常规版本  | 维护      | 2025/12/30 | 预计2026/06/30起无维护    |        |
| v7.2.RC1  |  常规版本  | 维护      | 2025/09/30 | 2026/03/30起无维护    |         |
| v7.1.RC1  |  常规版本  | 无维护      | 2025/06/30 | 2025/12/30起无维护    |         |
| v7.0.RC1  |  常规版本  | 无维护      | 2025/03/30 | 2025/9/30起无维护    |           |
| v6.0.0   |  常规版本  | 生命周期终止      | 2024/12/30 | 2025/6/30起无维护    |    2025/09/30  |
| v6.0.0-RC3 |  常规版本  | 生命周期终止      | 2024/09/30 | 2025/3/30起无维护    |   2025/06/30 |
| v6.0.0-RC2 |  常规版本  | 生命周期终止      | 2024/06/30 | 2024/12/30起无维护    |    2025/03/30 |
| v6.0.0-RC1 |  常规版本  | 生命周期终止  | 2024/03/30 | 2024/9/30起无维护           |    2024/12/30 |

## 常见问题

若在使用 Driving SDK 过程中遇到问题，可查看 [Driving SDK 常见问题](./docs/zh/faq/faq.md) 自助解决，或在 [Issues](https://gitcode.com/Ascend/DrivingSDK/issues) 中留言。

## 联系我们

未来规划会动态刷新在 [Driving SDK RoadMap](https://gitcode.com/Ascend/DrivingSDK/issues/132) 中，欢迎大家通过此链接进行互动并提出诉求。

为了交流开发经验、分享使用心得、及时获取项目更新，我们创建了Driving SDK 官方微信群。

无论你是正在使用这个项目，还是有奇思妙想，都欢迎加入👋

<p align="center"> <img src="./docs/zh/figures//DrivingSDK_wechat_qrcode.jpg" width="150"> </p>

## 安全声明

主要描述了Driving SDK的安全加固信息、公网地址信息及通信矩阵等内容。具体介绍，请参考[Driving SDK 安全声明](./SECURITYNOTE.md)。

## 免责声明

### 致Driving SDK使用者

1. Driving SDK提供的模型仅供您用于非商业目的。
2. 对于各模型，Driving SDK平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用Driving SDK模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。

### 致数据集所有者

如果您不希望您的数据集在Driving SDK中的模型被提及，或希望更新Driving SDK中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对Driving SDK的理解和贡献。

## License

* Driving SDK产品的使用许可证，具体请参见[LICENSE](./LICENSE)。<br>
* Driving SDK工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)。

## 贡献声明

贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)。
通常，Driving SDK一年会有4个正式release版本。
如果您遇到bug，请[提交issue](https://gitcode.com/Ascend/DrivingSDK/issues)。
如果您计划贡献bug-fixes，请提交Pull Requests，参见[具体要求](./CONTRIBUTING.md#pullrequest)。
如果您计划贡献新特性、功能，请先创建issue与我们讨论。写明模型或算子名称，需求背景/目的，如何设计，对现有仓的影响。未经讨论提交PR可能会导致请求被拒绝，因为项目演进方向可能与您的想法存在偏差。
更详细的贡献流程，请参考[贡献指导](./CONTRIBUTING.md)。

## 致谢

Driving SDK由华为公司的下列部门联合贡献：

* 昇腾计算训练开发部

感谢来自社区的每一个PR，欢迎贡献Driving SDK！
