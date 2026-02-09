# Driving SDK 介绍

Driving SDK自动驾驶训练加速套件，基于昇腾AI集群系统开发，提供高性能算子库、自动驾驶优选模型库，以及一键patcher快速迁移工具。Driving SDK 旨在为自动驾驶行业客户和开发者提供基于昇腾算力的自动驾驶系统开发的底座能力。

<p align="center"> <img src="./figures/DrivingSDK_Arc.png"> </p>

## Driving SDK 组成部分 
### 高性能算子库
提供50+算子自动驾驶场景高性能算子，涵盖典型智驾算子如DCN、MSDA、SparseConv3D等，快速打通迁移训练链路。

### 优选模型库
Driving SDK自动驾驶加速套件提供50+主流场景模型，包含BEVFormer、MapTR、FlashOCC、LMDrive、StreamPETR和UniAD等高性能自动驾驶模型，涵盖BEV感知、OCC感知、Lidar感知、预测规划、通用目标检测等多个场景。

<p align="center"> <img src="./figures/DrivingSDK_Models.png"> </p>

### 一键Patcher
DrivingSDK内置一键Patcher，可以帮助用户快速将GPU工程迁移到NPU，默认通用优化自动使能，同时集成性能采集分析工具，帮助快速分析定位问题。
