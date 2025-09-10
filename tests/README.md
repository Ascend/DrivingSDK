# Tests
该目录存放了DrivingSDK的各类测试脚本

## 安装
### 前提条件
在完成根目录下README安装步骤后，应当完成了：
* CANN包
* torch_npu包
* 根目录下requirements.txt里列出的依赖
* 源码编译并安装了的drivingsdk包

### 额外依赖

然而具体模型或具体算子可涵盖额外的依赖，如需跑通所有单元测试脚本，需补充以下依赖：
* 需源码编译的依赖
  * mmcv == 1.7.2
    ```Bash
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OP=1 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OP=1 FORCE_NPU=1 python setup.py install
    cd ../
    ```
* 可通过pip安装的依赖，已列于当前目录下的requirements.txt
  ```Bash
  pip install requirements.txt
  ```

### 运行测试脚本
对于`onnx/`目录下的UT脚本，数量较少，可直接运行具体的Unit Test脚本的`.py`文件。

对于`torch/`目录下的UT脚本，可通过运行`run_test.py`执行torch里面的所有UT测试脚本，可运行：
```Bash
python [Path_to_DrivingSDK]/tests/torch/run_test.py
```

### 注意事项
* 算子UT脚本可能在`tests/torch/`生产`data_cache`文件夹存放缓存
* 如遇到golden_data_cache相关的报错信息，可尝试删除`data_cache`文件夹并重新运行测试脚本
* 在根目录下运行UT，pip会使用根目录下的mx_driving包的信息，而不是conda环境安装路径上