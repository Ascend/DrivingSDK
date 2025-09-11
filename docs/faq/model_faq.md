# DrivingSDK常见模型问题FAQ
如果你在使用DrivingSDK/model_examples/中的模型时，遇到报错问题，可查看本文档或者去issue中留言。

## 目录
- [DrivingSDK常见模型问题FAQ](#drivingsdk常见模型问题faq)
  - [通用问题速查](#通用问题速查)
  - [模型特定问题速查](#模型特定问题速查)
  - [目录](#目录)
    - [1. 编译错误](#1-编译错误)
    - [2. 数据集错误](#2-数据集错误)
    - [3. 组件及依赖错误](#3-组件及依赖错误)
    - [4. 训练错误](#4-训练错误)
    - [5. 环境变量及依赖错误](#5-环境变量及依赖错误)
    - [6. LTO及PGO编译优化错误](#6-lto及pgo编译优化错误)
    - [7. 其他错误及问题](#7-其他错误及问题)

## 通用问题速查

<table align=center>
    <tr>
        <td align=center>问题类型</td>
        <td align=center>报错关键字及解决方法</td>
    </tr>
    <tr>
        <td rowspan=1, align=center>编译错误</td>
        <td align=center><a href="#1-1">ModuleNotFoundError: No module named 'torch'</a></td>
    </tr>
    <tr>
        <td rowspan=3, align=center>数据集错误</td>
        <td align=center><a href="#2-1">训练时报错无pkl格式文件</a></td>
    </tr>
    <tr>
        <td align=center><a href="#2-2">需要每一次训练模型时都重新预处理文件吗？</a></td>
    </tr>
    <tr>
        <td align=center><a href="#2-3">训练或验证过程中突然报错缺少数据集文件</a></td>
    </tr>
    <tr>
        <td rowspan=12, align=center>组件及依赖错误</td>
        <td align=center><a href="#3-1">yapf组件报错：`EOFError: Ran out of input`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-2">blas组件报错：`ImportError: libblas.so.3`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-3">protobuf组件报错：`DistributionNotFound: The 'protobuf' distribution was not found`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-16">`mm`相关组件报错`mmcv_full`组件与其要求的最大兼容版本不匹配</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-17">`av2`组件报错：`TypeError: Type subscription requires python >= 3.9`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-5">`ImportError: cannot import name 'gcd' from 'fraction'`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-6">`ImportError: libGL.so.1`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-7">`libc.so.6: version 'GLIBC_xxx' not found`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-9">脚本评估性能时，报错：`syntax error at or near`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-12">模型训练过程报错：`torch has no attribute: uint64_t`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-14">模型训练过程报错：`AttributeError: module 'attr' has no attribute 's'`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-15">模型训练过程中，`numpy`组件报无属性、无函数或其他类似错误</a></td>
    </tr>
    <tr>
        <td rowspan=2, align=center>训练错误</td>
        <td align=center><a href="#4-1">`AttributeError: 'int' object has no attribute 'type'`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#4-4">模型训练过程偶现`AssertionError`，导致模型训练中断</a></td>
    </tr>
    <tr>
        <td rowspan=3, align=center>环境变量及依赖错误</td>
        <td align=center><a href="#5-1">模型训练过程报错：`ImportError:/usr/local/gcc-7.5.0/lib64/libgomp.so.1:cannot allocate memory in static TLS block`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#5-2">模型训练过程报错：`ImportError: {conda_env_path}/bin/../lib/libgomp.so.1:cannot allocate memory in static TLS block`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#5-3">模型训练过程报错：`ImportError: {conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`</a></td>
    </tr>
    <tr>
        <td rowspan=2, align=center>LTO及PGO编译优化错误</td>
        <td align=center><a href="#6-1">为什么要使用编译优化</a></td>
    </tr>
    <tr>
        <td align=center><a href="#6-2">编译优化`torch_npu`时，报错`undefind symbol`</a></td>
    </tr>
    <tr>
        <td rowspan=3, align=center>其他错误及问题</td>
        <td align=center><a href="#7-1">模型所需的预训练权重文件因网络问题下载失败</a></td>
    </tr>
    <tr>
        <td align=center><a href="#7-2">模型训练配置是否可以自行更改</a></td>
    </tr>
    <tr>
        <td align=center><a href="#7-3">可以使用同一套环境管理所有模型吗？</a></td>
    </tr>
    
</table>

## 模型特定问题速查

<table align=center>
    <tr>
        <td align=center>模型名称</td>
        <td align=center>报错关键字及解决方法</td>
    </tr>
    <tr>
        <td rowspan=2, align=center>CenterPoint</td>
        <td align=center><a href="#1-2">'cumm'编译过程报ccimport错误</a></td>
    </tr>
    <tr>
        <td align=center><a href="#1-3">源码编译`OpenPCDet`解决方法</a></td>
    </tr>
    <tr>
        <td rowspan=2, align=center>PointPillar</td>
        <td align=center><a href="#1-2">'cumm'编译过程报ccimport错误</a></td>
    </tr>
    <tr>
        <td align=center><a href="#4-3">`KeyError:'road_plane'`</a></td>
    </tr>
    <tr>
        <td align=center>Diffusion-Planner</td>
        <td align=center><a href="#3-4">安装`nuplan-devkit`，报错`No such file or directory: 'gdal-config'</a></td>
    </tr>
    <tr>
        <td align=center>GameFormer</td>
        <td align=center><a href="#3-8">安装`Openexr`过程失败</a></td>
    </tr>
    <tr>
        <td rowspan=2, align=center>HiVT</td>
        <td align=center><a href="#3-10">安装`omegaconf==2.1.0`组件报错：`ERROR: Could not find a version that satisfies the requirement`</a></td>
    </tr>
    <tr>
        <td align=center><a href="#3-11">安装`h5py`组件，报错：`ERROR: Could not build wheels for h5py`</a></td>
    </tr>
    <tr>
        <td align=center>PivotNet</td>
        <td align=center><a href="#3-13">安装`loguru`组件，报错：`error: subpross-exited-with-error`</a></td>
    </tr>
    <tr>
        <td align=center>CenterNet</td>
        <td align=center><a href="#4-2">无法找到`datasets`组件</a></td>
    </tr>

</table>

### 1. 编译错误
<a id="1-1"></a>
#### 1.1 源码编译`mmdet3d`组件时，编译过程报错：`File".../setuptools/build_meta.py" ModuleNotFoundError: No module named 'torch'`
该报错可通过改变`setuptools`组件的版本解决，建议使用`pip install setuptools==75.3.0`。详细的报错原因和解决方法可参考[开源社区Issue：No module named 'torch', why?](https://github.com/facebookresearch/pytorch3d/issues/1892)

<a id="1-2"></a>
#### 1.2 源码编译`cumm`组件时，编译过程报错：`TypeError: ccimport() got multiple values for argument 'std'`
该报错可参考以下语句安装编译所需文件：
```
pip install ccimport==0.3.7
```

<a id="1-3"></a>
#### 1.3 CenterPoint2D或CenterPoint3D模型源码编译`OpenPCDet`时，执行`python setup.py develop`语句时报错：`subprocess.CalledProcessError: Command ['which', 'c++'] return non-zero exit status 1.`
该报错是由于操作系统的GCC版本过高，推荐使用[DrivingSDK仓库README文档](https://gitee.com/ascend/DrivingSDK/blob/master/README.md)中的建议版本`gcc 10.2`进行编译，或使用[CenterPoint模型README文档](https://gitee.com/ascend/DrivingSDK/blob/master/model_examples/CenterPoint/README.md)中的建议版本`gcc 7.5`进行编译。

### 2. 数据集错误
<a id="2-1"></a>
#### 2.1 已经按照README要求在指定路径下放置数据集，训练时为何会报错无pkl格式文件？
目前大部分模型需要使用预处理后的数据集训练，通常在模型README文件的“准备数据集”一节说明预处理步骤。

<a id="2-2"></a>
#### 2.2 需要每一次训练模型时都重新预处理文件吗？
不需要。数据集只需预处理一次即可。

<a id="2-3"></a>
#### 2.3 模型已训练或验证一些Iter，但却在训练或验证过程中突然报错缺少数据集文件，如`FileNotFoundError: [Error2] No such file or directory: 'dataset/xxx.pcb.bin'`
可能在数据集下载或解压过程中缺少文件，需检查数据集是否完整。

### 3. 组件及依赖错误
<a id="3-1"></a>
#### 3.1 模型训练时，yapf组件报错：`EOFError: Ran out of input`
该报错的原因是，yapf组件会创建`~/.cache/YAPF`缓存，在多进程环境中，部分进程创建该缓存后，还未向缓存文件写入内容时，其他进程识别到缓存文件存在，并试图读取文件中的内容，从而报出`EOFError: Ran out of input`错误。遇见此报错时，重新拉起模型训练即可解决。更详细的报错原因及解决方案可参考[开源社区issue[Bug] [Crash][Reproducible] EOFError: Ran out of input when import yapf with multiprocess](https://github.com/google/yapf/issues/1204)。

<a id="3-2"></a>
#### 3.2 模型训练时，blas组件报错：`ImportError: libblas.so.3: cannot open shared object file: No such file or directory`
该问题原因为操作系统未安装openblas依赖，导致依赖缺失，以下给出OpenEuler操作系统的解决方法：
```
yum install openblas
find / -name libopenblas*so
ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/libblas.so.3
ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/liblapack.so.3
```

<a id="3-3"></a>
#### 3.3 模型训练时，protobuf组件报错：`pkg_resources.DistributionNotFound: The 'protobuf' distribution was not found and is required by the application`
该问题的解决方法为：
```
pip install protobuf
```

<a id="3-4"></a>
#### 3.4 `Diffusion-Planner`模型安装模型环境时，需安装`nuplan-devkit`，报错`No such file or directory: 'gdal-config'`
该问题是由于操作系统未安装gmp、mpfr、OpenBLAS、sqlite3、curl、PROJ、GDAL等C++依赖库，以下给出OpenEuler操作系统相关依赖库的安装方式：
```
进入https://ftp.swin.edu.au/gnu/gmp/并下载gmp-6.1.0.tar.bz2
yum install m4, libcurl-devel, libtiff-devel
tar -jxvf gmp-6.1.0.tar.bz2
cd gmp-6.1.0
./configure --prefix=/usr/local/gmp
make -j128
make install
cd ../
wget https://ftp.swin.edu.au/gnu/mpfr/mpfr-4.1.1.tar.gz
tar -zxvf mpfr-4.1.1.tar.gz
cd mpfr-4.1.1
./configure --prefix=/usr/local/mpfr --with-gmp=/usr/local/gmp # 该步骤若报错，替换命令为：./configure --with-gmp=/usr/local/gmp
make -j128
make install
cd ../
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.zip
unzip v0.3.24.zip
cd OpenBLAS-0.3.24
make -j128
make PREFIX=/usr/local install
cd ../
wget https://github.com/sqlite/sqlite/archive/refs/tags/version-3.36.0.tar.gz
tar -xzvf version-3.36.0.tar.gz
cd sqlite-version-3.36.0
CFLAGS="-DSQLITE_ENABLE_COLUMN_METADATA=1" ./configure
make -j128
make install
cd ../
wget https://github.com/OSGeo/PROJ/archive/refs/tags/7.2.0.tar.gz
tar -xzvf 7.2.0.tar.gz
cd PROJ-7.2.0
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
cd ../
git clone https://github.com/OSGeo/gdal.git
cd gdal
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```

<a id="3-5"></a>
#### 3.5 模型训练过程报错：`ImportError: cannot import name 'gcd' from 'fraction'`
该问题由`networkx`组件版本与模型不匹配引起，使用`pip install networkx==3.1`升级依赖版本即可。

<a id="3-6"></a>
#### 3.6 模型训练过程报错：`ImportError: libGL.so.1, cannot open shared object file: No such file or directory`
当模型安装`opencv-python`组件时，需配套安装相同版本的`opencv-python-headless`组件，安装`opencv-contrib-python`组件时，需配套安装相同版本的`opencv-contrib-python-headless`组件。

<a id="3-7"></a>
#### 3.7 模型训练过程报错：`libc.so.6: version 'GLIBC_xxx' not found`
该报错由操作系统GLIBC组件版本过低引起。
```
ldd --version # 查看系统GLIBC版本
```
若GLIBC组件版本低于2.31，需升级组件，以下给出OpenEuler操作系统升级GLIBC组件的命令：
```
yum upgrade glibc glibc-devel
```

<a id="3-8"></a>
#### 3.8 `GameFormer`模型安装模型环境时，需安装`Openexr`，安装过程失败
该报错是由于操作系统未安装OpenEXR和OpenEXR-devel等依赖库，以下给出OpenEuler操作系统相关依赖库的安装方式：
```
sudo yum makecache
sudo yum install gcc gcc-c++ cmake
sudo yum install OpenEXR
sudo yum install OpenEXR-devel
```

<a id="3-9"></a>
#### 3.9 模型训练结束后，脚本评估性能时，报错：`syntax error at or near`
该问题是由于许多模型训练脚本使用`awk`正则表达式获取性能、精度等数据，而操作系统不支持`awk`的拓展正则表达式。需安装`gawk`依赖提供支持，以下给出OpenEuler操作系统相关依赖库的安装方式：
```
yum install -y gawk
```

<a id="3-10"></a>
#### 3.10 `HiVT`模型安装模型环境时，需安装`omegaconf==2.1.0`组件，报错：`ERROR: Could not find a version that satisfies the requirement omegaconf==2.1.0`
该问题是由于`pip`版本过低，无法正确安装组件，需升级`pip`至最新版本。
```
pip install --upgrade pip
```

<a id="3-11"></a>
#### 3.11 `HiVT`模型安装模型环境时，需安装`h5py`组件，报错：`ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects`
该问题推荐使用anaconda管理模型环境，并使用`conda install h5py`代替`pip`安装此依赖。

<a id="3-12"></a>
#### 3.12 模型训练过程报错：`torch has no attribute: uint64_t`
报错原因是`safetensors`版本与`PyTorch`版本不匹配，`PyTorch`版本为2.1.0，需匹配0.6.0以下的`safetensors`，使用`pip install safetensors==0.5.1`改变依赖版本即可。

<a id="3-13"></a>
#### 3.13 `PivotNet`模型安装模型环境时，需安装`loguru`组件，报错：`error: subpross-exited-with-error pip subprocee to install build dependencies did not run successfully.`
报错原因是由于`setuptools`组件版本与`loguru`组件版本不匹配，可使用`pip install loguru==0.7.2`解决报错，更详细的报错原因和解决方法可参考[开源社区Issue：pip subprocess to install build dependencies did not run successfully](https://github.com/pypa/packaging-problems/issues/721)。

<a id="3-14"></a>
#### 3.14 模型训练过程报错：`AttributeError: module 'attr' has no attribute 's'`
报错原因是`attr`组件安装出错。以下给出解决方法：
```
pip uninstall attr
pip install attrs
```

<a id="3-15"></a>
#### 3.15 模型训练过程中，`numpy`组件报无属性、无函数或其他类似错误
报错原因是`Numpy`组件版本与模型不匹配，通常使用`pip install numpy==1.23.5`可解决，过高的numpy版本会导致代码中numpy部分被废弃用法不可用。

<a id="3-16"></a>
#### 3.16 模型安装环境时，安装`mmcv_full==1.7.2`后，安装`mmdet`、`mmdet3d`或其他`mm`相关组件时，报错`mmcv_full`组件与其要求的最大兼容版本不匹配
需按照模型README要求，应用对应patch文件，或按照环境安装步骤进行修改适配。

<a id="3-17"></a>
#### 3.17 模型训练过程中，`av2`组件报错：`TypeError: Type subscription requires python >= 3.9`
报错原因是`av2`组件版本与`Python`不匹配，若使用`python==3.8`，`pip install av2==0.2.1`即可解决。

### 4. 训练错误

<a id="4-1"></a>
#### 4.1 模型依赖`mmcv_full==1.7.2`，训练过程报错：`File ".../torch/nn/parallel/_functions.py", line 117, in _get_stream: if device.dtype == "cpu": AttributeError: 'int' object has no attribute 'type'`
该报错可按照以下方式修改：
```
pip show mmcv_full # 获取mmcv_full安装路径，将路径记为mmcv_install_path
cd mmcv_install_path
vim mmcv/parallel/_functions.py
```
在文件第8行新增语句：
```
from packaging import version
```
将文件第74行`streams = [_get_stream(device) for device in target_gpus]`修改为：
```
if version.parse(torch.__version__) >= version.parse('2.1.0'):
   streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
else:
   streams = [_get_stream(device) for device in target_gpus]
```
详细的报错原因见[开源社区Issue：AttributeError: 'int' object has no attribute 'type'](https://github.com/open-mmlab/mmdetection/issues/10720)。

<a id="4-2"></a>
#### 4.2 CenterNet模型训练过程报错：无法找到`datasets`组件
该报错是由于环境中存在与模型目录下同名的`datasets`组件，导致模型不能找到模型目录下的`datasets`。需卸载模型环境中的同名三方库。

<a id="4-3"></a>
#### 4.3 PointPillar模型训练过程报错：`File ".../PointPillar/OpenDCDet/tools/../pcdet/datasets/augmentor/database_sampler.py", line 372, in add_sampled_boxes_to_scene: KeyError:'road_plane'`
该报错需修改`tools/cfgs/kitti_models/pointpillar.yaml`，将文件中`USE_ROAD_PLANE`设置为`False`。

<a id="4-4"></a>
#### 4.4 模型训练过程偶现`AssertionError`，导致模型训练中断
该问题重新拉起训练即可解决，具体问题原因及解决方法可参考[开源社区Issue: Assertion Error On Finiteness](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus/issues/4)。

### 5. 环境变量及依赖错误

<a id="5-1"></a>
#### 5.1 模型训练过程报错：`ImportError:/usr/local/gcc-7.5.0/lib64/libgomp.so.1:cannot allocate memory in static TLS block`
该问题由glibc版本兼容性引起，可升级glibc版本或者手动导入环境变量：
```
export LD_PRELOAD=/usr/local/gcc-7.5.0/lib64/libgomp.so.1
```

<a id="5-2"></a>
#### 5.2 模型训练过程报错：`ImportError: {conda_env_path}/bin/../lib/libgomp.so.1:cannot allocate memory in static TLS block`
该问题与5.1中的错误类似，可手动导入环境变量：
```
export LD_PRELOAD={conda_env_path}/bin/../lib/libgomp.so.1:$LD_PRELOAD
```

<a id="5-3"></a>
#### 5.3 模型训练过程报错：`ImportError: {conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`
该问题与5.1、5.2中的错误类似，可手动导入环境变量:
```
export LD_PRELOAD={conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
```

### 6. LTO及PGO编译优化错误
<a id="6-1"></a>
#### 6.1 为什么要使用编译优化，能否不使用编译优化？
编译优化技术在数据库、分布式存储等数据和计算密集型等前端瓶颈较高的场景效果显著，性能可得到显著的提升。通过毕昇编译器对源码构建编译Python、PyTorch、torch_npu（Ascend Extension for PyTorch）三个组件，可以有效提升模型性能。如果不需要追求编译优化后的更高模型性能，那么可以不使用编译优化。

<a id="6-2"></a>
#### 6.2 编译优化`torch_npu`时，报错：`ImportError：.../torch_npu/lib/libtorch_npu.so: undefind symbol`
该问题是由于编译优化对于GCC等编译依赖的版本要求较高，推荐使用Pytorch和torch_npu编译优化专有镜像编译
，具体镜像使用和编译优化步骤请参考[昇腾文档：PyTorch 训练模型迁移调优指南-编译优化](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0061.html)。

### 7. 其他错误及问题
<a id="7-1"></a>
#### 7.1 模型所需的预训练权重文件因网络问题下载失败如何解决？
预训练权重文件下载失败，可以根据报错链接，手动下载，拷贝到用户名对应目录：
   ```
   wget ckpt_file # 将预训练权重文件的链接记为ckpt_file
   cp ckpt_file {root}/.cache/torch/hub/checkpoints/resnet-*.pth # 将用户根目录记为{root}
   ```

<a id="7-2"></a>
#### 7.2 模型训练配置是否可以自行更改？
推荐按照模型README文件中提供的训练配置进行模型训练。

<a id="7-3"></a>
#### 7.3 训练不同模型时，必须为每个模型新建一个环境吗？可以使用同一套环境管理所有模型吗？
不建议使用同一套环境管理所有模型。每个模型所使用的组件和依赖版本不完全相同，且部分模型应用tcmalloc高性能内存库、编译优化技术提升模型的性能，若使用同一套环境，可能影响未应用tcmalloc高性能内存库和编译优化技术的模型性能。
