## Driving SDK仓编译指导
如果你遇到编译问题，可查看[FAQ](#faq)或者去issue中留言。
### 前置依赖
Driving SDK仓编译依赖以下组件：
1. Ascend-cann-toolkit开发套件包，可前往官网下载，并在安装后使能环境变量（假设安装在`/usr/local/Ascend/`下）：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. cmake >= 3.19.0，如过你的环境中没有满足条件的cmake，可以通过`pip install cmake`的方式快速安装。
3. requirements.txt下的python包，可以通过`pip install -r requirements.txt`的方式快速安装。
4. torch,torch_npu
### Release 模式
该模式下可使用
```
bash ci/build --python=3.8
或者
python3.8 setup.py bdist_wheel
```
适合在ci服务器上构建Relase版的wheel包，编译产物生成在`dist`文件夹下，之后需要用`pip install`的方式进行安装。
### Develop 模式
该模式适合开发调试使用，在该模式下编译选项与`Release`有些许不同，性能不如`Release`。使用
```
python3.8 setup.py develop
```
即可进行编译全仓，编译产物在当前工作目录下，无需后续安装。
#### Develop 模式编译特定算子
如果你想编译一个或多个算子，比如`DeformableConv2d`和`MultiScaleDeformbaleAttn`,算子名为`op_host/xx.cpp`中的`OpDef`定义的名字，可以使用`--kernel-name`参数：
```
python3.8 setup.py develop --kernel-name="DeformableConv2d;MultiScaleDeformableAttn"
```
注意，kernel-name用`;`分隔。

## FAQ
1. Q: fatal error: proto/onnx/ge_onnx.pb.h: No such file or directory
A:如果你不需要使用`onnx`进行推理，请在`CMakePresets.json`中关闭`ENABLE_ONNX`选项，将`True`改为`False`。
如果需要`onnx`可尝试执行`bash ci/docker/ARM/build_protobuf.sh`安装`protobuf`。
2. Q: third_party/acl/inc/acl/acl_base.h: No such file or directory
A: 你可能没有成功安装torch_npu，重新安装即可。
3. Q: undefinde symbol: _ZN2at4_ops4view4callERKNS_6TensorEN3c108ArrayRefIlEE
A: torch 与torch_npu的版本可能不配套。
4. Q: opbuild ops error: Invalid socVersion ascend910_93 of xxx
A: 更换最新的Ascend-cann-toolkit套件
