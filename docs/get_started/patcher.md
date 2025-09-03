# 一键Patcher
一键Patcher特性通过Python自身的猴子补丁（Monkey Patch）机制，提供了非侵入式的代码替换框架和通用补丁（Patch），可将原本基于GPU平台的代码实现，简洁快速地迁移到适配昇腾的NPU亲和优化实现上。Python的Monkey Patch本质上是一种在程序运行时（Runtime）动态替换修改模块、类、函数或方法等属性的技术，它无需修改原始源代码即可改变程序的行为。该特性主要包含以下内容：

## 主要功能

1. Patcher框架: 提供对补丁的封装、管理、应用、构建、异常处理等机制
2. 预定义Patch: 总结归纳了高泛用性的Patch实现，主要针对mmcv、torch、numpy等包，并封装好对应的预定义补丁供用户快速迁移优化模型
3. 默认Patcher：提供了提升易用性的`default_patcher_builder`的预定义类实例，帮助用户**仅通过添加几行代码**即可快速迁移模型到昇腾NPU上运行
4. 自定义Patch：用户可通过一键patcher的接口实现针对某个模型专用的自定义补丁，并结合通用公共补丁以兼容各式各类的模型迁移
5. Utility功能：通过补丁机制提供一些易用性提升的功能，例如训练early brake和采集profiling等实用功能

## 当前已支持特性

* [x] 支持default patcher快速迁移
* [x] 支持自定义patcher
* [x] 支持黑名单机制禁用特定patch
* [x] 支持应用具体Patch失败时的日志告警与异常处理
* [x] 支持一键迁移torch至torch_npu（默认关闭私有格式）
* [x] 支持通用CV算子的NPU亲和实现：
  * [x] MultiScaleDeformableAttention（msda）
  * [x] DeformableConv2d （dcn）
  * [x] ModulatedDeformConv2d （mdc）
  * [x] ResNet优化：
    * [x] npu_add_relu
    * [x] npu_max_pool
    * [x] 支持FP16数据格式
* [x] 支持替换index_bool索引操作为masked_select
* [x] 支持Nuscenes数据集操作的NPU亲和优化
  * [x] output_to_nusc_box四元数旋转优化
* [x] 支持优化器的NPU亲和优化
  * [x] 梯度累积优化
  * [x] FP16训练的适配
  * [x] 支持clip_grad_norm_fused_
* [x] 支持mmdet.pseudo_sampler去除unique操作的优化
* [x] 支持兼容性适配
  * [x] 提供patch_mmcv_version解决mmcv与mmdet的版本冲突
  * [x] 支持高版本numpy bool类型
  * [x] 修复mmcv 1.x 里stream对npu环境的兼容性问题
  * [x] 修复mmcv 1.x 里ddp对npu环境的兼容性问题
* [x] 支持易用性提升的Utility功能
  * [x] 支持标准化、简易化采集性能profiling
  * [x] 支持early brake提前终止训练
  * [x] profiling与brake功能的组合

* 预定义Patch的源码实现参考该路径`DrivingSDK/mx_driving/patcher/`下的`[模块名]_patch.py`文件
* 由于MMCV 1.x 变迁到MMCV 2.x时，Runner, Hook, Parallel, Config, FileIO等模块迁移到了MMEngine下，部分目标替换对象属于的MMCV 1.x下的预定义Patch归类到了`mmengine_patch.py`内

## Default Patcher快速迁移

为方便用户快速迁移原本基于GPU/CUDA生态实现的模型至昇腾生态，一键Patcher提供了一个预定义的PatcherBuilder实例，帮助用户仅需添加几行代码即可使模型在昇腾NPU上进行训练。

具体使用方法分为两步：
（前提：需完成CANN、torch_npu、drivingsdk以及相关依赖的安装）

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

然后照常运行模型即可。

* default patcher会逐一尝试应用预定义Patch，每个Patch的应用成功与否因具体模型而异
  
  * 反馈机制：当Patcher成功匹配到了要被替换的目标对象，default patcher会将它替换，并打印应用成功的信息
  * Fail-safe机制：如果模型未匹配到Patch尝试替换的对象（通常由于模型未使用到Patch要替换的第三方库，或第三方库的版本差异导致相关属性命名差异导致），异常报错信息会以warning形式打印，并不会阻塞程序运行
  * 预定义Patch并不保证能在所有模型上适用，default patcher本质上是对预定义Patch逐一尝试应用，部分Patch应用失败是正常现象
* 迁移完备性：部分模型光靠default patcher无法保证迁移完备性，通常当要迁移的模型自身有专有的自定义配置、算子、模块，且这些配置、算子、模块与NPU不兼容时（例如CUDA算子），需要再开发对应的自定义Patch，详情参考下文
* 模型成功跑通后，可添加更定制化的自定义Patcher和以及额外环境变量设置进行深度优化
* default patcher定义在该文件下：`DrivingSDK/mx_driving/patcher/__init__.py`

## 自定义Patch

一键Patcher支持用户自定义Patch，具体方法如下：
创建`patch.py`文件（文件命名可自定义，只需保证训练脚本入口可通过import调用其内容即可）

### 识别修改点

当识别到模型原本的代码里有在昇腾NPU环境下运行时不兼容、不亲和、非最优的实现，可参考：

* `DrivingSDK/model_examples`目录下丰富的模型迁移优化案例
* `DrivingSDK/docs/get_started/model_optimization.md`模型优化指导文档
* `DrivingSDK/docs/api/README.md`里的算子清单
* 昇腾社区上相关文档

### Patch封装

识别并设计好修改点后，假设：

* 识别到的需要修改的地方在一个名为`gpu_affine_func`里面，可将其内容复制出来
* 在`patch.py`文件里定义新函数并粘贴其内容，假设新函数名为`npu_affine_func`
* `gpu_affine_func`所在模块的import路径为：`root_module.xxx.yyy.zzz.gpu_affine_func`
  * `root_module`通常是`mmcv`、`mmdet`、`torch`、`numpy`等第三方依赖的库名，或者是模型自身的`projects`目录

然后使用一键Patcher的Patch类进行封装，并推荐以下格式开发Patch的内容，以符合Patch类封装的格式：

```Python
def my_patch(root_module: ModuleType, options: Dict):
	if hasattr(root_module.xxx.yyy.zzz, "gpu_affine_func"):
		def npu_affine_func(......):
			......
		
		# Monkey Patching i.e. dynamic attribute replacement
		root_module.xxx.yyy.zzz.gpu_affine_func = npu_affine_func
	else:
		raise AttributeError("root_module.xxx.yyy.zzz.gpu_affine_func not found")
```

* 补丁函数`my_patch`定义好后，即可通过`Patch(my_patch)`封装成供Patcher统一管理和应用的补丁单元
* `DrivingSDK/mx_driving/patcher/`目录下的存放预定义Patch的文件，例如`mmcv_patch.py`，包含了例如`def msda(mmcv: ModuleType, options: Dict)`这样的补丁函数作为预定义Patch的实现，可作为实现`my_patch`的参考

### 自定义PatcherBuilder

类似`default_patcher_builder`，可在`patch.py`里定义`my_patcher_builder`，并通过`add_module_patch`方法来逐步添加Patch：

```Python
from mx_driving.patcher import Patch, Patcher, PatcherBuilder

def my_patch(root_module: ModuleType, options: Dict):
	...... # 参考上文

my_patcher_builder = PatcherBuilder()
my_patcher_builder.add_module_patch('root_module', Patch(my_patch, {'option1': xxx, 'option2': xxx, ...}))
```

### 应用Context

然后在训练脚本入口处添加：

```Python
from path_to_my_patch_py import my_patcher_builder
#.....
if __name__ == '__main__':
	with my_patcher_builder.build() as patcher:
		main()
```

### 与Default Patcher混用

通常当模型存在一些模型自有的CUDA算子或非NPU亲和的操作，光靠预定义Patch无法保证迁移完备性，可在使用`default_patch_builder`的基础上添加自定义Patch:

patch.py:

```Python
from mx_driving.patcher import Patch, Patcher, PatcherBuilder, default_patcher_builder

def my_patch(root_module: ModuleType, options: Dict):
   	...... # 参考上文

my_patcher_builder = (
	default_patch_builder
	.add_module_patch('root_module', Patch(my_patch, {'option1': xxx, 'option2': xxx, ...}))
)
```

### 与预定义Patch混用

具体要迁移的模型往往并不会使用到`default_patch_builder`内所有的预定义Patch，更优雅的写法是从中挑选出模型实际用的到的Patch:
patch.py:

```Python
from mx_driving.patcher.mmengine_patch import stream, ddp, optimizer_hooks, optimizer_wrapper 
from mx_driving.patcher.mmcv_patch import dc, mdc, msda, patch_mmcv_version 
from mx_driving.patcher.mmdet_patch import pseudo_sampler, resnet_add_relu, resnet_maxpool, resnet_fp16
from mx_driving.patcher.mmdet3d_patch import nuscenes_dataset, nuscenes_metric
from mx_driving.patcher.numpy_patch import numpy_type
from mx_driving.patcher.torch_patch import index, batch_matmul

from mx_driving.patcher.patcher import Patch, Patcher, PatcherBuilder

def my_patch(root_module: ModuleType, options: Dict):
      	...... # 参考上文

my_patcher_builder = (
    PatcherBuilder()
    .add_module_patch("mmcv", Patch(msda), Patch(dc), Patch(mdc), Patch(stream), Patch(ddp))
    .add_module_patch("torch", Patch(index), Patch(batch_matmul))
    .add_module_patch("numpy", Patch(numpy_type))
    .add_module_patch("mmdet", Patch(pseudo_sampler), Patch(resnet_add_relu), Patch(resnet_maxpool))
    .add_module_patch("mmdet3d", Patch(nuscenes_dataset), Patch(nuscenes_metric))

    .add_module_patch('root_module', Patch(my_patch, {'option1': xxx, 'option2': xxx, ...}))
    # ......
)
```

## 实用Utility功能

### 采集性能Profiling

该框架以补丁形式提供采集NPU profiling的功能，具体用法如下：

```Python
# 需放在“with my_patcher_builder.build() as patcher”之前
profiling_path = "path/to/profiling/"
profiling_level = 1 
my_patcher_builder.with_profiling(profiling_path, profiling_level)
```

* profiling_level当前有三个level
  * level 0: 最小膨胀，只记录NPU活动，记录的过程本身的时延最小
  * level 1: 记录NPU和CPU活动
  * level 2: 记录NPU和CPU活动，并打印调用栈
* 当前采集profiling的许多配置暂时未暴露接口，近期下个commit会加以补充，敬请期待

### 训练早停Early Brake

类似的：

```Python
# 需放在“with my_patcher_builder.build() as patcher”之前
brake_step = 1000 # 在第1000个训练step提前sys.exit(0)
my_patcher_builder.with_brake(brake_step)
```

### Profiling与Brake组合

仅使用Profiling，在采集完成后，不会自动早停，可以同时使用Profiling和Brake实现采集完profiling后停止，未有参数指定的情况下，Profiling默认值是在第21个step采集

## 一键Patcher框架

### `Patch`类

打补丁操作的基本封装单元，用于封装一个具体补丁，并确保其可配置性、优先级、状态跟踪等
主要API：

* `Patch.__init__(self, func: Callable, options: Optional[Dict] = None,  priority: int = 0, patch_failure_warning: bool = True)`
  * priority参数决定patch的优先级，在patch顺序敏感的场景里可用到，数字小的优先级高（当前Patch依照优先级排序仅发生在PatchBuilder里，下个commit会补充进Patcher里）
  * patch_failure_warning参数默认为True，可设置为False让其不报warning

用法例子：

```Python
def my_patch(root_module: ModuleType, options: Dict):
	...... # 参考上文

my_options = {}
my_options['key1'] = val1
my_options['key2'] = val2
# .......

#封装补丁
my_wrapped_patch = Patch(func=my_patch, 
					     options=my_options, 
					     priority=0, 
					     patch_failure_warning=True)
```

### `Patcher`类

该框架的核心管理器，负责加载模块、应用修补集合并处理NPU相关的配置
主要API：

* `__init__(self, module_patches: Dict[str, List[Patch]], blacklist: Set[str], allow_internal_format: bool = False)`
  * blacklist可以指定要diable的Patch，需传入`Patch(func)`内的func函数名
  * allow_internal_format默认关闭私有格式，可设为True开启
* `apply(self)`
  * 应用所有的非黑名单上的Patch
  * 传入的module如果无法import，或者Patcher无法找到要替换的目标属性，均会以warnings的形式报错，不会阻塞程序
  * 每个patch应用成功后会打印一行反馈

用法例子：

```Python
def my_patch1(.....):
	.......
def my_patch2(.....):
	.......
my_wrapped_patch1 = Patch(my_patch1)
module_name1 = "mmcv"
my_wrapped_patch2 = Patch(my_patch2)
module_name2 = "torch"
my_patches = {}
my_patches[module_name1] = my_wrapped_patch1
my_patches[module_name2] = my_wrapped_patch2

#.......
black_list = {"my_patch2"}
patcher = Patcher(my_patches, black_list)
with patcher:
	# train model here
	main()
```

### `PatcherBuilder`类

采用常用设计模式里的Builder Pattern，提供了流畅接口（Fluent Interface）来逐步构建和配置最终的 `Patcher`实例
主要API：

* `add_module_patch(self, module_name: str, *patches: Patch) -> PatcherBuilder`
  * module_name是被替换目标属性所在的模块路径的根模块，通常为第三方库的库名，或模型自身代码模块的根目录名，例如`mmcv`, `torch`，`projects`等
  * \*patches可以传入多组Patch，例如`.add_module_patch("mmcv", Patch(msda), Patch(dc), Patch(mdc) .....)`
* `disable_patches(self, *patch_names: str) -> PatcherBuilder`
  * 可用`Patch(func)`内的func函数名来添加黑名单确立要被disable的Patch
* `with_profiling(self, path: str, level: int = 0) -> PatcherBuilder`
  * 详细用法参考上文
* `brake_at(self, brake_step: int) -> PatcherBuilder`
  * 详细用法参考上文
* `build(self, allow_internal_format: bool = False)`
  * allow_internal_format默认关闭私有格式，可设为True开启，该参数会透传给Patcher类的__init___，仅在此处设置即可

## 应用范式

### 少量侵入式修改

当前位于`DrivingSDK/model_example`目录下的模型案例里，使用到一键Patcher特性的example很多是少量diff生成的`.patch`文件进行侵入式修改，结合一键Patcher的纯Python文件共同使用，通常需要做一下几步:

1. 在`tools/`里新建`patch.py`，编写一键patcher范式的补丁函数以及应用所有补丁的patcher_builder
2. 侵入式修改训练脚本入口函数，通常为`train.py`，增加patcher的import和`with patcher_builder.build():`的context
3. 对于只在gpu环境下使用，但在npu下未安装的依赖，通常为三方CUDA算子包，将其import删除，并在模型使用到它的位置替换成npu对应的算子或Patch
4. 侵入式修改非python文件的配置，例如json，或者较为特殊的python文件，例如mmcv的config文件
5. 侵入式修改特定模型特定场景下，一键patcher无法应用的修改
6. 编写包含NPU专有的环境变量设置的训练启动shell脚本，通常命名为`train_8p.sh`，统一调用训练脚本

### 无侵入式修改

位于`DrivingSDK/model_example`目录下的模型案例里，配有`migrate_to_ascend`文件夹的模型，通常为纯血一键patcher，无需改动模型官方的任何代码，仅需拷贝该文件到模型源码目录下，即可运行其内部脚本仅需模型训练，步骤如下：

1. 新建独立文件夹`migrate_to_ascend`
2. 在`migrate_to_ascend`下创建`patch.py`，开发相应的补丁和补丁Builder
3. 在`patch.py`里可通过sys.modules提前注册那些在npu环境下未安装的依赖，欺Python的import机制使其误以为依赖已经import，在Python解释器执行到官方源码里对它们的import时，就会自动略过，以避免import报错（不过前提是要保证要么模型内部压根没用到这个依赖，或者使用的位置已经开发有相应的补丁，才可确保模型正常运作），具体例子：

```Python
#在patch.py里添加：
def _init():
    # block dependencies that are not used nor installed
    sys.modules['mmdet3d.ops.scatter_v2'] = ModuleType('mmdet3d.ops.scatter_v2')
    sys.modules['torch_scatter'] = ModuleType('torch_scatter')
    # .......
_init()
```

4. 将官方源码的训练脚本文件，整个拷贝到`migrate_to_ascend`下，再在里面的拷贝里自由的修改，同时在训练入口函数处添加补丁Builder的context
5. 编写包含NPU专有的环境变量设置的训练启动shell脚本，通常命名为`train_8p.sh`，统一调用训练脚本
6. 针对非python的少量配置文件，同样的，直接拷贝到migrate_to_ascend下进行修改
7. 针对mmcv的config文件，可以由以下几种方式修改
   * 通过mmcv框架自带的训练脚本命令行传参`--cfg-options`来修改
   * 新建config文件继承原config文件，然后覆盖要修改的参数，也是mmcv自带的机制
   * 在shell脚本里备份config文件，通过sed命令强行修改，然后执行训练任务，完成后或者异常中断后再复原