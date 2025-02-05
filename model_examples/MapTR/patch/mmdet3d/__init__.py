# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

import mmdet
import mmseg
from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.5.2'
mmcv_maximum_version = '1.7.2'
mmcv_version = digit_version(mmcv.__version__)


if (mmcv_version < digit_version(mmcv_minimum_version)
        or mmcv_version > digit_version(mmcv_maximum_version)):
    raise Exception(f'MMCV=={mmcv.__version__} is used but incompatible. Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.')

mmdet_minimum_version = '2.14.0'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)
if (mmdet_version < digit_version(mmdet_minimum_version)
        or mmdet_version > digit_version(mmdet_maximum_version)):
    raise Exception(f'MMDET=={mmdet.__version__} is used but incompatible. Please install mmdet>={mmdet_minimum_version}, <={mmdet_maximum_version}.')

mmseg_minimum_version = '0.14.1'
mmseg_maximum_version = '1.0.0'
mmseg_version = digit_version(mmseg.__version__)
if (mmseg_version < digit_version(mmseg_minimum_version)
        or mmseg_version > digit_version(mmseg_maximum_version)):
    raise Exception(f'MMSEG=={mmseg.__version__} is used but incompatible. Please install mmseg>={mmseg_minimum_version}, <={mmseg_maximum_version}.')

__all__ = ['__version__', 'short_version']
