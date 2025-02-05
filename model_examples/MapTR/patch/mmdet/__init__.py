import mmcv

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


mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.7.2'
mmcv_version = digit_version(mmcv.__version__)


if (mmcv_version < digit_version(mmcv_minimum_version)
        or mmcv_version > digit_version(mmcv_maximum_version)):
    raise Exception(f'MMCV=={mmcv.__version__} is used but incompatible. Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.')

__all__ = ['__version__', 'short_version']
