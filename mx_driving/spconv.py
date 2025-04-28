import warnings

from .modules.sparse_conv import SparseConv3d, SubMConv3d
from .modules.sparse_modules import SparseConvTensor, SparseModule, SparseSequential

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)