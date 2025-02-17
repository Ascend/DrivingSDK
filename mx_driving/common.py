import warnings

from .ops.three_interpolate import three_interpolate
from .ops.scatter_max import scatter_max
from .ops.knn import knn
from .ops.three_nn import three_nn
from .ops.scatter_mean import scatter_mean
from .ops.hypot import hypot
from .ops.assign_score_withk import assign_score_withk

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)