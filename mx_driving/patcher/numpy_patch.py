# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Numpy compatibility patches for numpy 1.24+ removed aliases."""
from typing import List

from mx_driving.patcher.patch import BasePatch, LegacyPatch, Patch


class NumpyCompat(Patch):
    """Numpy compatibility patch for deprecated aliases (np.bool, np.float, np.int)."""

    name = "numpy_compat"
    legacy_name = "numpy_type"
    target_module = "numpy"
    apply_before_collect = True

    @staticmethod
    def _patch_deprecated_aliases(np_module, _options):
        """
        1.0 等效写法：显式补齐缺失别名。

        这样 `np.bool` / `np.float` / `np.int` 会真实存在于
        `numpy.__dict__`（`dir()` 也可见），
        避免 hook `numpy.__getattr__` 带来的潜在边界差异。
        """
        changed = False

        if not hasattr(np_module, "bool"):
            np_module.bool = bool
            changed = True

        if not hasattr(np_module, "float"):
            np_module.float = float
            changed = True

        if not hasattr(np_module, "int"):
            np_module.int = int
            changed = True

        # If nothing changed, surface as SKIPPED via LegacyPatch semantics.
        if not changed:
            raise AttributeError(
                "deprecated aliases already exist in this NumPy runtime; "
                "numpy_compat is not needed"
            )

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            LegacyPatch(cls._patch_deprecated_aliases, target_module="numpy", options=options),
        ]
