# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
python patches for NPU adaptation.

Provides NPU-compatible replacements for python operators including:
- CollectionsCompat
"""

from mx_driving.patcher.patch import (
    LegacyPatch,
    Patch,
)


class CollectionsCompat(Patch):
    """Iterable patch for collections."""

    name = "Iterable"
    legacy_name = "Iterable"
    target_module = "python"
    apply_before_collect = True

    @staticmethod
    def _patch_deprecated_aliases(collections, _options):
        changed = False

        if not hasattr(collections, "Iterable"):
            # In python3.11, the Iterable move to collections.abc, which may cause import error
            collections.Iterable = collections.abc.Iterable
            changed = True

        # If nothing changed, surface as SKIPPED via LegacyPatch semantics.
        if not changed:
            raise AttributeError(
                "deprecated aliases already exist in this python runtime; CollectionsCompat is not needed"
            )

    @classmethod
    def patches(cls, options=None):
        return [
            LegacyPatch(cls._patch_deprecated_aliases, target_module="collections", options=options),
        ]
