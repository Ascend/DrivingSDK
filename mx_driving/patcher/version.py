# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Version detection utilities for patcher framework.

Provides generic version detection and comparison utilities.

Usage:
    from mx_driving.patcher.version import get_version, check_version

    # Get package version
    version = get_version("mmcv")  # Returns "2.1.0" or None

    # Check version in precheck
    def precheck():
        return check_version("mmcv", major=2)  # True if mmcv 2.x
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Tuple


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse version string to tuple of integers.

    Args:
        version_str: Version string like "2.1.0", "1.7.0.post1"

    Returns:
        Tuple of version numbers, e.g., (2, 1, 0)
    """
    # Extract numeric parts from version string
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version_str)
    if match:
        parts = [int(x) for x in match.groups() if x is not None]
        return tuple(parts)
    return ()


@lru_cache(maxsize=32)
def get_version(package: str) -> Optional[str]:
    """
    Get version string of a package.

    Args:
        package: Package name (e.g., "mmcv", "torch", "numpy")

    Returns:
        Version string (e.g., "2.1.0") or None if not installed.

    Example:
        >>> get_version("mmcv")
        "2.1.0"
        >>> get_version("nonexistent")
        None
    """
    try:
        module = __import__(package)
        # Try common version attributes
        for attr in ("__version__", "version", "VERSION"):
            version = getattr(module, attr, None)
            if version is not None:
                if callable(version):
                    version = version()
                return str(version)
        return None
    except ImportError:
        return None


def check_version(package: str, major: Optional[int] = None, minor: Optional[int] = None) -> bool:
    """
    Check if a package version matches the specified criteria.

    Use this in precheck functions to conditionally apply patches based on version.

    Args:
        package: Package name (e.g., "mmcv", "torch")
        major: Required major version (e.g., 2 for "2.x.x")
        minor: Required minor version (e.g., 1 for "x.1.x")

    Returns:
        True if package is installed and version matches, False otherwise.

    Example:
        >>> check_version("mmcv", major=2)      # True if mmcv 2.x
        >>> check_version("mmcv", major=1)      # True if mmcv 1.x
        >>> check_version("torch", major=2, minor=0)  # True if torch 2.0.x
    """
    version = get_version(package)
    if version is None:
        return False

    parts = _parse_version(version)
    if not parts:
        return False

    if major is not None and (len(parts) < 1 or parts[0] != major):
        return False
    if minor is not None and (len(parts) < 2 or parts[1] != minor):
        return False

    return True


# =============================================================================
# Convenience functions for common packages
# =============================================================================

class _MMCVVersion:
    """
    MMCV version info object with is_v1x/is_v2x properties.

    Usage:
        from mx_driving.patcher.version import mmcv_version

        if mmcv_version.is_v1x:
            # mmcv 1.x specific code
            pass

        if mmcv_version.is_v2x:
            # mmcv 2.x specific code
            pass

        # Get version string
        version_str = mmcv_version.version  # e.g., "2.1.0" or None

        # Also callable for backward compatibility
        version_str = mmcv_version()  # e.g., "2.1.0" or None
    """

    def __init__(self):
        self._cached = False
        self._version_cache: Optional[str] = None
        self._is_v1x_cache: Optional[bool] = None
        self._is_v2x_cache: Optional[bool] = None

    def _ensure_cached(self):
        """Ensure version info is cached."""
        if not self._cached:
            self._version_cache = get_version("mmcv")
            self._is_v1x_cache = check_version("mmcv", major=1)
            self._is_v2x_cache = check_version("mmcv", major=2)
            self._cached = True

    @property
    def version(self) -> Optional[str]:
        """Get mmcv version string."""
        self._ensure_cached()
        return self._version_cache

    @property
    def is_v1x(self) -> bool:
        """Check if mmcv 1.x."""
        self._ensure_cached()
        return self._is_v1x_cache

    @property
    def is_v2x(self) -> bool:
        """Check if mmcv 2.x."""
        self._ensure_cached()
        return self._is_v2x_cache

    @property
    def has_mmcv(self) -> bool:
        """Check if mmcv is installed."""
        return self.version is not None

    @property
    def has_mmengine(self) -> bool:
        """Check if mmengine is installed."""
        return get_version("mmengine") is not None

    @property
    def available(self) -> bool:
        """Check if mmcv is available (alias for has_mmcv)."""
        return self.has_mmcv

    def __bool__(self) -> bool:
        """Allow using mmcv_version in boolean context."""
        return self.has_mmcv

    def __call__(self) -> Optional[str]:
        """Allow calling mmcv_version() for backward compatibility."""
        return self.version

    def __repr__(self) -> str:
        return f"MMCVVersion(version={self.version!r}, is_v1x={self.is_v1x}, is_v2x={self.is_v2x})"


# Singleton instance for mmcv version detection
mmcv_version = _MMCVVersion()


def is_mmcv_v1x() -> bool:
    """Check if mmcv 1.x."""
    return mmcv_version.is_v1x


def is_mmcv_v2x() -> bool:
    """Check if mmcv 2.x."""
    return mmcv_version.is_v2x
