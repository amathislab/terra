"""Retargeting method identifiers shared by lightweight public interfaces."""

from __future__ import annotations

from typing import Literal, cast

RetargetingMethod = Literal["terra", "omniretarget", "gmr", "smpl"]
SUPPORTED_METHODS: tuple[RetargetingMethod, ...] = ("terra", "omniretarget", "gmr", "smpl")


def validate_method(method: str) -> RetargetingMethod:
    """Return a supported retargeting method or raise an actionable error."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown retargeting method {method!r}; choose one of {SUPPORTED_METHODS}")
    return cast(RetargetingMethod, method)


__all__ = ["SUPPORTED_METHODS", "RetargetingMethod", "validate_method"]
