"""HumanTouch reader surface.

The first migration keeps the proven parsing implementation in the package
level ``readers`` module. This module provides the dataset-local import surface
so the implementation can be split into format-specific private functions
without changing callers or numerical behavior later.
"""

from ...readers import (
    egopro_frame_key,
    read_egopro_reference,
    read_humantouch2_real_robot_reference,
    read_humantouch2_reference,
)

__all__ = [
    "egopro_frame_key",
    "read_egopro_reference",
    "read_humantouch2_real_robot_reference",
    "read_humantouch2_reference",
]
