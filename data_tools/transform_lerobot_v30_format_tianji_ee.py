"""Spark HDF5 -> LeRobot v3: H264 video, camera extrinsics, tactile and MANO.

The converter lives in ``data_tools.data_alignment.storage.lerobot``.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "data_tools"

from .data_alignment.storage.lerobot import main

if __name__ == "__main__":
    main()
