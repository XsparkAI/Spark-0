"""Read Spark / RoboDojo xspark-v1 HDF5 episodes into nested dicts.

Camera ``colors`` are decoded to RGB by ``spark0.utils.image_codec.decode_image_bit``.
JPEG frame sequences are decoded one element at a time; this keeps single-frame and
frame-sequence HDF5 datasets compatible with the same RGB decoder.
Metadata-only reads never load images or episode arrays.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
from spark0.utils.image_codec import decode_image_bit


def _maybe_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _dataset_value(key: str, value, decode_images: bool):
    if key == "colors" and isinstance(value, np.ndarray):
        if not decode_images:
            return value
        # The decoder treats a 1-D ndarray as one encoded image.
        # HDF5 camera streams are commonly a 1-D sequence of JPEG byte strings,
        # so decode each frame separately and retain the existing RGB decoder.
        if value.ndim == 1 and value.dtype.kind in {"S", "O", "U"}:
            frames = [decode_image_bit(frame) for frame in value]
            return np.stack(frames, axis=0)
        if value.ndim == 3 and value.shape[-1] == 3:
            return value
        return decode_image_bit(value)

    if isinstance(value, (bytes, np.bytes_)):
        try:
            return _maybe_json(value.decode("utf-8"))
        except UnicodeDecodeError:
            return value

    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        item = value.item() if value.size == 1 else value
        if isinstance(item, (bytes, np.bytes_)):
            try:
                return _maybe_json(item.decode("utf-8"))
            except UnicodeDecodeError:
                return value
        if isinstance(item, str):
            return _maybe_json(item)

    return value


def _group_to_dict(group: h5py.Group, decode_images: bool) -> dict:
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _group_to_dict(item, decode_images)
        elif isinstance(item, h5py.Dataset):
            result[key] = _dataset_value(key, item[()], decode_images)
    return result


def load_xspark_v1(hdf5_path, decode_images: bool = True) -> dict:
    with h5py.File(hdf5_path, "r") as file:
        return _group_to_dict(file, decode_images)


def _validate_format(data_type: str, data_version: str) -> None:
    # RoboDojo sim_cloud HDF5 uses the same xspark v1.0 layout.
    if data_type not in {"xspark", "RoboDojo"}:
        raise NotImplementedError(f"{data_type} is not valid data type. ")
    if data_version != "v1.0":
        raise NotImplementedError(f"{data_version} is not valid in {data_type} .")


def load_metadata(data_path, data_type: str = "xspark", data_version: str = "v1.0") -> dict:
    """Read additional_info without loading episode arrays or decoding images."""
    _validate_format(data_type, data_version)
    with h5py.File(data_path, "r") as file:
        item = file.get("additional_info")
        if item is None:
            return {}
        value = (
            _group_to_dict(item, decode_images=False)
            if isinstance(item, h5py.Group)
            else _dataset_value("additional_info", item[()], decode_images=False)
        )
    return {"additional_info": value}


def load(data_path, data_type: str = "xspark", data_version: str = "v1.0") -> dict:
    _validate_format(data_type, data_version)
    return load_xspark_v1(data_path, decode_images=True)
