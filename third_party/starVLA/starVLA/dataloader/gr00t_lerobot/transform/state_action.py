# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import random
from typing import Any, ClassVar, Optional

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..schema import DatasetMetadata, RotationType, StateActionMetadata
from .base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


# `hold_centered_perrow` only: lower bound on a row's scale, as a fraction of
# the widest row's scale. A channel whose motion is rarer than the 1%/99%
# quantiles on some row would otherwise get a (near-)zero span there and have
# its few moving samples amplified without bound. 0.02 caps the amplification
# at 50x the widest row; the measured row0/row49 ratio of the shipped Tianji
# delta stats is ~0.05, so the floor does not bind in normal operation.
HOLD_CENTERED_PERROW_MIN_SCALE_RATIO = 0.02

# Every bounded normalization mode (`q99`, `hold_centered*`) saturates here, so
# no training target ever exceeds it and the head's output distribution lives
# inside [-LIMIT, LIMIT]. Named because the deploy side has to know the bound:
# the RTC guidance leftover is normalized through this same forward path, and a
# saturated guidance row silently aims the frozen prefix at a shortened target
# (see PolicyServerWrapper._normalize_prev_chunk).
NORMALIZED_CLAMP_LIMIT = 2.2


class Normalizer:
    valid_modes = [
        "q99",
        "hold_centered",
        "hold_centered_perrow",
        "mean_std",
        "min_max",
        "binary",
    ]

    def __init__(
        self,
        mode: str,
        statistics: dict,
        binary_threshold: float = 0.5,
        hold: list | None = None,
    ):
        self.mode = mode
        self.statistics = statistics
        self.binary_threshold = binary_threshold
        for key, value in self.statistics.items():
            # Optional stats (the *_per_row fields) are absent for every key
            # that keeps the pooled statistics.
            self.statistics[key] = None if value is None else torch.tensor(value)
        # `hold_centered*` only: the representation-defined "robot holds its
        # current pose" vector (a constant, not a dataset statistic).
        self.hold = None if hold is None else torch.tensor(hold, dtype=torch.float32)
        if self.mode in ("hold_centered", "hold_centered_perrow"):
            if self.hold is None:
                raise ValueError(f"{self.mode} normalization requires a `hold` vector")
            for stat in self._required_quantile_keys():
                value = self.statistics.get(stat)
                if value is None:
                    raise ValueError(
                        f"{self.mode} normalization requires {stat} statistics"
                    )
                if value.shape[-1] != len(self.hold):
                    raise ValueError(
                        f"hold vector length {len(self.hold)} != {stat} width "
                        f"{value.shape[-1]}"
                    )
            if self.mode == "hold_centered_perrow":
                q01 = self.statistics["q01_per_row"]
                q99 = self.statistics["q99_per_row"]
                if q01.ndim != 2 or q01.shape != q99.shape:
                    raise ValueError(
                        "hold_centered_perrow needs (H, D) q01_per_row/q99_per_row "
                        f"of equal shape, got {tuple(q01.shape)} and {tuple(q99.shape)}"
                    )

    def _required_quantile_keys(self) -> tuple[str, ...]:
        if self.mode == "hold_centered_perrow":
            return ("q01_per_row", "q99_per_row")
        return ("q01", "q99")

    def _hold_centered_scale(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(hold, scale)`` with ``scale = max(|q01-hold|, |q99-hold|)``.

        Symmetric about ``hold``, so the normalized value of ``hold`` is
        exactly 0 for every channel.
        """
        hold = self.hold.to(dtype)
        q01 = self.statistics["q01"].to(dtype)
        q99 = self.statistics["q99"].to(dtype)
        scale = torch.maximum((q01 - hold).abs(), (q99 - hold).abs())
        return hold, scale

    def _hold_centered_perrow_scale(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(hold, scale)`` with one ``(H, D)`` scale per horizon row.

        Same origin as ``hold_centered`` (so every row still maps hold to
        exactly 0) but the span is taken from that row's own quantiles, which
        equalizes the target amplitude across the chunk instead of letting the
        far rows set one pooled scale for all of them.
        """
        hold = self.hold.to(dtype)
        q01 = self.statistics["q01_per_row"].to(dtype)
        q99 = self.statistics["q99_per_row"].to(dtype)
        scale = torch.maximum((q01 - hold).abs(), (q99 - hold).abs())
        floor = HOLD_CENTERED_PERROW_MIN_SCALE_RATIO * scale.amax(dim=0, keepdim=True)
        return hold, torch.maximum(scale, floor)

    def _check_per_row_shape(self, x: torch.Tensor, scale: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[-2:] != scale.shape:
            raise ValueError(
                f"{self.mode} expects a chunk of shape (..., {scale.shape[0]}, "
                f"{scale.shape[1]}) to match the per-row statistics, got "
                f"{tuple(x.shape)}. The action horizon of the statistics and of "
                "the model must be identical."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(
                normalized, -NORMALIZED_CLAMP_LIMIT, NORMALIZED_CLAMP_LIMIT
            )

        elif self.mode == "hold_centered":
            # Offset-free variant of q99: hold maps to exactly 0, so the flow
            # matching prior mean (0) and any regression shrinkage decode back
            # to "hold" instead of q99's midpoint (q01+q99)/2. For chunk-wise
            # delta pose targets that midpoint is a real ~11 deg + 5 cm motion,
            # which the deploy-side composition accumulates across chunks.
            hold, scale = self._hold_centered_scale(x.dtype)

            # scale == 0 means q01 == q99 == hold: the channel is constant at
            # hold, so 0 is both its only value and an exact inverse.
            mask = scale != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - hold[mask]) / scale[mask]

            normalized = torch.clamp(
                normalized, -NORMALIZED_CLAMP_LIMIT, NORMALIZED_CLAMP_LIMIT
            )

        elif self.mode == "hold_centered_perrow":
            # hold_centered with a per-horizon-row span: under one pooled span
            # the normalized amplitude of a chunk-wise delta target grows ~20x
            # from row 0 to row H-1, so the near rows -- the only ones the
            # robot executes before re-inferring -- get a fraction of the
            # resolution and of the loss budget. Scaling each row by its own
            # quantile span removes that shared-scale bias without moving the
            # origin. It does not assert that every row has equal prediction
            # difficulty or equal residual MSE.
            hold, scale = self._hold_centered_perrow_scale(x.dtype)
            self._check_per_row_shape(x, scale)

            # scale == 0 means the channel never leaves hold on that row, so 0
            # is both its only value and an exact inverse.
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            normalized = torch.where(
                nonzero, (x - hold) / safe_scale, torch.zeros_like(x)
            )

            normalized = torch.clamp(
                normalized, -NORMALIZED_CLAMP_LIMIT, NORMALIZED_CLAMP_LIMIT
            )

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "hold_centered":
            hold, scale = self._hold_centered_scale(x.dtype)
            return x * scale + hold
        elif self.mode == "hold_centered_perrow":
            hold, scale = self._hold_centered_perrow_scale(x.dtype)
            self._check_per_row_shape(x, scale)
            return x * scale + hold
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    hold_values: dict[str, list[float]] = Field(
        default_factory=dict,
        description=(
            "Per-key 'hold' vector for the `hold_centered*` normalization "
            "modes: the constant value that means 'the robot does not move' in "
            "this representation (e.g. the identity delta). Required for every "
            "key whose normalization mode is `hold_centered` or "
            "`hold_centered_perrow`, ignored otherwise."
        ),
    )
    binary_threshold: float = Field(
        default=0.5, description="Threshold for binary normalization mode."
    )
    per_row_horizon: Optional[int] = Field(
        default=None,
        description=(
            "Expected number of horizon rows for `hold_centered_perrow` keys, "
            "i.e. the run's action_horizon. The statistics carry one span per "
            "row, so a chunk length that disagrees with them rescales every "
            "commanded row; checking it here fails at dataset/policy-server "
            "construction instead of at the first batch. None skips the check."
        ),
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode in ("q99", "hold_centered", "hold_centered_perrow"):
                    # `hold_centered_perrow` is scaled by the per-row quantiles
                    # but still validated against the pooled pair, which every
                    # key carries.
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for {normalization_mode} normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                    if normalization_mode == "hold_centered_perrow":
                        for stat in ("q01_per_row", "q99_per_row"):
                            per_row = normalization_statistics.get(stat)
                            assert per_row is not None and len(per_row) > 0, (
                                f"hold_centered_perrow normalization requires {stat} "
                                f"statistics for {modality_key}; recompute the "
                                "converted-pose statistics cache"
                            )
                            assert len(per_row[0]) == len(
                                normalization_statistics["q01"]
                            ), (
                                f"{stat} for {modality_key} must be (H, D) with the "
                                f"same D as q01, got width {len(per_row[0])} vs "
                                f"{len(normalization_statistics['q01'])}"
                            )
                    if normalization_mode in ("hold_centered", "hold_centered_perrow"):
                        assert modality_key in self.hold_values, (
                            f"{normalization_mode} normalization requires hold_values[{modality_key}]"
                        )
                        assert len(self.hold_values[modality_key]) == len(
                            normalization_statistics["q01"]
                        ), (
                            f"hold vector for {modality_key} must match the statistics length, "
                            f"but got {len(self.hold_values[modality_key])} and "
                            f"{len(normalization_statistics['q01'])}"
                        )
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
            if (
                self.normalization_modes[key] in ("hold_centered", "hold_centered_perrow")
                and key not in self.hold_values
            ):
                raise ValueError(
                    f"{key} uses {self.normalization_modes[key]} normalization but no "
                    "hold vector was supplied (StateActionTransform.hold_values)"
                )
            if self.normalization_modes[key] == "hold_centered_perrow":
                self._check_per_row_horizon(key, statistics)
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics,
                binary_threshold=self.binary_threshold,
                hold=self.hold_values.get(key),
            )

    def _check_per_row_horizon(self, key: str, statistics: dict) -> None:
        """Statistics rows must match the run's chunk length, or every row is
        scaled by another row's span."""
        if self.per_row_horizon is None:
            return
        for stat in ("q01_per_row", "q99_per_row"):
            rows = statistics.get(stat)
            if rows is None:
                raise ValueError(
                    f"{key} uses hold_centered_perrow but the statistics carry no "
                    f"{stat}; recompute the converted-pose statistics cache."
                )
            if len(rows) != self.per_row_horizon:
                raise ValueError(
                    f"{key}: {stat} has {len(rows)} horizon rows but this run's "
                    f"action_horizon is {self.per_row_horizon}. The statistics and "
                    "the model must agree on the chunk length — recompute the "
                    "converted-pose statistics, or (at eval) stop overriding "
                    "framework.action_model.action_horizon."
                )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            # Rotate the state
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            # Unnormalize the state
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data
