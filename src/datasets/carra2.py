# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cftime
import cv2
import numpy as np
import torch
import xarray as xr
import zarr
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, Subset

from datasets.base import ChannelMetadata, DownscalingDataset
from datasets.img_utils import reshape_fields
from datasets.norm import denormalize, normalize

logger = logging.getLogger(__file__)
EPS = 1e-6


@dataclass(frozen=True)
class VariableGroups:
    dynamic_inputs: list[str]
    static_inputs: list[str]
    targets: list[str]


class ZarrVariableDiscovery:
    def __init__(
        self,
        *,
        target: str | None = None,
        static_candidates: Sequence[str] = ("x_lsm", "x_orog"),
    ):
        self.targets = [target, ]
        self.static_candidates = tuple(static_candidates)

    def discover(self, data: xr.Dataset) -> VariableGroups:
        dynamic_inputs = sorted(
            [
                name
                for name, array in data.data_vars.items()
                if name.startswith("x_") and array.ndim == 3
            ]
        )
        static_inputs = [
            name
            for name in self.static_candidates
            if name in data and data[name].ndim == 2
        ]
        if self.targets is None:
            self.targets = sorted(
                [
                    name
                    for name, array in data.data_vars.items()
                    if name.startswith("y_") and array.ndim == 3
                ]
            )

        if not dynamic_inputs:
            raise ValueError(
                "No dynamic predictor variables found. Expected 3D variables starting with 'x_'."
            )

        return VariableGroups(
            dynamic_inputs=dynamic_inputs,
            static_inputs=static_inputs,
            targets=self.targets,
        )


class ZarrNormalizationStats:
    def __init__(self, stats_path: str | Path):
        self.stats_path = Path(stats_path)
        self.stats = xr.open_zarr(self.stats_path)

    def tensor_stats(
        self, variables: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means: list[float] = []
        stds: list[float] = []
        for variable in variables:
            mean, std = self._extract_mean_std(variable)
            means.append(mean)
            stds.append(max(std, EPS))

        mean_tensor = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)
        return mean_tensor, std_tensor

    def _extract_mean_std(self, variable: str) -> tuple[float, float]:
        mean_key = f"{variable}_mean"
        std_key = f"{variable}_std"
        if mean_key in self.stats and std_key in self.stats:
            return float(self.stats[mean_key].values), float(self.stats[std_key].values)

        raise ValueError(
            f"Could not infer mean/std format for variable '{variable}' in {self.stats_path}."
        )


class ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file

    This dataset should not be modified to add image processing contributions.
    """

    path: str

    def __init__(
        self,
        data_path: str,
        stats_file: str | Path,
    ):
        super().__init__()
        self.path = Path(data_path)

        # chunks here are dask chunks, not zarr chunks
        data = xr.open_zarr(self.path, consolidated=True, chunks=None)
        self.data = None

        discovery = ZarrVariableDiscovery(target="y_t2m")
        self.variable_groups = discovery.discover(data)

        self.input_vars = (
            self.variable_groups.dynamic_inputs + self.variable_groups.static_inputs
        )
        self.target_vars = self.variable_groups.targets
        total_samples = int(data[self.variable_groups.dynamic_inputs[0]].shape[0])
        self.indices = np.asarray(np.arange(total_samples), dtype=np.int64)

        self.static_tensor = self._load_static_tensor(data)

        stats = ZarrNormalizationStats(stats_file)
        self.input_means, self.input_stds = stats.tensor_stats(self.input_vars)
        self.target_means, self.target_stds = stats.tensor_stats(self.target_vars)

    def _load_static_tensor(self, data) -> torch.Tensor:
        static = np.stack(
            [data[var].values for var in self.variable_groups.static_inputs], axis=0
        )
        return torch.tensor(static, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _ensure_open(self):
        if self.data is None:
            self.data = xr.open_zarr(self.path, consolidated=True, chunks=None)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        time_index = int(self.indices[index])

        dynamic = np.stack(
            [
                self.data[var].isel({"time": time_index}).values
                for var in self.variable_groups.dynamic_inputs
            ],
            axis=0,
        )
        dynamic_tensor = torch.tensor(dynamic, dtype=torch.float32)
        x = torch.cat([dynamic_tensor, self.static_tensor], dim=0)

        targets = np.stack(
            [
                self.data[var].isel({"time": time_index}).values
                for var in self.variable_groups.targets
            ],
            axis=0,
        )
        y = torch.tensor(targets, dtype=torch.float32)

        return y, x

    def normalize_input(self, x):
        """Convert input from physical units to normalized data."""
        return normalize(x, self.input_means, self.input_stds)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        return denormalize(x, self.input_means, self.input_stds)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        return normalize(x, self.target_means, self.target_stds)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        return denormalize(x, self.target_means, self.target_stds)

    def longitude(self):
        """The longitude. useful for plotting"""
        self._ensure_open()
        return self.data["longitude"]

    def latitude(self):
        """The latitude. useful for plotting"""
        self._ensure_open()
        return self.data["latitude"]

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        return [ChannelMetadata(name) for name in self.input_vars]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        return [ChannelMetadata(name) for name in self.target_vars]

    def _read_time(self):
        """The vector of time coordinate has length (self)"""
        self._ensure_open()
        return cftime.num2date(
            self.data["time"], units=self.data["time"].attrs["units"]
        )

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time.tolist()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        self._ensure_open()
        return self.data["y_t2m"].shape[-2:]

    def info(self):
        return "none"


def get_zarr_dataset(*, data_path, **kwargs):
    """Get a Zarr dataset for training or evaluation."""
    data_path = to_absolute_path(data_path)
    return ZarrDataset(data_path=data_path, **kwargs)
