# Copyright 2026 Google LLC
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

"""Dataset utilities for TimesFM finetuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..timesfm_2p5 import timesfm_2p5_base
from .config import WindowingConfig


class TimeSeriesWindowDataset(Dataset):
  """Sliding-window dataset over one or many univariate time series."""

  def __init__(self, series_list: Sequence[Sequence[float]], config: WindowingConfig):
    if not series_list:
      raise ValueError("series_list must contain at least one series.")
    if config.context_len <= 0 or config.horizon_len <= 0:
      raise ValueError("context_len and horizon_len must be positive.")
    if config.stride <= 0:
      raise ValueError("stride must be positive.")

    self.config = config
    self.samples: list[tuple[np.ndarray, np.ndarray]] = []

    total = config.context_len + config.horizon_len
    for raw in series_list:
      arr = np.array(raw, dtype=np.float32)
      arr = timesfm_2p5_base.strip_leading_nans(arr)
      arr = timesfm_2p5_base.linear_interpolation(arr)
      if arr.size < total:
        if config.drop_last:
          continue
        raise ValueError(
          f"Series length {arr.size} is shorter than required window {total}."
        )

      max_start = arr.size - total
      for start in range(0, max_start + 1, config.stride):
        context_end = start + config.context_len
        target_end = context_end + config.horizon_len
        context = arr[start:context_end]
        target = arr[context_end:target_end]
        self.samples.append((context, target))

    if not self.samples:
      raise ValueError("No training samples were created. Check dataset/window config.")

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    context, target = self.samples[index]
    return {
      "context": torch.tensor(context, dtype=torch.float32),
      "target": torch.tensor(target, dtype=torch.float32),
    }


def create_collate_fn(patch_len: int) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
  """Creates a collate function that right-aligns context and pads to patch size."""

  if patch_len <= 0:
    raise ValueError("patch_len must be positive.")

  def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
      raise ValueError("Batch is empty.")

    bsize = len(batch)
    contexts = [item["context"] for item in batch]
    targets = [item["target"] for item in batch]

    max_context = max(x.shape[0] for x in contexts)
    padded_context = ((max_context + patch_len - 1) // patch_len) * patch_len

    context_tensor = torch.zeros((bsize, padded_context), dtype=torch.float32)
    context_mask = torch.ones((bsize, padded_context), dtype=torch.bool)

    for i, context in enumerate(contexts):
      c_len = context.shape[0]
      context_tensor[i, -c_len:] = context
      context_mask[i, -c_len:] = False

    horizon = targets[0].shape[0]
    if any(t.shape[0] != horizon for t in targets):
      raise ValueError("All targets in a batch must have the same horizon length.")
    target_tensor = torch.stack(targets, dim=0)

    return {
      "context": context_tensor,
      "context_mask": context_mask,
      "target": target_tensor,
    }

  return _collate
