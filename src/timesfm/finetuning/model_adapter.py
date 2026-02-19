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

"""Training adapters for TimesFM torch models."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from ..torch import util
from ..timesfm_2p5 import timesfm_2p5_torch

revin = util.revin


@dataclasses.dataclass(frozen=True)
class FinetuneForwardOutput:
  """Model outputs returned by training forward pass."""

  point_forecast: torch.Tensor
  full_forecast: torch.Tensor


class BaseTrainAdapter(nn.Module):
  """Base class for finetuning model adapters."""

  patch_len: int
  point_channel_index: int
  quantiles: tuple[float, ...]

  def forward_train(
    self,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    horizon: int,
  ) -> FinetuneForwardOutput:
    raise NotImplementedError


class TimesFMTorchTrainAdapter(BaseTrainAdapter):
  """Training wrapper around TimesFM_2p5_200M_torch_module.

  Phase-1 constraint: only horizon <= output patch length is supported.
  """

  def __init__(self, model: timesfm_2p5_torch.TimesFM_2p5_200M_torch_module):
    super().__init__()
    self.model = model
    self.patch_len = model.p
    self.max_horizon = model.o
    self.point_channel_index = model.aridx
    self.quantiles = tuple(model.config.quantiles)

  def _running_context_stats(
    self,
    patched_inputs: torch.Tensor,
    patched_masks: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_patches, _ = patched_inputs.shape
    n = torch.zeros(batch_size, device=patched_inputs.device)
    mu = torch.zeros(batch_size, device=patched_inputs.device)
    sigma = torch.zeros(batch_size, device=patched_inputs.device)
    patch_mu = []
    patch_sigma = []

    for i in range(num_patches):
      (n, mu, sigma), _ = util.update_running_stats(
        n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
      )
      patch_mu.append(mu)
      patch_sigma.append(sigma)

    return torch.stack(patch_mu, dim=1), torch.stack(patch_sigma, dim=1)

  def forward_train(
    self,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    horizon: int,
  ) -> FinetuneForwardOutput:
    if horizon <= 0:
      raise ValueError("horizon must be positive.")
    if horizon > self.max_horizon:
      raise ValueError(
        f"Phase-1 training supports horizon <= {self.max_horizon}, got {horizon}."
      )
    if context.shape != context_mask.shape:
      raise ValueError("context and context_mask must have matching shapes.")
    if context.shape[1] % self.patch_len != 0:
      raise ValueError("context length must be a multiple of patch length.")

    batch_size = context.shape[0]
    patched_inputs = torch.reshape(context, (batch_size, -1, self.patch_len))
    patched_masks = torch.reshape(context_mask, (batch_size, -1, self.patch_len))

    context_mu, context_sigma = self._running_context_stats(patched_inputs, patched_masks)
    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    (_, _, normed_outputs, _), _ = self.model(
      normed_inputs,
      patched_masks,
      decode_caches=None,
    )

    renormed_outputs = torch.reshape(
      revin(normed_outputs, context_mu, context_sigma, reverse=True),
      (batch_size, -1, self.max_horizon, self.model.q),
    )

    full_forecast = renormed_outputs[:, -1, :horizon, :]
    point_forecast = full_forecast[..., self.point_channel_index]
    return FinetuneForwardOutput(
      point_forecast=point_forecast,
      full_forecast=full_forecast,
    )
