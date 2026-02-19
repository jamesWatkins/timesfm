from __future__ import annotations

import types

import torch
from torch import nn

from timesfm.finetuning import TimesFMTorchTrainAdapter


class _FakeModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.p = 2
    self.o = 4
    self.q = 3
    self.aridx = 1
    self.scale = nn.Parameter(torch.tensor(1.0))
    self.config = types.SimpleNamespace(quantiles=[0.1, 0.5])

  def forward(self, inputs: torch.Tensor, masks: torch.Tensor, decode_caches=None):
    del masks, decode_caches
    b, n, _ = inputs.shape
    flattened = inputs.repeat_interleave(self.o * self.q // self.p, dim=-1)
    flattened = flattened * self.scale
    return (inputs, inputs, flattened, flattened), []


def test_adapter_forward_shapes() -> None:
  model = _FakeModel()
  adapter = TimesFMTorchTrainAdapter(model)

  context = torch.arange(16, dtype=torch.float32).reshape(2, 8)
  mask = torch.zeros_like(context, dtype=torch.bool)
  out = adapter.forward_train(context=context, context_mask=mask, horizon=3)

  assert out.point_forecast.shape == (2, 3)
  assert out.full_forecast.shape == (2, 3, 3)
