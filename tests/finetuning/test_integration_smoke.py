from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from timesfm.finetuning import (
  BaseTrainAdapter,
  FinetuneForwardOutput,
  FinetuningConfig,
  TimeSeriesWindowDataset,
  TimesFMFinetuner,
  WindowingConfig,
)


class _SmokeAdapter(BaseTrainAdapter):
  def __init__(self):
    super().__init__()
    self.patch_len = 4
    self.point_channel_index = 0
    self.quantiles = ()
    self.bias = nn.Parameter(torch.tensor(0.0))

  def forward_train(self, context: torch.Tensor, context_mask: torch.Tensor, horizon: int):
    del context_mask
    point = context[:, -horizon:] + self.bias
    full = point[..., None]
    return FinetuneForwardOutput(point_forecast=point, full_forecast=full)


def test_smoke_end_to_end(tmp_path: Path) -> None:
  series = np.sin(np.linspace(0.0, 20.0, 300))
  cfg = WindowingConfig(context_len=32, horizon_len=8, stride=4)
  train_ds = TimeSeriesWindowDataset([series[:220]], cfg)
  val_ds = TimeSeriesWindowDataset([series[180:]], cfg)

  ft_cfg = FinetuningConfig(
    batch_size=8,
    num_epochs=1,
    learning_rate=1e-2,
    output_dir=str(tmp_path),
    log_every_n_steps=0,
  )
  trainer = TimesFMFinetuner(_SmokeAdapter(), ft_cfg)
  result = trainer.finetune(train_ds, val_ds)

  assert len(result.train_losses) == 1
  assert result.checkpoint_path is not None
  assert (tmp_path / "epoch_1.pt").exists()
