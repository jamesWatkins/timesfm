from __future__ import annotations

import torch

from timesfm.finetuning import TimeSeriesWindowDataset, WindowingConfig, create_collate_fn


def test_window_dataset_length_and_shapes() -> None:
  cfg = WindowingConfig(context_len=4, horizon_len=2, stride=2)
  ds = TimeSeriesWindowDataset([list(range(12))], cfg)

  assert len(ds) == 4
  sample = ds[0]
  assert sample["context"].shape == (4,)
  assert sample["target"].shape == (2,)


def test_collate_pads_to_patch_multiple() -> None:
  cfg = WindowingConfig(context_len=5, horizon_len=2, stride=1)
  ds = TimeSeriesWindowDataset([list(range(12))], cfg)

  collate = create_collate_fn(patch_len=4)
  batch = collate([ds[0], ds[1]])

  assert batch["context"].shape == (2, 8)
  assert batch["context_mask"].dtype == torch.bool
  assert batch["target"].shape == (2, 2)
  assert torch.all(batch["context_mask"][:, -5:] == torch.tensor(False))
