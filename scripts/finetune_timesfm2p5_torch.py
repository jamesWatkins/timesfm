#!/usr/bin/env python3
"""Command-line torch finetuning entrypoint for TimesFM 2.5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from timesfm import TimesFM_2p5_200M_torch
from timesfm.finetuning import (
  FinetuningConfig,
  TimeSeriesWindowDataset,
  TimesFMFinetuner,
  TimesFMTorchTrainAdapter,
  WindowingConfig,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Finetune TimesFM 2.5 (torch)")
  parser.add_argument("--series-npy", required=True, help="Path to 1D numpy .npy time series")
  parser.add_argument("--context-len", type=int, default=256)
  parser.add_argument("--horizon-len", type=int, default=64)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--output-dir", default="outputs/timesfm_finetune")
  parser.add_argument("--use-mlflow", action="store_true")
  parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
  parser.add_argument("--mlflow-experiment-name", default="timesfm-finetuning")
  parser.add_argument("--mlflow-run-name", default=None)
  parser.add_argument(
    "--model-id",
    default="google/timesfm-2.5-200m-pytorch",
    help="Hugging Face model id or local model directory.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  series = np.load(args.series_npy)
  if series.ndim != 1:
    raise ValueError("--series-npy must contain a 1D array.")

  split = int(0.8 * len(series))
  train_series = [series[:split]]
  val_series = [series[split - args.context_len - args.horizon_len :]]

  window_cfg = WindowingConfig(
    context_len=args.context_len,
    horizon_len=args.horizon_len,
    stride=args.stride,
  )
  train_ds = TimeSeriesWindowDataset(train_series, window_cfg)
  val_ds = TimeSeriesWindowDataset(val_series, window_cfg)

  wrapper = TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
  adapter = TimesFMTorchTrainAdapter(wrapper.model)

  run_name = args.mlflow_run_name
  if args.use_mlflow and not run_name:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"timesfm-npy-{ts}"

  ft_cfg = FinetuningConfig(
    batch_size=args.batch_size,
    num_epochs=args.epochs,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    output_dir=args.output_dir,
    use_mlflow=args.use_mlflow,
    mlflow_tracking_uri=args.mlflow_tracking_uri,
    mlflow_experiment_name=args.mlflow_experiment_name,
    mlflow_run_name=run_name,
  )

  trainer = TimesFMFinetuner(adapter, ft_cfg)
  result = trainer.finetune(train_ds, val_ds)
  print(result)
  print(f"Artifacts written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
  main()
