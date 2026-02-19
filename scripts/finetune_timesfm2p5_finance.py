#!/usr/bin/env python3
"""Finance-focused TimesFM 2.5 torch finetuning script.

This replicates the v1 notebook-style flow with a script that is easier to run
and debug end-to-end.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

# Ensure Matplotlib works in headless/sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

import timesfm
from timesfm.finetuning import (
  FinetuningConfig,
  TimeSeriesWindowDataset,
  TimesFMFinetuner,
  TimesFMTorchTrainAdapter,
  WindowingConfig,
  create_collate_fn,
)
from timesfm.timesfm_2p5 import timesfm_2p5_torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Finetune TimesFM 2.5 on financial data (ticker or CSV)."
  )
  parser.add_argument("--ticker", default="AAPL")
  parser.add_argument("--start", default="2010-01-01")
  parser.add_argument("--end", default="2019-01-01")
  parser.add_argument(
    "--csv",
    default=None,
    help="Optional local CSV fallback if ticker download fails.",
  )
  parser.add_argument("--csv-column", default="Close")
  parser.add_argument(
    "--model-id",
    default="google/timesfm-2.5-200m-pytorch",
    help="HF model id or local model directory containing model.safetensors.",
  )
  parser.add_argument(
    "--local-model-file",
    default=None,
    help=(
      "Optional direct path to a local model.safetensors file. "
      "If set, bypasses Hugging Face download APIs."
    ),
  )
  parser.add_argument("--context-len", type=int, default=256)
  parser.add_argument("--horizon-len", type=int, default=64)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--train-split", type=float, default=0.8)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--output-dir", default="outputs/finetune_finance")
  parser.add_argument("--use-mlflow", action="store_true")
  parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
  parser.add_argument("--mlflow-experiment-name", default="timesfm-finetuning")
  parser.add_argument("--mlflow-run-name", default=None)
  parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--no-inference-check",
    action="store_true",
    help="Skip post-training inference sanity check.",
  )
  return parser.parse_args()


def _extract_close_values(df: pd.DataFrame, csv_column: str) -> np.ndarray:
  if df.empty:
    raise RuntimeError("Input dataframe is empty.")

  # yfinance can produce either normal or MultiIndex columns.
  if isinstance(df.columns, pd.MultiIndex):
    if csv_column in df.columns.get_level_values(0):
      sub = df[csv_column]
      if isinstance(sub, pd.DataFrame):
        series = sub.iloc[:, 0]
      else:
        series = sub
    else:
      raise KeyError(
        f"Column '{csv_column}' not found in MultiIndex columns {list(df.columns)}."
      )
  else:
    if csv_column not in df.columns:
      raise KeyError(
        f"Column '{csv_column}' not found in columns {list(df.columns)}."
      )
    series = df[csv_column]

  values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float32)
  if values.size == 0:
    raise RuntimeError("No numeric values were found in selected column.")
  return values


def load_series(args: argparse.Namespace) -> np.ndarray:
  if args.csv:
    csv_path = Path(args.csv)
    if not csv_path.exists():
      raise FileNotFoundError(f"CSV fallback file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    values = _extract_close_values(df, args.csv_column)
    print(f"Loaded {len(values)} points from CSV {csv_path}.")
    return values

  try:
    df = yf.download(
      args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False
    )
    values = _extract_close_values(df, args.csv_column)
    print(f"Downloaded {len(values)} points for ticker {args.ticker}.")
    return values
  except Exception as exc:  # pragma: no cover - network-dependent branch
    if not args.csv:
      raise RuntimeError(
        "Ticker download failed and no --csv fallback was provided. "
        "Pass --csv /path/to/file.csv --csv-column Close."
      ) from exc
    print(f"Ticker download failed ({exc}). Falling back to CSV: {args.csv}")

  raise RuntimeError("Unreachable")


def build_datasets(
  values: np.ndarray,
  context_len: int,
  horizon_len: int,
  stride: int,
  train_split: float,
) -> tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, WindowingConfig]:
  if not (0.0 < train_split < 1.0):
    raise ValueError("--train-split must be in (0, 1).")
  if context_len <= 0 or horizon_len <= 0:
    raise ValueError("--context-len and --horizon-len must be positive.")
  if stride <= 0:
    raise ValueError("--stride must be positive.")

  min_needed = context_len + horizon_len
  if len(values) < min_needed:
    raise ValueError(
      f"Series too short for requested windows: len={len(values)} < {min_needed}."
    )

  split_idx = int(len(values) * train_split)
  split_idx = max(split_idx, min_needed)
  split_idx = min(split_idx, len(values) - horizon_len)

  train_values = values[:split_idx]
  val_start = max(0, split_idx - context_len - horizon_len)
  val_values = values[val_start:]

  window_cfg = WindowingConfig(
    context_len=context_len,
    horizon_len=horizon_len,
    stride=stride,
  )
  train_ds = TimeSeriesWindowDataset([train_values], window_cfg)
  val_ds = TimeSeriesWindowDataset([val_values], window_cfg)
  return train_ds, val_ds, window_cfg


def _resolve_device(device_arg: str) -> str:
  if device_arg == "auto":
    return "cuda" if torch.cuda.is_available() else "cpu"
  if device_arg == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device=cuda requested but CUDA is not available.")
  return device_arg


def plot_loss_curve(result, output_dir: Path) -> Path:
  path = output_dir / "loss_curve.png"
  xs = np.arange(1, len(result.train_losses) + 1)
  plt.figure(figsize=(8, 4))
  plt.plot(xs, result.train_losses, label="train_loss", linewidth=2)
  plt.plot(xs, result.val_losses, label="val_loss", linewidth=2)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Finetuning Loss Curve")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path)
  plt.close()
  return path


def plot_prediction_example(
  adapter: TimesFMTorchTrainAdapter,
  dataset: TimeSeriesWindowDataset,
  horizon_len: int,
  output_dir: Path,
) -> Path:
  path = output_dir / "prediction_plot.png"
  adapter.eval()

  collate = create_collate_fn(adapter.patch_len)
  batch = collate([dataset[0]])

  param = next(adapter.parameters())
  context = batch["context"].to(param.device)
  context_mask = batch["context_mask"].to(param.device)
  target = batch["target"][0].cpu().numpy()

  with torch.no_grad():
    out = adapter.forward_train(
      context=context, context_mask=context_mask, horizon=horizon_len
    )
  forecast = out.point_forecast[0].detach().cpu().numpy()

  valid_start = np.where(~batch["context_mask"][0].numpy())[0][0]
  context_vals = context[0].detach().cpu().numpy()[valid_start:]

  x0 = np.arange(len(context_vals))
  x1 = np.arange(len(context_vals), len(context_vals) + len(target))

  plt.figure(figsize=(10, 4))
  plt.plot(x0, context_vals, label="Context", linewidth=2)
  plt.plot(x1, target, label="Ground Truth", linestyle="--", linewidth=2)
  plt.plot(x1, forecast, label="Forecast", linewidth=2)
  plt.title("TimesFM 2.5 Finetuning Prediction")
  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path)
  plt.close()
  return path


def run_inference_sanity(model_id: str, series: np.ndarray) -> None:
  infer_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
  infer_model.compile(
    timesfm.ForecastConfig(
      max_context=256,
      max_horizon=64,
      normalize_inputs=True,
    )
  )
  point, quantile = infer_model.forecast(horizon=8, inputs=[series[-200:]])
  print("Inference sanity output shapes:", point.shape, quantile.shape)


def load_finetune_adapter(
  model_id: str,
  local_model_file: str | None,
) -> TimesFMTorchTrainAdapter:
  if local_model_file:
    model_path = Path(local_model_file)
    if not model_path.exists():
      raise FileNotFoundError(f"--local-model-file does not exist: {model_path}")
    module = timesfm_2p5_torch.TimesFM_2p5_200M_torch_module()
    module.load_checkpoint(str(model_path), torch_compile=False)
    return TimesFMTorchTrainAdapter(module)

  try:
    wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
  except Exception as exc:
    raise RuntimeError(
      "Failed to load model from Hugging Face. "
      "If running offline, pass --local-model-file /path/to/model.safetensors."
    ) from exc
  return TimesFMTorchTrainAdapter(wrapper.model)


def main() -> None:
  args = parse_args()
  device = _resolve_device(args.device)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  values = load_series(args)
  train_ds, val_ds, window_cfg = build_datasets(
    values=values,
    context_len=args.context_len,
    horizon_len=args.horizon_len,
    stride=args.stride,
    train_split=args.train_split,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
  print(f"Window config: {window_cfg}")

  adapter = load_finetune_adapter(args.model_id, args.local_model_file)
  if args.horizon_len > adapter.max_horizon:
    raise ValueError(
      f"--horizon-len={args.horizon_len} exceeds adapter max_horizon={adapter.max_horizon}."
    )

  run_name = args.mlflow_run_name
  if args.use_mlflow and not run_name:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"timesfm-finance-{ts}"

  ft_cfg = FinetuningConfig(
    batch_size=args.batch_size,
    num_epochs=args.epochs,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    mixed_precision=(device == "cuda"),
    output_dir=str(output_dir),
    device=device,
    use_mlflow=args.use_mlflow,
    mlflow_tracking_uri=args.mlflow_tracking_uri,
    mlflow_experiment_name=args.mlflow_experiment_name,
    mlflow_run_name=run_name,
  )

  trainer = TimesFMFinetuner(adapter, ft_cfg)
  result = trainer.finetune(train_ds, val_ds)
  print("Finetuning result:", result)

  try:
    loss_path = plot_loss_curve(result, output_dir)
    pred_path = plot_prediction_example(adapter, val_ds, args.horizon_len, output_dir)
    trainer.log_artifact(loss_path)
    trainer.log_artifact(pred_path)
    print(f"Saved loss curve: {loss_path.resolve()}")
    print(f"Saved prediction plot: {pred_path.resolve()}")
  except Exception as exc:
    print(f"Plotting failed: {exc}")
  if result.checkpoint_path:
    print(f"Best checkpoint: {Path(result.checkpoint_path).resolve()}")

  if args.local_model_file and not args.no_inference_check:
    print(
      "Skipping inference sanity check when --local-model-file is used. "
      "Use --model-id with network access for wrapper-level inference validation."
    )
  elif not args.no_inference_check:
    run_inference_sanity(args.model_id, values)


if __name__ == "__main__":
  main()
