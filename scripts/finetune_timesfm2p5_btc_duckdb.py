#!/usr/bin/env python3
"""Finetune TimesFM 2.5 using BTCUSDT 1-minute close data from DuckDB."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timezone
from pathlib import Path

# Ensure Matplotlib works in headless/sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from timesfm.finetuning import (
  FinetuningConfig,
  TimeSeriesWindowDataset,
  TimesFMFinetuner,
  TimesFMTorchTrainAdapter,
  WindowingConfig,
  create_collate_fn,
)
from timesfm.timesfm_2p5 import timesfm_2p5_torch
from timesfm.data.crypto_duckdb import connect_db, get_canonical_series


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Finetune TimesFM 2.5 on BTCUSDT 1-minute close prices from DuckDB."
  )
  parser.add_argument("--db-path", default="data/crypto.duckdb")
  parser.add_argument("--table", default="btc_1m_canonical")
  parser.add_argument("--symbol", default="BTCUSDT")
  parser.add_argument("--start", default=None, help="Optional ISO datetime inclusive")
  parser.add_argument("--end", default=None, help="Optional ISO datetime exclusive")
  parser.add_argument("--drop-nulls", action="store_true")
  parser.add_argument("--min-points", type=int, default=5000)

  parser.add_argument(
    "--model-id",
    default="google/timesfm-2.5-200m-pytorch",
    help="HF model id or local model directory containing model.safetensors.",
  )
  parser.add_argument(
    "--local-model-file",
    default=None,
    help="Direct path to local model.safetensors. Bypasses Hugging Face APIs.",
  )
  parser.add_argument("--context-len", type=int, default=256)
  parser.add_argument("--horizon-len", type=int, default=64)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--train-split", type=float, default=0.8)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-2)
  parser.add_argument("--output-dir", default="outputs/finetune_btc_duckdb")
  parser.add_argument("--use-mlflow", action="store_true")
  parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
  parser.add_argument("--mlflow-experiment-name", default="timesfm-finetuning")
  parser.add_argument("--mlflow-run-name", default=None)
  parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def _parse_ts(value: str | None) -> datetime | None:
  if value is None:
    return None
  ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
  if ts.tzinfo is None:
    ts = ts.replace(tzinfo=UTC)
  return ts.astimezone(UTC)


def _resolve_device(device_arg: str) -> str:
  if device_arg == "auto":
    return "cuda" if torch.cuda.is_available() else "cpu"
  if device_arg == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device=cuda requested but CUDA is not available.")
  return device_arg


def _load_close_series(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
  conn = connect_db(args.db_path)
  # Keep table arg for interface stability while current pipeline is fixed.
  if args.table != "btc_1m_canonical":
    raise ValueError("This v1 path supports --table btc_1m_canonical only.")

  df = get_canonical_series(
    conn=conn,
    symbol=args.symbol,
    start=_parse_ts(args.start),
    end=_parse_ts(args.end),
  )
  if df.empty:
    raise RuntimeError("No rows found in canonical table for the selected range.")

  null_count = int(df["close"].isna().sum())
  if args.drop_nulls:
    close_vals = df["close"].dropna().to_numpy(dtype=np.float32)
  else:
    close_vals = df["close"].to_numpy(dtype=np.float32)

  if close_vals.size < args.min_points:
    raise ValueError(
      f"Insufficient points: got {close_vals.size}, require >= {args.min_points}."
    )

  meta = {
    "rows": int(len(df)),
    "null_rows": null_count,
    "start": str(df["timestamp_utc"].iloc[0]),
    "end": str(df["timestamp_utc"].iloc[-1]),
  }
  return close_vals, meta


def build_datasets(
  values: np.ndarray,
  context_len: int,
  horizon_len: int,
  stride: int,
  train_split: float,
) -> tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, WindowingConfig]:
  if not (0.0 < train_split < 1.0):
    raise ValueError("--train-split must be in (0, 1).")

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
  plt.ylabel("Price")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path)
  plt.close()
  return path


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

  wrapper = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(model_id)
  return TimesFMTorchTrainAdapter(wrapper.model)


def main() -> None:
  args = parse_args()
  device = _resolve_device(args.device)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  values, metadata = _load_close_series(args)
  print("Loaded canonical BTC series:", metadata)

  train_ds, val_ds, _window_cfg = build_datasets(
    values=values,
    context_len=args.context_len,
    horizon_len=args.horizon_len,
    stride=args.stride,
    train_split=args.train_split,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  adapter = load_finetune_adapter(args.model_id, args.local_model_file)
  if args.horizon_len > adapter.max_horizon:
    raise ValueError(
      f"--horizon-len={args.horizon_len} exceeds adapter max_horizon={adapter.max_horizon}."
    )

  run_name = args.mlflow_run_name
  if args.use_mlflow and not run_name:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"timesfm-btc-{ts}"

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
    mlflow_tags={"dataset": "btc_1m_canonical", "symbol": args.symbol},
  )

  trainer = TimesFMFinetuner(adapter, ft_cfg)
  if args.use_mlflow:
    trainer.log_params(
      {
        "db_path": args.db_path,
        "rows": metadata["rows"],
        "null_rows": metadata["null_rows"],
        "data_start": metadata["start"],
        "data_end": metadata["end"],
      }
    )

  result = trainer.finetune(train_ds, val_ds)
  print("Finetuning result:", result)

  loss_path = plot_loss_curve(result, output_dir)
  pred_path = plot_prediction_example(adapter, val_ds, args.horizon_len, output_dir)
  if args.use_mlflow:
    trainer.log_artifact(loss_path)
    trainer.log_artifact(pred_path)

  print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
  main()
