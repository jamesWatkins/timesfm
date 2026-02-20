# Bitcoin 1-Minute DuckDB Pipeline

This pipeline ingests Binance `BTCUSDT` 1-minute OHLCV history into DuckDB, then materializes a canonical minute grid for finetuning.

## 1) Backfill and build canonical table

```bash
python scripts/build_btc_duckdb.py \
  --db-path data/crypto.duckdb \
  --mode full \
  --rebuild-canonical \
  --verbose
```

## 2) Incremental update

```bash
python scripts/build_btc_duckdb.py \
  --db-path data/crypto.duckdb \
  --mode incremental \
  --rebuild-canonical
```

## 3) Sanity SQL checks

```sql
SELECT COUNT(*), MIN(open_ts_utc), MAX(open_ts_utc)
FROM btc_klines_raw
WHERE symbol = 'BTCUSDT';

SELECT COUNT(*) AS rows,
       SUM(CASE WHEN is_missing THEN 1 ELSE 0 END) AS missing_rows,
       MIN(timestamp_utc),
       MAX(timestamp_utc)
FROM btc_1m_canonical
WHERE symbol = 'BTCUSDT';
```

## 4) Finetune from DuckDB

```bash
python scripts/finetune_timesfm2p5_btc_duckdb.py \
  --db-path data/crypto.duckdb \
  --symbol BTCUSDT \
  --context-len 256 \
  --horizon-len 64 \
  --epochs 1 \
  --batch-size 8 \
  --local-model-file /local/google-timesfm-2.5-200m-pytorch/model.safetensors \
  --use-mlflow
```

Notes:
- Canonical table keeps missing minutes as `NULL` (`is_missing = true`).
- The finetuning dataset path interpolates missing values through existing TimesFM preprocessing in `TimeSeriesWindowDataset`.
