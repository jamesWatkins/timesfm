"""DuckDB utilities for exchange OHLCV ingestion."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

if TYPE_CHECKING:
  import duckdb


@dataclass(frozen=True)
class IngestStats:
  fetched_rows: int
  inserted_rows: int
  first_open_time_ms: int | None
  last_open_time_ms: int | None


def connect_db(db_path: str):
  try:
    import duckdb
  except ImportError as exc:  # pragma: no cover
    raise RuntimeError("duckdb is required. Install with `pip install -e .[finetune]`.") from exc
  path = Path(db_path)
  if path.parent and not path.parent.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
  # DuckDB can create a new DB file itself, but a pre-created empty file is invalid.
  if path.exists() and path.stat().st_size == 0:
    path.unlink()
  return duckdb.connect(database=str(path))


def ensure_schema(conn) -> None:
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS btc_klines_raw (
      exchange VARCHAR NOT NULL,
      symbol VARCHAR NOT NULL,
      interval VARCHAR NOT NULL,
      open_time_ms BIGINT NOT NULL,
      open_ts_utc TIMESTAMP NOT NULL,
      open DOUBLE,
      high DOUBLE,
      low DOUBLE,
      close DOUBLE,
      volume DOUBLE,
      close_time_ms BIGINT,
      quote_volume DOUBLE,
      trade_count BIGINT,
      taker_buy_base DOUBLE,
      taker_buy_quote DOUBLE,
      ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      ingest_batch_id VARCHAR,
      UNIQUE(exchange, symbol, interval, open_time_ms)
    )
    """
  )
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS btc_1m_canonical (
      timestamp_utc TIMESTAMP NOT NULL,
      symbol VARCHAR NOT NULL,
      close DOUBLE,
      source_exchange VARCHAR NOT NULL,
      is_missing BOOLEAN NOT NULL,
      materialized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (timestamp_utc, symbol)
    )
    """
  )
  conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_btc_klines_raw_symbol_ts ON btc_klines_raw(symbol, open_ts_utc)"
  )


def create_exchange_client(exchange_id: str = "binanceus") -> Any:
  try:
    import ccxt
  except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ccxt is required. Install with `pip install -e .[finetune]`.") from exc

  if not hasattr(ccxt, exchange_id):
    raise ValueError(f"Unsupported exchange id: {exchange_id}")
  exchange_cls = getattr(ccxt, exchange_id)
  return exchange_cls({"enableRateLimit": True})


def create_binance_client() -> Any:
  """Backward-compatible helper. Uses binanceus by default for US regions."""
  return create_exchange_client("binanceus")


def to_ccxt_symbol(symbol: str) -> str:
  if "/" in symbol:
    return symbol
  if symbol.endswith("USDT") and len(symbol) > 4:
    return f"{symbol[:-4]}/USDT"
  if symbol.endswith("USD") and len(symbol) > 3:
    return f"{symbol[:-3]}/USD"
  raise ValueError(
    f"Unsupported symbol format '{symbol}'. Use forms like BTCUSDT or BTC/USD."
  )


def discover_earliest_binance_1m(exchange: Any, symbol: str = "BTCUSDT") -> datetime:
  rows = exchange.fetch_ohlcv(symbol=symbol, timeframe="1m", since=0, limit=1)
  if not rows:
    raise RuntimeError(f"No OHLCV data returned for symbol={symbol} timeframe=1m.")
  return datetime.fromtimestamp(rows[0][0] / 1000.0, tz=UTC)


def _normalize_kline_row(
  exchange: str,
  symbol: str,
  interval: str,
  row: Sequence[float],
  batch_id: str,
) -> tuple[Any, ...]:
  def _maybe_int(v: Any) -> int | None:
    return int(v) if v is not None else None

  def _maybe_float(v: Any) -> float | None:
    return float(v) if v is not None else None

  open_time_ms = int(row[0])
  return (
    exchange,
    symbol,
    interval,
    open_time_ms,
    datetime.fromtimestamp(open_time_ms / 1000.0, tz=UTC),
    float(row[1]),
    float(row[2]),
    float(row[3]),
    float(row[4]),
    float(row[5]),
    _maybe_int(row[6]) if len(row) > 6 else None,
    _maybe_float(row[7]) if len(row) > 7 else None,
    _maybe_int(row[8]) if len(row) > 8 else None,
    _maybe_float(row[9]) if len(row) > 9 else None,
    _maybe_float(row[10]) if len(row) > 10 else None,
    datetime.now(tz=UTC),
    batch_id,
  )


def fetch_klines_paginated(
  exchange: Any,
  symbol: str,
  start_ms: int,
  end_ms: int,
  limit: int = 1000,
  max_retries: int = 5,
  sleep_ms: int = 200,
) -> list[dict[str, Any]]:
  """Fetches 1-minute klines in [start_ms, end_ms)."""
  if end_ms <= start_ms:
    return []

  rows: list[dict[str, Any]] = []
  since = start_ms
  while since < end_ms:
    attempt = 0
    page = None
    while attempt <= max_retries:
      try:
        page = exchange.fetch_ohlcv(
          symbol=symbol,
          timeframe="1m",
          since=since,
          limit=limit,
        )
        break
      except Exception:
        if attempt == max_retries:
          raise
        backoff_ms = min(5000, (2**attempt) * sleep_ms)
        time.sleep(backoff_ms / 1000.0)
        attempt += 1

    if not page:
      break

    for kline in page:
      open_time_ms = int(kline[0])
      if open_time_ms >= end_ms:
        continue
      rows.append(
        {
          "open_time_ms": open_time_ms,
          "open": float(kline[1]),
          "high": float(kline[2]),
          "low": float(kline[3]),
          "close": float(kline[4]),
          "volume": float(kline[5]),
          "close_time_ms": int(kline[6]) if len(kline) > 6 else None,
          "quote_volume": float(kline[7]) if len(kline) > 7 else None,
          "trade_count": int(kline[8]) if len(kline) > 8 else None,
          "taker_buy_base": float(kline[9]) if len(kline) > 9 else None,
          "taker_buy_quote": float(kline[10]) if len(kline) > 10 else None,
        }
      )

    last_open_ms = int(page[-1][0])
    next_since = last_open_ms + 60_000
    if next_since <= since:
      break
    since = next_since
    time.sleep(sleep_ms / 1000.0)

  return rows


def upsert_raw_klines(
  conn,
  rows: Sequence[dict[str, Any]],
  symbol: str = "BTCUSDT",
  exchange: str = "binance",
  interval: str = "1m",
  batch_id: str | None = None,
) -> IngestStats:
  if not rows:
    return IngestStats(0, 0, None, None)

  batch_id = batch_id or str(uuid.uuid4())
  normalized = [
    _normalize_kline_row(
      exchange=exchange,
      symbol=symbol,
      interval=interval,
      row=(
        r["open_time_ms"],
        r["open"],
        r["high"],
        r["low"],
        r["close"],
        r["volume"],
        r.get("close_time_ms"),
        r.get("quote_volume"),
        r.get("trade_count"),
        r.get("taker_buy_base"),
        r.get("taker_buy_quote"),
      ),
      batch_id=batch_id,
    )
    for r in rows
  ]

  before = conn.execute("SELECT COUNT(*) FROM btc_klines_raw").fetchone()[0]
  conn.executemany(
    """
    INSERT OR IGNORE INTO btc_klines_raw (
      exchange, symbol, interval, open_time_ms, open_ts_utc,
      open, high, low, close, volume,
      close_time_ms, quote_volume, trade_count, taker_buy_base, taker_buy_quote,
      ingested_at, ingest_batch_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    normalized,
  )
  after = conn.execute("SELECT COUNT(*) FROM btc_klines_raw").fetchone()[0]

  open_times = [r["open_time_ms"] for r in rows]
  return IngestStats(
    fetched_rows=len(rows),
    inserted_rows=int(after - before),
    first_open_time_ms=min(open_times),
    last_open_time_ms=max(open_times),
  )


def materialize_canonical_1m(
  conn,
  symbol: str = "BTCUSDT",
  source_exchange: str = "binance",
) -> dict[str, int]:
  bounds = conn.execute(
    """
    SELECT MIN(open_ts_utc) AS min_ts, MAX(open_ts_utc) AS max_ts
    FROM btc_klines_raw
    WHERE symbol = ? AND interval = '1m'
    """,
    [symbol],
  ).fetchone()
  if not bounds or bounds[0] is None or bounds[1] is None:
    return {"materialized_rows": 0}

  conn.execute("DELETE FROM btc_1m_canonical WHERE symbol = ?", [symbol])
  conn.execute(
    """
    INSERT INTO btc_1m_canonical (
      timestamp_utc, symbol, close, source_exchange, is_missing, materialized_at
    )
    WITH minute_grid AS (
      SELECT ts
      FROM generate_series(?, ?, INTERVAL 1 MINUTE) AS g(ts)
    ), close_by_minute AS (
      SELECT open_ts_utc AS ts, close
      FROM btc_klines_raw
      WHERE symbol = ? AND interval = '1m'
    )
    SELECT
      g.ts AS timestamp_utc,
      ? AS symbol,
      c.close,
      ? AS source_exchange,
      c.close IS NULL AS is_missing,
      CURRENT_TIMESTAMP AS materialized_at
    FROM minute_grid g
    LEFT JOIN close_by_minute c
    ON g.ts = c.ts
    ORDER BY g.ts
    """,
    [bounds[0], bounds[1], symbol, symbol, source_exchange],
  )
  count = conn.execute(
    "SELECT COUNT(*) FROM btc_1m_canonical WHERE symbol = ?",
    [symbol],
  ).fetchone()[0]
  return {"materialized_rows": int(count)}


def get_canonical_series(
  conn,
  symbol: str = "BTCUSDT",
  start: datetime | None = None,
  end: datetime | None = None,
) -> pd.DataFrame:
  clauses = ["symbol = ?"]
  params: list[Any] = [symbol]

  if start is not None:
    clauses.append("timestamp_utc >= ?")
    params.append(start)
  if end is not None:
    clauses.append("timestamp_utc < ?")
    params.append(end)

  where_clause = " AND ".join(clauses)
  return conn.execute(
    f"""
    SELECT timestamp_utc, close, is_missing
    FROM btc_1m_canonical
    WHERE {where_clause}
    ORDER BY timestamp_utc
    """,
    params,
  ).df()
