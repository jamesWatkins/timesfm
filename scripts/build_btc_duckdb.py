#!/usr/bin/env python3
"""Builds/updates a local DuckDB database of BTCUSDT 1-minute bars."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from timesfm.data import (
  connect_db,
  create_exchange_client,
  discover_earliest_binance_1m,
  ensure_schema,
  fetch_klines_paginated,
  materialize_canonical_1m,
  to_ccxt_symbol,
  upsert_raw_klines,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Build BTCUSDT 1m DuckDB dataset.")
  parser.add_argument("--db-path", default="data/crypto.duckdb")
  parser.add_argument("--exchange", default="binanceus")
  parser.add_argument("--symbol", default="BTCUSDT")
  parser.add_argument(
    "--market-symbol",
    default=None,
    help="Optional CCXT market symbol override, e.g. BTC/USDT.",
  )
  parser.add_argument("--interval", default="1m", choices=["1m"])
  parser.add_argument("--mode", default="full", choices=["backfill", "incremental", "full"])
  parser.add_argument("--start", default=None, help="ISO datetime, e.g. 2020-01-01T00:00:00Z")
  parser.add_argument("--end", default=None, help="ISO datetime (exclusive)")
  parser.add_argument("--batch-limit", type=int, default=1000)
  parser.add_argument("--sleep-ms", type=int, default=200)
  parser.add_argument("--rebuild-canonical", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  return parser.parse_args()


def _parse_ts(value: str | None) -> datetime | None:
  if value is None:
    return None
  ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
  if ts.tzinfo is None:
    ts = ts.replace(tzinfo=UTC)
  return ts.astimezone(UTC)


def _floor_to_completed_minute(ts: datetime) -> datetime:
  ts = ts.astimezone(UTC).replace(second=0, microsecond=0)
  return ts


def _ingest_window(
  mode: str,
  earliest: datetime,
  raw_max_ms: int | None,
  cli_start: datetime | None,
  cli_end: datetime,
) -> tuple[datetime, datetime]:
  if cli_start is not None:
    start = cli_start
  elif mode == "incremental":
    if raw_max_ms is None:
      start = earliest
    else:
      start = datetime.fromtimestamp((raw_max_ms + 60_000) / 1000.0, tz=UTC)
  else:
    start = earliest

  return start, cli_end


def main() -> None:
  args = parse_args()
  if args.symbol != "BTCUSDT":
    raise ValueError("This v1 pipeline supports --symbol BTCUSDT only.")
  market_symbol = args.market_symbol or to_ccxt_symbol(args.symbol)

  cli_start = _parse_ts(args.start)
  cli_end = _parse_ts(args.end)
  now_completed = _floor_to_completed_minute(datetime.now(tz=UTC))
  end = cli_end or now_completed

  conn = connect_db(args.db_path)
  ensure_schema(conn)

  exchange = create_exchange_client(args.exchange)
  try:
    earliest = discover_earliest_binance_1m(exchange, symbol=market_symbol)
  except Exception as exc:
    raise RuntimeError(
      f"Failed to query {args.exchange} for {market_symbol}. "
      "If your region is blocked on this venue, try --exchange binanceus "
      "and --market-symbol BTC/USDT."
    ) from exc
  raw_max = conn.execute(
    "SELECT MAX(open_time_ms) FROM btc_klines_raw WHERE symbol = ?",
    [args.symbol],
  ).fetchone()[0]

  start, end = _ingest_window(args.mode, earliest, raw_max, cli_start, end)
  start_ms = int(start.timestamp() * 1000)
  end_ms = int(end.timestamp() * 1000)
  if end_ms <= start_ms:
    print("No ingest needed: end <= start.")
  else:
    rows = fetch_klines_paginated(
      exchange=exchange,
      symbol=market_symbol,
      start_ms=start_ms,
      end_ms=end_ms,
      limit=args.batch_limit,
      sleep_ms=args.sleep_ms,
    )
    stats = upsert_raw_klines(conn, rows, symbol=args.symbol, exchange=args.exchange)
    print(
      "Ingest complete:",
      {
        "fetched_rows": stats.fetched_rows,
        "inserted_rows": stats.inserted_rows,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
      },
    )

  do_rebuild = args.rebuild_canonical or args.mode in {"backfill", "full"}
  if do_rebuild:
    cstats = materialize_canonical_1m(
      conn, symbol=args.symbol, source_exchange=args.exchange
    )
    print("Canonical materialization:", cstats)

  if args.verbose:
    summary = conn.execute(
      """
      SELECT
        COUNT(*) AS raw_rows,
        MIN(open_ts_utc) AS raw_min_ts,
        MAX(open_ts_utc) AS raw_max_ts
      FROM btc_klines_raw
      WHERE symbol = ?
      """,
      [args.symbol],
    ).fetchone()
    canon = conn.execute(
      """
      SELECT
        COUNT(*) AS canonical_rows,
        SUM(CASE WHEN is_missing THEN 1 ELSE 0 END) AS missing_rows,
        MIN(timestamp_utc) AS canon_min_ts,
        MAX(timestamp_utc) AS canon_max_ts
      FROM btc_1m_canonical
      WHERE symbol = ?
      """,
      [args.symbol],
    ).fetchone()
    print(
      "Summary:",
      {
        "raw_rows": summary[0],
        "raw_min_ts": str(summary[1]),
        "raw_max_ts": str(summary[2]),
        "canonical_rows": canon[0],
        "missing_rows": canon[1],
        "canonical_min_ts": str(canon[2]),
        "canonical_max_ts": str(canon[3]),
      },
    )


if __name__ == "__main__":
  main()
