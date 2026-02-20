from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from timesfm.data.crypto_duckdb import connect_db, ensure_schema, upsert_raw_klines


def test_schema_creation_and_insert_ignore(tmp_path: Path) -> None:
  db_path = tmp_path / "btc.duckdb"
  conn = connect_db(str(db_path))
  ensure_schema(conn)

  rows = [
    {
      "open_time_ms": 1700000000000,
      "open": 1.0,
      "high": 2.0,
      "low": 0.5,
      "close": 1.5,
      "volume": 10.0,
      "close_time_ms": 1700000059999,
      "quote_volume": 15.0,
      "trade_count": 1,
      "taker_buy_base": 5.0,
      "taker_buy_quote": 7.5,
    }
  ]

  s1 = upsert_raw_klines(conn, rows)
  s2 = upsert_raw_klines(conn, rows)

  assert s1.fetched_rows == 1
  assert s1.inserted_rows == 1
  assert s2.fetched_rows == 1
  assert s2.inserted_rows == 0

  n = conn.execute("SELECT COUNT(*) FROM btc_klines_raw").fetchone()[0]
  assert n == 1
