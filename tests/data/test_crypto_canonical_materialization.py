from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from timesfm.data.crypto_duckdb import (
  connect_db,
  ensure_schema,
  get_canonical_series,
  materialize_canonical_1m,
  upsert_raw_klines,
)


def test_materialize_canonical_preserves_missing_minutes(tmp_path: Path) -> None:
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
    },
    {
      "open_time_ms": 1700000120000,
      "open": 1.4,
      "high": 1.6,
      "low": 1.1,
      "close": 1.2,
      "volume": 11.0,
    },
  ]
  upsert_raw_klines(conn, rows)

  stats = materialize_canonical_1m(conn)
  assert stats["materialized_rows"] == 3

  df = get_canonical_series(conn)
  assert len(df) == 3
  assert int(df["is_missing"].sum()) == 1
  assert df["close"].isna().sum() == 1
