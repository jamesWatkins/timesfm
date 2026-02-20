from __future__ import annotations

from datetime import UTC, datetime

from timesfm.data.crypto_duckdb import (
  discover_earliest_binance_1m,
  fetch_klines_paginated,
)


class _FakeExchange:
  def __init__(self) -> None:
    self.calls = 0

  def fetch_ohlcv(self, symbol, timeframe, since, limit):
    assert symbol == "BTCUSDT"
    assert timeframe == "1m"
    self.calls += 1

    if since <= 0:
      return [[60_000, 1.0, 1.0, 1.0, 1.0, 10.0, 119_999, 10.0, 1, 5.0, 5.0]]

    if since == 60_000:
      return [
        [60_000, 1.0, 1.0, 1.0, 1.0, 10.0, 119_999, 10.0, 1, 5.0, 5.0],
        [120_000, 1.1, 1.2, 1.0, 1.15, 11.0, 179_999, 12.0, 2, 6.0, 6.0],
      ]
    if since == 180_000:
      return [[180_000, 1.2, 1.3, 1.1, 1.25, 9.0, 239_999, 9.0, 3, 4.5, 4.5]]
    return []


def test_discover_earliest() -> None:
  exchange = _FakeExchange()
  earliest = discover_earliest_binance_1m(exchange)
  assert earliest == datetime.fromtimestamp(60_000 / 1000.0, tz=UTC)


def test_fetch_paginated_respects_end_exclusive() -> None:
  exchange = _FakeExchange()
  rows = fetch_klines_paginated(
    exchange=exchange,
    symbol="BTCUSDT",
    start_ms=60_000,
    end_ms=180_000,
    limit=1000,
    sleep_ms=0,
  )
  assert [r["open_time_ms"] for r in rows] == [60_000, 120_000]
