"""Series preparation before the model sees the data.

The FTGM expects an **equally-spaced, aggregated** series (monthly by default). Raw
retail data arrives daily and noisy, so this module:

* aggregates daily observations into seasonal buckets (month / quarter / week) and
  fills calendar gaps with zeros, producing a contiguous, equally-spaced series; and
* repairs stock-out periods, where an observed zero/low value is *censored demand*
  (we could not sell what we did not have), by interpolating from neighbouring periods.

Both steps matter for forecast quality: gaps break the equal-spacing assumption, and
treating stock-out zeros as genuine demand biases the seasonal shape downwards.

Only the seasonal periods 12 (monthly), 4 (quarterly) and 52 (weekly) trigger calendar
aggregation; any other period assumes the caller already supplies one value per period.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Seasonal period -> "is this a calendar frequency we know how to bucket?"
_CALENDAR_PERIODS = {12, 4, 52}


def _bucket_start(d: date, period: int) -> date:
    """First day of the seasonal bucket that ``d`` falls into."""
    if period == 12:
        return date(d.year, d.month, 1)
    if period == 4:
        quarter_first_month = ((d.month - 1) // 3) * 3 + 1
        return date(d.year, quarter_first_month, 1)
    if period == 52:
        return d - timedelta(days=d.weekday())  # Monday of that week
    return d


def _next_bucket(d: date, period: int) -> date:
    """Start date of the bucket immediately after the one starting at ``d``."""
    if period == 12:
        return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
    if period == 4:
        month = d.month + 3
        year, month = (d.year + (month - 1) // 12, (month - 1) % 12 + 1)
        return date(year, month, 1)
    if period == 52:
        return d + timedelta(days=7)
    return _add_months(d, 1)  # safe fallback for non-calendar periods


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year, month = d.year + month // 12, month % 12 + 1
    return date(year, month, 1)


def aggregate(
    dates: list[date], demand: FloatArray, stockout: BoolArray, period: int
) -> tuple[list[date], FloatArray, BoolArray]:
    """Aggregate observations into contiguous seasonal buckets.

    Returns ``(bucket_starts, summed_demand, stockout_flags)`` sorted in time, with
    calendar gaps filled by zero-demand periods. A bucket is flagged as a stock-out if
    *any* observation inside it was flagged.
    """
    demand = np.asarray(demand, dtype=np.float64).ravel()
    stockout = np.asarray(stockout, dtype=bool).ravel()
    if not (len(dates) == demand.size == stockout.size):
        raise ValueError("dates, demand and stockout must have the same length")
    if demand.size == 0:
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    # Sum demand (and OR the stock-out flag) within each bucket.
    sums: dict[date, float] = {}
    flags: dict[date, bool] = {}
    for d, value, flag in zip(dates, demand, stockout):
        key = _bucket_start(d, period)
        sums[key] = sums.get(key, 0.0) + float(value)
        flags[key] = flags.get(key, False) or bool(flag)

    ordered = sorted(sums)
    if period not in _CALENDAR_PERIODS:
        # Already one value per period: no calendar gap filling possible.
        return ordered, np.array([sums[k] for k in ordered]), np.array([flags[k] for k in ordered])

    # Walk the calendar from first to last bucket, inserting zeros for missing periods.
    out_dates: list[date] = []
    out_vals: list[float] = []
    out_flags: list[bool] = []
    cursor, last = ordered[0], ordered[-1]
    while cursor <= last:
        out_dates.append(cursor)
        out_vals.append(sums.get(cursor, 0.0))
        out_flags.append(flags.get(cursor, False))
        cursor = _next_bucket(cursor, period)
    return out_dates, np.array(out_vals, dtype=np.float64), np.array(out_flags, dtype=bool)


def impute_stockouts(demand: FloatArray, stockout: BoolArray) -> FloatArray:
    """Replace stock-out periods with linearly interpolated demand.

    A stock-out period carries censored demand, so its recorded value is unreliable. We
    interpolate it from the nearest non-stock-out neighbours (clamped at the edges). If
    fewer than two clean periods exist, the series is returned unchanged.
    """
    demand = np.asarray(demand, dtype=np.float64).ravel()
    stockout = np.asarray(stockout, dtype=bool).ravel()
    clean = ~stockout
    if clean.sum() < 2 or stockout.sum() == 0:
        return demand.copy()

    idx = np.arange(demand.size, dtype=np.float64)
    repaired = demand.copy()
    repaired[stockout] = np.interp(idx[stockout], idx[clean], demand[clean])
    return repaired


def future_period_dates(last_start: date, horizon: int, period: int) -> list[date]:
    """Bucket start dates for the ``horizon`` periods following ``last_start``."""
    out: list[date] = []
    cursor = last_start
    for _ in range(horizon):
        cursor = _next_bucket(cursor, period)
        out.append(cursor)
    return out
