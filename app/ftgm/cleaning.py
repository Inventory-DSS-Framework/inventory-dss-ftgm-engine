"""Robust series diagnostics: outlier treatment, seasonality strength and trend.

These helpers run *after* aggregation and censored-demand repair, before the FTGM sees
the data:

* :func:`hampel_clean` — a Hampel filter on the **deseasonalised** series, so a genuine
  seasonal peak (e.g. December) is not mistaken for an outlier, while one-off spikes
  (a bulk order, a data-entry error) are replaced by the local robust level.
* :func:`seasonality_strength` — Hyndman's ``F_s = max(0, 1 - Var(R) / Var(S + R))``
  from a classical additive decomposition (needs two full seasons).
* :func:`trend_per_period` — robust linear slope of the deseasonalised series, as a
  share of the mean level per period.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MAD_SCALE = 1.4826  # MAD -> sigma under normality
_MIN_POINTS = 8
#: A point must also deviate by this share of the local level to count as an outlier
#: (avoids flagging tiny wiggles when the local MAD is almost zero).
_MIN_REL_DEVIATION = 0.35


def _centered_moving_average(x: FloatArray, period: int) -> FloatArray:
    """Centered moving average of length ``period`` (2xMA for even periods); NaN at edges."""
    n = x.size
    out = np.full(n, np.nan)
    if period < 2 or n < period + 1:
        return out
    if period % 2 == 1:
        kernel = np.ones(period) / period
        half = period // 2
    else:
        kernel = np.concatenate([[0.5], np.ones(period - 1), [0.5]]) / period
        half = period // 2
    conv = np.convolve(x, kernel, mode="valid")
    out[half : half + conv.size] = conv
    return out


def seasonal_component(x: FloatArray, period: int) -> FloatArray | None:
    """Additive seasonal profile (sums to ~0 over a season) or None if < 2 seasons."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if period < 2 or x.size < 2 * period:
        return None
    trend = _centered_moving_average(x, period)
    detrended = x - trend
    profile = np.zeros(period)
    for pos in range(period):
        vals = detrended[pos::period]
        vals = vals[np.isfinite(vals)]
        profile[pos] = float(np.median(vals)) if vals.size else 0.0
    profile -= profile.mean()
    reps = int(np.ceil(x.size / period))
    return np.tile(profile, reps)[: x.size]


def seasonality_strength(x: FloatArray, period: int) -> float | None:
    """Hyndman's seasonal strength in [0, 1]; None when the history is too short."""
    x = np.asarray(x, dtype=np.float64).ravel()
    season = seasonal_component(x, period)
    if season is None:
        return None
    trend = _centered_moving_average(x, period)
    valid = np.isfinite(trend)
    if valid.sum() < period:
        return None
    detrended = (x - trend)[valid]
    remainder = detrended - season[valid]
    var_sr = float(np.var(detrended))
    if var_sr <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - np.var(remainder) / var_sr)))


def hampel_clean(
    x: FloatArray,
    period: int,
    protected: BoolArray | None = None,
    half_window: int | None = None,
    n_sigmas: float = 3.0,
) -> tuple[FloatArray, BoolArray]:
    """Replace outliers of the deseasonalised series by the local rolling median.

    ``protected`` buckets (e.g. repaired stock-outs) are never modified. Returns the
    cleaned series (non-negative) and the outlier mask.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    mask = np.zeros(n, dtype=bool)
    if n < _MIN_POINTS:
        return x.copy(), mask
    k = half_window if half_window is not None else (4 if period >= 52 else 3)
    season = seasonal_component(x, period)
    base = x - season if season is not None else x.copy()

    cleaned = x.copy()
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        window = np.delete(base[lo:hi], i - lo)  # exclude the point itself
        if window.size < 3:
            continue
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med))) * _MAD_SCALE
        deviation = abs(base[i] - med)
        scale = max(mad, 1e-9)
        level = max(abs(med), 1e-9)
        if deviation > n_sigmas * scale and deviation > _MIN_REL_DEVIATION * level:
            if protected is not None and protected[i]:
                continue
            mask[i] = True
            cleaned[i] = med + (season[i] if season is not None else 0.0)
    return np.clip(cleaned, 0.0, None), mask


def trend_per_period(x: FloatArray, period: int) -> float | None:
    """Least-squares slope of the (deseasonalised) recent history, as % of the mean."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return None
    season = seasonal_component(x, period)
    base = x - season if season is not None else x
    window = base[-min(base.size, 2 * period if period > 1 else base.size) :]
    mean = float(np.mean(window))
    if mean <= 1e-9:
        return None
    t = np.arange(window.size, dtype=np.float64)
    slope = float(np.polyfit(t, window, 1)[0])
    return 100.0 * slope / mean
