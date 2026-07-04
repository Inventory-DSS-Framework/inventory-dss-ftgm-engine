"""Forecast orchestration.

Glues the pure FTGM core to the HTTP contract. For each product in the batch it runs the
full pipeline: aggregate -> repair stock-outs -> select Fourier order -> fit -> forecast
-> metrics. Failure isolation is per product:

* series too short / unstable model  -> transparent **seasonal-naive fallback** with the
  reason recorded in ``fallback_reason``;
* empty / unusable series            -> ``status="skipped"`` entry with a warning —
  one bad product can never fail the whole batch.

The response also carries the aggregated in-sample ``history`` (observed, cleaned and
fitted values per bucket) plus ``diagnostics`` (Algorithm 1 evidence: candidate scores,
validation size), so callers can chart *real vs model* and explain the selection.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
from numpy.typing import NDArray

from app.baselines import seasonal_naive_forecast
from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order
from app.ftgm import metrics as M
from app.ftgm import preprocessing as prep
from app.presentation.schemas import (
    ForecastMetrics,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    ProductDiagnostics,
    ProductForecast,
    ProductSeries,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Sentinel order reported when we fall back to the baseline forecaster.
_BASELINE_ORDER = 0
_ROUND_METRIC = 4
_ROUND_VALUE = 3
# Residual band z-score for the baseline fallback (matches FTGMConfig.interval_z).
_BASELINE_Z = 1.96


class ForecastService:
    """Produces forecasts for a batch of product series."""

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        config = FTGMConfig(period=request.period)
        forecasts = [
            self._forecast_product(series, request, config) for series in request.series
        ]
        return ForecastResponse(period=request.period, forecasts=forecasts)

    # ------------------------------------------------------------------ per product
    def _forecast_product(
        self, series: ProductSeries, request: ForecastRequest, config: FTGMConfig
    ) -> ProductForecast:
        dates = [p.date for p in series.points]
        demand = np.array([p.demand for p in series.points], dtype=np.float64)
        stockout = np.array([p.stockout_flag for p in series.points], dtype=bool)

        # 1. Aggregate to seasonal buckets and 2. repair censored stock-out demand.
        bucket_dates, agg_demand, agg_flags = prep.aggregate(dates, demand, stockout, config.period)
        clean = prep.impute_stockouts(agg_demand, agg_flags)

        if clean.size == 0:
            return self._skipped(
                series, reason="La serie no contiene observaciones utilizables."
            )

        future_dates = prep.future_period_dates(bucket_dates[-1], request.horizon, config.period)
        imputed = int(np.sum(agg_flags & (clean != agg_demand)))
        base_diag = ProductDiagnostics(
            n_input_points=len(series.points),
            n_periods=int(clean.size),
            stockout_periods=int(agg_flags.sum()),
            imputed_periods=imputed,
        )
        warnings = self._series_warnings(clean, config.period)

        try:
            return self._forecast_with_ftgm(
                series.product_id, request, clean, future_dates, config,
                bucket_dates, agg_demand, agg_flags, base_diag, warnings,
            )
        except FTGMError as exc:
            # Series too short / unstable -> transparent baseline fallback.
            return self._forecast_with_baseline(
                series.product_id, request.horizon, clean, future_dates, config,
                bucket_dates, agg_demand, agg_flags, base_diag, warnings, reason=str(exc),
            )
        except Exception as exc:  # defensive: never let one product fail the batch
            return self._skipped(series, reason=f"Error inesperado: {exc}")

    def _forecast_with_ftgm(
        self,
        product_id: str,
        request: ForecastRequest,
        clean: FloatArray,
        future_dates: list[date],
        config: FTGMConfig,
        bucket_dates: list[date],
        observed: FloatArray,
        flags: BoolArray,
        diag: ProductDiagnostics,
        warnings: list[str],
    ) -> ProductForecast:
        selection = select_order(
            clean, config,
            validation_size=request.validation_size,
            max_order=request.max_order,
        )
        model = FTGM(order=selection.best_order, config=config).fit(clean)
        result = model.predict(len(future_dates))

        assert model.fitted_ is not None
        diag.validation_size = selection.validation_size
        diag.order_scores = {
            k: (round(v, _ROUND_METRIC) if math.isfinite(v) else None)
            for k, v in selection.scores.items()
        }
        best_score = selection.scores.get(selection.best_order)
        diag.validation_rmse = (
            round(best_score, _ROUND_METRIC)
            if best_score is not None and math.isfinite(best_score)
            else None
        )

        return ProductForecast(
            product_id=product_id,
            model=request.model,
            order_selected=selection.best_order,
            status="ok",
            warnings=warnings,
            points=self._build_points(future_dates, result.point, result.lower, result.upper),
            history=self._build_history(bucket_dates, observed, clean, model.fitted_, flags),
            metrics=self._in_sample_metrics(clean, model.fitted_, config.period),
            diagnostics=diag,
        )

    def _forecast_with_baseline(
        self,
        product_id: str,
        horizon: int,
        clean: FloatArray,
        future_dates: list[date],
        config: FTGMConfig,
        bucket_dates: list[date],
        observed: FloatArray,
        flags: BoolArray,
        diag: ProductDiagnostics,
        warnings: list[str],
        *,
        reason: str,
    ) -> ProductForecast:
        point = seasonal_naive_forecast(clean, len(future_dates) or horizon, config.period)
        # In-sample one-step seasonal-naive fit, for comparable metrics.
        season = config.period if clean.size > config.period else 1
        fitted = (
            np.concatenate([clean[:season], clean[:-season]]) if clean.size > season else clean.copy()
        )
        # Honest uncertainty: residual band that widens with the horizon (like the FTGM),
        # instead of a zero-width interval pretending certainty.
        residuals = clean - fitted
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        steps = np.arange(1, point.size + 1, dtype=np.float64)
        half = _BASELINE_Z * sigma * np.sqrt(steps)
        lower = np.clip(point - half, 0.0, None)
        upper = point + half

        return ProductForecast(
            product_id=product_id,
            model="SeasonalNaive",
            order_selected=_BASELINE_ORDER,
            status="fallback",
            fallback_reason=reason,
            warnings=warnings,
            points=self._build_points(future_dates, point, lower, upper),
            history=self._build_history(bucket_dates, observed, clean, fitted, flags),
            metrics=self._in_sample_metrics(clean, fitted, config.period),
            diagnostics=diag,
        )

    @staticmethod
    def _skipped(series: ProductSeries, *, reason: str) -> ProductForecast:
        return ProductForecast(
            product_id=series.product_id,
            model="None",
            order_selected=_BASELINE_ORDER,
            status="skipped",
            fallback_reason=reason,
            warnings=[reason],
            points=[],
            history=[],
            metrics=ForecastMetrics(),
            diagnostics=ProductDiagnostics(
                n_input_points=len(series.points), n_periods=0
            ),
        )

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _series_warnings(clean: FloatArray, period: int) -> list[str]:
        warnings: list[str] = []
        if clean.size < 2 * period:
            warnings.append(
                f"La serie tiene {clean.size} periodos (<2 estaciones de {period}); "
                "la selección del orden de Fourier es menos fiable."
            )
        if float(np.count_nonzero(clean)) / clean.size < 0.5:
            warnings.append(
                "Más de la mitad de los periodos tienen demanda cero (demanda intermitente); "
                "el MAPE puede no estar definido."
            )
        return warnings

    @staticmethod
    def _build_points(
        dates: list[date], point: FloatArray, lower: FloatArray, upper: FloatArray
    ) -> list[ForecastPoint]:
        return [
            ForecastPoint(
                date=d,
                predicted_demand=round(float(p), _ROUND_VALUE),
                lower_bound=round(float(lo), _ROUND_VALUE),
                upper_bound=round(float(hi), _ROUND_VALUE),
            )
            for d, p, lo, hi in zip(dates, point, lower, upper)
        ]

    @staticmethod
    def _build_history(
        dates: list[date],
        observed: FloatArray,
        cleaned: FloatArray,
        fitted: FloatArray,
        flags: BoolArray,
    ) -> list[HistoryPoint]:
        return [
            HistoryPoint(
                date=d,
                observed=round(float(o), _ROUND_VALUE),
                cleaned=round(float(c), _ROUND_VALUE),
                fitted=round(float(f), _ROUND_VALUE),
                is_stockout=bool(s),
            )
            for d, o, c, f, s in zip(dates, observed, cleaned, fitted, flags)
        ]

    @staticmethod
    def _in_sample_metrics(actual: FloatArray, fitted: FloatArray, period: int) -> ForecastMetrics:
        def clean(value: float) -> float | None:
            # NaN/inf (e.g. MAPE on all-zero demand) -> null, to keep JSON valid.
            return round(value, _ROUND_METRIC) if math.isfinite(value) else None

        return ForecastMetrics(
            mae=clean(M.mean_absolute_error(actual, fitted)),
            rmse=clean(M.root_mean_squared_error(actual, fitted)),
            mape=clean(M.mean_absolute_percentage_error(actual, fitted)),
            mase=clean(M.mean_absolute_scaled_error(actual, fitted, actual, period)),
            rmsse=clean(M.root_mean_squared_scaled_error(actual, fitted, actual, period)),
        )
