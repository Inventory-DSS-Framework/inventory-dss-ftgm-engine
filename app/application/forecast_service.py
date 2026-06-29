"""Forecast orchestration.

Glues the pure FTGM core to the HTTP contract. For each product in the batch it runs the
full pipeline: aggregate -> repair stock-outs -> select Fourier order -> fit -> forecast
-> metrics. If a series is too short or the model diverges, it degrades gracefully to the
seasonal-naive baseline instead of failing the whole request.
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
    ProductForecast,
    ProductSeries,
)

FloatArray = NDArray[np.float64]

# Sentinel order reported when we fall back to the baseline forecaster.
_BASELINE_ORDER = 0
_ROUND_METRIC = 4
_ROUND_VALUE = 3


class ForecastService:
    """Produces forecasts for a batch of product series."""

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        config = FTGMConfig(period=request.period)
        forecasts = [
            self._forecast_product(series, request.horizon, request.model, config)
            for series in request.series
        ]
        return ForecastResponse(period=request.period, forecasts=forecasts)

    # ------------------------------------------------------------------ per product
    def _forecast_product(
        self, series: ProductSeries, horizon: int, model_name: str, config: FTGMConfig
    ) -> ProductForecast:
        dates = [p.date for p in series.points]
        demand = np.array([p.demand for p in series.points], dtype=np.float64)
        stockout = np.array([p.stockout_flag for p in series.points], dtype=bool)

        # 1. Aggregate to seasonal buckets and 2. repair censored stock-out demand.
        bucket_dates, agg_demand, agg_flags = prep.aggregate(dates, demand, stockout, config.period)
        clean = prep.impute_stockouts(agg_demand, agg_flags)

        future_dates = (
            prep.future_period_dates(bucket_dates[-1], horizon, config.period)
            if bucket_dates
            else []
        )

        try:
            return self._forecast_with_ftgm(
                series.product_id, model_name, clean, horizon, future_dates, config
            )
        except FTGMError:
            # Series too short / unstable -> transparent baseline fallback.
            return self._forecast_with_baseline(
                series.product_id, clean, horizon, future_dates, config
            )

    def _forecast_with_ftgm(
        self,
        product_id: str,
        model_name: str,
        clean: FloatArray,
        horizon: int,
        future_dates: list[date],
        config: FTGMConfig,
    ) -> ProductForecast:
        selection = select_order(clean, config)
        model = FTGM(order=selection.best_order, config=config).fit(clean)
        result = model.predict(horizon)

        assert model.fitted_ is not None
        metrics = self._in_sample_metrics(clean, model.fitted_, config.period)
        points = self._build_points(future_dates, result.point, result.lower, result.upper)
        return ProductForecast(
            product_id=product_id,
            model=model_name,
            order_selected=selection.best_order,
            points=points,
            metrics=metrics,
        )

    def _forecast_with_baseline(
        self,
        product_id: str,
        clean: FloatArray,
        horizon: int,
        future_dates: list[date],
        config: FTGMConfig,
    ) -> ProductForecast:
        point = seasonal_naive_forecast(clean, horizon, config.period)
        # In-sample one-step seasonal-naive fit, for comparable metrics.
        season = config.period if clean.size > config.period else 1
        fitted = np.concatenate([clean[:season], clean[:-season]]) if clean.size > season else clean
        metrics = self._in_sample_metrics(clean, fitted, config.period)
        points = self._build_points(future_dates, point, point, point)
        return ProductForecast(
            product_id=product_id,
            model="SeasonalNaive",
            order_selected=_BASELINE_ORDER,
            points=points,
            metrics=metrics,
        )

    # ------------------------------------------------------------------- helpers
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
