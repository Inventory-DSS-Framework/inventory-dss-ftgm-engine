"""Forecast orchestration.

Glues the pure FTGM core to the HTTP contract. For each product in the batch it runs:

    1. frequency   — legacy ``period``, or auto: monthly (>= 24 months), weekly
                     (>= 26 weeks), else a short-history baseline;
    2. bucketize   — contiguous buckets up to the last *complete* period (``as_of``);
    3. censoring   — repair demand in stock-out periods (scale up / interpolate);
    4. intermittency — > 50% zero periods -> Croston-SBA instead of the FTGM;
    5. outliers    — Hampel filter on the deseasonalised series;
    6. FTGM        — Fourier order chosen by Algorithm 1 inside a history-length cap,
                     fit, forecast;
    7. validation  — rolling-origin hold-out vs seasonal naive (honest accuracy, guard
                     against a clearly worse model, empirical prediction interval);
    8. diagnostics — every decision explained in plain Spanish.

Failure isolation is per product: too short / unstable -> transparent baseline with the
reason in ``fallback_reason``; empty -> ``status="skipped"``. One bad product never fails
the batch. Forecasts are always finite and non-negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from app.baselines import seasonal_naive_forecast
from app.baselines.intermittent import croston_sba, moving_average_forecast
from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order
from app.ftgm import cleaning
from app.ftgm import metrics as M
from app.ftgm import preprocessing as prep
from app.ftgm.validation import HoldoutScore, rolling_origin
from app.presentation.schemas import (
    ForecastMetrics,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    HoldoutMetrics,
    ProductDiagnostics,
    ProductForecast,
    ProductSeries,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_BASELINE_ORDER = 0
_ROUND_METRIC = 4
_ROUND_VALUE = 3
#: Prediction interval level reported to clients (z for a two-sided 90% band).
_INTERVAL_LEVEL = 0.90
_INTERVAL_Z = 1.645
#: Auto-frequency thresholds.
_MIN_MONTHS_FOR_MONTHLY = 24
_MIN_WEEKS_FOR_WEEKLY = 26
#: Share of zero periods above which demand is treated as intermittent.
_INTERMITTENT_ZERO_SHARE = 0.5
#: Practical caps on the Fourier order per seasonal period (Nyquist allows more, but
#: high harmonics on short retail series overfit noise).
_FREQ_ORDER_CAP = {12: 5, 52: 6, 4: 1}
#: Guard: abandon the FTGM only when it is clearly worse than seasonal naive on the
#: hold-out AND inaccurate in absolute terms.
_GUARD_RATIO = 1.5
_GUARD_REL_RMSE = 0.25

_FREQ_LABEL = {12: "mensual", 52: "semanal", 4: "trimestral"}
_PERIOD_WORD = {12: ("mes", "meses"), 52: ("semana", "semanas"), 4: ("trimestre", "trimestres")}


def _r(value: float | None, digits: int = _ROUND_METRIC) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _plural(n: int, period: int) -> str:
    one, many = _PERIOD_WORD.get(period, ("periodo", "periodos"))
    return f"{n} {one if n == 1 else many}"


@dataclass
class _Plan:
    period: int
    mode: str  # "ftgm" | "short"
    reason: str | None
    series: prep.BucketedSeries


class ForecastService:
    """Produces forecasts for a batch of product series."""

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        forecasts = [self._forecast_product(series, request) for series in request.series]
        periods = {f.period for f in forecasts if f.period}
        response_period = request.period if request.frequency is None else (
            periods.pop() if len(periods) == 1 else request.period
        )
        return ForecastResponse(period=response_period, forecasts=forecasts)

    # ------------------------------------------------------------------ planning
    @staticmethod
    def _plan(series: ProductSeries, request: ForecastRequest) -> _Plan:
        dates = [p.date for p in series.points]
        demand = np.array([p.demand for p in series.points], dtype=np.float64)
        flags = np.array([p.stockout_flag for p in series.points], dtype=bool)

        def bucket(period: int) -> prep.BucketedSeries:
            return prep.bucketize(dates, demand, flags, period, request.as_of)

        freq = request.frequency
        if freq is None:
            return _Plan(request.period, "ftgm", None, bucket(request.period))
        if freq in prep.FREQUENCY_PERIOD:
            period = prep.FREQUENCY_PERIOD[freq]
            return _Plan(period, "ftgm", f"Frecuencia {_FREQ_LABEL[period]} elegida por el usuario.", bucket(period))

        monthly = bucket(12)
        if monthly.size >= _MIN_MONTHS_FOR_MONTHLY:
            return _Plan(
                12, "ftgm",
                f"Automática: {monthly.size} meses completos de historia (≥ {_MIN_MONTHS_FOR_MONTHLY}) → mensual.",
                monthly,
            )
        weekly = bucket(52)
        if weekly.size >= _MIN_WEEKS_FOR_WEEKLY:
            return _Plan(
                52, "ftgm",
                f"Automática: {monthly.size} meses (< {_MIN_MONTHS_FOR_MONTHLY}) pero {weekly.size} semanas "
                f"completas (≥ {_MIN_WEEKS_FOR_WEEKLY}) → semanal.",
                weekly,
            )
        return _Plan(
            52, "short",
            f"Automática: solo {weekly.size} semanas completas de historia (< {_MIN_WEEKS_FOR_WEEKLY}); "
            "no alcanza para identificar estacionalidad → promedio móvil semanal.",
            weekly,
        )

    @staticmethod
    def _horizon_periods(request: ForecastRequest, period: int) -> int:
        if request.horizon_days is None or request.frequency is None:
            return int(request.horizon or 1)
        days = prep.AVG_DAYS_PER_PERIOD.get(period, 30.4375)
        return int(max(1, min(104, math.ceil(request.horizon_days / days - 1e-6))))

    # ------------------------------------------------------------------ per product
    def _forecast_product(self, series: ProductSeries, request: ForecastRequest) -> ProductForecast:
        try:
            plan = self._plan(series, request)
        except Exception as exc:  # defensive
            return self._skipped(series, reason=f"Error inesperado: {exc}")
        period = plan.period
        buckets = plan.series
        if buckets.size == 0:
            reason = (
                "Solo hay ventas en el periodo en curso; aún no hay periodos completos."
                if series.points
                else "La serie no contiene observaciones utilizables."
            )
            return self._skipped(series, reason=reason)

        observed = buckets.demand
        repaired, flags, imputed = prep.repair_censored(observed, buckets.stockout_share)
        horizon = self._horizon_periods(request, period)
        future_dates = prep.future_period_dates(buckets.dates[-1], horizon, period)
        zero_share = float(np.mean(repaired <= 0.0))

        diag = ProductDiagnostics(
            n_input_points=len(series.points),
            n_periods=int(repaired.size),
            stockout_periods=int(flags.sum()),
            imputed_periods=int(imputed.sum()),
            frequency=prep.PERIOD_FREQUENCY.get(period, str(period)),
            period=period,
            frequency_reason=plan.reason,
            history_start=buckets.dates[0],
            history_end=buckets.dates[-1],
            dropped_incomplete_period=buckets.dropped_incomplete,
            zero_share=_r(zero_share),
            interval_level=_INTERVAL_LEVEL,
        )
        explain = diag.explanations
        if plan.reason:
            explain.append(plan.reason)
        explain.append(
            f"Se usaron {_plural(repaired.size, period)} de historia "
            f"({buckets.dates[0].isoformat()} → {buckets.dates[-1].isoformat()})."
        )
        if buckets.dropped_incomplete:
            explain.append("El periodo en curso se excluyó porque aún no termina (sesgaría la demanda a la baja).")
        if flags.any():
            explain.append(
                f"{_plural(int(flags.sum()), period)} con quiebre de stock: la venta registrada subestima la demanda, "
                f"así que se reconstruyó ({int(imputed.sum())} ajustado(s))."
            )
        warnings = self._series_warnings(repaired, period)

        try:
            # Intermittent demand: the FTGM (a smooth ODE) is the wrong tool.
            if repaired.size >= 4 and zero_share > _INTERMITTENT_ZERO_SHARE:
                diag.intermittent = True
                return self._forecast_intermittent(
                    series.product_id, repaired, observed, flags, buckets, future_dates, period, diag, warnings,
                )

            clean, outliers = cleaning.hampel_clean(repaired, period, protected=flags)
            diag.outliers_cleaned = int(outliers.sum())
            if outliers.any():
                explain.append(
                    f"{_plural(int(outliers.sum()), period)} con valores atípicos (picos o caídas aisladas) "
                    "se suavizaron con un filtro de Hampel sobre la serie desestacionalizada."
                )
            strength = cleaning.seasonality_strength(clean, period)
            diag.seasonality_strength = _r(strength, 3)
            diag.trend_pct_per_period = _r(cleaning.trend_per_period(clean, period), 2)
            self._explain_shape(diag, period)

            if plan.mode == "short":
                return self._forecast_short(
                    series.product_id, clean, observed, flags, outliers, buckets, future_dates, period, diag,
                    warnings,
                )
            return self._forecast_with_ftgm(
                series.product_id, request, clean, observed, flags, outliers, buckets, future_dates, period,
                diag, warnings,
            )
        except Exception as exc:  # defensive: never let one product fail the batch
            return self._skipped(series, reason=f"Error inesperado: {exc}")

    # ------------------------------------------------------------------ FTGM path
    def _forecast_with_ftgm(
        self,
        product_id: str,
        request: ForecastRequest,
        clean: FloatArray,
        observed: FloatArray,
        flags: BoolArray,
        outliers: BoolArray,
        buckets: prep.BucketedSeries,
        future_dates: list[date],
        period: int,
        diag: ProductDiagnostics,
        warnings: list[str],
    ) -> ProductForecast:
        config = FTGMConfig(period=period)
        n = clean.size
        nyquist = max(1, math.ceil(period / 2) - 1)
        season_cap = max(1, int(round(2.0 * n / period)))
        cap = min(nyquist, _FREQ_ORDER_CAP.get(period, nyquist), season_cap)
        if request.max_order is not None:
            cap = min(cap, request.max_order)
        diag.max_order_allowed = cap

        validation_size = request.validation_size
        if validation_size is None and period >= 52:
            validation_size = max(1, min(13, n // 4))

        try:
            selection = select_order(clean, config, validation_size=validation_size, max_order=cap)
            model = FTGM(order=selection.best_order, config=config).fit(clean)
            result = model.predict(len(future_dates))
        except FTGMError as exc:
            diag.explanations.append(
                f"El FTGM no pudo ajustarse de forma estable ({exc}); se usó el baseline estacional."
            )
            return self._forecast_seasonal_naive(
                product_id, clean, observed, flags, outliers, buckets, future_dates, period, diag, warnings,
                reason=str(exc),
            )
        assert model.fitted_ is not None

        diag.validation_size = selection.validation_size
        diag.order_scores = {k: _r(v) for k, v in selection.scores.items()}
        best = selection.scores.get(selection.best_order)
        diag.validation_rmse = _r(best) if best is not None else None
        diag.explanations.append(
            f"Algoritmo 1: se probaron órdenes de Fourier 1…{cap} (tope por largo de historia y Nyquist) "
            f"y se eligió N = {selection.best_order} por menor RMSE en validación."
        )

        # Rolling-origin hold-out: FTGM (fixed order) vs seasonal naive on the same origins.
        h_eval = max(1, min(len(future_dates), 6 if period < 52 else 8))
        n_origins = 3 if period < 52 else 4
        min_train = max(8, 2 + 4 * selection.best_order + 3)

        def ftgm_fc(train: FloatArray, h: int) -> FloatArray:
            return FTGM(order=selection.best_order, config=config).fit(train).predict(h).point

        def naive_fc(train: FloatArray, h: int) -> FloatArray:
            return seasonal_naive_forecast(train, h, period)

        hold = rolling_origin(clean, ftgm_fc, horizon=h_eval, n_origins=n_origins, min_train=min_train, period=period)
        naive = rolling_origin(clean, naive_fc, horizon=h_eval, n_origins=n_origins, min_train=min_train, period=period)
        diag.holdout = self._holdout(hold)
        diag.naive_holdout = self._holdout(naive)
        if hold is not None and naive is not None and naive.rmse > 0:
            diag.skill_vs_naive = _r(1.0 - hold.rmse / naive.rmse, 3)

        mean_level = float(np.mean(clean)) if clean.size else 0.0
        if (
            hold is not None
            and naive is not None
            and naive.rmse > 0
            and hold.rmse > _GUARD_RATIO * naive.rmse
            and mean_level > 0
            and hold.rmse / mean_level > _GUARD_REL_RMSE
        ):
            reason = (
                f"En validación el FTGM (RMSE {hold.rmse:.1f}) fue claramente peor que el baseline estacional "
                f"(RMSE {naive.rmse:.1f}); se usa el baseline para no arriesgar la reposición."
            )
            diag.explanations.append(reason)
            return self._forecast_seasonal_naive(
                product_id, clean, observed, flags, outliers, buckets, future_dates, period, diag, warnings,
                reason=reason,
            )

        if hold is not None:
            mape_txt = f"MAPE {hold.mape:.1f}%" if hold.mape is not None else f"RMSE {hold.rmse:.1f}"
            diag.explanations.append(
                f"Validación rolling-origin ({hold.origins} orígenes, {_plural(hold.horizon, period)} adelante): "
                f"{mape_txt}"
                + (
                    f"; {abs(diag.skill_vs_naive) * 100:.0f}% {'mejor' if diag.skill_vs_naive >= 0 else 'peor'} "
                    "que el baseline estacional."
                    if diag.skill_vs_naive is not None
                    else "."
                )
            )

        residuals = clean - model.fitted_
        point = np.clip(result.point, 0.0, None)
        lower, upper = self._interval(point, residuals, hold)
        self._explain_forecast(diag, point, clean, period)
        return ProductForecast(
            product_id=product_id,
            model=request.model,
            order_selected=selection.best_order,
            status="ok",
            warnings=warnings,
            frequency=diag.frequency,
            period=period,
            points=self._build_points(future_dates, point, lower, upper),
            history=self._build_history(buckets, observed, clean, model.fitted_, flags, outliers),
            metrics=self._in_sample_metrics(clean, model.fitted_, period),
            diagnostics=diag,
        )

    # ------------------------------------------------------------------ baselines
    def _forecast_seasonal_naive(
        self,
        product_id: str,
        clean: FloatArray,
        observed: FloatArray,
        flags: BoolArray,
        outliers: BoolArray,
        buckets: prep.BucketedSeries,
        future_dates: list[date],
        period: int,
        diag: ProductDiagnostics,
        warnings: list[str],
        *,
        reason: str,
    ) -> ProductForecast:
        point = seasonal_naive_forecast(clean, len(future_dates), period)
        season = period if clean.size > period else 1
        fitted = np.concatenate([clean[:season], clean[:-season]]) if clean.size > season else clean.copy()
        lower, upper = self._interval(point, clean - fitted, None)
        self._explain_forecast(diag, point, clean, period)
        return ProductForecast(
            product_id=product_id,
            model="SeasonalNaive",
            order_selected=_BASELINE_ORDER,
            status="fallback",
            fallback_reason=reason,
            warnings=warnings,
            frequency=diag.frequency,
            period=period,
            points=self._build_points(future_dates, point, lower, upper),
            history=self._build_history(buckets, observed, clean, fitted, flags, outliers),
            metrics=self._in_sample_metrics(clean, fitted, period),
            diagnostics=diag,
        )

    def _forecast_intermittent(
        self,
        product_id: str,
        repaired: FloatArray,
        observed: FloatArray,
        flags: BoolArray,
        buckets: prep.BucketedSeries,
        future_dates: list[date],
        period: int,
        diag: ProductDiagnostics,
        warnings: list[str],
    ) -> ProductForecast:
        point, fitted = croston_sba(repaired, len(future_dates))
        residuals = repaired - fitted
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        lower = np.clip(point - _INTERVAL_Z * sigma, 0.0, None)
        upper = point + _INTERVAL_Z * sigma
        reason = (
            f"Demanda intermitente: {diag.zero_share * 100:.0f}% de los periodos sin ventas (> 50%). "
            "Se usa Croston-SBA, que pronostica una tasa de demanda estable y es más adecuado que el FTGM "
            "para ventas esporádicas."
            if diag.zero_share is not None
            else "Demanda intermitente: se usa Croston-SBA."
        )
        diag.explanations.append(reason)
        self._explain_forecast(diag, point, repaired, period)
        return ProductForecast(
            product_id=product_id,
            model="CrostonSBA",
            order_selected=_BASELINE_ORDER,
            status="fallback",
            fallback_reason=reason,
            warnings=warnings,
            frequency=diag.frequency,
            period=period,
            points=self._build_points(future_dates, point, lower, upper),
            history=self._build_history(buckets, observed, repaired, fitted, flags, np.zeros_like(flags)),
            metrics=self._in_sample_metrics(repaired, fitted, period),
            diagnostics=diag,
        )

    def _forecast_short(
        self,
        product_id: str,
        clean: FloatArray,
        observed: FloatArray,
        flags: BoolArray,
        outliers: BoolArray,
        buckets: prep.BucketedSeries,
        future_dates: list[date],
        period: int,
        diag: ProductDiagnostics,
        warnings: list[str],
    ) -> ProductForecast:
        window = 8 if period >= 52 else 3
        point, fitted = moving_average_forecast(clean, len(future_dates), window)
        lower, upper = self._interval(point, clean - fitted, None)
        reason = diag.frequency_reason or "Historia corta: se usa un promedio móvil."
        self._explain_forecast(diag, point, clean, period)
        return ProductForecast(
            product_id=product_id,
            model="MovingAverage",
            order_selected=_BASELINE_ORDER,
            status="fallback",
            fallback_reason=reason,
            warnings=warnings,
            frequency=diag.frequency,
            period=period,
            points=self._build_points(future_dates, point, lower, upper),
            history=self._build_history(buckets, observed, clean, fitted, flags, outliers),
            metrics=self._in_sample_metrics(clean, fitted, period),
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
                n_input_points=len(series.points), n_periods=0, explanations=[reason]
            ),
        )

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _holdout(score: HoldoutScore | None) -> HoldoutMetrics | None:
        if score is None:
            return None
        return HoldoutMetrics(
            origins=score.origins,
            horizon=score.horizon,
            mae=_r(score.mae),
            rmse=_r(score.rmse),
            mape=_r(score.mape),
            mase=_r(score.mase),
        )

    @staticmethod
    def _interval(
        point: FloatArray, residuals: FloatArray, hold: HoldoutScore | None
    ) -> tuple[FloatArray, FloatArray]:
        """Empirical band: max(in-sample sigma·sqrt(h), out-of-sample RMSE at step h)."""
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        steps = np.arange(1, point.size + 1, dtype=np.float64)
        spread = sigma * np.sqrt(steps)
        if hold is not None and hold.rmse_by_step:
            by_step = np.array(hold.rmse_by_step, dtype=np.float64)
            k = by_step.size
            oos = np.array(
                [by_step[s - 1] if s <= k else by_step[-1] * math.sqrt(s / k) for s in range(1, point.size + 1)]
            )
            spread = np.maximum(spread, oos)
        half = _INTERVAL_Z * spread
        return np.clip(point - half, 0.0, None), point + half

    @staticmethod
    def _explain_shape(diag: ProductDiagnostics, period: int) -> None:
        s = diag.seasonality_strength
        if s is None:
            diag.explanations.append(
                "Aún no hay dos temporadas completas para medir la estacionalidad con confianza."
            )
        else:
            label = "fuerte" if s >= 0.6 else "moderada" if s >= 0.3 else "débil"
            diag.explanations.append(f"Estacionalidad {label} (fuerza {s:.2f} de 1).")
        t = diag.trend_pct_per_period
        if t is not None:
            unit = _PERIOD_WORD.get(period, ("periodo", ""))[0]
            if abs(t) < 0.5:
                diag.explanations.append("Tendencia estable en la historia reciente.")
            else:
                word = "creciente" if t > 0 else "decreciente"
                diag.explanations.append(f"Tendencia {word}: {t:+.1f}% por {unit}.")

    @staticmethod
    def _explain_forecast(diag: ProductDiagnostics, point: FloatArray, history: FloatArray, period: int) -> None:
        h = point.size
        recent = history[-h:] if history.size >= h else history
        if recent.size and float(np.sum(recent)) > 0:
            scaled = float(np.sum(recent)) * (h / recent.size)
            pct = 100.0 * (float(np.sum(point)) - scaled) / scaled
            diag.forecast_vs_recent_pct = _r(pct, 1)
            if abs(pct) >= 3:
                diag.explanations.append(
                    f"Se proyectan {float(np.sum(point)):.0f} unidades en {_plural(h, period)}, "
                    f"{abs(pct):.0f}% {'más' if pct > 0 else 'menos'} que los últimos {_plural(h, period)}."
                )
            else:
                diag.explanations.append(
                    f"Se proyectan {float(np.sum(point)):.0f} unidades en {_plural(h, period)}, "
                    "en línea con el nivel reciente."
                )

    @staticmethod
    def _series_warnings(clean: FloatArray, period: int) -> list[str]:
        warnings: list[str] = []
        if clean.size < 2 * period:
            warnings.append(
                f"La serie tiene {clean.size} periodos (<2 estaciones de {period}); "
                "la selección del orden de Fourier es menos fiable."
            )
        if clean.size and float(np.count_nonzero(clean)) / clean.size < 0.5:
            warnings.append(
                "Más de la mitad de los periodos tienen demanda cero (demanda intermitente); "
                "el MAPE puede no estar definido."
            )
        return warnings

    @staticmethod
    def _build_points(
        dates: list[date], point: FloatArray, lower: FloatArray, upper: FloatArray
    ) -> list[ForecastPoint]:
        out: list[ForecastPoint] = []
        for d, p, lo, hi in zip(dates, point, lower, upper):
            p_val = max(0.0, float(p)) if math.isfinite(float(p)) else 0.0
            lo_val = max(0.0, min(float(lo), p_val)) if math.isfinite(float(lo)) else 0.0
            hi_val = max(float(hi), p_val) if math.isfinite(float(hi)) else p_val
            out.append(
                ForecastPoint(
                    date=d,
                    predicted_demand=round(p_val, _ROUND_VALUE),
                    lower_bound=round(lo_val, _ROUND_VALUE),
                    upper_bound=round(hi_val, _ROUND_VALUE),
                )
            )
        return out

    @staticmethod
    def _build_history(
        buckets: prep.BucketedSeries,
        observed: FloatArray,
        cleaned: FloatArray,
        fitted: FloatArray,
        flags: BoolArray,
        outliers: BoolArray,
    ) -> list[HistoryPoint]:
        return [
            HistoryPoint(
                date=d,
                observed=round(float(o), _ROUND_VALUE),
                cleaned=round(float(c), _ROUND_VALUE),
                fitted=round(float(f), _ROUND_VALUE) if math.isfinite(float(f)) else None,
                is_stockout=bool(s),
                is_outlier=bool(out),
                stockout_share=round(float(sh), 3),
            )
            for d, o, c, f, s, out, sh in zip(
                buckets.dates, observed, cleaned, fitted, flags, outliers, buckets.stockout_share
            )
        ]

    @staticmethod
    def _in_sample_metrics(actual: FloatArray, fitted: FloatArray, period: int) -> ForecastMetrics:
        if actual.size == 0:
            return ForecastMetrics()
        return ForecastMetrics(
            mae=_r(M.mean_absolute_error(actual, fitted)),
            rmse=_r(M.root_mean_squared_error(actual, fitted)),
            mape=_r(M.mean_absolute_percentage_error(actual, fitted)),
            mase=_r(M.mean_absolute_scaled_error(actual, fitted, actual, period)),
            rmsse=_r(M.root_mean_squared_scaled_error(actual, fitted, actual, period)),
        )
