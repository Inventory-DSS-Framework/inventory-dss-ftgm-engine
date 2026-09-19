"""Forecast orchestration.

Glues the pure FTGM core to the HTTP contract. For each product in the batch it runs:

    1. frequency   — legacy ``period``, or auto: monthly (>= 24 months), weekly
                     (>= 26 weeks), else a short-history baseline;
    2. bucketize   — contiguous buckets up to the last *complete* period (``as_of``);
    3. censoring   — repair demand in stock-out periods (scale up / interpolate);
    4. intermittency — > 50% zero periods -> Croston-SBA instead of the FTGM;
    5. outliers    — Hampel filter on the deseasonalised series;
    6. FTGM        — Fourier order chosen by Algorithm 1 inside a history-length cap
                     (ridge-regularised estimation, see ``FTGMConfig.ridge``);
    7. tournament  — rolling-origin hold-out of the FTGM, the FTGM combined with the
                     seasonal naive, a damped-trend smoother, the seasonal naive and a
                     seasonal smoother on the *same* past origins; the fewest units missed
                     wins, technical ties go to the FTGM family. The winner's hold-out
                     gives the plain accuracy (100 - WAPE of the horizon total) and the
                     empirical prediction interval;
    8. diagnostics — every decision explained in plain Spanish.

Failure isolation is per product: too short / unstable -> transparent baseline with the
reason in ``fallback_reason``; empty -> ``status="skipped"``. One bad product never fails
the batch. Forecasts are always finite and non-negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from app.baselines import seasonal_naive_forecast
from app.baselines.intermittent import croston_sba, moving_average_forecast
from app.baselines.smoothing import damped_trend_forecast, seasonal_damped_forecast
from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order
from app.ftgm import cleaning
from app.ftgm import metrics as M
from app.ftgm import preprocessing as prep
from app.ftgm.validation import HoldoutScore, rolling_origin
from app.presentation.schemas import (
    CandidateScore,
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
#: Model tournament: rolling origins per period and the FTGM tie margin (MAE ratio).
_ORIGINS = {12: 6, 52: 8, 4: 4}
_FTGM_TIE = 1.03
#: Recent window (periods) the forecast is compared with to describe the trend.
_TREND_WINDOW = {12: 3, 52: 12, 4: 2}
#: Ridge penalties tried by Algorithm 1 together with the Fourier order (0 = paper OLS).
_RIDGE_GRID = (0.0, 0.5, 3.0, 10.0)
_FTGM_FAMILY = ("FTGM", "FTGMCombo")
_MODEL_WORD = {
    "FTGM": "FTGM",
    "FTGMCombo": "FTGM combinado",
    "DampedTrend": "suavizado de nivel",
    "SeasonalDamped": "suavizado estacional",
    "SeasonalNaive": "repetir la temporada anterior",
    "CrostonSBA": "Croston-SBA",
    "MovingAverage": "promedio móvil",
}

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
        """Model tournament: the FTGM against honest challengers on the same past data.

        Every contender is re-fitted at several past origins and scored on the periods it
        had not seen (rolling origin). The one that missed the fewest units wins; a
        technical tie (within ``_FTGM_TIE``) goes to the FTGM family, the thesis model.
        The winner is then fitted on the full history to produce the forecast.
        """
        config = FTGMConfig(period=period)
        n = clean.size
        horizon = len(future_dates)
        nyquist = max(1, math.ceil(period / 2) - 1)
        season_cap = max(1, int(round(2.0 * n / period)))
        cap = min(nyquist, _FREQ_ORDER_CAP.get(period, nyquist), season_cap)
        if request.max_order is not None:
            cap = min(cap, request.max_order)
        diag.max_order_allowed = cap

        validation_size = request.validation_size
        if validation_size is None and period >= 52:
            validation_size = max(1, min(13, n // 4))

        # --- Algorithm 1: Fourier order for the FTGM contender --------------------
        order: int | None = None
        try:
            selection = select_order(
                clean, config, validation_size=validation_size, max_order=cap, ridges=_RIDGE_GRID
            )
            order = selection.best_order
            config = replace(config, ridge=selection.best_ridge)
            diag.validation_size = selection.validation_size
            diag.order_scores = {k: _r(v) for k, v in selection.scores.items()}
            best = selection.scores.get(order)
            diag.validation_rmse = _r(best) if best is not None else None
            diag.explanations.append(
                f"Algoritmo 1: se probaron órdenes de Fourier 1…{cap} (tope por largo de historia y Nyquist) "
                f"y se eligió N = {order} por menor RMSE en validación"
                + (
                    f", con regularización ridge λ = {selection.best_ridge:g} (evita que el ruido se lea "
                    "como temporada)."
                    if selection.best_ridge > 0
                    else " (estimación por mínimos cuadrados, como en el paper)."
                )
            )
        except FTGMError as exc:
            diag.explanations.append(f"El FTGM no pudo ajustarse de forma estable ({exc}).")

        # --- contenders --------------------------------------------------------------
        h_eval = max(1, min(horizon, 6 if period < 52 else 8))
        n_origins = _ORIGINS.get(period, 6)
        min_train = max(8, 2 + 4 * (order or 1) + 3)

        # Every origin's train set is a prefix of ``clean``, so (length, h) identifies it:
        # the FTGM combo reuses the FTGM forecasts instead of re-solving the ODE.
        ftgm_cache: dict[tuple[int, int], FloatArray] = {}

        def ftgm_fc(train: FloatArray, h: int) -> FloatArray:
            assert order is not None
            key = (train.size, h)
            if key not in ftgm_cache:
                ftgm_cache[key] = FTGM(order=order, config=config).fit(train).predict(h).point
            return ftgm_cache[key]

        def damped_fc(train: FloatArray, h: int) -> FloatArray:
            return damped_trend_forecast(train, h)[0]

        def combo_fc(train: FloatArray, h: int) -> FloatArray:
            return 0.5 * (np.clip(ftgm_fc(train, h), 0.0, None) + naive_fc(train, h))

        def naive_fc(train: FloatArray, h: int) -> FloatArray:
            return seasonal_naive_forecast(train, h, period)

        def seasonal_fc(train: FloatArray, h: int) -> FloatArray:
            return seasonal_damped_forecast(train, h, period)[0]

        contenders: dict[str, Callable[[FloatArray, int], FloatArray]] = {}
        if order is not None:
            contenders["FTGM"] = ftgm_fc
            contenders["FTGMCombo"] = combo_fc
        contenders["DampedTrend"] = damped_fc
        contenders["SeasonalNaive"] = naive_fc
        # Seasonal decomposition needs two full seasons in every training window.
        if n - h_eval - (n_origins - 1) >= 2 * period:
            contenders["SeasonalDamped"] = seasonal_fc

        scores: dict[str, HoldoutScore] = {}
        for name, fc in contenders.items():
            score = rolling_origin(clean, fc, horizon=h_eval, n_origins=n_origins, min_train=min_train, period=period)
            if score is not None:
                scores[name] = score

        winner = self._pick_winner(scores, order is not None)
        naive = scores.get("SeasonalNaive")
        hold = scores.get(winner) if winner else None
        diag.holdout = self._holdout(hold)
        diag.naive_holdout = self._holdout(naive)
        if hold is not None and naive is not None and naive.rmse > 0:
            diag.skill_vs_naive = _r(1.0 - hold.rmse / naive.rmse, 3)
        diag.accuracy_pct = _r(hold.accuracy_pct, 1) if hold is not None else None
        diag.candidates = [
            CandidateScore(
                model=name,
                mae=_r(s.mae),
                wape=_r(s.wape, 2),
                accuracy_pct=_r(s.accuracy_pct, 1),
                chosen=name == winner,
            )
            for name, s in sorted(scores.items(), key=lambda kv: kv[1].mae)
        ]
        if scores:
            ranking = ", ".join(
                f"{_MODEL_WORD.get(c.model, c.model)} {c.accuracy_pct:.0f}%"
                for c in diag.candidates
                if c.accuracy_pct is not None
            )
            diag.explanations.append(
                f"Competencia de modelos en {hold.origins if hold else n_origins} cortes del pasado "
                f"({_plural(h_eval, period)} adelante), precisión sobre el total: {ranking}."
            )

        # --- final fit of the winner on the whole history ----------------------------
        if winner is None:
            winner = "FTGM" if order is not None else "DampedTrend"
        try:
            point, fitted = self._fit_final(winner, clean, horizon, period, config, order)
        except (FTGMError, ValueError, np.linalg.LinAlgError) as exc:
            diag.explanations.append(f"El modelo elegido falló al ajustarse con toda la historia ({exc}).")
            winner = "DampedTrend"
            hold = scores.get(winner)
            point, fitted = damped_trend_forecast(clean, horizon)

        is_ftgm = winner in _FTGM_FAMILY
        reason = None
        if not is_ftgm:
            ftgm_score = scores.get("FTGM")
            versus = ""
            if (
                hold is not None
                and ftgm_score is not None
                and hold.accuracy_pct is not None
                and ftgm_score.accuracy_pct is not None
            ):
                versus = f" ({hold.accuracy_pct:.0f}% vs {ftgm_score.accuracy_pct:.0f}% de precisión)"
            reason = (
                f"Con tus ventas pasadas, {_MODEL_WORD.get(winner, winner)} acertó más que el FTGM{versus}; "
                "se usa ese modelo para no arriesgar la reposición."
            )
            diag.explanations.append(reason)
        elif winner == "FTGMCombo":
            diag.explanations.append(
                "Ganó el FTGM combinado: el promedio del FTGM y de la misma temporada del año anterior, "
                "que reduce el ruido de las ventas pequeñas sin perder la forma estacional."
            )

        residuals = clean - fitted
        point = np.clip(np.nan_to_num(point, nan=0.0), 0.0, None)
        lower, upper = self._interval(point, residuals, hold)
        self._explain_forecast(diag, point, clean, period)
        return ProductForecast(
            product_id=product_id,
            model=(request.model if winner == "FTGM" else winner),
            order_selected=(order or _BASELINE_ORDER) if is_ftgm else _BASELINE_ORDER,
            status="ok" if is_ftgm else "fallback",
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
    def _pick_winner(scores: dict[str, HoldoutScore], ftgm_available: bool) -> str | None:
        if not scores:
            return None
        best = min(scores, key=lambda k: scores[k].mae)
        if best in _FTGM_FAMILY or not ftgm_available:
            return best
        family = [k for k in _FTGM_FAMILY if k in scores]
        if family:
            top = min(family, key=lambda k: scores[k].mae)
            if scores[top].mae <= _FTGM_TIE * scores[best].mae:
                return top
        return best

    @staticmethod
    def _fit_final(
        winner: str, clean: FloatArray, horizon: int, period: int, config: FTGMConfig, order: int | None
    ) -> tuple[FloatArray, FloatArray]:
        if winner in _FTGM_FAMILY:
            assert order is not None
            model = FTGM(order=order, config=config).fit(clean)
            assert model.fitted_ is not None
            point = np.clip(model.predict(horizon).point, 0.0, None)
            fitted = model.fitted_
            if winner == "FTGMCombo":
                s_point = seasonal_naive_forecast(clean, horizon, period)
                season = period if clean.size > period else 1
                s_fit = np.concatenate([clean[:season], clean[:-season]]) if clean.size > season else clean.copy()
                return 0.5 * (point + s_point), 0.5 * (fitted + s_fit)
            return point, fitted
        if winner == "SeasonalDamped":
            return seasonal_damped_forecast(clean, horizon, period)
        if winner == "SeasonalNaive":
            point = seasonal_naive_forecast(clean, horizon, period)
            season = period if clean.size > period else 1
            fitted = np.concatenate([clean[:season], clean[:-season]]) if clean.size > season else clean.copy()
            return point, fitted
        return damped_trend_forecast(clean, horizon)

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
        hold = self._score_single(
            "CrostonSBA", lambda tr, h: croston_sba(tr, h)[0], repaired, len(future_dates), period, diag,
        )
        lower, upper = self._interval(point, repaired - fitted, hold)
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
        hold = self._score_single(
            "MovingAverage", lambda tr, h: moving_average_forecast(tr, h, window)[0], clean, len(future_dates),
            period, diag,
        )
        lower, upper = self._interval(point, clean - fitted, hold)
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

    def _score_single(
        self,
        name: str,
        forecaster: Callable[[FloatArray, int], FloatArray],
        x: FloatArray,
        horizon: int,
        period: int,
        diag: ProductDiagnostics,
    ) -> HoldoutScore | None:
        """Honest past accuracy for the non-tournament paths (intermittent / short)."""
        h_eval = max(1, min(horizon, 6 if period < 52 else 8))
        hold = rolling_origin(
            x, forecaster, horizon=h_eval, n_origins=_ORIGINS.get(period, 6), min_train=6, period=period
        )
        if hold is None:
            return None
        diag.holdout = self._holdout(hold)
        diag.accuracy_pct = _r(hold.accuracy_pct, 1)
        diag.candidates = [
            CandidateScore(
                model=name, mae=_r(hold.mae), wape=_r(hold.wape, 2), accuracy_pct=diag.accuracy_pct, chosen=True
            )
        ]
        return hold

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
            wape=_r(score.wape, 2),
            total_wape=_r(score.total_wape, 2),
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
        # Compare with a steadier recent level (>= 3 months / 12 weeks), not the single
        # last period: one noisy month would otherwise read as a +/-30% "trend".
        window = max(h, _TREND_WINDOW.get(period, 3))
        recent = history[-window:] if history.size >= window else history
        if recent.size and float(np.sum(recent)) > 0:
            scaled = float(np.sum(recent)) * (h / recent.size)
            pct = 100.0 * (float(np.sum(point)) - scaled) / scaled
            diag.forecast_vs_recent_pct = _r(pct, 1)
            if abs(pct) >= 3:
                diag.explanations.append(
                    f"Se proyectan {float(np.sum(point)):.0f} unidades en {_plural(h, period)}, "
                    f"{abs(pct):.0f}% {'más' if pct > 0 else 'menos'} que el ritmo de los últimos "
                    f"{_plural(recent.size, period)}."
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
