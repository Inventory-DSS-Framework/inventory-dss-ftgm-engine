"""HTTP request/response models for the FTGM engine.

This is the **engine side of the contract** shared with ``inventory-dss-api``. The
engine accepts a *batch* of product series (so the backend can forecast a whole catalog
in one call) and returns, per product, the point forecast with a prediction interval,
the Fourier order that was selected, the aggregated in-sample history (with the model
fit), accuracy metrics, and diagnostics explaining what the pipeline did.

Two ways to call it:

* **Legacy** — ``period`` (seasonal period T) + ``horizon`` (periods). Every product is
  aggregated with that period.
* **Frequency-aware** — ``frequency`` (``auto`` | ``monthly`` | ``weekly``) +
  ``horizon_days`` + ``as_of``. The engine decides the bucket size per product (monthly
  with >= 24 months of history, weekly with >= 26 weeks, otherwise a short-history
  baseline), drops the period in progress and converts the day horizon to periods.

Backward compatibility: every field added on top of the original contract has a
default, so an older client simply ignores the extras.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ``model`` is a meaningful field name here; opt out of pydantic's ``model_`` guard.
_ALLOW_MODEL_FIELD = ConfigDict(protected_namespaces=())

Frequency = Literal["auto", "monthly", "weekly"]


class ObservationPoint(BaseModel):
    """A single observed demand value at a given date."""

    date: date
    demand: float
    stockout_flag: bool = False


class ProductSeries(BaseModel):
    """The historical demand series for one product."""

    product_id: str
    points: list[ObservationPoint]


class ForecastRequest(BaseModel):
    """Batch forecast request."""

    model_config = _ALLOW_MODEL_FIELD

    model: str = "FTGM"
    period: int = Field(default=12, gt=0, description="Seasonal period T (12 = monthly)")
    horizon: int | None = Field(
        default=None, gt=0, le=104, description="Number of periods to forecast (legacy)"
    )
    series: list[ProductSeries]
    frequency: Frequency | None = Field(
        default=None, description="Bucket size; None keeps the legacy `period` behaviour"
    )
    horizon_days: int | None = Field(
        default=None, gt=0, le=730, description="Horizon in days (converted per frequency)"
    )
    as_of: date | None = Field(
        default=None, description="Cut-off: the period containing this date is excluded"
    )
    # Optional overrides for Algorithm 1 (defaults follow the paper).
    validation_size: int | None = Field(
        default=None, gt=0, description="Hold-out size for order selection (default: T/2)"
    )
    max_order: int | None = Field(
        default=None, gt=0, description="Cap on the candidate Fourier orders"
    )

    @model_validator(mode="after")
    def _horizon_given(self) -> "ForecastRequest":
        if self.horizon is None and self.horizon_days is None:
            raise ValueError("either `horizon` or `horizon_days` is required")
        return self


class ForecastPoint(BaseModel):
    """One forecasted period with its prediction interval."""

    date: date
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class HistoryPoint(BaseModel):
    """One in-sample period: what was observed, what the model consumed and produced.

    ``observed``   – aggregated demand for the bucket (raw sum).
    ``cleaned``    – demand after stock-out repair and outlier treatment (model input).
    ``fitted``     – the model's in-sample fit for the bucket (None for skipped series).
    ``is_stockout``– the bucket was censored by a stock-out (>= 10% of it).
    ``is_outlier`` – the bucket was treated as an outlier (Hampel filter).
    """

    date: date
    observed: float
    cleaned: float
    fitted: float | None = None
    is_stockout: bool = False
    is_outlier: bool = False
    stockout_share: float = 0.0


class ForecastMetrics(BaseModel):
    """In-sample accuracy metrics for the fitted model."""

    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    rmsse: float | None = None


class HoldoutMetrics(BaseModel):
    """Out-of-sample accuracy from rolling-origin evaluation."""

    origins: int
    horizon: int
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    wape: float | None = None
    total_wape: float | None = None


class CandidateScore(BaseModel):
    """One contender of the model tournament, scored on the same rolling origins."""

    model: str
    mae: float | None = None
    wape: float | None = None
    accuracy_pct: float | None = None
    chosen: bool = False


class ProductDiagnostics(BaseModel):
    """Evidence of what the pipeline did for one product (Algorithm 1 transparency)."""

    n_input_points: int = Field(description="Raw observations received")
    n_periods: int = Field(description="Aggregated seasonal buckets used for fitting")
    stockout_periods: int = Field(default=0, description="Buckets flagged as stock-out")
    imputed_periods: int = Field(default=0, description="Buckets repaired by interpolation")
    validation_size: int | None = Field(
        default=None, description="Hold-out length used by order selection"
    )
    order_scores: dict[int, float | None] = Field(
        default_factory=dict,
        description="Validation RMSE per candidate Fourier order (null = unstable order)",
    )
    validation_rmse: float | None = Field(
        default=None, description="Validation RMSE of the selected order"
    )
    # --- frequency-aware pipeline evidence ------------------------------------------
    frequency: str | None = None
    period: int | None = None
    frequency_reason: str | None = None
    history_start: date | None = None
    history_end: date | None = None
    dropped_incomplete_period: bool = False
    outliers_cleaned: int = 0
    zero_share: float | None = None
    intermittent: bool = False
    seasonality_strength: float | None = None
    trend_pct_per_period: float | None = None
    max_order_allowed: int | None = None
    holdout: HoldoutMetrics | None = None
    naive_holdout: HoldoutMetrics | None = None
    skill_vs_naive: float | None = Field(
        default=None, description="1 - RMSE(model)/RMSE(seasonal naive) on the same origins"
    )
    interval_level: float | None = None
    accuracy_pct: float | None = Field(
        default=None,
        description="Plain accuracy of the chosen model on past data: 100 - WAPE of the horizon total",
    )
    candidates: list[CandidateScore] = Field(default_factory=list)
    forecast_vs_recent_pct: float | None = Field(
        default=None, description="Forecast total vs the same number of recent periods (%)"
    )
    explanations: list[str] = Field(default_factory=list)


class ProductForecast(BaseModel):
    """Forecast bundle for a single product."""

    model_config = _ALLOW_MODEL_FIELD

    product_id: str
    model: str
    order_selected: int = Field(description="Fourier order chosen; 0 = baseline fallback")
    status: Literal["ok", "fallback", "skipped"] = "ok"
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    frequency: str | None = None
    period: int | None = None
    points: list[ForecastPoint]
    history: list[HistoryPoint] = Field(default_factory=list)
    metrics: ForecastMetrics
    diagnostics: ProductDiagnostics | None = None


class ForecastResponse(BaseModel):
    """Batch forecast response."""

    period: int
    forecasts: list[ProductForecast]
