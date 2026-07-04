"""HTTP request/response models for the FTGM engine.

This is the **engine side of the contract** shared with ``inventory-dss-api``. The
engine accepts a *batch* of product series (so the backend can forecast a whole catalog
in one call) and returns, per product, the point forecast with a prediction interval,
the Fourier order that was selected, the aggregated in-sample history (with the model
fit), accuracy metrics, and diagnostics explaining what the pipeline did.

Backward compatibility: every field added on top of the original contract has a
default, so an older client simply ignores the extras.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ``model`` is a meaningful field name here; opt out of pydantic's ``model_`` guard.
_ALLOW_MODEL_FIELD = ConfigDict(protected_namespaces=())


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
    horizon: int = Field(gt=0, le=104, description="Number of periods to forecast")
    series: list[ProductSeries]
    # Optional overrides for Algorithm 1 (defaults follow the paper).
    validation_size: int | None = Field(
        default=None, gt=0, description="Hold-out size for order selection (default: T/2)"
    )
    max_order: int | None = Field(
        default=None, gt=0, description="Cap on the candidate Fourier orders"
    )


class ForecastPoint(BaseModel):
    """One forecasted period with its prediction interval."""

    date: date
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class HistoryPoint(BaseModel):
    """One in-sample period: what was observed, what the model consumed and produced.

    ``observed``   – aggregated demand for the bucket (raw sum).
    ``cleaned``    – demand after stock-out imputation (what the model was fit on).
    ``fitted``     – the model's in-sample fit for the bucket (None for skipped series).
    ``is_stockout``– the bucket contained at least one stock-out observation.
    """

    date: date
    observed: float
    cleaned: float
    fitted: float | None = None
    is_stockout: bool = False


class ForecastMetrics(BaseModel):
    """In-sample accuracy metrics for the fitted model."""

    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    rmsse: float | None = None


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


class ProductForecast(BaseModel):
    """Forecast bundle for a single product."""

    model_config = _ALLOW_MODEL_FIELD

    product_id: str
    model: str
    order_selected: int = Field(description="Fourier order chosen; 0 = baseline fallback")
    status: Literal["ok", "fallback", "skipped"] = "ok"
    fallback_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    points: list[ForecastPoint]
    history: list[HistoryPoint] = Field(default_factory=list)
    metrics: ForecastMetrics
    diagnostics: ProductDiagnostics | None = None


class ForecastResponse(BaseModel):
    """Batch forecast response."""

    period: int
    forecasts: list[ProductForecast]
