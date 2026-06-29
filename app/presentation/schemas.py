"""HTTP request/response models for the FTGM engine.

This is the **engine side of the contract** shared with ``inventory-dss-api``. The
engine accepts a *batch* of product series (so the backend can forecast a whole catalog
in one call) and returns, per product, the point forecast with a prediction interval,
the Fourier order that was selected, and accuracy metrics.
"""
from __future__ import annotations

from datetime import date

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


class ForecastPoint(BaseModel):
    """One forecasted period with its prediction interval."""

    date: date
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class ForecastMetrics(BaseModel):
    """In-sample accuracy metrics for the fitted model."""

    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mase: float | None = None
    rmsse: float | None = None


class ProductForecast(BaseModel):
    """Forecast bundle for a single product."""

    model_config = _ALLOW_MODEL_FIELD

    product_id: str
    model: str
    order_selected: int = Field(description="Fourier order chosen; 0 = baseline fallback")
    points: list[ForecastPoint]
    metrics: ForecastMetrics


class ForecastResponse(BaseModel):
    """Batch forecast response."""

    period: int
    forecasts: list[ProductForecast]
