from pydantic import BaseModel, Field

class TimeSeriesPoint(BaseModel):
    period: str
    value: float
    stockout_flag: bool = False

class ForecastRequest(BaseModel):
    product_id: str
    series: list[TimeSeriesPoint]
    horizon: int = Field(gt=0, le=52)

class ForecastPoint(BaseModel):
    period: str
    value: float

class ForecastMetrics(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None

class ForecastResponse(BaseModel):
    product_id: str
    model: str
    forecast: list[ForecastPoint]
    metrics: ForecastMetrics
