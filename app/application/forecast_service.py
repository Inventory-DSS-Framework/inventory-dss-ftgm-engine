from app.ftgm.model import FTGMModel
from app.presentation.schemas import ForecastMetrics, ForecastRequest, ForecastResponse

class ForecastService:
    def __init__(self) -> None:
        self._model = FTGMModel()

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        forecast_points = self._model.forecast(
            series=request.series,
            horizon=request.horizon
        )

        return ForecastResponse(
            product_id=request.product_id,
            model="FTGM",
            forecast=forecast_points,
            metrics=ForecastMetrics()
        )
