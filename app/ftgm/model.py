from app.presentation.schemas import ForecastPoint, TimeSeriesPoint

class FTGMModel:
    """
    Placeholder for the Fourier Time-Varying Grey Model implementation.

    The real mathematical implementation will be added in a later phase.
    """

    def forecast(self, series: list[TimeSeriesPoint], horizon: int) -> list[ForecastPoint]:
        if not series:
            return []

        last_value = series[-1].value

        return [
            ForecastPoint(period=f"t+{step}", value=last_value)
            for step in range(1, horizon + 1)
        ]
