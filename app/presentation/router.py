from fastapi import APIRouter

from app.application.forecast_service import ForecastService
from app.presentation.schemas import ForecastRequest, ForecastResponse

router = APIRouter(tags=["FTGM Engine"])

@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "inventory-dss-ftgm-engine"
    }

@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    service = ForecastService()
    return service.forecast(request)
