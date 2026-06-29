"""HTTP routes for the FTGM engine."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.application.forecast_service import ForecastService
from app.ftgm import FTGMError
from app.presentation.schemas import ForecastRequest, ForecastResponse

router = APIRouter(tags=["FTGM Engine"])

# Stateless service — safe to share across requests.
_service = ForecastService()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "inventory-dss-ftgm-engine"}


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Run the FTGM forecast pipeline for a batch of product series."""
    if not request.series:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="series must contain at least one product",
        )
    try:
        return _service.forecast(request)
    except FTGMError as exc:  # defensive: per-product fallbacks should prevent this
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
