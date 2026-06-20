from fastapi import FastAPI

from app.config import settings
from app.presentation.router import router

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        version="0.1.0"
    )

    app.include_router(router, prefix=settings.api_v1_prefix)

    return app

app = create_app()
