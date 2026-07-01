# FTGM analytical engine (FastAPI + numpy/scipy). No database.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY . .
RUN pip install .

EXPOSE 8010
# Bind to $PORT when the platform injects one (Render/Railway), else 8010 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010}"]
