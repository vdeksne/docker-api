FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --no-deps \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       --wheel-dir /wheels \
       -r requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/

EXPOSE 5000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "5000"]
