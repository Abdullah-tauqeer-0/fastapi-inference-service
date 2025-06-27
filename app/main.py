from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import load_settings
from app.logging_config import configure_logging
from app.metrics import (
    HTTP_REQUEST_COUNT,
    HTTP_REQUEST_LATENCY_SECONDS,
    PREDICTION_REQUEST_COUNT,
)
from app.model_runner import ModelLoadError, ModelRegistry
from app.schemas import (
    BatchPredictionItem,
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

load_dotenv()
settings = load_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

model_registry = ModelRegistry(models_root=settings.models_root)


def _resolve_model_version(override_header: str | None) -> str:
    version = (override_header or settings.model_version).strip()
    return version or settings.model_version


def _ensure_runner(model_version: str):
    try:
        return model_registry.load(model_version)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{model_version}' is unavailable",
        ) from exc
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model version '{model_version}' failed to load",
        ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _ensure_runner(settings.model_version)
        logger.info(
            "default_model_loaded",
            extra={"model_version": settings.model_version},
        )
    except HTTPException:
        logger.exception(
            "default_model_load_failed",
            extra={"model_version": settings.model_version},
        )
    yield


app = FastAPI(
    title="FastAPI Inference Service",
    description="Production-style template for local ML inference serving.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid4())
    model_version = _resolve_model_version(request.headers.get("X-Model-Version"))
    endpoint = request.url.path
    method = request.method

    request.state.request_id = request_id
    request.state.model_version = model_version

    start_time = perf_counter()
    response: Response
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        latency_seconds = perf_counter() - start_time
        latency_ms = round(latency_seconds * 1000, 3)
        HTTP_REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, endpoint=endpoint).observe(
            latency_seconds
        )
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "model_version": model_version,
            },
        )
        raise

    latency_seconds = perf_counter() - start_time
    latency_ms = round(latency_seconds * 1000, 3)

    response.headers["X-Request-ID"] = request_id
    HTTP_REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, status_code=str(status_code)
    ).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, endpoint=endpoint).observe(
        latency_seconds
    )
    logger.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "model_version": model_version,
        },
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse | JSONResponse:
    loaded = model_registry.is_loaded(settings.model_version)
    payload = ReadyResponse(
        status="ok" if loaded else "not_ready",
        model_loaded=loaded,
        model_version=settings.model_version,
    )
    if not loaded:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    x_model_version: str | None = Header(default=None, alias="X-Model-Version"),
) -> PredictResponse:
    model_version = _resolve_model_version(x_model_version)
    request.state.model_version = model_version

    runner = _ensure_runner(model_version)
    result = runner.predict(payload.features)
    PREDICTION_REQUEST_COUNT.labels(
        model_version=model_version, endpoint="/predict"
    ).inc()

    return PredictResponse(
        prediction=result.label,
        score=result.score,
        model_version=model_version,
        request_id=request.state.request_id,
    )


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(
    payload: PredictBatchRequest,
    request: Request,
    x_model_version: str | None = Header(default=None, alias="X-Model-Version"),
) -> PredictBatchResponse:
    model_version = _resolve_model_version(x_model_version)
    request.state.model_version = model_version

    runner = _ensure_runner(model_version)
    features = [item.features for item in payload.items]
    results = runner.predict_batch(features)
    PREDICTION_REQUEST_COUNT.labels(
        model_version=model_version, endpoint="/predict-batch"
    ).inc()

    return PredictBatchResponse(
        predictions=[
            BatchPredictionItem(prediction=item.label, score=item.score)
            for item in results
        ],
        count=len(results),
        model_version=model_version,
        request_id=request.state.request_id,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
