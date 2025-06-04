from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Numeric feature vector (size must match model input dimension).",
    )


class PredictResponse(BaseModel):
    prediction: int
    score: float
    model_version: str
    request_id: str


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=512)


class BatchPredictionItem(BaseModel):
    prediction: int
    score: float


class PredictBatchResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    count: int
    model_version: str
    request_id: str

