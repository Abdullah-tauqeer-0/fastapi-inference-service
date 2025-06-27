from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests processed",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Prediction request count by model version and endpoint",
    ["model_version", "endpoint"],
)

