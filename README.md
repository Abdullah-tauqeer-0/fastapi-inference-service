# fastapi-inference-service

Production-style FastAPI inference service template for ML/GenAI engineers.

This project is designed to be:

- Deployable (`Dockerfile`, `docker-compose.yml`)
- Tested (`pytest`)
- Observable (Prometheus metrics + structured JSON logs)
- Versioned (`models/<version>/model.npz`)
- Runnable locally with no external dependencies

## Features

- `GET /health` liveness endpoint
- `GET /ready` readiness endpoint (ready only when default model is loaded)
- `POST /predict` single prediction
- `POST /predict-batch` batch prediction
- `GET /metrics` Prometheus metrics
- Request ID propagation via `X-Request-ID`
- Per-request model version override via `X-Model-Version`
- Structured JSON logs with:
  - `ts`, `level`, `msg`, `request_id`, `endpoint`, `status_code`, `latency_ms`, `model_version`

## Repository structure

```text
fastapi-inference-service/
+-- .github/workflows/ci.yml
+-- app/
|   +-- config.py
|   +-- logging_config.py
|   +-- main.py
|   +-- metrics.py
|   +-- model_runner.py
|   +-- schemas.py
+-- models/
|   +-- v1/model.npz
+-- scripts/
|   +-- load_test_async.py
|   +-- load_test_k6.js
+-- tests/test_api.py
+-- .env.example
+-- .ruff.toml
+-- docker-compose.yml
+-- Dockerfile
+-- Makefile
+-- requirements-dev.txt
+-- requirements.txt
```

## Quickstart (local)

1. Install dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

1. Configure environment:

```bash
cp .env.example .env
```

1. Run API:

```bash
uvicorn app.main:app --reload
```

1. Verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[1.0,0.5,-0.2]}'
```

## Quickstart (Docker)

```bash
docker compose up --build
```

Then test:

```bash
curl http://127.0.0.1:8000/health
```

## Endpoints

### `GET /health`

- Purpose: liveness
- Response: `200 {"status":"ok"}`

### `GET /ready`

- Purpose: readiness of default model (`MODEL_VERSION`)
- Responses:
  - `200` when model is loaded
  - `503` when model is not loaded

Example success payload:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1"
}
```

### `POST /predict`

- Request body:

```json
{
  "features": [1.0, 0.5, -0.2]
}
```

- Headers:
  - Optional `X-Request-ID`: if absent, server generates UUID
  - Optional `X-Model-Version`: override model version for this request
- Response includes:
  - `X-Request-ID` header
  - JSON with prediction, score, model version, and request id

### `POST /predict-batch`

- Request body:

```json
{
  "items": [
    { "features": [1.0, 0.5, -0.2] },
    { "features": [0.2, -0.1, 0.9] }
  ]
}
```

- Returns predictions for all items plus count.

### `GET /metrics`

- Prometheus-format metrics, including:
  - `http_requests_total{method,endpoint,status_code}`
  - `http_request_latency_seconds{method,endpoint}`
  - `prediction_requests_total{model_version,endpoint}`

## Model versioning

- Artifacts live in `models/<version>/model.npz`
- Default version is set by environment variable:
  - `MODEL_VERSION=v1` (from `.env.example`)
- Override per request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-Model-Version: v1" \
  -d '{"features":[1.0,0.5,-0.2]}'
```

## Replace the toy model

`ModelRunner` expects each `.npz` artifact to contain:

- `weights`: 1D float array (input dimension)
- `bias`: float scalar
- `threshold`: float scalar (e.g., `0.5`)

Example generation:

```python
import numpy as np
from pathlib import Path

out = Path("models/v2")
out.mkdir(parents=True, exist_ok=True)
np.savez(
    out / "model.npz",
    weights=np.array([0.8, -0.4, 0.2], dtype=np.float64),
    bias=np.float64(-0.1),
    threshold=np.float64(0.5),
)
```

Then run with:

```bash
MODEL_VERSION=v2 uvicorn app.main:app --reload
```

## Testing, linting, CI

- Run tests:

```bash
pytest
```

- Run lint:

```bash
ruff check .
```

- GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
  - Ruff lint
  - Pytest

## Load testing

### Python async script

```bash
python scripts/load_test_async.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 20
```

Outputs p50 and p95 latency in milliseconds.

### k6 script

```bash
k6 run scripts/load_test_k6.js
```

Optional:

```bash
BASE_URL=http://127.0.0.1:8000 k6 run scripts/load_test_k6.js
```

## Makefile targets

- `make install`
- `make install-dev`
- `make run`
- `make test`
- `make lint`
- `make docker-build`
- `make docker-up`
