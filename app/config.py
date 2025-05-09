from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Settings:
    model_version: str = "v1"
    log_level: str = "INFO"
    models_root: Path = BASE_DIR / "models"


def load_settings() -> Settings:
    model_version = os.getenv("MODEL_VERSION", "v1").strip() or "v1"
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"

    models_root_env = os.getenv("MODELS_ROOT")
    if models_root_env:
        models_root = Path(models_root_env).expanduser().resolve()
    else:
        models_root = BASE_DIR / "models"

    return Settings(
        model_version=model_version,
        log_level=log_level,
        models_root=models_root,
    )

