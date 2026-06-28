"""Service configuration: port, data dir, deterministic seeds.

Values are read from the environment so the Rust orchestrator and docker-compose
can configure the worker without code changes. All defaults are safe for local dev.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Global default seed. Every stochastic component (LightGBM, numpy RNGs, CV shuffles)
# threads this through so runs are reproducible. Requests may override per-job.
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    port: int
    data_dir: Path
    artifact_dir: Path
    seed: int

    @staticmethod
    def from_env() -> Config:
        port = int(os.environ.get("ML_WORKER_PORT", "8088"))
        data_dir = Path(os.environ.get("ML_WORKER_DATA_DIR", "/tmp/se_ml/data")).resolve()
        artifact_dir = Path(
            os.environ.get("ML_WORKER_ARTIFACT_DIR", "/tmp/se_ml/artifacts")
        ).resolve()
        seed = int(os.environ.get("ML_WORKER_SEED", str(DEFAULT_SEED)))
        data_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return Config(port=port, data_dir=data_dir, artifact_dir=artifact_dir, seed=seed)


# Module-level singleton; cheap to construct, safe to import everywhere.
CONFIG = Config.from_env()
