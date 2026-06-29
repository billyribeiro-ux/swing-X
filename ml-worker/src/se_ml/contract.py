"""The Rust<->Python contract: pydantic v2 job + result models.

The Rust `se-mlclient` crate mirrors these EXACT field names (snake_case). Bulk data
(features + labels) is never embedded in JSON; it is exchanged as Parquet/Arrow by
`dataset_uri` (a local path or file:// URI). Bodies carry only metadata + metrics.

JSON contract per endpoint
==========================

GET /health
    -> HealthResponse
       {"status": "ok", "version": "0.1.0"}

POST /fit
    FitRequest:
       {"dataset_uri": "/data/spy_swing.parquet",
        "horizon": "swing",
        "model_params": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
        "seed": 42}
    -> FitResult:
       {"model_id": "gbm-3f2a...",
        "artifact_uri": "/artifacts/gbm-3f2a....joblib",
        "in_sample_metrics": {"expectancy": 0.12, "profit_factor": 1.8,
                              "sharpe": 1.4, "cvar5": -0.9, "mar": 2.1, "n": 1200}}

POST /validate
    ValidateRequest:
       {"dataset_uri": "/data/spy_swing.parquet",
        "horizon": "swing",
        "fold_spec": {"n_groups": 8, "k_test_groups": 2,
                      "embargo_bars": 5, "purge": true},
        "n_trials": 50}
    -> ValidateResult:
       {"dsr": 0.61, "pbo": 0.08, "oos_expectancy_cost_aware": 0.04,
        "profit_factor": 1.35, "cvar5": -0.7, "mar": 1.2,
        "regime_contrib": {"bull": 0.06, "bear": 0.01, "chop": 0.03},
        "n_regimes_positive": 3, "passed_gate": true,
        "precision_oos": 0.58, "recall_oos": 0.41,
        "act_threshold": 0.62, "n_acted_oos": 73}

POST /calibrate
    CalibrateRequest:
       {"dataset_uri": "/data/spy_swing.parquet", "model_id": "gbm-3f2a..."}
    -> CalibrateResult:
       {"calibration_map": {"method": "isotonic",
                            "x": [...], "y": [...]},
        "reliability_points": [{"predicted": 0.1, "realized": 0.08, "count": 40}, ...],
        "brier": 0.18}

POST /importance
    ImportanceRequest:
       {"dataset_uri": "/data/spy_swing.parquet", "model_id": "gbm-3f2a..."}
    -> ImportanceResult:
       {"per_feature": {"rsi_14": {"shap": 0.21, "permutation": 0.18}, ...},
        "per_layer":   {"momentum": {"shap": 0.40, "permutation": 0.35}, ...}}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_SEED


class _Strict(BaseModel):
    """Base model: snake_case only, forbid unexpected fields to catch contract drift."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
class HealthResponse(_Strict):
    status: str = "ok"
    version: str


# --------------------------------------------------------------------------- #
# /fit
# --------------------------------------------------------------------------- #
class FitRequest(_Strict):
    dataset_uri: str = Field(..., description="Parquet/Arrow path or file:// URI (features+labels)")
    horizon: str = Field(..., description="Horizon profile id, e.g. 'swing'.")
    model_params: dict[str, Any] = Field(default_factory=dict)
    seed: int = DEFAULT_SEED


class InSampleMetrics(_Strict):
    expectancy: float
    profit_factor: float
    sharpe: float
    cvar5: float
    mar: float
    n: int


class FitResult(_Strict):
    model_id: str
    artifact_uri: str
    in_sample_metrics: InSampleMetrics


# --------------------------------------------------------------------------- #
# /validate
# --------------------------------------------------------------------------- #
class FoldSpec(_Strict):
    n_groups: int = Field(8, ge=2, description="Number of contiguous time groups.")
    k_test_groups: int = Field(2, ge=1, description="Groups held out as test per CPCV combination.")
    embargo_bars: int = Field(5, ge=0, description="Bars embargoed AFTER each test block.")
    purge: bool = Field(True, description="Purge train labels overlapping any test span.")


class ValidateRequest(_Strict):
    dataset_uri: str
    horizon: str
    fold_spec: FoldSpec = FoldSpec()
    n_trials: int = Field(1, ge=1, description="Number of strategy trials (for DSR deflation).")


class ValidateResult(_Strict):
    dsr: float
    pbo: float
    oos_expectancy_cost_aware: float
    profit_factor: float
    cvar5: float
    mar: float
    regime_contrib: dict[str, float]
    n_regimes_positive: int
    passed_gate: bool
    # Precision-optimized meta-labeling acting layer (measured out-of-sample at tau*).
    precision_oos: float = Field(
        ..., description="Fraction of ACTED OOS trades that were profitable (R > 0) at tau*."
    )
    recall_oos: float = Field(
        ...,
        description="Fraction of all profitable OOS opportunities captured at tau* (coverage).",
    )
    act_threshold: float = Field(
        ..., description="tau*, the selected meta-labeling acting threshold (probability in [0,1])."
    )
    n_acted_oos: int = Field(
        ..., description="Number of OOS-reporting-half trades acted on at tau*."
    )


# --------------------------------------------------------------------------- #
# /calibrate
# --------------------------------------------------------------------------- #
class CalibrateRequest(_Strict):
    dataset_uri: str
    model_id: str


class ReliabilityPoint(_Strict):
    predicted: float
    realized: float
    count: int


class CalibrationMap(_Strict):
    method: str
    x: list[float]
    y: list[float]


class CalibrateResult(_Strict):
    calibration_map: CalibrationMap
    reliability_points: list[ReliabilityPoint]
    brier: float


# --------------------------------------------------------------------------- #
# /importance
# --------------------------------------------------------------------------- #
class ImportanceScore(_Strict):
    shap: float
    permutation: float


class ImportanceRequest(_Strict):
    dataset_uri: str
    model_id: str


class ImportanceResult(_Strict):
    per_feature: dict[str, ImportanceScore]
    per_layer: dict[str, ImportanceScore]
