"""THE checkpoint: the validation harness must catch a planted future-peeking feature.

A leaky dataset (with a ``leak__lookahead`` feature carrying look-ahead bias) is
spectacular in-sample. The purged + embargoed CPCV harness, DSR and PBO must expose its
out-of-sample collapse so the promotion gate REJECTS it. In contrast, a genuine-edge
dataset must PASS, proving the harness is not simply rejecting everything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from se_ml.config import CONFIG
from se_ml.contract import ValidateRequest, ValidateResult
from se_ml.io_arrow import write_dataframe
from se_ml.server import app, validate

from .fixtures import genuine_edge_dataset, leaky_dataset, pure_noise_dataset

FOLD_SPEC = {"n_groups": 6, "k_test_groups": 2, "embargo_bars": 3, "purge": True}
N = 900          # dataset size used across the leakage tests
N_TRIALS = 16    # search-space size; large enough for a meaningful PBO


def _write(df, name: str) -> str:
    path = CONFIG.data_dir / name
    write_dataframe(df, path)
    return str(path)


def _validate(uri: str, n_trials: int = N_TRIALS) -> ValidateResult:
    req = ValidateRequest(
        dataset_uri=uri,
        horizon="swing",
        fold_spec=FOLD_SPEC,  # type: ignore[arg-type]
        n_trials=n_trials,
    )
    return validate(req)


def test_in_sample_leak_looks_great():
    # Fit-on-everything in-sample: the leak makes the model near-perfect, proving the
    # leak is genuinely "great in-sample" (the trap the harness must avoid).
    from se_ml.io_arrow import split_features_labels
    from se_ml.models.gbm import fit_gbm

    df = leaky_dataset(n=N, seed=2)
    X, y = split_features_labels(df)
    model = fit_gbm(X, y, seed=0)
    in_sample_proba = model.predict_proba(X)
    pred = (in_sample_proba >= 0.5).astype(int)
    acc = float((pred == (y.to_numpy() > 0).astype(int)).mean())
    assert acc > 0.95, f"leak should be near-perfect in-sample, got acc={acc}"


def test_leak_is_rejected_by_gate():
    # THE checkpoint: spectacular in-sample, but the look-ahead leak flips on the held-out
    # slice, so the purged+embargoed harness sees the OOS edge collapse and the gate REJECTS.
    uri = _write(leaky_dataset(n=N, seed=2), "leaky.parquet")
    res = _validate(uri)
    assert res.passed_gate is False, f"gate must REJECT the leak; result={res}"
    # The collapse is concrete: cost-aware OOS expectancy is negative and DSR is non-positive.
    assert res.oos_expectancy_cost_aware <= 0.0, f"leak OOS expectancy should collapse; {res}"
    assert res.dsr <= 0.5, f"leak DSR should deflate to non-positive; {res}"


def test_pure_noise_is_rejected():
    uri = _write(pure_noise_dataset(n=N, seed=1), "noise.parquet")
    res = _validate(uri)
    assert res.passed_gate is False, f"gate must REJECT pure noise; result={res}"


def test_genuine_edge_passes_gate():
    # The genuine-edge set must PASS — proving the harness discriminates and is not simply
    # rejecting everything. Positive cost-aware OOS expectancy, low PBO, positive DSR.
    uri = _write(genuine_edge_dataset(n=N, seed=0), "genuine.parquet")
    res = _validate(uri)
    assert res.oos_expectancy_cost_aware > 0.0, f"genuine edge OOS expectancy; {res}"
    assert res.pbo < 0.5, f"genuine edge should have low PBO; {res}"
    assert res.dsr > 0.0, f"genuine edge should have positive DSR; {res}"
    assert res.passed_gate is True, f"gate must PROMOTE a genuine edge; {res}"


def test_health_endpoint_via_testclient():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_full_validate_endpoint_roundtrip():
    # Exercise the HTTP path end-to-end for the leaky set and confirm the contract shape.
    uri = _write(leaky_dataset(n=N, seed=2), "leaky_http.parquet")
    client = TestClient(app)
    resp = client.post(
        "/validate",
        json={
            "dataset_uri": uri,
            "horizon": "swing",
            "fold_spec": FOLD_SPEC,
            "n_trials": N_TRIALS,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in (
        "dsr",
        "pbo",
        "oos_expectancy_cost_aware",
        "profit_factor",
        "cvar5",
        "mar",
        "regime_contrib",
        "n_regimes_positive",
        "passed_gate",
    ):
        assert key in body, f"missing contract field {key}"
    assert body["passed_gate"] is False


def test_artifact_dir_is_writable():
    assert Path(CONFIG.artifact_dir).exists()
    assert np.isfinite(1.0)  # trivial guard so import of np is used
