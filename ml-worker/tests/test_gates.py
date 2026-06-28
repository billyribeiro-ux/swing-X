"""Tests for the promotion gate (pure function, source of truth for the Rust side)."""

from __future__ import annotations

from se_ml.gates import (
    DSR_MIN,
    MIN_POSITIVE_REGIMES,
    OOS_EXPECTANCY_MIN,
    PBO_MAX,
    evaluate,
)


def test_all_conditions_pass():
    r = evaluate(dsr=0.6, pbo=0.1, oos_expectancy_cost_aware=0.05, n_regimes_positive=3)
    assert r["dsr_ok"] and r["pbo_ok"] and r["oos_expectancy_ok"] and r["regimes_ok"]
    assert r["passed"] is True


def test_fails_when_dsr_not_positive():
    r = evaluate(dsr=0.0, pbo=0.1, oos_expectancy_cost_aware=0.05, n_regimes_positive=3)
    assert r["dsr_ok"] is False
    assert r["passed"] is False


def test_fails_when_pbo_too_high():
    r = evaluate(dsr=0.6, pbo=0.5, oos_expectancy_cost_aware=0.05, n_regimes_positive=3)
    assert r["pbo_ok"] is False
    assert r["passed"] is False


def test_fails_when_oos_expectancy_not_positive():
    r = evaluate(dsr=0.6, pbo=0.1, oos_expectancy_cost_aware=0.0, n_regimes_positive=3)
    assert r["oos_expectancy_ok"] is False
    assert r["passed"] is False


def test_fails_when_too_few_positive_regimes():
    r = evaluate(dsr=0.6, pbo=0.1, oos_expectancy_cost_aware=0.05, n_regimes_positive=1)
    assert r["regimes_ok"] is False
    assert r["passed"] is False


def test_boundary_thresholds_are_strict():
    # exactly at thresholds -> fail (strict inequalities), except regimes which is >=.
    assert evaluate(DSR_MIN, 0.1, 0.05, 3)["dsr_ok"] is False
    assert evaluate(0.6, PBO_MAX, 0.05, 3)["pbo_ok"] is False
    assert evaluate(0.6, 0.1, OOS_EXPECTANCY_MIN, 3)["oos_expectancy_ok"] is False
    assert evaluate(0.6, 0.1, 0.05, MIN_POSITIVE_REGIMES)["regimes_ok"] is True


def test_keys_match_contract():
    r = evaluate(0.6, 0.1, 0.05, 3)
    assert set(r.keys()) == {
        "dsr_ok",
        "pbo_ok",
        "oos_expectancy_ok",
        "regimes_ok",
        "passed",
    }
