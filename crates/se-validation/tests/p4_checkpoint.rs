//! THE P4 CHECKPOINT (Rust side, end-to-end against the live ml-worker).
//!
//! We build TWO in-memory datasets and run both through the [`ValidationHarness`]:
//!
//!   (a) a GENUINE-EDGE dataset — features causally drive the label across the whole
//!       timeline, so a purged+embargoed CPCV model keeps an edge out-of-sample;
//!   (b) a LEAKY dataset — a `leak__lookahead` feature equals the label's signal on the
//!       IN-SAMPLE (early) half but FLIPS SIGN on the held-out (later) half. In-sample it
//!       looks spectacular; out-of-sample, once the leak can't see the future, it collapses.
//!
//! ASSERTIONS:
//!   * the LEAKY dataset is REJECTED by the gate (DSR <= 0 or PBO >= 0.5 -> not passed),
//!   * the GENUINE-EDGE dataset PASSES.
//!
//! This mirrors `ml-worker/tests/test_leakage_fixture.py` from the Rust side, proving
//! leakage is caught through the real HTTP boundary.
//!
//! If the worker is not reachable, the test SKIPS with a clear message (so absence of the
//! sidecar never fails CI).

use std::collections::BTreeMap;

use chrono::{DateTime, TimeZone, Utc};
use se_core::HorizonProfile;
use se_mlclient::{DatasetRow, MlClient};
use se_validation::ValidationHarness;

// Mirror the Python fixtures' shape.
const N: usize = 900;
const N_TRIALS: u32 = 16;
const N_GROUPS: u32 = 6;
const K_TEST_GROUPS: u32 = 2;
const REGIMES: [&str; 3] = ["bull", "bear", "chop"];

/// A tiny deterministic PRNG (SplitMix64) — we do NOT need to reproduce numpy's RNG, only
/// to generate reproducible draws with the right *statistical structure*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        // 53-bit mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn ts(i: usize) -> DateTime<Utc> {
    // Daily grid; `t1` is +3 days so adjacent label windows overlap (exercises purging).
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i as i64)
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Genuine edge: a stable linear signal of three features drives the R label across the
/// whole timeline. (Mirrors `fixtures.genuine_edge_dataset`.)
fn genuine_edge_dataset(seed: u64) -> Vec<DatasetRow> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let momentum = rng.normal();
        let trend = rng.normal();
        let vol = rng.normal();
        let noise_feat = rng.normal();

        let signal = 0.9 * momentum + 0.6 * trend - 0.3 * vol;
        let label = clip(signal + rng.normal(), -1.0, 2.0);

        let regime = REGIMES[(rng.next_u64() % 3) as usize];
        let features = BTreeMap::from([
            ("momentum__signal".to_string(), momentum),
            ("trend__slope".to_string(), trend),
            ("volatility__atr_norm".to_string(), vol),
            ("momentum__noise".to_string(), noise_feat),
        ]);
        rows.push(DatasetRow {
            ts: ts(i),
            t1: ts(i + 3),
            label,
            regime: Some(regime.to_string()),
            features,
        });
    }
    rows
}

/// Leaky look-ahead: `leak__lookahead` tracks the label's signal early, then FLIPS sign on
/// the later (held-out) half. The genuine features are pure noise, so a naive search locks
/// onto the leak in-sample and collapses OOS. (Mirrors `fixtures.leaky_dataset`.)
fn leaky_dataset(seed: u64) -> Vec<DatasetRow> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(N);
    let half = N / 2;
    for i in 0..N {
        let base = rng.normal();
        let label = clip(base + rng.normal(), -1.5, 1.5); // symmetric: no free drift

        let sign = if i < half { 1.0 } else { -1.0 };
        let leak = sign * base + 0.01 * rng.normal();

        let a = rng.normal();
        let b = rng.normal();
        let c = rng.normal();

        let regime = REGIMES[(rng.next_u64() % 3) as usize];
        let features = BTreeMap::from([
            ("leak__lookahead".to_string(), leak),
            ("momentum__a".to_string(), a),
            ("trend__b".to_string(), b),
            ("volatility__c".to_string(), c),
        ]);
        rows.push(DatasetRow {
            ts: ts(i),
            t1: ts(i + 3),
            label,
            regime: Some(regime.to_string()),
            features,
        });
    }
    rows
}

/// Probe the worker; return a ready harness, or `None` (skip) if it's not reachable/healthy.
async fn harness_if_up() -> Option<ValidationHarness> {
    let client = MlClient::from_env().ok()?;
    match client.health().await {
        Ok(h) if h.status == "ok" => {
            let dir = std::env::temp_dir().join("se_validation_p4");
            Some(ValidationHarness::new(client, dir))
        }
        _ => None,
    }
}

#[tokio::test]
async fn p4_checkpoint_leak_rejected_genuine_passes() {
    let harness = match harness_if_up().await {
        Some(h) => h,
        None => {
            eprintln!(
                "SKIP: ml-worker not reachable at {} (set ML_WORKER_URL and start the sidecar). \
                 This is expected when the sidecar is absent; not a failure.",
                MlClient::from_env()
                    .map(|c| c.base_url().to_string())
                    .unwrap_or_default()
            );
            return;
        }
    };

    let profile = HorizonProfile::swing();

    // --- (b) LEAKY dataset: must be REJECTED ---
    let leaky = leaky_dataset(2);
    let leaky_out = harness
        .evaluate(
            &leaky,
            &profile,
            "p4_leaky.parquet",
            N_GROUPS,
            K_TEST_GROUPS,
            N_TRIALS,
        )
        .await
        .expect("leaky validation call failed");

    println!(
        "[P4] LEAKY:   dsr={:.4} pbo={:.4} oos_exp={:.4} n_regimes_pos={} passed={} reasons={:?}",
        leaky_out.validation.dsr,
        leaky_out.validation.pbo,
        leaky_out.validation.oos_expectancy_cost_aware,
        leaky_out.validation.n_regimes_positive,
        leaky_out.decision.passed,
        leaky_out.decision.reasons,
    );

    // --- (a) GENUINE-EDGE dataset: must PASS ---
    let genuine = genuine_edge_dataset(0);
    let genuine_out = harness
        .evaluate(
            &genuine,
            &profile,
            "p4_genuine.parquet",
            N_GROUPS,
            K_TEST_GROUPS,
            N_TRIALS,
        )
        .await
        .expect("genuine validation call failed");

    println!(
        "[P4] GENUINE: dsr={:.4} pbo={:.4} oos_exp={:.4} n_regimes_pos={} passed={} reasons={:?}",
        genuine_out.validation.dsr,
        genuine_out.validation.pbo,
        genuine_out.validation.oos_expectancy_cost_aware,
        genuine_out.validation.n_regimes_positive,
        genuine_out.decision.passed,
        genuine_out.decision.reasons,
    );

    // ASSERT: the leak is REJECTED by the gate. This is the primary checkpoint — the
    // promotion gate must NOT pass a look-ahead-leaky strategy.
    assert!(
        !leaky_out.decision.passed,
        "LEAK MUST BE REJECTED by the gate, got {:?} for {:?}",
        leaky_out.decision, leaky_out.validation
    );

    // ASSERT: the rejection reflects a genuine OUT-OF-SAMPLE COLLAPSE (not a spurious
    // failure of an unrelated condition). The leak's edge evaporates once the held-out
    // half flips sign, so at least one of the edge metrics must be non-promotable:
    //   * cost-aware OOS expectancy goes non-positive (the headline collapse), OR
    //   * the deflated Sharpe deflates away (dsr <= 0.5, matching the Python fixture's
    //     `assert res.dsr <= 0.5`), OR
    //   * PBO crosses the coin-flip line (>= 0.5).
    let v = &leaky_out.validation;
    let collapsed = v.oos_expectancy_cost_aware <= 0.0 || v.dsr <= 0.5 || v.pbo >= 0.5;
    assert!(
        collapsed,
        "leak OOS-collapse signature expected (oos_exp<=0 or dsr<=0.5 or pbo>=0.5), \
         got dsr={} pbo={} oos_exp={}",
        v.dsr, v.pbo, v.oos_expectancy_cost_aware
    );
    // The concrete collapse here is the most direct: cost-aware OOS expectancy is negative.
    assert!(
        v.oos_expectancy_cost_aware <= 0.0,
        "leak cost-aware OOS expectancy should collapse non-positive, got {}",
        v.oos_expectancy_cost_aware
    );

    // ASSERT: genuine edge PASSES (proves the harness is not rejecting everything).
    assert!(
        genuine_out.decision.passed,
        "GENUINE EDGE MUST PASS the gate, got {:?} for {:?}",
        genuine_out.decision, genuine_out.validation
    );

    println!("[P4] OK: leak REJECTED, genuine-edge PASSED.");
}
