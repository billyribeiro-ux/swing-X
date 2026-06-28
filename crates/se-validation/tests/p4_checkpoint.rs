//! THE P4 CHECKPOINT (Rust side, end-to-end against the live ml-worker).
//!
//! Runs the two `se_validation::fixtures` datasets through the [`ValidationHarness`]:
//!   (a) a GENUINE-EDGE dataset — features causally drive the label across the whole
//!       timeline, so a purged+embargoed CPCV model keeps an edge out-of-sample;
//!   (b) a LEAKY dataset — `leak__lookahead` tracks the label in-sample then flips sign
//!       on the held-out half: spectacular in-sample, collapses out-of-sample.
//!
//! ASSERTIONS: the LEAKY dataset is REJECTED by the gate; the GENUINE-EDGE dataset PASSES.
//! Mirrors `ml-worker/tests/test_leakage_fixture.py` through the real HTTP boundary.
//! Skips with a clear message when the worker is unreachable (absence never fails CI).

use se_core::HorizonProfile;
use se_mlclient::MlClient;
use se_validation::fixtures::{genuine_edge_dataset, leaky_dataset};
use se_validation::ValidationHarness;

const N_TRIALS: u32 = 16;
const N_GROUPS: u32 = 6;
const K_TEST_GROUPS: u32 = 2;

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
                 Expected when the sidecar is absent; not a failure.",
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

    // The leak MUST be rejected by the gate (primary checkpoint).
    assert!(
        !leaky_out.decision.passed,
        "LEAK MUST BE REJECTED by the gate, got {:?} for {:?}",
        leaky_out.decision, leaky_out.validation
    );
    // ...and the rejection must reflect a genuine OUT-OF-SAMPLE collapse, not a spurious
    // failure of an unrelated condition.
    let v = &leaky_out.validation;
    assert!(
        v.oos_expectancy_cost_aware <= 0.0 || v.dsr <= 0.5 || v.pbo >= 0.5,
        "leak OOS-collapse signature expected, got dsr={} pbo={} oos_exp={}",
        v.dsr,
        v.pbo,
        v.oos_expectancy_cost_aware
    );

    // Genuine edge MUST pass (proves the harness discriminates, not rejects-everything).
    assert!(
        genuine_out.decision.passed,
        "GENUINE EDGE MUST PASS the gate, got {:?} for {:?}",
        genuine_out.decision, genuine_out.validation
    );

    println!("[P4] OK: leak REJECTED, genuine-edge PASSED.");
}
