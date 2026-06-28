//! `inject-leak-test` — the §8 / P4 deliberate-leakage checkpoint, operator-runnable.
//!
//! Runs the same two `se_validation::fixtures` datasets that CI verifies through the live
//! ML worker: a genuine-edge dataset (must PASS the gate) and a look-ahead-leaky dataset
//! (must be REJECTED). If the harness ever PROMOTES the leak, the harness is broken — this
//! command exits non-zero so it can gate a pipeline.

use anyhow::{bail, Result};
use se_core::HorizonProfile;
use se_mlclient::MlClient;
use se_validation::fixtures::{genuine_edge_dataset, leaky_dataset};
use se_validation::{HarnessOutcome, ValidationHarness};

const N_TRIALS: u32 = 16;
const N_GROUPS: u32 = 6;
const K_TEST_GROUPS: u32 = 2;

fn print_row(tag: &str, o: &HarnessOutcome) {
    let v = &o.validation;
    println!(
        "  {tag:<8} dsr={:.4} pbo={:.4} oos_exp={:+.4} regimes_pos={} gate={}",
        v.dsr,
        v.pbo,
        v.oos_expectancy_cost_aware,
        v.n_regimes_positive,
        if o.decision.passed { "PASS" } else { "REJECT" },
    );
    if !o.decision.reasons.is_empty() {
        for r in &o.decision.reasons {
            println!("           • {r}");
        }
    }
}

pub async fn run() -> Result<()> {
    let client = MlClient::from_env().map_err(|e| anyhow::anyhow!("ml client: {e}"))?;
    let base = client.base_url().to_string();

    // Fail clearly (not silently) if the sidecar isn't up.
    match client.health().await {
        Ok(h) if h.status == "ok" => {}
        _ => bail!(
            "ML worker not reachable at {base}. Start it first:\n  \
             cd ml-worker && uv run uvicorn se_ml.server:app --port 8088"
        ),
    }

    let dir = std::env::temp_dir().join("se_inject_leak_test");
    let harness = ValidationHarness::new(client, dir);
    let profile = HorizonProfile::swing();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" inject-leak-test │ worker={base} │ horizon=swing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let leaky = harness
        .evaluate(&leaky_dataset(2), &profile, "leak_leaky.parquet", N_GROUPS, K_TEST_GROUPS, N_TRIALS)
        .await
        .map_err(|e| anyhow::anyhow!("leaky validation failed: {e}"))?;
    print_row("LEAKY", &leaky);

    let genuine = harness
        .evaluate(&genuine_edge_dataset(0), &profile, "leak_genuine.parquet", N_GROUPS, K_TEST_GROUPS, N_TRIALS)
        .await
        .map_err(|e| anyhow::anyhow!("genuine validation failed: {e}"))?;
    print_row("GENUINE", &genuine);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let leak_caught = !leaky.decision.passed;
    let genuine_ok = genuine.decision.passed;
    if leak_caught && genuine_ok {
        println!(" ✓ CHECKPOINT PASS: leak REJECTED, genuine-edge PASSED — the harness has teeth.");
        Ok(())
    } else {
        println!(" ✗ CHECKPOINT FAIL:");
        if !leak_caught {
            println!("   the LEAKY dataset was PROMOTED — the validation harness is broken. Fix before anything else.");
        }
        if !genuine_ok {
            println!("   the GENUINE-EDGE dataset was REJECTED — the harness rejects everything (useless).");
        }
        bail!("leakage checkpoint failed");
    }
}
