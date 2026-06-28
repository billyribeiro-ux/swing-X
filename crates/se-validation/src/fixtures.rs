//! Deterministic datasets for the deliberate-leakage checkpoint (§8 / P4).
//!
//! Two generators with opposite truth:
//!   * [`genuine_edge_dataset`] — features causally drive the label across the whole
//!     timeline, so a purged+embargoed CPCV model keeps an edge out-of-sample.
//!   * [`leaky_dataset`] — a `leak__lookahead` feature equals the label's signal on the
//!     in-sample (early) half then FLIPS SIGN on the held-out (later) half. In-sample it
//!     looks spectacular; out-of-sample it collapses. The promotion gate MUST reject it.
//!
//! Shared by the integration test and the `inject-leak-test` CLI command so the operator
//! runs exactly what CI verifies.

use std::collections::BTreeMap;

use chrono::{DateTime, TimeZone, Utc};
use se_mlclient::DatasetRow;

/// Number of rows in each fixture.
pub const N: usize = 900;
/// Regime labels cycled through the fixtures (for the ≥2-regime gate condition).
pub const REGIMES: [&str; 3] = ["bull", "bear", "chop"];

/// A tiny deterministic PRNG (SplitMix64) — reproducible draws with the right
/// statistical structure (we do not need to reproduce numpy's RNG).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) using a 53-bit mantissa.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
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
/// whole timeline.
pub fn genuine_edge_dataset(seed: u64) -> Vec<DatasetRow> {
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
/// onto the leak in-sample and collapses out-of-sample.
pub fn leaky_dataset(seed: u64) -> Vec<DatasetRow> {
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
