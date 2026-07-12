# ML / Signal-Generation Engine — Audit & Improvement Roadmap

Produced by an 80-agent adversarial audit (audit → design → adversarial-refute → synthesize),
corroborated by a direct forward-in-time holdout measurement. **Every proposal that claimed to
RAISE headline OOS precision was refuted and the refutations were correct.** The durable work is
to make the reported number *honest* and to *tighten selection* — not to chase a higher precision.

## The core finding (hard evidence)

The reported ~0.51 avg / 0.73 best OOS precision is **not a forward number**. It is the maximum of
a large, undeflated genetic search over ONE frozen partition (`server.py` `mid = t//2`, reused to
rank every genome and every nightly run), drawn from a present-day survivor universe
(`fmp.rs:441` `isActivelyTrading & marketCap>$10B`) in a single 2020–2026 bull window
(`search_cmd.rs:109` `from = to − 730d`). **It is a training-set statistic wearing an OOS label.**

Empirical confirmation: on a strict train-past / test-future split, CPCV precision ~0.36 collapsed
to **forward precision ~0.28 with negative forward expectancy (−0.23R)** — the edge does not survive
forward testing. Two independent lines of evidence converge.

## Honest ceiling

A durable, cost-aware, survivorship-free, multi-regime OOS precision — P(net profit | acted) at a
selective τ\* — realistically tops out around **0.55–0.62**, NOT 0.73. After de-biasing, the honest
average likely settles near net break-even (~0.5). Reasons: semi-strong efficiency at the most
liquid megacaps; costs/execution geometry the labeler doesn't see (enters at close, fills next-bar
open); and the mechanical precision↔expectancy trade-off. **"Win rate beyond human" is a category
error** — a high hit rate is trivially manufacturable with tiny targets/wide stops (net-negative),
which is exactly why `win_rate` is banned. The objective is cost-aware **expectancy** with a
lower-bounded, regime-robust precision.

## Ranked roadmap (all adversarially verified; none add overfit surface)

| # | Item | Effort | Overfit risk | Status |
|---|------|--------|--------------|--------|
| 1 | **Locked out-of-time TEST era**, firewalled from all search+nightly, scored once, REPORT-ONLY (the selection-bias meter). Purge by label `t1`, not entry-ts. | M | zero (measures only) | ✅ DONE end-to-end: firewall (`f058cdd`) + `se test-era-score` (`befedd8`) + **first calibration recorded** (era 2025-07-01→2026-07-09; see below) |
| 2 | **Wilson lower-bound precision gate** + per-regime acted-sufficiency floor (kill optimizer's-curse on n≈8 cohorts). | S–M | low | ✅ done (`97e570f`) — Wilson LB gate on promote + live floor |
| 3 | **Precision NET of cost** (`threshold_r = cost`) wired into conviction + live floor. | S | zero | ✅ done (`607d9e3`) |
| 4 | **Deflate DSR for the TRUE cumulative genome count** + set a real DSR threshold (currently `dsr>0`, always true). | M | low | true-count deflation ✅ shipped (`e3e3979`, `befedd8`); threshold still `dsr>0` — the first calibration sample (2 clean strategies, n_test 7/15) is too small to set a bar from; accumulate more once-only measurements first |
| 5 | **Aggregate combinatorial CPCV paths**; split on calendar dates, not row-count. | S | low | ✅ done (`84f3185`) — path-averaged proba (calendar split still open) |
| 6 | **Point-in-time, survivorship-free universe** (delisted names, as-of membership, vintaged fundamentals) — the de-bias keystone. | XL | zero (de-bias) | blocked on data vendor (CRSP/Norgate/Sharadar) |
| 7 | **Deep multi-regime history** (≥10–15y incl. 2008/2018/2022 bears), GATED on #6; macro-regime tag report-only. | M | low | blocked on #6 |
| 8 | **Sample-uniqueness weighting + effective-N** into DSR/PBO (overlapping-label honesty). | M | low | ✅ done (`84f3185`) — LdP weights in fit + effective-N into DSR |

## Explicitly OFF the roadmap (refuted as sub-noise lift that adds overfit surface)

Cross-sectional feature standardization; vol-regime features; microstructure proxies; a stacked
primary→secondary meta-model; calibration wiring; fractional-Kelly sizing; conviction shrinkage.
Each was refuted as either below the noise floor, downstream of the real problem (selection), or
adding fit surface without durable benefit.

## First recorded calibration (era 2025-07-01 → 2026-07-09, data through 2026-07-09)

Clean set = strategies promoted by FIREWALLED searches (training + selection never saw the era),
each scored exactly once in-era (`test_era_scores`):

| Strategy | Reported OOS (pre-era) | Locked era (never seen) | Gap |
|---|---|---|---|
| `96540337` | prec 0.624, +1.23R (n=133) | **prec 0.638, +1.27R (n=116)** | −0.014 |
| `b4336198` | prec 0.588, +0.66R (n=233) | prec 0.429, +0.79R (n=7) | +0.159 |
| `d524a599` | prec 0.525, +0.22R (n=221) | prec 0.533, +0.34R (n=15) | −0.008 |

Reading: all three held POSITIVE cost-aware expectancy in a full unseen year; the lead strategy
held precision ~0.64 over a substantive in-era cohort (n=116; Wilson LB ≈ 0.55). Mean clean gap
≈ +0.05 — the discipline stack (Wilson-LB promote, net-of-cost precision, regime floor,
true-count DSR, uniqueness weights, calendar split) is selecting strategies whose reported
numbers approximately survive forward. Contrast: the pre-hardening engine's promotions collapsed
to ~0.28 precision / negative expectancy on the same style of forward test.

Caveats that stand: ONE era (itself broadly favorable), a survivor universe, 3 clean strategies.
The DSR bar stays `dsr>0` until more eras/strategies accumulate; the decisive stress (a real
bear era) is blocked on the survivorship-free multi-regime dataset (#6/#7).

## What "done based on hard evidence" means here

Not a higher precision number. It means: (a) a locked test era that *measures* the selection gap;
(b) selection discipline (Wilson LB, true-count DSR deflation, effective-N) that shrinks the gap;
(c) a survivorship-free, multi-regime dataset so the number is *believable forward*. The engine's
value is that it refuses to fool itself — this roadmap makes it refuse harder.
