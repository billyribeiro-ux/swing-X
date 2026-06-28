# ADR 0001 — Runtime SQL queries (vs. sqlx compile-time-checked macros)

- Status: Accepted
- Date: 2026-06-28
- Scope: all crates touching Postgres/TimescaleDB via `sqlx`

## Context

The data layer uses `sqlx` with **runtime** queries — `sqlx::query("…")` / `query_scalar`
plus `.bind(…)` — rather than the compile-time-checked `query!` / `query_as!` macros. The
macros validate SQL against a live database (or committed offline `.sqlx` metadata from
`cargo sqlx prepare`) at build time, catching schema drift and type mismatches before the
binary is produced. We forgo that build-time guarantee.

The query surface is small and centralized: ~11 call sites across 8 crates
(`se-store`, `se-api`, `se-search`, `se-cli`, `se-journal`, `se-monitor`, `se-regime`,
`se-signal`), most of them in `se-store` and the read-model API.

## Decision

Keep runtime queries. Do **not** require a live database (or committed offline query
metadata) at compile time.

## Rationale

- **Build ergonomics.** `query!` macros require either a reachable `DATABASE_URL` at
  `cargo build`/`clippy` time or a committed `.sqlx` offline cache that must be regenerated
  (and reviewed) on every schema change. That couples a pure `cargo build` to database state
  or to a generated artifact, which is friction for a fast-moving research engine.
- **The risk it guards against is already caught.** Schema drift surfaces as a **test
  failure today**: `rust.yml` spins a fresh `timescaledb:latest-pg16` service, applies all
  migrations, and `cargo test --workspace` executes every real query against it (the PIT
  store, validation, journal, monitor, and API read-model tests all hit the DB). A column
  rename or type change breaks those tests in CI — the same signal the macros would give,
  one stage later.
- **Small, centralized surface.** With ~11 call sites, the marginal safety of compile-time
  checking over the existing integration coverage is modest.

## Consequences

- Drift is caught at **test time in CI**, not at compile time locally. A developer can
  `cargo build` a query that is wrong for the current schema; `cargo test` (local or CI)
  will fail it.
- The PIT invariant is **not** weakened by this choice: every feature read still funnels
  through `se-store`'s `PitQuery`, whose `as_of_ts <= decision_ts` predicate is enforced in
  Rust regardless of how the SQL is type-checked.

## Revisiting

If the query surface grows substantially or schema churn becomes a source of CI flakiness,
migrate to `query!`/`query_as!` with a committed offline cache:

1. `cargo sqlx prepare --workspace` against a migrated DB to generate `.sqlx/`.
2. Commit `.sqlx/` and add `SQLX_OFFLINE=true` to the build.
3. Add a CI step `cargo sqlx prepare --check` so a schema change without a regenerated cache
   fails the build.

Until then, the DB-backed integration tests are the accepted mitigation.
