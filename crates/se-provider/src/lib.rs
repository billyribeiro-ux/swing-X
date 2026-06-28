//! `se-provider` — the `DataProvider` abstraction and its adapters.
//!
//! v1 adapters:
//! * [`MockProvider`] — deterministic synthetic data (tests / offline).
//! * `FmpProvider` — Financial Modeling Prep (primary; `/stable` REST).
//! * `FredProvider` — FRED (free; credit spreads + net liquidity).
//!
//! Data FMP/FRED cannot supply (dealer GEX/charm/vanna/walls, tick order-flow, DIX)
//! lives behind [`ProprietaryProvider`] hooks that return `Unavailable` until the
//! operator wires a private feed — the system never fabricates them.

mod fmp;
mod fred;
mod http_util;
mod mock;
mod proprietary;
mod provider;
mod types;

pub use fmp::FmpProvider;
pub use fred::FredProvider;
pub use mock::MockProvider;
pub use proprietary::{GexSnapshot, NullProprietary, ProprietaryProvider};
pub use provider::DataProvider;
pub use types::{Capabilities, EtfProfile, MacroPoint, MacroSeries, ProviderKind, Quote};
