//! The feature-engine abstraction. Every layer implements [`FeatureModule`] and
//! reads ONLY through the [`FeatureContext`] — a PIT-safe handle plus the
//! proprietary hooks — so no module can reach live/future data.

use async_trait::async_trait;
use se_core::{Feature, HorizonProfile, Layer, Result};
use se_provider::ProprietaryProvider;
use se_store::PitContext;

/// What a feature module is allowed to see when computing for one decision bar.
pub struct FeatureContext<'a> {
    /// PIT-safe reads (bars + features with `as_of <= decision_ts`).
    pub pit: &'a PitContext<'a>,
    /// Proprietary data hooks; return `Unavailable` unless the operator wired a feed.
    pub proprietary: &'a dyn ProprietaryProvider,
    /// Active horizon profile (barrier/cadence/cost constants).
    pub profile: HorizonProfile,
}

impl<'a> FeatureContext<'a> {
    pub fn new(
        pit: &'a PitContext<'a>,
        proprietary: &'a dyn ProprietaryProvider,
        profile: HorizonProfile,
    ) -> Self {
        FeatureContext {
            pit,
            proprietary,
            profile,
        }
    }
}

#[async_trait]
pub trait FeatureModule: Send + Sync {
    fn layer(&self) -> Layer;
    fn name(&self) -> &str;
    /// Compute this module's features at the context's decision bar.
    async fn compute(&self, ctx: &FeatureContext<'_>) -> Result<Vec<Feature>>;
}
