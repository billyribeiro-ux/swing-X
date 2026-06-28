//! Strongly-typed UUID identifiers, so a `StrategyId` can never be passed where a
//! `SignalId` is expected.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

macro_rules! id_type {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                $name(Uuid::new_v4())
            }
            pub fn from_uuid(u: Uuid) -> Self {
                $name(u)
            }
            pub fn inner(self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id_type!(
    /// Identifies a candidate/promoted strategy in the population.
    StrategyId
);
id_type!(
    /// Identifies a surfaced, executable signal.
    SignalId
);
id_type!(
    /// Identifies a fitted model artifact in the registry.
    ModelId
);
id_type!(
    /// Identifies a paper/live trade in the journal.
    TradeId
);
id_type!(
    /// Identifies a triple-barrier label.
    LabelId
);
