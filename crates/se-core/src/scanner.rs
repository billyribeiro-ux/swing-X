//! Which scanner a strategy belongs to. The engine runs two parallel populations over two
//! universes — the curated index/sector ETFs, and individual equities — that are searched,
//! scored, and surfaced independently. A strategy is tagged with its [`Scanner`] so the two
//! never mix on the scoreboard, in the journal, or in promotion.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

/// The scanner (universe family) a strategy was searched and scored on.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Scanner {
    /// The curated 10-ticker index/sector-ETF universe (v1).
    #[default]
    Etf,
    /// Individual US equities (TSLA, AAPL, …) with earnings-blackout handling.
    Equity,
}

impl Scanner {
    pub const ALL: [Scanner; 2] = [Scanner::Etf, Scanner::Equity];

    pub const fn as_str(self) -> &'static str {
        match self {
            Scanner::Etf => "etf",
            Scanner::Equity => "equity",
        }
    }

    /// Human label for the dashboard / CLI headers.
    pub const fn label(self) -> &'static str {
        match self {
            Scanner::Etf => "ETF Scanner",
            Scanner::Equity => "Equity Scanner",
        }
    }
}

impl fmt::Display for Scanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Scanner {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "etf" | "etfs" => Ok(Scanner::Etf),
            "equity" | "equities" | "stock" | "stocks" => Ok(Scanner::Equity),
            other => Err(Error::Parse(format!("unknown scanner: {other}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_default() {
        assert_eq!(Scanner::default(), Scanner::Etf);
        for s in Scanner::ALL {
            assert_eq!(s.as_str().parse::<Scanner>().unwrap(), s);
        }
        assert_eq!("equities".parse::<Scanner>().unwrap(), Scanner::Equity);
        assert!("bonds".parse::<Scanner>().is_err());
    }

    #[test]
    fn serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&Scanner::Equity).unwrap(),
            "\"equity\""
        );
        assert_eq!(
            serde_json::from_str::<Scanner>("\"etf\"").unwrap(),
            Scanner::Etf
        );
    }
}
