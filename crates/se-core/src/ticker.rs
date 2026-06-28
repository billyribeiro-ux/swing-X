//! The fixed v1 universe. A closed enum so the universe can't silently drift,
//! and so every match over tickers is exhaustive.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

/// The curated 10-ticker US equity-index ETF universe (v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub enum Ticker {
    Spy,
    Qqq,
    Iwm,
    Dia,
    Xlf,
    Xlk,
    Xle,
    Smh,
    Xlv,
    Xlu,
}

impl Ticker {
    /// All universe members, in canonical order (broad indices first, then sectors).
    pub const ALL: [Ticker; 10] = [
        Ticker::Spy,
        Ticker::Qqq,
        Ticker::Iwm,
        Ticker::Dia,
        Ticker::Xlf,
        Ticker::Xlk,
        Ticker::Xle,
        Ticker::Smh,
        Ticker::Xlv,
        Ticker::Xlu,
    ];

    /// The broad-market benchmark used for relative-strength features.
    pub const BENCHMARK: Ticker = Ticker::Spy;

    pub const fn as_str(self) -> &'static str {
        match self {
            Ticker::Spy => "SPY",
            Ticker::Qqq => "QQQ",
            Ticker::Iwm => "IWM",
            Ticker::Dia => "DIA",
            Ticker::Xlf => "XLF",
            Ticker::Xlk => "XLK",
            Ticker::Xle => "XLE",
            Ticker::Smh => "SMH",
            Ticker::Xlv => "XLV",
            Ticker::Xlu => "XLU",
        }
    }

    /// True for the broad indices (vs. the sector ETFs).
    pub const fn is_broad_index(self) -> bool {
        matches!(self, Ticker::Spy | Ticker::Qqq | Ticker::Iwm | Ticker::Dia)
    }
}

impl fmt::Display for Ticker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Ticker {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let up = s.trim().to_ascii_uppercase();
        Ticker::ALL
            .into_iter()
            .find(|t| t.as_str() == up)
            .ok_or_else(|| Error::UnknownTicker(s.to_string()))
    }
}

impl From<Ticker> for String {
    fn from(t: Ticker) -> Self {
        t.as_str().to_string()
    }
}

impl TryFrom<String> for Ticker {
    type Error = Error;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all() {
        for t in Ticker::ALL {
            assert_eq!(t.as_str().parse::<Ticker>().unwrap(), t);
        }
    }

    #[test]
    fn case_insensitive_parse() {
        assert_eq!("spy".parse::<Ticker>().unwrap(), Ticker::Spy);
        assert_eq!(" QqQ ".parse::<Ticker>().unwrap(), Ticker::Qqq);
    }

    #[test]
    fn unknown_rejected() {
        assert!("TSLA".parse::<Ticker>().is_err());
    }

    #[test]
    fn benchmark_is_broad() {
        assert!(Ticker::BENCHMARK.is_broad_index());
    }
}
