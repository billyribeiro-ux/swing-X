//! A market symbol. Originally a closed 10-ETF enum; now a `Copy` newtype that accepts any
//! valid US ticker, so the engine can scan individual equities as well as the index/sector ETFs.
//!
//! The 10 ETFs remain first-class constants ([`Ticker::SPY`] … [`Ticker::XLU`]), the ETF
//! universe is still [`Ticker::ALL`], and [`Ticker::BENCHMARK`] is SPY — so the existing ETF
//! scanner is unchanged. Arbitrary symbols (e.g. `TSLA`, `AAPL`) are constructed via
//! [`Ticker::new`] / `parse`, validated to a sane symbol shape but no longer restricted to a
//! closed set. The on-the-wire and in-DB representation is the uppercase symbol string.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::Error;

/// Max symbol length we store inline. US tickers are ≤ 5 chars; class shares / some ADRs add a
/// suffix (e.g. `BRK.B`), so 15 bytes is comfortable headroom and keeps `Ticker` `Copy`.
const MAX_LEN: usize = 15;

/// A market symbol — `Copy`, inline (no allocation), comparable and hashable so it can key maps
/// and sort deterministically. Always holds an uppercase ASCII symbol of 1..=15 chars.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct Ticker {
    /// Zero-padded uppercase ASCII bytes; only `bytes[..len]` are meaningful.
    bytes: [u8; MAX_LEN],
    len: u8,
}

impl Ticker {
    /// Construct from a known-good static symbol (used for the ETF constants). Panics in const
    /// evaluation if the string is empty or too long — only ever called with literals below.
    const fn from_static(s: &str) -> Ticker {
        let b = s.as_bytes();
        assert!(
            !b.is_empty() && b.len() <= MAX_LEN,
            "ticker literal out of range"
        );
        let mut bytes = [0u8; MAX_LEN];
        let mut i = 0;
        while i < b.len() {
            bytes[i] = b[i];
            i += 1;
        }
        Ticker {
            bytes,
            len: b.len() as u8,
        }
    }

    // --- The 10-ETF universe (v1), still first-class constants. ---
    pub const SPY: Ticker = Ticker::from_static("SPY");
    pub const QQQ: Ticker = Ticker::from_static("QQQ");
    pub const IWM: Ticker = Ticker::from_static("IWM");
    pub const DIA: Ticker = Ticker::from_static("DIA");
    pub const XLF: Ticker = Ticker::from_static("XLF");
    pub const XLK: Ticker = Ticker::from_static("XLK");
    pub const XLE: Ticker = Ticker::from_static("XLE");
    pub const SMH: Ticker = Ticker::from_static("SMH");
    pub const XLV: Ticker = Ticker::from_static("XLV");
    pub const XLU: Ticker = Ticker::from_static("XLU");

    /// The curated ETF universe, in canonical order (broad indices first, then sectors). This is
    /// the *ETF scanner's* universe — the equity scanner supplies its own list at runtime.
    pub const ALL: [Ticker; 10] = [
        Ticker::SPY,
        Ticker::QQQ,
        Ticker::IWM,
        Ticker::DIA,
        Ticker::XLF,
        Ticker::XLK,
        Ticker::XLE,
        Ticker::SMH,
        Ticker::XLV,
        Ticker::XLU,
    ];

    /// The broad indices (vs. the sector ETFs).
    const BROAD: [Ticker; 4] = [Ticker::SPY, Ticker::QQQ, Ticker::IWM, Ticker::DIA];

    /// The broad-market benchmark used for relative-strength features and market breadth.
    pub const BENCHMARK: Ticker = Ticker::SPY;

    /// Validate and construct an arbitrary symbol. Accepts 1..=15 chars of `A–Z 0–9 . -`
    /// (case-insensitive on input; stored uppercase). Rejects empty/oversized/garbage symbols.
    pub fn new(s: &str) -> Result<Ticker, Error> {
        let up = s.trim().to_ascii_uppercase();
        let ok = !up.is_empty()
            && up.len() <= MAX_LEN
            && up
                .bytes()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'.' || c == b'-');
        if !ok {
            return Err(Error::UnknownTicker(s.to_string()));
        }
        let mut bytes = [0u8; MAX_LEN];
        bytes[..up.len()].copy_from_slice(up.as_bytes());
        Ok(Ticker {
            bytes,
            len: up.len() as u8,
        })
    }

    /// The uppercase symbol string. Borrowed from the inline buffer.
    pub fn as_str(&self) -> &str {
        // SAFETY-equivalent: bytes[..len] is always valid ASCII by construction.
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }

    /// True for the broad indices (SPY/QQQ/IWM/DIA); false for sectors and individual equities.
    pub fn is_broad_index(self) -> bool {
        Ticker::BROAD.contains(&self)
    }

    /// True if this symbol is one of the known v1 ETFs (vs. an individual equity).
    pub fn is_etf(self) -> bool {
        Ticker::ALL.contains(&self)
    }
}

impl fmt::Debug for Ticker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ticker({})", self.as_str())
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
        Ticker::new(s)
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
        Ticker::new(&s)
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
        assert_eq!("spy".parse::<Ticker>().unwrap(), Ticker::SPY);
        assert_eq!(" QqQ ".parse::<Ticker>().unwrap(), Ticker::QQQ);
    }

    #[test]
    fn accepts_arbitrary_equities() {
        // The universe is no longer closed: individual stocks now parse.
        for sym in ["TSLA", "AAPL", "META", "NVDA", "BRK.B"] {
            let t: Ticker = sym.parse().unwrap();
            assert_eq!(t.as_str(), sym);
        }
        assert_eq!("tsla".parse::<Ticker>().unwrap().as_str(), "TSLA");
    }

    #[test]
    fn rejects_garbage_symbols() {
        assert!("".parse::<Ticker>().is_err());
        assert!("  ".parse::<Ticker>().is_err());
        assert!("TS LA".parse::<Ticker>().is_err()); // space
        assert!("WAYTOOLONGSYMBOL".parse::<Ticker>().is_err()); // > 15 chars
        assert!("AB$C".parse::<Ticker>().is_err()); // illegal char
    }

    #[test]
    fn benchmark_is_broad_and_etf() {
        assert!(Ticker::BENCHMARK.is_broad_index());
        assert!(Ticker::BENCHMARK.is_etf());
        assert!(!"TSLA".parse::<Ticker>().unwrap().is_etf());
        assert!(!Ticker::XLF.is_broad_index()); // sector ETF
    }

    #[test]
    fn ordering_is_deterministic_and_lexicographic() {
        let mut v = ["TSLA", "AAPL", "SPY", "META"]
            .map(|s| s.parse::<Ticker>().unwrap())
            .to_vec();
        v.sort();
        assert_eq!(
            v.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
            ["AAPL", "META", "SPY", "TSLA"]
        );
    }
}
