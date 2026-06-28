//! Pure indicator math shared across feature layers. No I/O, no PIT concerns —
//! inputs are already PIT-clean slices pulled via a `PitContext`.

use se_core::Bar;

/// Simple bar-to-bar arithmetic returns from a close series.
pub fn simple_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .map(|w| if w[0] == 0.0 { 0.0 } else { w[1] / w[0] - 1.0 })
        .collect()
}

/// Natural-log returns from a close series.
pub fn log_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .filter(|w| w[0] > 0.0 && w[1] > 0.0)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Population standard deviation.
pub fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Annualized realized volatility (daily bars; 252 trading days) over the last
/// `lookback` returns.
pub fn realized_vol(closes: &[f64], lookback: usize) -> f64 {
    let rets = log_returns(closes);
    let n = rets.len();
    if n == 0 {
        return 0.0;
    }
    let slice = &rets[n.saturating_sub(lookback)..];
    stddev(slice) * (252f64).sqrt()
}

/// Simple moving average of the last `n` values.
pub fn sma(vals: &[f64], n: usize) -> Option<f64> {
    if n == 0 || vals.len() < n {
        return None;
    }
    Some(vals[vals.len() - n..].iter().sum::<f64>() / n as f64)
}

/// Exponential moving average over the whole series with span `n`.
pub fn ema(vals: &[f64], n: usize) -> Option<f64> {
    if n == 0 || vals.is_empty() {
        return None;
    }
    let k = 2.0 / (n as f64 + 1.0);
    let mut e = vals[0];
    for &v in &vals[1..] {
        e = v * k + e * (1.0 - k);
    }
    Some(e)
}

/// Wilder's Average True Range over `lookback` bars (needs `lookback + 1` bars).
pub fn atr(bars: &[Bar], lookback: usize) -> Option<f64> {
    if lookback == 0 || bars.len() <= lookback {
        return None;
    }
    let trs: Vec<f64> = bars
        .windows(2)
        .map(|w| w[1].true_range(w[0].close))
        .collect();
    let slice = &trs[trs.len() - lookback..];
    Some(slice.iter().sum::<f64>() / lookback as f64)
}

/// Average daily dollar volume (close * volume) over the last `lookback` bars.
pub fn dollar_adv(bars: &[Bar], lookback: usize) -> f64 {
    if bars.is_empty() {
        return 0.0;
    }
    let slice = &bars[bars.len().saturating_sub(lookback)..];
    slice.iter().map(|b| b.close * b.volume).sum::<f64>() / slice.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_and_vol() {
        let closes = [100.0, 101.0, 102.0, 101.0, 103.0];
        let r = simple_returns(&closes);
        assert_eq!(r.len(), 4);
        assert!(realized_vol(&closes, 4) >= 0.0);
    }

    #[test]
    fn sma_ema_basic() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(sma(&v, 2), Some(3.5));
        assert!(ema(&v, 2).is_some());
        assert_eq!(sma(&v, 9), None);
    }
}
