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

/// Wilder's Average Directional Index (ADX) over `period` bars.
///
/// Trend-strength oscillator in roughly `[0, 100]`: low (<~20) = no trend /
/// chop, high (>~25) = a directional trend (up OR down — ADX is unsigned).
/// Implemented with Wilder's smoothing (RMA), matching the canonical definition:
///   * +DM / -DM from successive highs/lows, true range from `Bar::true_range`,
///   * smoothed +DI / -DI, then DX = 100 * |+DI − −DI| / (+DI + −DI),
///   * ADX = Wilder-smoothed DX.
///
/// Needs at least `2 * period + 1` bars to seed and smooth; returns `None` otherwise.
pub fn adx(bars: &[Bar], period: usize) -> Option<f64> {
    if period == 0 || bars.len() < 2 * period + 1 {
        return None;
    }
    let n = bars.len();
    // Per-bar directional movement and true range (length n-1).
    let mut plus_dm = Vec::with_capacity(n - 1);
    let mut minus_dm = Vec::with_capacity(n - 1);
    let mut tr = Vec::with_capacity(n - 1);
    for w in bars.windows(2) {
        let (prev, cur) = (&w[0], &w[1]);
        let up_move = cur.high - prev.high;
        let down_move = prev.low - cur.low;
        let p = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let m = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };
        plus_dm.push(p);
        minus_dm.push(m);
        tr.push(cur.true_range(prev.close));
    }

    // Wilder-smoothed (RMA) running sums over `period`.
    let seed = |xs: &[f64]| -> f64 { xs[..period].iter().sum() };
    let mut tr_s = seed(&tr);
    let mut plus_s = seed(&plus_dm);
    let mut minus_s = seed(&minus_dm);

    let dx_at = |tr_s: f64, plus_s: f64, minus_s: f64| -> Option<f64> {
        if tr_s <= 0.0 {
            return None;
        }
        let plus_di = 100.0 * plus_s / tr_s;
        let minus_di = 100.0 * minus_s / tr_s;
        let denom = plus_di + minus_di;
        if denom <= 0.0 {
            Some(0.0)
        } else {
            Some(100.0 * (plus_di - minus_di).abs() / denom)
        }
    };

    let mut dxs = Vec::new();
    if let Some(dx) = dx_at(tr_s, plus_s, minus_s) {
        dxs.push(dx);
    }
    // Continue Wilder smoothing across the remaining bars.
    for i in period..tr.len() {
        tr_s = tr_s - tr_s / period as f64 + tr[i];
        plus_s = plus_s - plus_s / period as f64 + plus_dm[i];
        minus_s = minus_s - minus_s / period as f64 + minus_dm[i];
        if let Some(dx) = dx_at(tr_s, plus_s, minus_s) {
            dxs.push(dx);
        }
    }

    if dxs.len() < period {
        // Not enough DX points to seed the ADX average; fall back to mean DX.
        if dxs.is_empty() {
            return None;
        }
        return Some(dxs.iter().sum::<f64>() / dxs.len() as f64);
    }
    // ADX = Wilder-smoothed DX: seed with the first `period` DXs, then RMA.
    let mut adx = dxs[..period].iter().sum::<f64>() / period as f64;
    for &dx in &dxs[period..] {
        adx = (adx * (period as f64 - 1.0) + dx) / period as f64;
    }
    Some(adx)
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

    use se_core::Ticker;

    fn bar(o: f64, h: f64, l: f64, c: f64) -> Bar {
        Bar {
            ticker: Ticker::Spy,
            ts: chrono::Utc::now(),
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 1.0,
        }
    }

    #[test]
    fn adx_needs_enough_bars() {
        // 2*period+1 minimum; below it -> None.
        let bars: Vec<Bar> = (0..10)
            .map(|i| bar(i as f64, i as f64 + 1.0, i as f64 - 1.0, i as f64))
            .collect();
        assert!(adx(&bars, 14).is_none());
    }

    #[test]
    fn adx_high_in_strong_trend_low_in_chop() {
        // Strong, persistent uptrend: each bar steps up by 1, tight ranges.
        let mut up = Vec::new();
        let mut base = 100.0;
        for _ in 0..60 {
            let o = base;
            let c = base + 1.0;
            up.push(bar(o, c + 0.1, o - 0.1, c));
            base = c;
        }
        let adx_up = adx(&up, 14).expect("enough bars");
        assert!(
            adx_up > 40.0,
            "strong trend should have high ADX, got {adx_up}"
        );

        // Chop: oscillate around a level with no net direction.
        let mut chop = Vec::new();
        for i in 0..60 {
            let mid = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
            chop.push(bar(100.0, mid + 0.6, mid - 0.6, mid));
        }
        let adx_chop = adx(&chop, 14).expect("enough bars");
        assert!(
            adx_chop < adx_up,
            "chop ADX ({adx_chop}) must be below trend ADX ({adx_up})"
        );
    }

    #[test]
    fn adx_bounded() {
        let bars: Vec<Bar> = (0..60)
            .map(|i| {
                let c = 100.0 + i as f64;
                bar(c - 0.5, c + 0.5, c - 1.0, c)
            })
            .collect();
        let a = adx(&bars, 14).unwrap();
        assert!(
            (0.0..=100.0).contains(&a),
            "ADX must be in [0,100], got {a}"
        );
    }
}
