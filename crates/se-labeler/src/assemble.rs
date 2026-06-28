//! Assembling labeled events + per-entry features into the rows `se-mlclient` writes.
//!
//! The crucial invariant: each row's `t1` is the **barrier-end timestamp** ([`LabelEvent::t1`]),
//! so CPCV on the Python side can purge any training label whose window overlaps a test
//! span. The purge length therefore equals the label horizon (`max_hold_bars`) by
//! construction — the [`se_core::HorizonProfile`] is the single source of both.

use std::collections::BTreeMap;

use se_mlclient::DatasetRow;

use crate::triple_barrier::LabelEvent;

/// One assembled entry: a resolved [`LabelEvent`], its feature map (`layer__feature` ->
/// value), and an optional regime tag.
#[derive(Debug, Clone)]
pub struct LabeledEntry {
    pub event: LabelEvent,
    pub features: BTreeMap<String, f64>,
    pub regime: Option<String>,
}

/// Convert assembled entries into [`DatasetRow`]s ready for [`se_mlclient::write_dataset`].
///
/// `ts` is the entry timestamp, `t1` is the barrier-end timestamp (for purging), and
/// `label` is the realized R return. Rows are returned in input order; the caller is
/// responsible for providing entries sorted ascending by entry time (the Parquet writer
/// re-checks this and errors otherwise).
pub fn assemble_dataset(entries: &[LabeledEntry]) -> Vec<DatasetRow> {
    entries
        .iter()
        .map(|e| DatasetRow {
            ts: e.event.entry_ts,
            t1: e.event.t1,
            label: e.event.ret_r,
            regime: e.regime.clone(),
            features: e.features.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use se_core::Side;

    use crate::triple_barrier::Outcome;

    fn event(entry_secs: i64, t1_secs: i64, ret_r: f64) -> LabelEvent {
        LabelEvent {
            entry_ts: Utc.timestamp_opt(entry_secs, 0).unwrap(),
            t1: Utc.timestamp_opt(t1_secs, 0).unwrap(),
            side: Side::Long,
            entry_px: 100.0,
            target_px: 102.0,
            stop_px: 99.0,
            outcome: Outcome::Target,
            ret_r,
        }
    }

    #[test]
    fn t1_is_barrier_end_and_label_is_ret_r() {
        let entries = vec![LabeledEntry {
            event: event(1_000, 1_300, 2.0),
            features: BTreeMap::from([("momentum__a".to_string(), 0.5)]),
            regime: Some("bull".to_string()),
        }];
        let rows = assemble_dataset(&entries);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ts, Utc.timestamp_opt(1_000, 0).unwrap());
        assert_eq!(rows[0].t1, Utc.timestamp_opt(1_300, 0).unwrap());
        assert_eq!(rows[0].label, 2.0);
        assert_eq!(rows[0].regime.as_deref(), Some("bull"));
        assert_eq!(rows[0].features.get("momentum__a"), Some(&0.5));
    }

    #[test]
    fn preserves_input_order() {
        let entries = vec![
            LabeledEntry {
                event: event(1_000, 1_300, 1.0),
                features: BTreeMap::new(),
                regime: None,
            },
            LabeledEntry {
                event: event(2_000, 2_300, -1.0),
                features: BTreeMap::new(),
                regime: None,
            },
        ];
        let rows = assemble_dataset(&entries);
        assert_eq!(rows[0].label, 1.0);
        assert_eq!(rows[1].label, -1.0);
    }
}
