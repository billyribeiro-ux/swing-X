//! Writing a labeled feature matrix to Parquet at the Rust->Python handoff.
//!
//! The on-disk schema is the contract the Python `se_ml.io_arrow` module reads
//! (`ml-worker/src/se_ml/io_arrow.py`). REQUIRED columns:
//!
//!   * `ts`     — event timestamp (sorted ascending), microsecond UTC timestamp.
//!   * `t1`     — label-window end (barrier-touch) timestamp, used for CPCV purging.
//!   * `label`  — realized return in R units (`f64`).
//!   * `regime` — optional categorical regime tag (string) for the "positive in >= 2
//!     regimes" gate condition. Omitted entirely if no row carries one.
//!   * feature columns named `layer__feature` (`f64`), one per distinct feature key.
//!
//! Feature column order is STABLE (the union of all feature keys, sorted), so repeated
//! writes of the same logical dataset produce byte-identical schemas — important for
//! reproducible model fingerprints downstream.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use parquet::arrow::ArrowWriter;

use crate::error::MlError;

/// Reserved (non-feature) column names; mirrors `se_ml.io_arrow._RESERVED`.
pub const TS_COL: &str = "ts";
pub const T1_COL: &str = "t1";
pub const LABEL_COL: &str = "label";
pub const REGIME_COL: &str = "regime";

/// One labeled observation: a point in time, its label window, its R-unit label, an
/// optional regime tag, and its feature map (`layer__feature` -> value).
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetRow {
    /// Event timestamp. Rows MUST be provided sorted ascending by `ts`.
    pub ts: DateTime<Utc>,
    /// Label-window end (barrier touch) timestamp, for purging.
    pub t1: DateTime<Utc>,
    /// Realized return in R units.
    pub label: f64,
    /// Optional regime tag (e.g. `"bull"`).
    pub regime: Option<String>,
    /// Feature values keyed by `layer__feature` name.
    pub features: BTreeMap<String, f64>,
}

/// Collect the stable, sorted union of feature column names across all rows.
fn feature_columns(rows: &[DatasetRow]) -> Vec<String> {
    let mut set: BTreeSet<&str> = BTreeSet::new();
    for r in rows {
        for k in r.features.keys() {
            set.insert(k.as_str());
        }
    }
    set.into_iter().map(|s| s.to_string()).collect()
}

/// Build the Arrow [`Schema`] for `rows`. `regime` is included only if any row carries one.
fn build_schema(feature_cols: &[String], with_regime: bool) -> Schema {
    let tz: Option<Arc<str>> = None;
    let mut fields = vec![
        Field::new(
            TS_COL,
            DataType::Timestamp(TimeUnit::Microsecond, tz.clone()),
            false,
        ),
        Field::new(
            T1_COL,
            DataType::Timestamp(TimeUnit::Microsecond, tz),
            false,
        ),
        Field::new(LABEL_COL, DataType::Float64, false),
    ];
    if with_regime {
        // Nullable: rows without a regime carry null.
        fields.push(Field::new(REGIME_COL, DataType::Utf8, true));
    }
    for c in feature_cols {
        // Feature values are nullable so a missing key for a given row is expressible.
        fields.push(Field::new(c, DataType::Float64, true));
    }
    Schema::new(fields)
}

/// Write `rows` to a Parquet file at `path`, returning the resolved path.
///
/// Validates that `ts` is sorted ascending (a precondition of the CPCV purging logic on
/// the Python side) and that there is at least one row. Feature column order is stable.
pub fn write_dataset(rows: &[DatasetRow], path: impl AsRef<Path>) -> Result<PathBuf, MlError> {
    if rows.is_empty() {
        return Err(MlError::Dataset("cannot write an empty dataset".into()));
    }
    for w in rows.windows(2) {
        if w[1].ts < w[0].ts {
            return Err(MlError::Dataset(format!(
                "dataset `ts` must be sorted ascending; {} < {}",
                w[1].ts, w[0].ts
            )));
        }
    }

    let feature_cols = feature_columns(rows);
    let with_regime = rows.iter().any(|r| r.regime.is_some());
    let schema = Arc::new(build_schema(&feature_cols, with_regime));

    let ts: TimestampMicrosecondArray =
        rows.iter().map(|r| Some(r.ts.timestamp_micros())).collect();
    let t1: TimestampMicrosecondArray =
        rows.iter().map(|r| Some(r.t1.timestamp_micros())).collect();
    let label: Float64Array = rows.iter().map(|r| Some(r.label)).collect();

    let mut columns: Vec<ArrayRef> = vec![Arc::new(ts), Arc::new(t1), Arc::new(label)];

    if with_regime {
        let regime: StringArray = rows.iter().map(|r| r.regime.as_deref()).collect();
        columns.push(Arc::new(regime));
    }

    for c in &feature_cols {
        let col: Float64Array = rows.iter().map(|r| r.features.get(c).copied()).collect();
        columns.push(Arc::new(col));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| MlError::Dataset(format!("failed to build record batch: {e}")))?;

    let out = path.as_ref().to_path_buf();
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| MlError::Dataset(format!("create_dir_all {parent:?}: {e}")))?;
        }
    }
    let file = File::create(&out).map_err(|e| MlError::Dataset(format!("create {out:?}: {e}")))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| MlError::Dataset(format!("parquet writer: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| MlError::Dataset(format!("parquet write: {e}")))?;
    writer
        .close()
        .map_err(|e| MlError::Dataset(format!("parquet close: {e}")))?;

    Ok(out)
}

/// Convert a local filesystem path to a `dataset_uri` the worker accepts.
///
/// The Python side accepts both bare local paths and `file://` URIs; we hand it the
/// absolute path as a plain string, which is unambiguous on the shared local filesystem.
pub fn path_to_uri(path: impl AsRef<Path>) -> String {
    path.as_ref().to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    fn row(t: i64, label: f64, regime: Option<&str>, feats: &[(&str, f64)]) -> DatasetRow {
        DatasetRow {
            ts: ts(t),
            t1: ts(t + 100),
            label,
            regime: regime.map(|s| s.to_string()),
            features: feats.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn rejects_empty() {
        let dir = std::env::temp_dir().join("se_mlclient_empty");
        let err = write_dataset(&[], dir.join("x.parquet")).unwrap_err();
        assert!(matches!(err, MlError::Dataset(_)));
    }

    #[test]
    fn rejects_unsorted_ts() {
        let rows = vec![
            row(200, 0.1, Some("bull"), &[("momentum__a", 1.0)]),
            row(100, -0.2, Some("bear"), &[("momentum__a", 2.0)]),
        ];
        let path = std::env::temp_dir().join("se_mlclient_unsorted.parquet");
        let err = write_dataset(&rows, &path).unwrap_err();
        assert!(matches!(err, MlError::Dataset(_)));
    }

    #[test]
    fn feature_column_order_is_stable() {
        // Two rows with feature keys in different insertion orders must yield the same
        // (sorted) union of columns.
        let rows = vec![
            row(1, 0.1, Some("bull"), &[("z__b", 1.0), ("a__a", 2.0)]),
            row(2, 0.2, Some("bear"), &[("a__a", 3.0), ("m__c", 4.0)]),
        ];
        assert_eq!(
            feature_columns(&rows),
            vec!["a__a".to_string(), "m__c".to_string(), "z__b".to_string()]
        );
    }

    #[test]
    fn writes_and_roundtrips_via_parquet_reader() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let rows = vec![
            row(
                1,
                0.5,
                Some("bull"),
                &[("momentum__a", 1.0), ("trend__b", 2.0)],
            ),
            row(
                2,
                -1.0,
                Some("bear"),
                &[("momentum__a", 3.0), ("trend__b", 4.0)],
            ),
            row(3, 0.0, None, &[("momentum__a", 5.0), ("trend__b", 6.0)]),
        ];
        let path = std::env::temp_dir().join("se_mlclient_roundtrip.parquet");
        let written = write_dataset(&rows, &path).unwrap();
        assert_eq!(written, path);

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);

        let schema = batches[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["ts", "t1", "label", "regime", "momentum__a", "trend__b"]
        );
    }

    #[test]
    fn regime_column_omitted_when_absent() {
        let rows = vec![
            row(1, 0.1, None, &[("momentum__a", 1.0)]),
            row(2, 0.2, None, &[("momentum__a", 2.0)]),
        ];
        let path = std::env::temp_dir().join("se_mlclient_noregime.parquet");
        write_dataset(&rows, &path).unwrap();

        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let has_regime = batch.schema().fields().iter().any(|f| f.name() == "regime");
        assert!(!has_regime);
    }
}
