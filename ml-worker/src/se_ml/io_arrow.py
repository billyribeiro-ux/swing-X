"""Parquet/Arrow I/O at the Rust<->Python handoff boundary.

The service receives a `dataset_uri`, loads a pandas DataFrame, runs a job, and may
write an artifact. We keep the read/write surface tiny and explicit so the bitemporal
Rust store and the ML worker agree on exactly one on-disk format (Parquet).

Dataset column convention
-------------------------
A feature/label parquet is expected to contain:

  * ``ts``       : event timestamp (datetime64 or int64 epoch) — REQUIRED, sorted ascending.
  * ``t1``       : barrier-touch timestamp for the event's label window — REQUIRED for
                   purging in CPCV. (For non-event data it may equal ``ts``.)
  * ``label``    : target in R units (float) — REQUIRED for fit/validate.
  * ``regime``   : optional categorical regime tag (e.g. bull/bear/chop) used for the
                   "positive in >=2 regimes" gate condition.
  * everything else : feature columns. Column-name prefixes before the first ``__``
                      denote the feature "layer" (e.g. ``momentum__rsi_14``).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TS_COL = "ts"
T1_COL = "t1"
LABEL_COL = "label"
REGIME_COL = "regime"
# Reserved (non-feature) columns.
_RESERVED = frozenset({TS_COL, T1_COL, LABEL_COL, REGIME_COL})


def uri_to_path(uri: str) -> Path:
    """Resolve a local path or ``file://`` URI to a :class:`Path`."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path if parsed.scheme == "file" else uri)
    raise ValueError(f"Unsupported dataset URI scheme: {parsed.scheme!r} (only file/local paths)")


def read_dataset(uri: str) -> pd.DataFrame:
    """Read a parquet dataset into pandas, validating required columns are present."""
    path = uri_to_path(uri)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    df = pq.read_table(path).to_pandas()
    if TS_COL not in df.columns:
        raise ValueError(f"dataset missing required '{TS_COL}' column")
    return df


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, returning the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything that is not a reserved column)."""
    return [c for c in df.columns if c not in _RESERVED]


def layer_of(feature: str) -> str:
    """Map a feature name to its layer via the ``layer__feature`` naming convention.

    Features without a ``__`` separator are placed in the ``base`` layer.
    """
    return feature.split("__", 1)[0] if "__" in feature else "base"


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into the feature matrix X and label vector y (R units)."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"dataset missing required '{LABEL_COL}' column")
    feats = feature_columns(df)
    X = df[feats].copy()
    y = df[LABEL_COL].astype(float).copy()
    return X, y
