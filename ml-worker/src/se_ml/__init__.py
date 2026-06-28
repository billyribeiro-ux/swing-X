"""se_ml — Python ML sidecar for the swing-X self-learning swing-trading engine.

This package is the *authoritative* implementation of the financial-ML statistics
(triple-barrier labeling, Combinatorial Purged CV, Deflated Sharpe Ratio,
Probability of Backtest Overfit, calibration, importance, and the promotion gate).

The Rust orchestrator (`se-mlclient`) calls this service over HTTP. Bulk data is
exchanged as Parquet/Arrow by file URI; request/response bodies carry only metadata.
"""

__version__ = "0.1.0"
