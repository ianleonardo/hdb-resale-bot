"""
eval_v2.py — detailed post-training evaluation for the v2 model.

Loads saved artifacts (model_v2.pkl, preprocessor_v2.pkl) and the raw CSV,
then produces a diagnostic report:
  - Overall val / test metrics
  - Per-town and per-flat-type MAE breakdown
  - Comp coverage analysis: MAE with vs without comps
  - Error percentile distribution
  - Worst-performing segments

Run from the training/ directory after train_v2.py completes.
"""

import io
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_v2 import (
    ALL_FEATURES, TARGET, engineer_features, compute_comp_features,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/Resaleflatpricesbase.csv"
LOCAL_ARTIFACTS = Path("artifacts/")

SEP = "─" * 72


def load_artifacts():
    model        = joblib.load(LOCAL_ARTIFACTS / "model_v2.pkl")
    preprocessor = joblib.load(LOCAL_ARTIFACTS / "preprocessor_v2.pkl")
    return model, preprocessor


def load_eval_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (val_df, test_df) with comp features already attached."""
    gcs = storage.Client()
    csv_bytes = gcs.bucket(GCS_BUCKET).blob(DATA_BLOB).download_as_bytes()
    raw = pd.read_csv(io.BytesIO(csv_bytes))

    logger.info("Engineering features + computing comps...")
    df = engineer_features(raw)
    df = compute_comp_features(df)

    val  = df[(df["transaction_year"] == 2025) & (df["transaction_month"] <= 9)].copy()
    test = df[
        ((df["transaction_year"] == 2025) & (df["transaction_month"] >= 10)) |
        (df["transaction_year"] == 2026)
    ].copy()
    return val, test


def add_predictions(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    X = preprocessor.transform(df[ALL_FEATURES])
    df = df.copy()
    df["pred_price"] = np.expm1(model.predict(X))
    df["true_price"] = np.expm1(df[TARGET])
    df["abs_error"]  = (df["pred_price"] - df["true_price"]).abs()
    df["pct_error"]  = df["abs_error"] / df["true_price"] * 100
    return df


def overall_metrics(df: pd.DataFrame, label: str) -> dict:
    mae  = mean_absolute_error(df["true_price"], df["pred_price"])
    rmse = float(np.sqrt(mean_squared_error(df["true_price"], df["pred_price"])))
    mape = df["pct_error"].mean()
    r2   = r2_score(df["true_price"], df["pred_price"])
    print(f"\n{SEP}")
    print(f"  {label}  (n={len(df):,})")
    print(SEP)
    print(f"  MAE  : SGD {mae:>10,.0f}   target < 25,000")
    print(f"  RMSE : SGD {rmse:>10,.0f}")
    print(f"  MAPE :     {mape:>9.2f}%   target < 5%")
    print(f"  R²   :     {r2:>10.4f}   target > 0.96")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def segment_breakdown(df: pd.DataFrame, col: str, label: str, top_n: int = 10):
    seg = (
        df.groupby(col)
        .agg(n=("abs_error", "count"), mae=("abs_error", "mean"), mape=("pct_error", "mean"))
        .sort_values("mae", ascending=False)
    )
    print(f"\n  {label} breakdown (worst {top_n} by MAE):")
    print(f"  {'Segment':<30} {'N':>6}  {'MAE':>10}  {'MAPE':>7}")
    print(f"  {'-'*30} {'-'*6}  {'-'*10}  {'-'*7}")
    for seg_name, row in seg.head(top_n).iterrows():
        print(f"  {str(seg_name):<30} {int(row['n']):>6}  {row['mae']:>10,.0f}  {row['mape']:>6.1f}%")


def comp_coverage_analysis(df: pd.DataFrame):
    has_comps = df["comp_count"] > 0
    n_with    = has_comps.sum()
    n_without = (~has_comps).sum()

    mae_with    = mean_absolute_error(
        df.loc[has_comps, "true_price"], df.loc[has_comps, "pred_price"]
    ) if n_with else float("nan")
    mae_without = mean_absolute_error(
        df.loc[~has_comps, "true_price"], df.loc[~has_comps, "pred_price"]
    ) if n_without else float("nan")

    print(f"\n  Comp coverage analysis:")
    print(f"  {'':30} {'N':>6}  {'MAE':>10}")
    print(f"  {'-'*30} {'-'*6}  {'-'*10}")
    print(f"  {'With comps (count > 0)':<30} {n_with:>6}  {mae_with:>10,.0f}")
    print(f"  {'No comps (count = 0)':<30} {n_without:>6}  {mae_without:>10,.0f}")
    print(f"  {'MAE lift from comps':}")
    if not np.isnan(mae_with) and not np.isnan(mae_without):
        lift = mae_without - mae_with
        print(f"    comps reduce MAE by ~SGD {lift:,.0f} ({lift / mae_without * 100:.1f}%)")

    # Breakdown by comp_count bucket
    buckets = pd.cut(
        df["comp_count"],
        bins=[-1, 0, 5, 20, 50, np.inf],
        labels=["0", "1–5", "6–20", "21–50", "51+"],
    )
    bucket_stats = df.groupby(buckets)["abs_error"].agg(["count", "mean"])
    print(f"\n  MAE by comp_count bucket:")
    print(f"  {'comp_count':>10}  {'N':>6}  {'MAE':>10}")
    for bucket, row in bucket_stats.iterrows():
        print(f"  {str(bucket):>10}  {int(row['count']):>6}  {row['mean']:>10,.0f}")


def error_percentiles(df: pd.DataFrame):
    pcts = [50, 75, 90, 95, 99]
    vals = [np.percentile(df["abs_error"], p) for p in pcts]
    print(f"\n  Absolute-error percentiles:")
    for p, v in zip(pcts, vals):
        bar = "█" * int(v / 5000)
        print(f"  p{p:02d}: SGD {v:>8,.0f}  {bar}")


def main():
    logger.info("Loading artifacts...")
    model, preprocessor = load_artifacts()

    logger.info("Loading and processing eval data...")
    val_df, test_df = load_eval_data()

    val_df  = add_predictions(val_df,  model, preprocessor)
    test_df = add_predictions(test_df, model, preprocessor)

    for df, label in [(val_df,  "VALIDATION  2025 Jan–Sep"),
                      (test_df, "TEST        2025 Oct–2026")]:
        overall_metrics(df, label)
        comp_coverage_analysis(df)
        error_percentiles(df)
        segment_breakdown(df, "town",      "Town")
        segment_breakdown(df, "flat_type", "Flat type", top_n=7)

    # Cross-split: show best and worst towns on val for reference
    print(f"\n{SEP}")
    print("  Val — best towns by MAE:")
    seg = (
        val_df.groupby("town")["abs_error"]
        .mean()
        .sort_values()
        .head(5)
    )
    for town, mae in seg.items():
        print(f"    {town:<30} SGD {mae:>8,.0f}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
