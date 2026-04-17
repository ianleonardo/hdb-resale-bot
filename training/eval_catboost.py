"""
eval_catboost.py — detailed post-training evaluation for the CatBoost model.

Produces the same diagnostic report as eval_v2.py so results are directly
comparable between LightGBM and CatBoost.

Run from the training/ directory after train_catboost.py completes.
"""

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_catboost import (
    ALL_FEATURES, CAT_FEATURES, TARGET,
    engineer_features, compute_comp_features, prepare_X,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET      = "hdb-resale-artifacts"
DATA_BLOB       = "training/Resaleflatpricesbase.csv"
LOCAL_ARTIFACTS = Path("artifacts/")

SEP = "─" * 72


def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(LOCAL_ARTIFACTS / "model_catboost.cbm"))
    return model


def load_eval_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_predictions(df: pd.DataFrame, model: CatBoostRegressor) -> pd.DataFrame:
    X    = prepare_X(df)
    pool = Pool(X, cat_features=CAT_FEATURES)
    df   = df.copy()
    df["pred_price"] = np.expm1(model.predict(pool))
    df["true_price"] = np.expm1(df[TARGET])
    df["abs_error"]  = (df["pred_price"] - df["true_price"]).abs()
    df["pct_error"]  = df["abs_error"] / df["true_price"] * 100
    return df


def overall_metrics(df: pd.DataFrame, label: str):
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
    n_with, n_without = has_comps.sum(), (~has_comps).sum()
    mae_with    = mean_absolute_error(
        df.loc[has_comps, "true_price"], df.loc[has_comps, "pred_price"]
    ) if n_with else float("nan")
    mae_without = mean_absolute_error(
        df.loc[~has_comps, "true_price"], df.loc[~has_comps, "pred_price"]
    ) if n_without else float("nan")

    print(f"\n  Comp coverage analysis:")
    print(f"  {'With comps (count > 0)':<30} {n_with:>6}  {mae_with:>10,.0f}")
    print(f"  {'No comps (count = 0)':<30} {n_without:>6}  {mae_without:>10,.0f}")
    if not (np.isnan(mae_with) or np.isnan(mae_without)):
        lift = mae_without - mae_with
        print(f"  Comps reduce MAE by ~SGD {lift:,.0f} ({lift / mae_without * 100:.1f}%)")

    buckets = pd.cut(
        df["comp_count"],
        bins=[-1, 0, 5, 20, 50, np.inf],
        labels=["0", "1–5", "6–20", "21–50", "51+"],
    )
    bucket_stats = df.groupby(buckets)["abs_error"].agg(["count", "mean"])
    print(f"\n  MAE by comp_count bucket:")
    for bucket, row in bucket_stats.iterrows():
        print(f"  {str(bucket):>10}  n={int(row['count']):>6}  MAE={row['mean']:>10,.0f}")


def error_percentiles(df: pd.DataFrame):
    pcts = [50, 75, 90, 95, 99]
    print(f"\n  Absolute-error percentiles:")
    for p in pcts:
        v = np.percentile(df["abs_error"], p)
        bar = "█" * int(v / 5000)
        print(f"  p{p:02d}: SGD {v:>8,.0f}  {bar}")


def feature_importance(model: CatBoostRegressor, top_n: int = 15):
    fi = pd.Series(
        model.get_feature_importance(),
        index=ALL_FEATURES,
    ).sort_values(ascending=False)
    print(f"\n  Feature importance (top {top_n}):")
    for feat, imp in fi.head(top_n).items():
        bar = "█" * int(imp / 2)
        print(f"  {feat:<30} {imp:>6.2f}  {bar}")


def main():
    logger.info("Loading model...")
    model = load_model()

    logger.info("Loading and processing eval data...")
    val_df, test_df = load_eval_data()
    val_df  = add_predictions(val_df,  model)
    test_df = add_predictions(test_df, model)

    for df, label in [(val_df,  "VALIDATION  2025 Jan–Sep"),
                      (test_df, "TEST        2025 Oct–2026")]:
        overall_metrics(df, label)
        comp_coverage_analysis(df)
        error_percentiles(df)
        segment_breakdown(df, "town",      "Town")
        segment_breakdown(df, "flat_type", "Flat type", top_n=7)

    feature_importance(model)
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
