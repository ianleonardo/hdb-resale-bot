"""
arima_v4.py — ARIMA market baseline for HDB resale price prediction (v4).

Fits ARIMA time-series models on training data to capture market-level price dynamics
that CatBoost cannot represent natively (autocorrelation, trends, seasonality).

Two levels:
  Global  – one ARIMA on the overall monthly mean log-price series (all segments).
  Segment – one ARIMA per (town, flat_type) pair that has ≥ MIN_SERIES_LEN months.
            Sparse segments fall back to the global model.
  RPI     – one ARIMA on the quarterly HDB RPI series; extends the lagged-RPI feature
            beyond the last published quarter (critical for 2026 future-date queries).

Usage contract
--------------
  bundle = ARIMABundle().fit(train_df, rpi_df)   # train_df: 2017-2024 transactions
  feats  = bundle.get_arima_features(query_df)   # dict[str, np.ndarray]
  bundle.save(path)
  bundle = ARIMABundle.load(path)

All series are stored as 0-based RangeIndex numpy-backed pd.Series; start_period
integers are kept alongside to map positions back to tranc_period / quarterly idx.

Feature design note
-------------------
Both features are relative or static quantities — never absolute price levels.
  arima_seg_vs_global  — relative: segment level minus global level at the same
    period.  Even if both levels are systematically wrong for future periods, their
    difference still encodes valid cross-sectional information ("is this segment
    above or below the market?").
  arima_seg_series_std — static per segment: the historical log-price volatility
    of the (town, flat_type) cell computed from the training series.  This gives
    CatBoost a risk/uncertainty signal that is symmetric across train and val/test.

The _rpi_for_ym() method is kept for backend inference-time feature switching
(substitute arima_rpi_forecast for hdb_rpi when the official RPI series is stale)
but arima_rpi_forecast is no longer a CatBoost training feature.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

# ARIMA candidate orders — AIC selects best.  Keep list short for speed.
_GLOBAL_ORDERS  = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2)]
_SEGMENT_ORDERS = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
_RPI_ORDERS     = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]

MIN_SERIES_LEN  = 18   # min months before fitting a per-segment ARIMA
FORECAST_HORIZON = 48  # cap multi-step forecasts to this many steps

# Feature names produced by ARIMABundle.get_arima_features()
ARIMA_FEATURES = [
    "arima_seg_vs_global",   # segment level[M] − global level[M] (relative positioning)
    "arima_seg_series_std",  # historical log-price volatility per (town, flat_type)
]


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _fit_best(series: pd.Series, orders: list[tuple]) -> Any | None:
    """Try each ARIMA order, return the fit with the lowest AIC. None if all fail."""
    best_m, best_aic = None, float("inf")
    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(series, order=order).fit(method_kwargs={"warn_convergence": False})
            if m.aic < best_aic:
                best_aic, best_m = m.aic, m
        except Exception:
            continue
    return best_m


def _monthly_series(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """
    Aggregate df to a dense monthly mean log-price series.
    Returns (series_0indexed, start_tranc_period).
    Gaps are filled via linear interpolation then forward/back fill.
    """
    tmp = df[["Tranc_Year", "Tranc_Month", "resale_price"]].copy()
    tmp["log_price"] = np.log1p(tmp["resale_price"])
    tmp["tp"] = tmp["Tranc_Year"] * 12 + tmp["Tranc_Month"]

    raw   = tmp.groupby("tp")["log_price"].mean()
    tmin, tmax = int(raw.index.min()), int(raw.index.max())
    dense = raw.reindex(range(tmin, tmax + 1)).interpolate("linear").ffill().bfill()

    s = pd.Series(dense.values.astype(float), index=pd.RangeIndex(len(dense)))
    return s, tmin


def _rpi_series(rpi_df: pd.DataFrame) -> tuple[pd.Series, int]:
    """
    Build a 0-indexed quarterly RPI series.
    Quarterly index  q_idx = year * 4 + quarter − 1  (Jan-Mar 2020 → 8080).
    Returns (series_0indexed, start_q_idx).
    """
    tmp = rpi_df[["year", "quarter", "rpi"]].copy()
    tmp["q_idx"] = tmp["year"] * 4 + tmp["quarter"] - 1
    raw = tmp.set_index("q_idx")["rpi"].sort_index()

    qmin, qmax = int(raw.index.min()), int(raw.index.max())
    dense = raw.reindex(range(qmin, qmax + 1)).interpolate("linear").ffill().bfill()

    s = pd.Series(dense.values.astype(float), index=pd.RangeIndex(len(dense)))
    return s, qmin


def _forecast(
    model: Any | None,
    series: pd.Series,
    start: int,
    target: int,
) -> tuple[float, float]:
    """
    Produce a (point, ci_width) forecast for `target` (absolute period index).

    target <= last training period → return in-sample fitted value (ci_width=0).
    target  > last training period → return multi-step out-of-sample forecast.
    Falls back gracefully to the series mean if the model is None or forecast fails.
    """
    fallback = float(series.mean())
    if model is None:
        return fallback, 0.0

    pos = target - start           # 0-based position
    n   = len(series)

    if pos < 0:
        return fallback, 0.0

    if pos < n:
        # In-sample: use fitted value
        try:
            fv = model.fittedvalues
            val = float(fv.iloc[pos]) if pos < len(fv) else fallback
            return (fallback if np.isnan(val) else val), 0.0
        except Exception:
            return fallback, 0.0

    # Out-of-sample
    steps = min(pos - n + 1, FORECAST_HORIZON)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc  = model.get_forecast(steps=steps)
        point = float(fc.predicted_mean.iloc[-1])
        ci    = fc.conf_int(alpha=0.05)
        width = float(ci.iloc[-1, 1] - ci.iloc[-1, 0])
        return (fallback if np.isnan(point) else point), max(0.0, width)
    except Exception:
        return fallback, 0.0


# ── ARIMABundle ────────────────────────────────────────────────────────────────

class ARIMABundle:
    """
    Encapsulates all ARIMA models needed to generate v4 market-baseline features.

    After fit():
      .global_model / .global_series / .global_start
      .segment_data  : {(town, ft): (model, series_0idx, start_period)}
      .rpi_model / .rpi_series / .rpi_start
      .train_end_period : last tranc_period in the training data
    """

    def __init__(self):
        self.global_model   = None
        self.global_series  = None
        self.global_start   = None
        self.global_std     = 0.0   # log-price std of the global series

        self.segment_data: dict[tuple, tuple] = {}  # (town, ft) → (model, series, start)
        self.segment_std:  dict[tuple, float] = {}  # (town, ft) → log-price series std

        self.rpi_model  = None
        self.rpi_series = None
        self.rpi_start  = None

        self.train_end_period = None

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, rpi_df: pd.DataFrame) -> "ARIMABundle":
        logger.info("Fitting ARIMABundle …")

        # Global
        gseries, gstart = _monthly_series(train_df)
        self.global_series = gseries
        self.global_start  = gstart
        self.global_std    = float(gseries.std())
        self.global_model  = _fit_best(gseries, _GLOBAL_ORDERS)
        if self.global_model:
            logger.info(
                "  Global ARIMA order=%s  AIC=%.1f  n=%d months",
                self.global_model.model.order, self.global_model.aic, len(gseries),
            )

        # Per segment
        n_fit = n_fallback = 0
        for (town, ft), grp in train_df.groupby(["town", "flat_type"]):
            s, start = _monthly_series(grp)
            self.segment_std[(town, ft)] = float(s.std())
            if len(s) >= MIN_SERIES_LEN:
                m = _fit_best(s, _SEGMENT_ORDERS)
                if m is not None:
                    self.segment_data[(town, ft)] = (m, s, start)
                    n_fit += 1
                    continue
            # Fallback: store series + start but use global model at forecast time
            self.segment_data[(town, ft)] = (None, s, start)
            n_fallback += 1

        logger.info(
            "  Segments: %d ARIMA fit | %d fall back to global",
            n_fit, n_fallback,
        )

        self.train_end_period = int(
            train_df["Tranc_Year"].max() * 12 + train_df["Tranc_Month"].max()
        )

        # RPI
        rs, rstart = _rpi_series(rpi_df)
        self.rpi_series = rs
        self.rpi_start  = rstart
        self.rpi_model  = _fit_best(rs, _RPI_ORDERS)
        if self.rpi_model:
            logger.info(
                "  RPI ARIMA  order=%s  AIC=%.1f  n=%d quarters",
                self.rpi_model.model.order, self.rpi_model.aic, len(rs),
            )

        return self

    # ── Feature extraction ─────────────────────────────────────────────────────

    def _rpi_for_ym(self, year: int, month: int) -> float:
        """
        ARIMA forecast of HDB RPI using the same 1-quarter lag as v2.
        For periods where the actual RPI is already in the training data,
        this returns the in-sample fitted value; for future periods it forecasts.
        """
        q      = (month - 1) // 3 + 1
        ly, lq = (year, q - 1) if q > 1 else (year - 1, 4)
        q_idx  = ly * 4 + lq - 1
        if self.rpi_model is None or self.rpi_series is None:
            return float("nan")
        val, _ = _forecast(self.rpi_model, self.rpi_series, self.rpi_start, q_idx)
        return val

    def get_arima_features(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Compute ARIMA features for every row in df.
        Returns a dict of two float32 arrays aligned to df's index order.
        """
        N = len(df)
        seg_vs_global  = np.zeros(N, dtype=np.float32)
        seg_series_std = np.zeros(N, dtype=np.float32)

        df_w = df.reset_index(drop=True)
        periods = (df_w["Tranc_Year"] * 12 + df_w["Tranc_Month"]).values.astype(int)

        # ── Global level cache ────────────────────────────────────────────────
        unique_p = np.unique(periods)
        global_level: dict[int, float] = {}
        for p in unique_p:
            gf, _ = _forecast(self.global_model, self.global_series, self.global_start, p)
            global_level[p] = gf

        # ── Segment level + std ───────────────────────────────────────────────
        for (town, ft), grp in df_w.groupby(["town", "flat_type"]):
            idx   = grp.index.values
            grp_p = periods[idx]
            uniq  = np.unique(grp_p)

            entry = self.segment_data.get((town, ft))
            if entry is not None:
                m, s, st = entry
                if m is None:
                    m, s, st = self.global_model, self.global_series, self.global_start
            else:
                m, s, st = self.global_model, self.global_series, self.global_start

            seg_cache: dict[int, float] = {}
            for p in uniq:
                seg_cache[p], _ = _forecast(m, s, st, p)

            std = self.segment_std.get((town, ft), self.global_std)

            for li, gi in enumerate(idx):
                p = grp_p[li]
                seg_vs_global[gi]  = seg_cache[p] - global_level[p]
                seg_series_std[gi] = std

        return {
            "arima_seg_vs_global":  seg_vs_global,
            "arima_seg_series_std": seg_series_std,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved ARIMABundle → %s", path)

    @classmethod
    def load(cls, path: Path) -> "ARIMABundle":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded ARIMABundle from %s", path)
        return obj
