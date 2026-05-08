# ARIMA + CatBoost in v4 (`arima_v4.py` + `train_v4.py`)

This document explains how the v4 training pipeline combines **ARIMA** (statsmodels) with **CatBoost**: what each ARIMA layer does, how features reach the gradient-boosted model, and what the **(p, d, q)** orders mean in this codebase.

---

## 1. High-level strategy

1. **ARIMA** is fit on **historical transaction data** (and official RPI CSV) to summarize **time dynamics** at the market level: trends, persistence, and smooth extrapolation into future months or quarters. Tree models like CatBoost do not naturally encode **long-horizon autocorrelation** in a single monthly index the way a dedicated time-series model does.

2. **CatBoost** is trained on **unit-level rows** (each resale transaction) with v2-style engineered features (spatial KDTree encodings, official lagged RPI, interactions, etc.) **plus two numeric columns** produced by the ARIMA bundle. CatBoost predicts **`log_resale_price`**; dollars are recovered with `expm1`.

3. **Leakage control**: the ARIMA bundle is fit only on raw rows with `Tranc_Year` from **`HIST_YEAR_START` (2017) through `TRAIN_YEAR_END` (2024)** — i.e. the same broad window as the “full history” used for ARIMA, **not** including validation or test years. CatBoost itself uses training years **2020–2024** with early stopping on **2025** validation MAE (see `features_v4.py` for year constants).

So: **ARIMA = cheap, explicit time-series prior on market aggregates**; **CatBoost = flexible function approximator** that learns how that prior interacts with flat size, lease, location, amenities, etc.

---

## 2. The three ARIMA “levels” in `ARIMABundle`

| Level | Series | Purpose in code |
|--------|--------|------------------|
| **Global** | One **monthly** series: mean of `log1p(resale_price)` over **all** transactions in the ARIMA fit window, indexed by `tranc_period = year×12 + month`. | Defines the **overall market level** through time. Used as the subtracted baseline in `arima_seg_vs_global` and as the **fallback** when a segment is too sparse to fit its own ARIMA. |
| **Segment** | Same monthly mean **log-price**, but grouped by **`(town, flat_type)`**. | Captures how a **local market cell** evolves vs the whole island. If the cell has fewer than **`MIN_SERIES_LEN` (18)** months of history after densification, the code stores the segment’s series for volatility (`arima_seg_series_std`) but uses the **global model** for forecasts (see below). |
| **RPI** | **Quarterly** HDB Resale Price Index from `data/hdb_rpi.csv`, indexed by `q_idx = year×4 + quarter − 1`, densified like the price series. | Fit so the bundle can produce **`_rpi_for_ym()`**: an ARIMA-based RPI value aligned with the **one-quarter lag** used in v2. This is intended for **backend inference-time switching** when the official series is stale (e.g. future transaction dates in 2026). **It is not one of the two CatBoost training columns** (`ARIMA_FEATURES` in `arima_v4.py`). Training still uses **`hdb_rpi`** from `add_official_rpi()` as in v2. |

**Why “global” vs “segment”?**  
Global answers: “What is the typical log-price level for the whole market this month?”  
Segment answers: “What is the typical log-price level for *this town and flat type* this month?”  
Their **difference** (segment minus global) is a **relative positioning** signal: is this cell hot or cold vs the aggregate, even if both absolute levels drift.

**Why RPI ARIMA if CatBoost already has `hdb_rpi`?**  
Official RPI stops at the last published quarter. For **queries beyond that**, a model that only reads the CSV would see a stale or missing macro. The RPI ARIMA path is there to **extend** the macro series in a principled way (smooth forecast from a quarterly ARIMA). The training pipeline deliberately keeps **actual** RPI for rows where it exists, and documents the ARIMA RPI hook for inference only.

---

## 3. How ARIMA features are combined with CatBoost

### 3.1 What CatBoost actually receives

From `arima_v4.py`, **`ARIMA_FEATURES`** are exactly:

| Feature | Definition | Role for CatBoost |
|---------|------------|-------------------|
| **`arima_seg_vs_global`** | For each row’s transaction month: **segment ARIMA level − global ARIMA level** (both in **log-price** space for the monthly mean series). | **Relative** macro/segment positioning through time. If both forecasts are biased for future months, the **difference** can still rank segments sensibly (“above vs below market”). |
| **`arima_seg_series_std`** | Standard deviation of that segment’s **training** monthly log-price series (or global std if unknown). | **Static** per `(town, flat_type)` risk/volatility signal; same value for train/val/test rows in that cell (no peeking at future volatility). |

These arrays are built in `ARIMABundle.get_arima_features()` and merged in `train_v4._add_arima_cols()`. `get_feature_cols()` in `features_v4.py` includes any column not in `DROP_COLS`; ARIMA columns are numeric and **not** dropped, so they enter **`prepare_X`** automatically.

### 3.2 Division of labour

- **ARIMA** encodes **slowly moving market structure** (monthly means, optional per-segment means) and **one summary volatility** per segment.
- **CatBoost** maps the full feature vector — including micro structure (floor area, lease, storey, spatial encodings, schools, MRT, etc.) — to **log price**, using the ARIMA columns as **context** for how “hot” the segment-month is vs the island and how noisy that segment historically was.

This is a **stacked / hybrid** design: time-series model first, tree model second, rather than a single end-to-end temporal network.

---

## 4. ARIMA mechanics in this repo

### 4.1 Series construction

- **Transactions → monthly log-price means**: `log1p(resale_price)` is averaged per calendar month (via `tranc_period`). Missing months in the span are **linearly interpolated**, then forward/back-filled (`_monthly_series`).
- **RPI → quarterly series**: same idea on `rpi` by quarter index (`_rpi_series`).

### 4.2 Model selection: **AIC over a small candidate set**

`_fit_best(series, orders)` loops over candidate **(p, d, q)** tuples, fits `statsmodels.tsa.arima.model.ARIMA(series, order=order)`, and keeps the fit with **lowest AIC**. Failed fits are skipped.

Candidate sets (from `arima_v4.py`):

- **Global**: `(0,1,1)`, `(1,1,0)`, `(1,1,1)`, `(2,1,1)`, `(2,1,2)`
- **Segment**: `(0,1,1)`, `(1,1,0)`, `(1,1,1)`, `(2,1,1)`
- **RPI**: `(0,1,1)`, `(1,1,0)`, `(1,1,1)`, `(2,1,1)`

Global gets one extra rich candidate `(2,1,2)` because the aggregate series is long and smooth enough that the extra parameters are sometimes worth the AIC cost.

### 4.3 Fitted value vs forecast (`_forecast`)

- If the row’s period index falls **inside** the fitted sample range: use **in-sample fitted value** (`fittedvalues`), CI width 0.
- If the period is **after** the training end of that series: use **`get_forecast`** stepping forward, with **`FORECAST_HORIZON = 48`** as a cap on how far ahead to project in one call (stability / runtime).
- On failure or missing model: fall back to the **mean** of that series.

**Sparse segments**: if `len(series) < MIN_SERIES_LEN` or every candidate fails, `segment_data[(town, ft)]` stores `(None, series, start)`. At feature time, **`None` is replaced by the global model and global series** for level forecasts, but **`segment_std`** still uses that segment’s own historical log-price std when available.

---

## 5. What **(p, d, q)** means — and why these candidates

An **ARIMA(p,d,q)** model (in the Box–Jenkins sense used by statsmodels) assumes that after **d** differences, the series behaves like an **AR(p)** plus **MA(q)** process on the **innovations** (errors).

| Parameter | Name | Meaning |
|-----------|------|--------|
| **p** | Autoregressive order | Differenced series depends linearly on **p** of its own **lagged values**. |
| **d** | Integration / differencing | Apply **d** rounds of differencing to remove **unit root / polynomial trend** so the rest of the model fits stationary-ish dynamics. Here **d = 1** is standard for **non-stationary levels** (prices and indices tend to drift). |
| **q** | Moving-average order | Current value depends on **q** lags of the **one-step-ahead forecast errors** (shocks). |

**Why d = 1 everywhere here?**  
Monthly mean log-prices and quarterly RPI are **persistent level** processes. First differencing targets **changes** (growth, cooling) rather than absolute level, which stabilizes estimation and matches common practice for economic series.

### 5.1 Intuition for the specific orders used

- **(0, 1, 1)** — **IMA(1,1)**, related to **simple exponential smoothing** of the differenced series.  
  - *Interpretation*: smoothed **local trend** with short memory in **innovations**.  
  - *When it wins*: smooth, slowly drifting level with noise; few AR parameters.

- **(1, 1, 0)** — **differenced AR(1)** (ARIMA(1,1,0)).  
  - *Interpretation*: **momentum / mean-reversion in growth rates** — today’s *change* depends on yesterday’s *change*.  
  - *When it wins*: growth rates show **autocorrelation** rather than a purely MA structure.

- **(1, 1, 1)** — **ARIMA(1,1,1)** — the “workhorse” mixed model.  
  - *Interpretation*: flexible blend of AR and MA in differences; can mimic either (0,1,1) or (1,1,0) as limits.  
  - *When it wins*: both **persistence in levels of differences** and **correlated shocks** matter.

- **(2, 1, 1)** / **(2, 1, 2)** — more **lags** in AR and/or MA on the differenced series.  
  - *Interpretation*: richer **cyclical or multi-step memory** in changes (e.g. overshoots, two-period cycles).  
  - *Risk*: overfit on short segment series — hence **segment** list omits `(2,1,2)`; **global** keeps it because the series is longer.

**Important:** the code does **not** fix one “true” order by theory alone. It **enumerates a small, standard menu** of low-order integrated models and lets **AIC** pick the trade-off between fit and complexity for each of global, segment, and RPI series.

---

## 6. Training flow (reference)

Rough order in `train_v4.main()`:

1. Load CSV, split train / val / test by year (`split_data`).
2. Engineer features + official RPI + interactions (`engineer_features`, `add_official_rpi`, `add_macro_interaction_features`).
3. Build spatial KDTree features; save `spatial_inference.pkl`.
4. Fit **`ARIMABundle`** on `raw` restricted to **`HIST_YEAR_START`–`TRAIN_YEAR_END`** and `hdb_rpi.csv`; save `arima_bundle_v4.pkl`.
5. **`get_arima_features`** on train / val / test DataFrames → inject the two ARIMA columns.
6. **`get_feature_cols`** → CatBoost **`Pool`**, train with early stopping on val.

---

## 7. Artifacts and inference

- **`arima_bundle_v4.pkl`**: pickled `ARIMABundle` (statsmodels models + pandas series + segment std map). Backend should load this and call **`get_arima_features(query_df)`** alongside the same engineered + spatial + RPI columns used in training.
- **`model_v4.cbm`**: CatBoost weights conditioned on those features.

Together they implement: **ARIMA summarizes the temporal envelope of the market; CatBoost prices the individual flat conditional on that envelope and micro covariates.**

---

## 8. Pros and cons of combining ARIMA with CatBoost (this design)

### Pros

- **Explicit time structure for aggregates** — ARIMA is built for autocorrelation and smooth extrapolation on the **monthly mean log-price** (and quarterly RPI). CatBoost sees only tabular rows; feeding it `arima_seg_vs_global` and `arima_seg_series_std` gives a **low-dimensional summary** of how the market and segment evolved through time without forcing the tree to rediscover long memory from sparse monthly dummies alone.

- **Relative features reduce level risk** — `arima_seg_vs_global` is a **difference** (segment minus global). If both ARIMA levels are wrong for a far-future month, the **gap** can still carry useful cross-sectional ranking (“this town/type vs island”), which is harder to get from raw price aggregates fed directly into a tree.

- **Interpretability and modularity** — You can inspect AIC-chosen orders, fitted vs forecast paths, and segment fallbacks independently of CatBoost. Training is **two-stage**: refit the bundle when history grows, then retrain CatBoost; no single giant temporal graph to debug.

- **Cheap inference path for macro extension** — The RPI ARIMA branch supports **`_rpi_for_ym()`** for stale-official-RPI scenarios without changing the core CatBoost feature set; the main model still trains on **actual** `hdb_rpi` where published.

- **Segment volatility as a stable risk signal** — `arima_seg_series_std` is a **training-era constant** per `(town, flat_type)`, so it does not leak future dispersion into val/test while still telling CatBoost which cells were historically noisier.

- **Controlled complexity** — Small **(p,d,q)** grids and **AIC** selection keep fitting fast and avoid exotic orders; sparse segments **fall back** to the global model instead of unstable high-parameter fits.

### Cons

- **Two models to ship and version** — Production needs **`arima_bundle_v4.pkl`** and **`model_v4.cbm`** in sync (and the same feature engineering). Mismatch or a forgotten `get_arima_features` call silently hurts accuracy.

- **ARIMA assumptions may be wrong** — Linear Gaussian ARIMA on **aggregated means** ignores structural breaks, policy shocks, and fat tails. AIC picks the best *within* a small family, not necessarily the best **real-world** process; bad multi-step forecasts still propagate into `arima_seg_vs_global` for future months (mitigated partly by using a **difference**, not raw level).

- **Segment definition is fixed** — Only **`(town, flat_type)`** gets its own series; micro-neighbourhoods within a town are not separate ARIMA cells. Very heterogeneous towns get one blended monthly mean, which can wash out local pockets.

- **Interpolation and fallbacks smooth or blunt signal** — Missing months are **linearly interpolated** then ffill/bfill; failed fits use the **series mean**. That avoids NaNs but can **inject synthetic dynamics** or flatten extremes, especially for thin segments.

- **Forecast horizon cap** — **`FORECAST_HORIZON = 48`** caps how far ahead `_forecast` steps in one go; very distant future months rely on the last step of that capped path, which can **damp or distort** very long-horizon behaviour compared to an unconstrained recursive forecast.

- **Pickle + statsmodels maintenance** — The bundle pickles fitted statsmodels objects; library upgrades can break load compatibility. CatBoost alone is often simpler to serialize and reproduce across environments.

- **RPI ARIMA not in training features** — CatBoost never sees **`arima_rpi_forecast`** during training, so if the backend swaps in ARIMA-extended RPI at inference, behaviour is **distribution-shift** relative to training unless you retrain or align that path deliberately.

- **Extra training time and CPU** — Fitting many segment ARIMAs plus CatBoost HPO/training adds wall-clock vs a single tree-only pipeline (though segment grids are intentionally small).

---

## 9. Example training metrics (reference run)

Snapshot from **`train_v4.py`** with splits **train 2020–2024**, **val 2025**, **test 2026** (`features_v4.py`), CatBoost **early stopping** on validation MAE (`use_best_model=True`):

```
Training done: 39.30s | best iter=2176

Train:       MAE=16,464 | RMSE=24,111 | MAPE=3.13% | R²=0.9805 | log_RMSE=0.044199
Validation:  MAE=26,236 | RMSE=38,668 | MAPE=3.89% | R²=0.9618 | log_RMSE=0.052578
Test:        MAE=29,584 | RMSE=44,985 | MAPE=4.42% | R²=0.9525 | log_RMSE=0.060948
```

### Reading these numbers

- **`best iter=2176`** — best checkpoint by **val MAE** (on log target); wall-clock **~39 s** here includes training only (feature + ARIMA fit times are logged separately in the full pipeline).
- **Train vs val** — Train MAE is **lower** than val (typical for boosted trees on the fit split). Metrics use **`expm1`** on predictions and labels for dollar MAE/RMSE/MAPE/R²; **log_RMSE** is on **`log_resale_price`**.
- **Val → test (2025 → 2026)** — Small degradation (e.g. test MAE **~29.6k** vs val **~26.2k**, MAPE **~4.4%** vs **~3.9%**) is normal **forward-year** behaviour; **R²** stays **~0.95+** on test.
- **Comparison anchor** — Use the same CSV splits when comparing to **v3 LSTM** or other baselines; absolute MAE depends on data volume and year mix.
