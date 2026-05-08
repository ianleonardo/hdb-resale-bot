# LSTM v3 training (`train_lstm_v3.py`, `features_v3.py`, `model_lstm_v3.py`)

This document describes how the v3 **hybrid LSTM + MLP** pipeline works, what an LSTM contributes here, pros and cons for **HDB resale price** prediction, and how to read training logs and reported metrics.

---

## 1. What problem v3 solves

Resale rows are **tabular** (flat attributes, location, lease, etc.), but each sale also sits in a **time evolving market**: segment-level prices, volumes, and macro (RPI) move month to month. **Gradient-boosted trees** (v2/v4) handle interactions well but only see **per-row** features unless you engineer explicit time summaries.

v3 adds a **sequence branch**: the **timeline** of months is still taken from **(town, flat_type)** (stable volume). For each timestep, if **(postal sector, flat_type)** has enough transactions that month (`≥ MIN_TXN_COUNT_FOR_SECTOR_SEQ`, default 3), that slot uses **sector** aggregates instead of town — otherwise it stays **town** (“safer hybrid”). **`postal_sector`** is also a **categorical embedding** on the static branch. An **LSTM** compresses the sequence; a **static MLP** encodes engineered + spatial features as v2; a **fusion head** predicts **`log_resale_price`** (`log1p` price).

---

## 2. End-to-end pipeline (high level)

1. **Load CSV** and split: **train 2020–2024**, **val 2025**, **test 2026** (`features_v3.split_data`).
2. **Engineer property rows** — `engineer_features`, `add_official_rpi`, `add_macro_interaction_features` (backend `inference_features`).
3. **KDTree spatial targets** — same temporal rules as v2 (`build_spatial_bundle_dict`, `compute_spatial_features`); save `spatial_inference.pkl`.
4. **Monthly aggregates** — `build_monthly_agg` returns **`{"town": …, "sector": …}`** from raw rows **`HIST_YEAR_START` (2017)–`TRAIN_YEAR_END` (2024)** (sector = `postal // 10000`). Same leakage rules as before.
5. **Sequences** — `build_sequences` walks the **town** monthly history; **hybrid** substitution per timestep when sector-month is dense enough; **`add_postal_sector`** adds string **`postal_sector`** for encoders.
6. **Preprocess** — `StandardScaler` on static numerics and sequence tensors; `OrdinalEncoder` for categoricals including **`postal_sector`** (`fit_preprocessors` / `apply_preprocessors`).
7. **Train** `HDBPriceLSTM` — **AdamW**, **cosine LR decay**, **L1 loss on log target** (MAE in log space), **early stopping on val loss**, **gradient clipping**.
8. **Artifacts** — `model_lstm_v3.pt`, `preprocessor_v3.pkl`, `config_lstm_v3.json`, `metrics_lstm_v3.json`.

---

## 3. What goes into the LSTM (sequence features)

Per **timestep**, seven features (`SEQ_FEATURES` in `features_v3.py`). Each timestep corresponds to a **calendar month** on the listing’s **(town, flat_type)** history; the **values** for that month are either **town** or **sector-(flat_type)** aggregates when sector volume ≥ **`MIN_TXN_COUNT_FOR_SECTOR_SEQ`** (default **3**), else **town**.

| Feature | Role |
|--------|------|
| `log_mean_price` | Segment typical **log price** that month |
| `log_std_price` | Dispersion of log prices (0 if only one transaction) |
| `log_volume` | `log1p` transaction count — liquidity |
| `log_mean_area` | Mean floor area — composition shift |
| `hdb_rpi` | Official RPI, **one-quarter lag** (same idea as v2) |
| `month_sin` / `month_cos` | Cyclical **calendar** encoding |

So the LSTM sees a **short panel** of local market dynamics; **invalid/missing postal** drops sector substitution (town-only). Training logs a line like **`Hybrid sequences: sector replaced X% of timestep slots`**.

---

## 4. What the neural network does (`model_lstm_v3.py`)

- **LSTM branch**: `input_size = 7`, **`seq_len = 12`**, stacked LSTM → **last timestep** hidden state → LayerNorm → linear projection to `head_hidden`.
- **Static branch**: concatenate **43 scaled numerics** + **categorical embeddings** (`flat_type`, `flat_model`, `town`, **`postal_sector`**, MRT/school names) → MLP → `head_hidden`.
- **Fusion head**: concat `[lstm_enc, static_enc]` → MLP → scalar **`log1p(price)`**.

So it is **not** “LSTM-only”; the static path carries **unit-level** information (area, lease, storey, spatial encodings, RPI interactions, etc.), matching the v2 feature philosophy.

---

## 5. A short primer on LSTM (why it here)

An **LSTM** (Long Short-Term Memory) is a recurrent unit that maintains a **hidden state** while scanning a sequence step by step. It learns **which past information to keep or forget** via gating, which helps with **smooth, correlated** series (e.g. month-to-month market drift) better than a plain feed-forward net that flattens 12×7 numbers without explicit temporal structure.

In this project, the sequence is **only 12 months** and relatively low-dimensional; the LSTM acts as a **learned summary** of recent segment dynamics (trend, volatility, seasonality) rather than a long-horizon language model.

---

## 6. Pros and cons for HDB resale prediction

### Pros

- **Explicit temporal context** for segment-level **mean price, volume, volatility**, and seasonality — aligned with how practitioners think about “market heating/cooling.”
- **Hybrid design** keeps **strong tabular signal** (static MLP + same engineered/spatial ideas as v2).
- ** Differentiable stack** — train end-to-end with MAE-on-log objective consistent with other models.
- **Interpretability of inputs** — sequence features are named, aggregated statistics (easier to sanity-check than raw IDs alone).

### Cons

- **Data and compute**: PyTorch training, GPU helpful; more moving parts than CatBoost-only (preprocessors, embeddings, deployment).
- **Segment granularity**: the **calendar spine** is **town × flat_type**; **sector** only fills in when supported by enough monthly volume — avoids an all-sector sequence that is mostly noise.
- **Risk of leakage if misconfigured**: monthly_agg must only use months **strictly before** each row’s transaction month (the code enforces `< curr_p`). Widening `monthly_agg` into future years without this rule would leak.
- **Distribution shift**: if 2026 market behaviour differs from training-era sequences, the LSTM’s temporal prior can be wrong even when static features look plausible.
- **Less interpretable than trees** for **global** feature importance (partial dependence is still possible but heavier).

---

## 7. Reading the training log (loss vs metrics)

- **`train` / `val` in each epoch** are **mean L1 loss on `log_resale_price`** (same scale as `log_RMSE` components — dimensionless in log space). They should **decrease** early; **val** is the early-stopping signal.
- **Learning rate** follows **cosine annealing** from initial LR down toward `eta_min` (`CosineAnnealingLR`).
- **Early stopping**: if **val** does not improve for **`PATIENCE` (15)** epochs, training stops; **best weights** are those at **`best_epoch`** (lowest val loss so far).

---

## 8. Assessing a concrete run (example log)

Reference run: **hybrid sequences** (town timeline + sector substitution when volume ≥ 3) + **`postal_sector`** embedding, and **correct train evaluation** (`train_eval_loader` with `shuffle=False`; see §8.4).

```
Epoch   1/100 | train=0.703238 | val=0.139567 | lr=2.81e-03
Epoch  10/100 | train=0.179559 | val=0.088894 | lr=2.74e-03
Epoch  20/100 | train=0.167672 | val=0.063859 | lr=2.54e-03
Epoch  30/100 | train=0.163722 | val=0.044710 | lr=2.23e-03
Epoch  40/100 | train=0.161425 | val=0.041288 | lr=1.85e-03
Epoch  50/100 | train=0.159316 | val=0.040716 | lr=1.42e-03
Epoch  60/100 | train=0.155998 | val=0.041479 | lr=9.89e-04
Early stop at epoch 65 (patience=15)
Training done: 68.42s | best_epoch=50 | best_val=0.040716

Train:       MAE=21,026 | RMSE=29,331 | MAPE=3.94% | R²=0.9712 | log_RMSE=0.052089
Validation:  MAE=27,032 | RMSE=38,335 | MAPE=4.08% | R²=0.9624 | log_RMSE=0.053335
Test:        MAE=30,141 | RMSE=41,332 | MAPE=4.67% | R²=0.9599 | log_RMSE=0.059778
```

### 8.1 Train / val / test line up

- **Train / val / test** are **aligned** (no shuffled-train bug). **Train** is a bit **stronger** than val on MAE (~21k vs ~27k) — expected with dropout in training and `eval()` at metric time.
- **Val vs test (2025 → 2026)**: MAE and MAPE **step up slightly**; **R²** stays high (~0.96). Typical **forward-year** gap, not a collapse.
- **log_RMSE** is close on **train and val** (~0.052–0.053); **test** ~0.060 — a bit harder year or sample mix.
- **best_val ≈ 0.041** at **epoch 50**; training stopped at **65** when val failed to improve for **15** epochs. Length of run varies with init and data order.
- **Vs. the previous non-hybrid run** (same shuffle fix, town-only sequence, no `postal_sector` cat): this example’s **val/test MAE and test MAPE improved** (e.g. test MAE in the **low 30k** SGD range vs **mid-30k**). Treat as one strong seed; **retrain** to confirm stability.

### 8.2 Epoch `train` vs `val` loss (not the same as table metrics)

Per-epoch **`train`** and **`val`** are **mean L1 on log** over batches. Early on, **train can be higher than val** (e.g. epoch 1 train 0.58 vs val 0.16) because: **train** uses **shuffle + dropout**; **val** is **no shuffle, no dropout** in `eval_epoch`. Trust **best_val** and the **final dollar/log metrics**, not a literal train–val gap in epoch 1.

### 8.3 Comparing runs

Absolute MAE/MAPE **vary between runs** (initialization, ordering). For production decisions, compare **v3 vs CatBoost/ARIMA v4** on the **same splits** and treat **val + test** together rather than a single lucky epoch curve.

### 8.4 Historical bug (fixed): shuffled train metrics

Older versions evaluated **train** predictions using the **training `DataLoader` with `shuffle=True`** while `y_train` stayed in **dataframe order**, which **permuted** predictions vs labels and produced nonsense train MAE/R² (e.g. negative R² with excellent val/test). **Fix:** use **`train_eval_loader(..., shuffle=False)`** only for the final `eval_epoch` / `compute_metrics` on train. Validation and test loaders were always correct.

---

## 9. Summary

v3 uses an **LSTM** over **12 months** of market statistics — **town-(flat_type) calendar spine** with **optional sector-(flat_type) cells** when monthly volume is high enough, plus a **`postal_sector`** static embedding — and a **deep branch** on v2-style features to predict **log price**. It targets **temporal structure** trees do not get “for free,” at the cost of **complexity and careful leakage handling**. Read **train / val / test** together (with **unshuffled** train eval); compare **val + test** to **CatBoost/ARIMA v4** on the same splits.
