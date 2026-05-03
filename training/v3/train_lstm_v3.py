"""
train_lstm_v3.py — HDB resale price LSTM training pipeline (v3).

Pipeline:
  1. Load raw CSV; split 2017-2024 train / 2025 val / 2026 test.
  2. Engineer property features (reuses v2 backend functions).
  3. Build KDTree spatial features (same as v2).
  4. Build monthly market aggregates + LSTM sequences (new in v3).
  5. Fit StandardScaler + OrdinalEncoder on training data.
  6. Train hybrid LSTM+MLP model with AdamW + cosine LR + early stopping.
  7. Evaluate and save artifacts.

Artifacts saved to training/v3/artifacts/:
  model_lstm_v3.pt        — best model state dict
  config_lstm_v3.json     — model + training hyperparameters
  preprocessor_v3.pkl     — scalers, encoder, monthly_agg, mall_dist_median
  spatial_inference.pkl   — KDTree bundle (same format as v2, usable by backend)
  metrics_lstm_v3.json    — evaluation metrics + split metadata
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.inference_features import (  # noqa: E402
    SPATIAL_FEATS,
    add_macro_interaction_features,
    add_official_rpi,
    build_spatial_bundle_dict,
    compute_spatial_features,
    engineer_features,
)
from features_v3 import (  # noqa: E402
    CAT_FEATURES,
    DATA_PATH,
    HIST_YEAR_START,
    N_SEQ_FEATURES,
    N_STATIC,
    RPI_PATH,
    SEQ_LEN,
    STATIC_NUM_FEATURES,
    TARGET,
    TRAIN_YEAR_END,
    TRAIN_YEAR_START,
    VAL_YEAR,
    TEST_YEAR_START,
    TEST_YEAR_END,
    apply_preprocessors,
    build_monthly_agg,
    build_sequences,
    fit_preprocessors,
    split_data,
)
from model_lstm_v3 import HDBPriceLSTM, LSTMConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
LOCAL_ARTIFACTS = Path(__file__).parent / "artifacts"
MLFLOW_URI      = "http://127.0.0.1:5005/"
EXPERIMENT_NAME = "HDB Resale Telegram Bot"
MODEL_VERSION   = "lstm-v3"
RANDOM_SEED     = 42

# Training hyperparameters (override via config_override in main() for HPO)
BATCH_SIZE    = 512
N_EPOCHS      = 100
LR            = 0.0028079495906642563
WEIGHT_DECAY  = 0.00015900447562343164
GRAD_CLIP     = 1.0
PATIENCE      = 15    # early stopping on val log-MAE

# Model architecture (tuned by hpo_lstm_v3.py, best trial #32)
LSTM_HIDDEN   = 32
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.37275704669262566
STATIC_HIDDEN = 512
HEAD_HIDDEN   = 256
DROPOUT       = 0.11692465131727794
CAT_EMB_DIM   = 16


# ── Dataset ────────────────────────────────────────────────────────────────────

class HDBDataset(Dataset):
    def __init__(
        self,
        sequences:  np.ndarray,          # [N, seq_len, seq_feats] float32
        static_num: np.ndarray,          # [N, n_static] float32
        cat_dict:   Dict[str, np.ndarray],  # {name: [N] int64}
        targets:    np.ndarray,          # [N] float32
    ):
        self.sequences  = torch.from_numpy(sequences)
        self.static_num = torch.from_numpy(static_num)
        self.cat_tensors = {k: torch.from_numpy(v) for k, v in cat_dict.items()}
        self.targets     = torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.static_num[idx],
            {k: t[idx] for k, t in self.cat_tensors.items()},
            self.targets[idx],
        )


def collate_fn(batch):
    seqs, statics, cats_list, targets = zip(*batch)
    return (
        torch.stack(seqs),
        torch.stack(statics),
        {name: torch.stack([c[name] for c in cats_list]) for name in cats_list[0]},
        torch.stack(targets),
    )


# ── Training utilities ─────────────────────────────────────────────────────────

def train_epoch(
    model: HDBPriceLSTM,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total = 0.0
    for seqs, statics, cats, targets in loader:
        seqs    = seqs.to(device)
        statics = statics.to(device)
        cats    = {k: v.to(device) for k, v in cats.items()}
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(seqs, statics, cats), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(targets)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: HDBPriceLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    total  = 0.0
    preds  = []
    for seqs, statics, cats, targets in loader:
        seqs    = seqs.to(device)
        statics = statics.to(device)
        cats    = {k: v.to(device) for k, v in cats.items()}
        targets = targets.to(device)
        p       = model(seqs, statics, cats)
        total  += criterion(p, targets).item() * len(targets)
        preds.append(p.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds)


def compute_metrics(y_log_true: np.ndarray, y_log_pred: np.ndarray, name: str) -> dict:
    true = np.expm1(y_log_true)
    pred = np.expm1(y_log_pred)
    mae      = float(mean_absolute_error(true, pred))
    rmse     = float(np.sqrt(mean_squared_error(true, pred)))
    mape     = float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100)
    r2       = float(r2_score(true, pred))
    log_rmse = float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))
    logger.info(
        "%s: MAE=%s | RMSE=%s | MAPE=%.2f%% | R²=%.4f | log_RMSE=%.6f",
        name, f"{mae:,.0f}", f"{rmse:,.0f}", mape, r2, log_rmse,
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "log_RMSE": log_rmse}


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(config_override: dict | None = None) -> tuple:
    """
    Full training pipeline. Pass config_override dict to override any of the
    module-level hyperparameters (used by hpo_lstm_v3.py).
    Returns (model, prep, metrics_out).
    """
    LOCAL_ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logger.info("Device: %s", device)

    cfg = config_override or {}
    batch_size    = cfg.get("batch_size",    BATCH_SIZE)
    n_epochs      = cfg.get("n_epochs",      N_EPOCHS)
    lr            = cfg.get("lr",            LR)
    weight_decay  = cfg.get("weight_decay",  WEIGHT_DECAY)
    patience      = cfg.get("patience",      PATIENCE)
    lstm_hidden   = cfg.get("lstm_hidden",   LSTM_HIDDEN)
    lstm_layers   = cfg.get("lstm_layers",   LSTM_LAYERS)
    lstm_dropout  = cfg.get("lstm_dropout",  LSTM_DROPOUT)
    static_hidden = cfg.get("static_hidden", STATIC_HIDDEN)
    head_hidden   = cfg.get("head_hidden",   HEAD_HIDDEN)
    dropout       = cfg.get("dropout",       DROPOUT)
    cat_emb_dim   = cfg.get("cat_emb_dim",   CAT_EMB_DIM)

    t_start = time.perf_counter()

    # ── 1. Load + split ────────────────────────────────────────────────────────
    logger.info("Loading %s", DATA_PATH)
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    logger.info("Loaded %d rows", len(raw))
    raw_train, raw_val, raw_test = split_data(raw)

    mall_dist_median = raw_train["Mall_Nearest_Distance"].median()

    # ── 2. Property feature engineering ───────────────────────────────────────
    logger.info("Engineering features...")
    t0 = time.perf_counter()

    def _prep_features(df):
        df = engineer_features(df, mall_dist_median)
        df = add_official_rpi(df, RPI_PATH)
        return add_macro_interaction_features(df)

    train_df = _prep_features(raw_train)
    val_df   = _prep_features(raw_val)
    test_df  = _prep_features(raw_test)
    t_feat   = time.perf_counter() - t0
    logger.info("Feature engineering: %.2fs", t_feat)

    # ── 3. Spatial KDTree features ─────────────────────────────────────────────
    logger.info("Building KDTree spatial features...")
    t0 = time.perf_counter()
    bundle       = build_spatial_bundle_dict(train_df)
    spatial_path = LOCAL_ARTIFACTS / "spatial_inference.pkl"
    joblib.dump(bundle, spatial_path)
    tr_sp, val_sp, te_sp = compute_spatial_features(train_df, val_df, test_df)
    for feat, vals in tr_sp.items():
        train_df[feat] = vals
    for feat, vals in val_sp.items():
        val_df[feat] = vals
    for feat, vals in te_sp.items():
        test_df[feat] = vals
    t_spatial = time.perf_counter() - t0
    logger.info("Spatial features: %.2fs", t_spatial)

    # ── 4. Monthly aggregates + LSTM sequences ─────────────────────────────────
    # monthly_agg is built from 2017-2024 training-era data ONLY.
    # Using val/test rows here would leak within-period prices into their own
    # sequences (e.g. a Feb-2025 row's sequence would include Jan-2025 val stats).
    # Loading 2017-TRAIN_YEAR_END (not just 2020) gives 2020 train rows
    # a full 12-month lookback without exposing any val/test information.
    logger.info("Building monthly aggregates and sequences (seq_len=%d)...", SEQ_LEN)
    t0 = time.perf_counter()
    raw_for_agg = raw[
        (raw["Tranc_Year"] >= HIST_YEAR_START) & (raw["Tranc_Year"] <= TRAIN_YEAR_END)
    ].reset_index(drop=True)
    monthly_agg = build_monthly_agg(raw_for_agg, RPI_PATH)
    tr_seq  = build_sequences(train_df, monthly_agg, SEQ_LEN)
    val_seq = build_sequences(val_df,   monthly_agg, SEQ_LEN)
    te_seq  = build_sequences(test_df,  monthly_agg, SEQ_LEN)
    t_seq   = time.perf_counter() - t0
    logger.info("Sequences built: %.2fs", t_seq)

    # ── 5. Fit preprocessors on train, apply to all splits ────────────────────
    logger.info("Fitting preprocessors...")
    prep = fit_preprocessors(train_df, tr_seq)
    tr_static,  tr_seq_s,  tr_cats  = apply_preprocessors(train_df, tr_seq,  prep)
    val_static, val_seq_s, val_cats = apply_preprocessors(val_df,   val_seq, prep)
    te_static,  te_seq_s,  te_cats  = apply_preprocessors(test_df,  te_seq,  prep)

    y_train = train_df[TARGET].values.astype(np.float32)
    y_val   = val_df[TARGET].values.astype(np.float32)
    y_test  = test_df[TARGET].values.astype(np.float32)

    prep_out  = {**prep, "monthly_agg": monthly_agg, "mall_dist_median": float(mall_dist_median)}
    prep_path = LOCAL_ARTIFACTS / "preprocessor_v3.pkl"
    joblib.dump(prep_out, prep_path)
    logger.info("Saved preprocessor -> %s", prep_path)

    # ── 6. DataLoaders ─────────────────────────────────────────────────────────
    pin = str(device) != "cpu"
    train_loader = DataLoader(
        HDBDataset(tr_seq_s,  tr_static,  tr_cats,  y_train),
        batch_size=batch_size, shuffle=True,  num_workers=0,
        collate_fn=collate_fn, pin_memory=pin,
    )
    val_loader = DataLoader(
        HDBDataset(val_seq_s, val_static, val_cats, y_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        HDBDataset(te_seq_s,  te_static,  te_cats,  y_test),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    # ── 7. Model ───────────────────────────────────────────────────────────────
    model_cfg = LSTMConfig(
        seq_features=N_SEQ_FEATURES,
        seq_len=SEQ_LEN,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        n_static=N_STATIC,
        static_hidden=static_hidden,
        head_hidden=head_hidden,
        dropout=dropout,
        cat_vocab_sizes=prep["cat_vocab_sizes"],
        cat_emb_dim=cat_emb_dim,
    )
    model     = HDBPriceLSTM(model_cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    criterion = nn.L1Loss()  # MAE on log target (consistent with v2 CatBoost MAE loss)

    # ── 8. Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    best_epoch    = 0

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lstm_v3_hybrid_temporal"):
        mlflow.log_params({
            "model":          MODEL_VERSION,
            "train_years":    f"{TRAIN_YEAR_START}-{TRAIN_YEAR_END}",
            "val_year":       VAL_YEAR,
            "test_years":     f"{TEST_YEAR_START}-{TEST_YEAR_END}",
            "n_train":        len(y_train),
            "n_val":          len(y_val),
            "n_test":         len(y_test),
            "seq_len":        SEQ_LEN,
            "n_seq_features": N_SEQ_FEATURES,
            "n_static":       N_STATIC,
            "batch_size":     batch_size,
            "n_epochs":       n_epochs,
            "lr":             lr,
            "weight_decay":   weight_decay,
            "lstm_hidden":    lstm_hidden,
            "lstm_layers":    lstm_layers,
            "lstm_dropout":   lstm_dropout,
            "static_hidden":  static_hidden,
            "head_hidden":    head_hidden,
            "dropout":        dropout,
            "cat_emb_dim":    cat_emb_dim,
            "n_params":       n_params,
            "device":         str(device),
        })
        mlflow.log_metrics({
            "t_feature_engineering_s": t_feat,
            "t_kdtree_s":              t_spatial,
            "t_seq_build_s":           t_seq,
        })

        logger.info("Training (%d epochs, patience=%d)...", n_epochs, patience)
        t_train_start = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            tr_loss           = train_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP)
            val_loss, _       = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            current_lr        = scheduler.get_last_lr()[0]

            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "lr": current_lr}, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | train=%.6f | val=%.6f | lr=%.2e",
                    epoch, n_epochs, tr_loss, val_loss, current_lr,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch    = epoch
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info("Early stop at epoch %d (patience=%d)", epoch, patience)
                    break

        t_training_s = time.perf_counter() - t_train_start
        logger.info("Training done: %.2fs | best_epoch=%d | best_val=%.6f", t_training_s, best_epoch, best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("t_training_s", t_training_s)

        # Restore best weights
        model.load_state_dict(best_state)

        # ── 9. Evaluation ──────────────────────────────────────────────────────
        _, tr_preds  = eval_epoch(model, train_loader, criterion, device)
        _, val_preds = eval_epoch(model, val_loader,   criterion, device)
        _, te_preds  = eval_epoch(model, test_loader,  criterion, device)

        train_metrics = compute_metrics(y_train, tr_preds,  "Train")
        val_metrics   = compute_metrics(y_val,   val_preds, "Validation")
        test_metrics  = compute_metrics(y_test,  te_preds,  "Test")

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}":   v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}":  v for k, v in test_metrics.items()})

        # ── 10. Save artifacts ─────────────────────────────────────────────────
        model_path   = LOCAL_ARTIFACTS / "model_lstm_v3.pt"
        config_path  = LOCAL_ARTIFACTS / "config_lstm_v3.json"
        metrics_path = LOCAL_ARTIFACTS / "metrics_lstm_v3.json"

        torch.save(best_state, model_path)

        config_dict = {
            "seq_features":   model_cfg.seq_features,
            "seq_len":        model_cfg.seq_len,
            "lstm_hidden":    model_cfg.lstm_hidden,
            "lstm_layers":    model_cfg.lstm_layers,
            "lstm_dropout":   model_cfg.lstm_dropout,
            "n_static":       model_cfg.n_static,
            "static_hidden":  model_cfg.static_hidden,
            "head_hidden":    model_cfg.head_hidden,
            "dropout":        model_cfg.dropout,
            "cat_vocab_sizes": model_cfg.cat_vocab_sizes,
            "cat_emb_dim":    model_cfg.cat_emb_dim,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        metrics_out = {
            "trained_at":  datetime.now(timezone.utc).isoformat(),
            "model":       MODEL_VERSION,
            "split": {
                "train_years": [TRAIN_YEAR_START, TRAIN_YEAR_END],
                "val_year":    VAL_YEAR,
                "test_years":  [TEST_YEAR_START, TEST_YEAR_END],
                "n_train":     int(len(y_train)),
                "n_val":       int(len(y_val)),
                "n_test":      int(len(y_test)),
            },
            "architecture":  config_dict,
            "training": {
                "batch_size":   batch_size,
                "n_epochs":     n_epochs,
                "best_epoch":   best_epoch,
                "lr":           lr,
                "weight_decay": weight_decay,
                "grad_clip":    GRAD_CLIP,
                "patience":     patience,
                "t_training_s": t_training_s,
                "n_params":     n_params,
            },
            "train":          train_metrics,
            "validation":     val_metrics,
            "test":           test_metrics,
            "static_features": STATIC_NUM_FEATURES,
            "cat_features":    CAT_FEATURES,
            "seq_features":    ["log_mean_price", "log_std_price", "log_volume",
                                 "log_mean_area", "hdb_rpi", "month_sin", "month_cos"],
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)

        for p in [model_path, config_path, metrics_path, prep_path, spatial_path]:
            mlflow.log_artifact(str(p))

        try:
            from google.cloud import storage  # noqa: PLC0415
            gcs    = storage.Client()
            bucket = gcs.bucket("hdb-resale-artifacts")
            for fname in ["model_lstm_v3.pt", "config_lstm_v3.json",
                          "metrics_lstm_v3.json", "preprocessor_v3.pkl", "spatial_inference.pkl"]:
                fp = LOCAL_ARTIFACTS / fname
                if fp.exists():
                    bucket.blob(f"models/{fname}").upload_from_filename(str(fp))
                    logger.info("Uploaded gs://hdb-resale-artifacts/models/%s", fname)
        except Exception as e:
            logger.warning("GCS upload skipped: %s", e)

        t_total = time.perf_counter() - t_start
        mlflow.log_metric("t_total_s", t_total)
        logger.info("Total pipeline: %.2fs", t_total)
        logger.info("Training complete ✅")

    return model, prep, metrics_out


if __name__ == "__main__":
    main()
