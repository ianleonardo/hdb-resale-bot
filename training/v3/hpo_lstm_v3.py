"""
hpo_lstm_v3.py — Hyperparameter optimisation for the HDB v3 LSTM model.

Strategy: rolling forward CV (folds: val years 2022, 2023, 2024).
Each trial trains for up to HPO_EPOCHS epochs with early stopping, then reports
median val log-MAE across folds × seeds.  TPE sampler + median pruner.

Usage:
    python hpo_lstm_v3.py              # run 60 trials (default)
    python hpo_lstm_v3.py --trials 30  # custom trial count
    python hpo_lstm_v3.py --fresh      # start a new study (ignore saved)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.inference_features import (  # noqa: E402
    add_macro_interaction_features,
    add_official_rpi,
    build_spatial_bundle_dict,
    compute_spatial_features,
    engineer_features,
)
from features_v3 import (  # noqa: E402
    DATA_PATH,
    HIST_YEAR_START,
    N_SEQ_FEATURES,
    RPI_PATH,
    SEQ_LEN,
    TARGET,
    add_postal_sector,
    apply_preprocessors,
    build_monthly_agg,
    build_sequences,
    fit_preprocessors,
)
from model_lstm_v3 import HDBPriceLSTM, LSTMConfig
from train_lstm_v3 import HDBDataset, collate_fn, eval_epoch, train_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOCAL_ARTIFACTS = Path(__file__).parent / "artifacts"
HPO_STUDY_PATH  = LOCAL_ARTIFACTS / "hpo_lstm_v3_study.pkl"

ROLLING_VAL_YEARS = [2022, 2023, 2024]
SEEDS             = [42, 142, 242]
HPO_EPOCHS        = 40
HPO_PATIENCE      = 8
HPO_BATCH         = 512
N_TRIALS          = 60


# ── Fold preparation ───────────────────────────────────────────────────────────

def _prepare_fold(raw_full: pd.DataFrame, val_year: int) -> dict | None:
    """
    Build train/val tensors for one rolling fold.
    train: 2020..(val_year-1)  |  val: val_year
    monthly_agg uses HIST_YEAR_START..(val_year-1) data only — no val leakage.
    Returns None if either split is empty.
    """
    # Model-fitting split: 2020..val_year-1 (mirrors TRAIN_YEAR_START=2020)
    train_raw = raw_full[
        (raw_full["Tranc_Year"] >= 2020) & (raw_full["Tranc_Year"] < val_year)
    ].reset_index(drop=True)
    val_raw = raw_full[raw_full["Tranc_Year"] == val_year].reset_index(drop=True)
    if len(train_raw) == 0 or len(val_raw) == 0:
        return None

    mall_med = train_raw["Mall_Nearest_Distance"].median()

    def _fe(df):
        df = engineer_features(df, mall_med)
        df = add_official_rpi(df, RPI_PATH)
        df = add_macro_interaction_features(df)
        return add_postal_sector(df)

    train_df = _fe(train_raw)
    val_df   = _fe(val_raw)

    empty = train_df.iloc[:0].copy()
    tr_sp, val_sp, _ = compute_spatial_features(train_df, val_df, empty)
    for feat, vals in tr_sp.items():
        train_df[feat] = vals
    for feat, vals in val_sp.items():
        val_df[feat] = vals

    # monthly_agg from 2017..(val_year-1) only — gives 2020 rows full 12-month
    # lookback without leaking any val-period prices into val sequences.
    agg_raw = raw_full[
        (raw_full["Tranc_Year"] >= HIST_YEAR_START) & (raw_full["Tranc_Year"] < val_year)
    ].reset_index(drop=True)
    monthly_agg = build_monthly_agg(agg_raw, RPI_PATH)
    tr_seq  = build_sequences(train_df, monthly_agg, SEQ_LEN)
    val_seq = build_sequences(val_df,   monthly_agg, SEQ_LEN)

    prep = fit_preprocessors(train_df, tr_seq)
    tr_static,  tr_seq_s,  tr_cats  = apply_preprocessors(train_df, tr_seq,  prep)
    val_static, val_seq_s, val_cats = apply_preprocessors(val_df,   val_seq, prep)

    return {
        "prep":       prep,
        "tr_seq":     tr_seq_s,
        "tr_static":  tr_static,
        "tr_cats":    tr_cats,
        "y_train":    train_df[TARGET].values.astype(np.float32),
        "val_seq":    val_seq_s,
        "val_static": val_static,
        "val_cats":   val_cats,
        "y_val":      val_df[TARGET].values.astype(np.float32),
    }


# ── Single trial × fold × seed ────────────────────────────────────────────────

def _run_fold(params: dict, fold: dict, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    cfg = LSTMConfig(
        seq_features=N_SEQ_FEATURES,
        seq_len=SEQ_LEN,
        lstm_hidden=params["lstm_hidden"],
        lstm_layers=params["lstm_layers"],
        lstm_dropout=params["lstm_dropout"],
        n_static=fold["tr_static"].shape[1],
        static_hidden=params["static_hidden"],
        head_hidden=params["head_hidden"],
        dropout=params["dropout"],
        cat_vocab_sizes=fold["prep"]["cat_vocab_sizes"],
        cat_emb_dim=params["cat_emb_dim"],
    )
    model     = HDBPriceLSTM(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS, eta_min=params["lr"] * 0.01)
    criterion = nn.L1Loss()

    train_loader = DataLoader(
        HDBDataset(fold["tr_seq"], fold["tr_static"], fold["tr_cats"], fold["y_train"]),
        batch_size=HPO_BATCH, shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        HDBDataset(fold["val_seq"], fold["val_static"], fold["val_cats"], fold["y_val"]),
        batch_size=HPO_BATCH, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    best_val = float("inf")
    bad      = 0
    for _ in range(HPO_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=1.0)
        val_loss, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            bad      = 0
        else:
            bad += 1
            if bad >= HPO_PATIENCE:
                break

    return best_val


# ── Optuna objective ───────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, folds: list[dict]) -> float:
    params = {
        "lstm_hidden":   trial.suggest_categorical("lstm_hidden",   [32, 64, 128, 256]),
        "lstm_layers":   trial.suggest_int("lstm_layers",           1, 3),
        "lstm_dropout":  trial.suggest_float("lstm_dropout",        0.0, 0.4),
        "static_hidden": trial.suggest_categorical("static_hidden", [128, 256, 512]),
        "head_hidden":   trial.suggest_categorical("head_hidden",   [64, 128, 256]),
        "dropout":       trial.suggest_float("dropout",             0.1, 0.4),
        "cat_emb_dim":   trial.suggest_categorical("cat_emb_dim",   [4, 8, 16]),
        "lr":            trial.suggest_float("lr",                  1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",        1e-4, 1e-1, log=True),
    }

    fold_scores: list[float] = []
    for step, fold in enumerate(folds):
        if fold is None:
            continue
        seed_scores: list[float] = []
        for seed in SEEDS:
            try:
                s = _run_fold(params, fold, seed)
                seed_scores.append(s)
            except Exception as e:
                logger.warning("Fold %d seed %d failed: %s", step, seed, e)
                return float("inf")
        fold_scores.append(float(np.median(seed_scores)))
        trial.report(float(np.median(fold_scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.median(fold_scores)) if fold_scores else float("inf")


# ── HPO entry point ────────────────────────────────────────────────────────────

def run_hpo(n_trials: int = N_TRIALS, resume: bool = True) -> optuna.Study:
    LOCAL_ARTIFACTS.mkdir(exist_ok=True)

    logger.info("Loading data for HPO (%s)...", DATA_PATH)
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    # Keep 2017-2024: HIST_YEAR_START for monthly_agg context; 2020+ for model fitting.
    raw = raw[(raw["Tranc_Year"] >= HIST_YEAR_START) & (raw["Tranc_Year"] <= 2024)].reset_index(drop=True)
    logger.info("HPO data: %d rows (%d-2024)", len(raw), HIST_YEAR_START)

    logger.info("Pre-building %d folds...", len(ROLLING_VAL_YEARS))
    folds = [_prepare_fold(raw, vy) for vy in ROLLING_VAL_YEARS]
    for vy, fold in zip(ROLLING_VAL_YEARS, folds):
        status = f"train={len(fold['y_train'])}, val={len(fold['y_val'])}" if fold else "EMPTY"
        logger.info("  val_year=%d | %s", vy, status)

    if resume and HPO_STUDY_PATH.exists():
        with open(HPO_STUDY_PATH, "rb") as f:
            study = pickle.load(f)
        logger.info("Resumed study: %d completed trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            study_name="hpo_lstm_v3",
        )
        logger.info("Created new study: hpo_lstm_v3")

    def _save_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        with open(HPO_STUDY_PATH, "wb") as f:
            pickle.dump(study, f)

    study.optimize(
        lambda trial: objective(trial, folds),
        n_trials=n_trials,
        callbacks=[_save_cb],
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("Best trial #%d | val_log_MAE=%.6f | params=%s", best.number, best.value, best.params)

    # Pretty-print best params as a config_override dict for train_lstm_v3.py
    print("\n# Best config_override for main():")
    print("{")
    for k, v in best.params.items():
        print(f'    "{k}": {v!r},')
    print("}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--fresh",  action="store_true", help="Start a new study")
    args = parser.parse_args()
    run_hpo(n_trials=args.trials, resume=not args.fresh)
