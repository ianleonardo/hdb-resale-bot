"""
model_lstm_v3.py — Hybrid LSTM + static MLP model for HDB resale price prediction.

Architecture:
  LSTM branch  : encodes SEQ_LEN months of per-(town, flat_type) market statistics
                 → temporal market context embedding
  Static branch: encodes engineered property features + categorical embeddings
                 → property context embedding
  Fusion head  : concatenates both embeddings → log_resale_price

The LSTM captures price trends, market momentum, and seasonality that tree models
cannot represent as naturally.  The static branch replicates v2 CatBoost features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    # Sequence (LSTM) branch
    seq_features: int = 7
    seq_len: int = 12
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2     # applied between LSTM layers (ignored when layers=1)

    # Static MLP branch
    n_static: int = 43
    static_hidden: int = 256

    # Categorical embeddings
    cat_vocab_sizes: Dict[str, int] = field(default_factory=dict)
    cat_emb_dim: int = 8          # shared embedding dim for all categorical features

    # Fusion head
    head_hidden: int = 128
    dropout: float = 0.2          # applied in static branch and fusion head


class HDBPriceLSTM(nn.Module):
    """
    Two-branch price prediction network.

    Forward inputs:
      sequences  : [B, seq_len, seq_features]  — scaled market time-series
      static_num : [B, n_static]               — scaled static property features
      cat_inputs : {name: [B] int64}           — category indices (0 = unknown/pad)

    Output: [B] — predicted log1p(resale_price)
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        # ── Categorical embeddings (vocab_size + 1 to reserve idx-0 for unknown) ──
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab + 1, config.cat_emb_dim, padding_idx=0)
            for name, vocab in config.cat_vocab_sizes.items()
        })
        cat_total_dim = len(self.embeddings) * config.cat_emb_dim

        # ── LSTM branch ──────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=config.seq_features,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(config.lstm_hidden)
        self.lstm_proj = nn.Linear(config.lstm_hidden, config.head_hidden)

        # ── Static branch ────────────────────────────────────────────────────────
        static_in = config.n_static + cat_total_dim
        self.static_branch = nn.Sequential(
            nn.Linear(static_in, config.static_hidden),
            nn.LayerNorm(config.static_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.static_hidden, config.head_hidden),
            nn.GELU(),
        )

        # ── Fusion head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(config.head_hidden * 2, config.head_hidden),
            nn.LayerNorm(config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(config.head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "embedding" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
            nn.init.zeros_(emb.weight[0])  # unknown token stays at zero

    def forward(
        self,
        sequences: torch.Tensor,
        static_num: torch.Tensor,
        cat_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # LSTM branch: use last hidden state as temporal context
        lstm_out, _ = self.lstm(sequences)
        lstm_enc = self.lstm_norm(lstm_out[:, -1, :])  # [B, lstm_hidden]
        lstm_enc = self.lstm_proj(lstm_enc)             # [B, head_hidden]

        # Static branch: concat numerical + embeddings
        emb_list = [self.embeddings[name](cat_inputs[name]) for name in self.embeddings]
        static_full = torch.cat([static_num] + emb_list, dim=-1)  # [B, static_in]
        static_enc  = self.static_branch(static_full)              # [B, head_hidden]

        # Fusion
        fused = torch.cat([lstm_enc, static_enc], dim=-1)  # [B, head_hidden*2]
        return self.head(fused).squeeze(-1)                 # [B]
