"""
Window-level economic signature models for direction A.

Two variants:
  - SimpleSummaryToken: [mean, std, max] of intensity -> linear -> conditioning token
  - ShapeSignatureToken: small causal encoder over full 168-step market window -> z_tau (d=16) -> conditioning token

Both inject via a single prepended conditioning token into the Transformer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SimpleSummaryToken(nn.Module):
    """Baseline: scalar intensity summary -> conditioning token."""

    def __init__(self, market_input_dim: int, d_model: int):
        super().__init__()
        # mean, std, max of intensity channel (last channel assumed)
        self.proj = nn.Linear(3, d_model)

    def forward(self, market_seq: torch.Tensor) -> torch.Tensor:
        # market_seq: (B, T, F), intensity is last channel
        intensity = market_seq[..., -1]  # (B, T)
        mean = intensity.mean(dim=1, keepdim=True)   # (B, 1)
        std = intensity.std(dim=1, keepdim=True)     # (B, 1)
        mx = intensity.max(dim=1).values.unsqueeze(1)  # (B, 1)
        summary = torch.cat([mean, std, mx], dim=1)  # (B, 3)
        return self.proj(summary)  # (B, d_model)


class ShapeSignatureEncoder(nn.Module):
    """Causal GRU encoder over full market window -> z_tau (d_sig)."""

    def __init__(self, market_input_dim: int, d_sig: int = 16, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(market_input_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_sig)

    def forward(self, market_seq: torch.Tensor) -> torch.Tensor:
        # market_seq: (B, T, F)
        _, h = self.gru(market_seq)  # h: (1, B, hidden)
        z = self.proj(h.squeeze(0))  # (B, d_sig)
        return z


class WindowSignatureHybrid(nn.Module):
    """
    Transformer + TCN hybrid conditioned on a window-level signature token.

    signature_mode:
      'simple'  -> SimpleSummaryToken
      'shape'   -> ShapeSignatureEncoder
    """

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_sig: int = 16,
        signature_mode: str = "shape",
        fusion_mode: str = "hybrid",
    ):
        super().__init__()
        assert fusion_mode in ("hybrid", "global_only"), "fusion_mode must be 'hybrid' or 'global_only'"
        self.signature_mode = signature_mode
        self.fusion_mode = fusion_mode

        if signature_mode == "simple":
            self.sig_encoder = SimpleSummaryToken(market_input_dim, d_model)
        else:
            self.sig_encoder = ShapeSignatureEncoder(market_input_dim, d_sig=d_sig)
            self.sig_proj = nn.Linear(d_sig, d_model)

        self.asset_proj = nn.Linear(asset_input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # TCN local branch (simple depthwise conv stack)
        self.tcn = nn.Sequential(
            nn.Conv1d(asset_input_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.gate = nn.Linear(d_model * 2, 1)
        self.head = nn.Linear(d_model, 1)

    def _sig_token(self, market_seq: torch.Tensor) -> torch.Tensor:
        if self.signature_mode == "simple":
            return self.sig_encoder(market_seq)  # (B, d_model)
        else:
            z = self.sig_encoder(market_seq)     # (B, d_sig)
            return self.sig_proj(z)              # (B, d_model)

    def forward(self, asset_src: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        # asset_src: (B, T, F_a), market_seq: (B, T, F_m)
        B, T, _ = asset_src.shape

        sig_tok = self._sig_token(market_seq).unsqueeze(1)  # (B, 1, d_model)
        x = self.asset_proj(asset_src)                       # (B, T, d_model)
        x_with_sig = torch.cat([sig_tok, x], dim=1)         # (B, T+1, d_model)

        global_out = self.transformer(x_with_sig)[:, 1:, :]  # (B, T, d_model) drop sig token
        global_feat = global_out[:, -1, :]                    # (B, d_model)

        if self.fusion_mode == "global_only":
            local_feat = torch.zeros_like(global_feat)
            gate_w = torch.ones(B, 1, device=asset_src.device, dtype=asset_src.dtype)
            fused = global_feat
        else:
            local_out = self.tcn(asset_src.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, d_model)
            local_feat = local_out[:, -1, :]                                     # (B, d_model)
            gate_w = torch.sigmoid(self.gate(torch.cat([global_feat, local_feat], dim=-1)))  # (B, 1)
            fused = gate_w * global_feat + (1 - gate_w) * local_feat                         # (B, d_model)
        pred = self.head(fused)                                                            # (B, 1)

        if return_diagnostics:
            diag = {
                "sig_token": sig_tok.squeeze(1),
                "gate": gate_w.squeeze(-1),
                "global_feat": global_feat,
                "local_feat": local_feat,
                "fused": fused,
                # stubs for compatibility with evaluate_model
                "tau": torch.zeros(B, T, device=asset_src.device),
                "step": torch.zeros(B, T, device=asset_src.device),
                "intensity": market_seq[..., -1],
                "qk_scores": torch.zeros(B, T, T, device=asset_src.device),
                "attention_map": torch.zeros(B, T, T, device=asset_src.device),
                "attention_importance": torch.zeros(B, T, device=asset_src.device),
                "joint_routing": torch.zeros(B, T, device=asset_src.device),
                "attention_bias": None,
                "pe_scale": torch.tensor(0.0),
                "alpha": torch.zeros(B),
                "context_gate": torch.zeros(B, T),
            }
            return pred, diag
        return pred
