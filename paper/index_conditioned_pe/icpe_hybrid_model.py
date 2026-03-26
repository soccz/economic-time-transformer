from __future__ import annotations

import torch
import torch.nn as nn

from paper.index_conditioned_pe.icpe_cvae import ICPECVAE
from paper.index_conditioned_pe.icpe_transformer import ICPETransformer


class SimpleGatedFusion(nn.Module):
    """Paper-specific scalar gate for global/local fusion."""

    def __init__(self, d_model: int, local_dim: int):
        super().__init__()
        self.local_proj = nn.Linear(local_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor):
        local_proj = self.local_proj(local_feat)
        gate = self.gate(torch.cat([global_feat, local_proj], dim=-1))
        fused = gate * global_feat + (1.0 - gate) * local_proj
        return fused, gate.squeeze(-1)


class PaperICPEHybrid(nn.Module):
    """
    Paper-specific hybrid model:
      Transformer encoder + routed TCN + gated fusion + point/quantile/CVAE head.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        dropout: float = 0.1,
        pe_mode: str = "cycle_pe",
        routing_mode: str = "attention",
        decoder_mode: str = "point",
        output_dim: int = 1,
    ):
        super().__init__()
        assert routing_mode in ("attention", "uniform", "random")
        self.decoder_mode = decoder_mode
        self.routing_mode = routing_mode

        self.transformer = build_transformer_encoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            context_dim=context_dim,
            use_causal_mask=False,
            pe_mode=pe_mode,
        )

        self.local_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=2, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 48, kernel_size=3, padding=4, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(48, local_dim, kernel_size=3, padding=8, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fusion = SimpleGatedFusion(d_model=d_model, local_dim=local_dim)

        if decoder_mode == "cvae":
            self.decoder = ICPECVAE(
                context_dim=d_model,
                output_dim=output_dim,
                latent_dim=16,
                hidden_dim=128,
                dropout=dropout,
            )
        elif decoder_mode == "quantile":
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )

    def _build_routing_weights(self, attention_importance: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "attention":
            return attention_importance
        if self.routing_mode == "uniform":
            seq_len = attention_importance.size(-1)
            return torch.full_like(attention_importance, 1.0 / seq_len)

        random_weights = torch.rand_like(attention_importance)
        return random_weights / random_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def encode(
        self,
        src: torch.Tensor,
        context: torch.Tensor,
        return_diagnostics: bool = False,
        disable_interaction: bool = False,
    ):
        transformer_out = self.transformer(
            src,
            context=context,
            return_attention=True,
            disable_xip_interaction=disable_interaction,
        )
        if len(transformer_out) == 3:
            encoded, attn_weights, transformer_diag = transformer_out
        else:
            encoded, attn_weights = transformer_out
            transformer_diag = {}
        global_feat = encoded[:, -1, :]

        last_attn = attn_weights[-1]
        attention_importance = torch.softmax(last_attn[:, -1, :], dim=-1)
        routing_weights = self._build_routing_weights(attention_importance)
        guided_src = src * routing_weights.unsqueeze(-1)
        local_feat = self.local_encoder(guided_src.permute(0, 2, 1)).squeeze(-1)

        fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            diag = {
                "attention_importance": attention_importance,
                "routing_weights": routing_weights,
                "routing_mode": self.routing_mode,
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
            diag.update(transformer_diag)
            return fused, diag
        return fused

    def forward(
        self,
        src: torch.Tensor,
        context: torch.Tensor,
        y_true: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        disable_interaction: bool = False,
    ):
        fused, diag = self.encode(
            src,
            context,
            return_diagnostics=True,
            disable_interaction=disable_interaction,
        )

        if self.decoder_mode == "cvae":
            if y_true is not None:
                pred, kl = self.decoder(fused, y_true)
            else:
                pred = self.decoder(fused)
                kl = torch.tensor(0.0, device=fused.device)
        elif self.decoder_mode == "quantile":
            pred = torch.sort(self.decoder(fused), dim=-1).values
            kl = torch.tensor(0.0, device=fused.device)
        else:
            pred = self.decoder(fused)
            kl = torch.tensor(0.0, device=fused.device)

        if return_diagnostics:
            diag["fused"] = fused
            diag["kl"] = kl
            return pred, diag
        return pred, kl

    @torch.no_grad()
    def predict_interval(
        self,
        src: torch.Tensor,
        context: torch.Tensor,
        n_samples: int = 100,
        alpha: float = 0.80,
    ):
        assert self.decoder_mode == "cvae", "predict_interval requires decoder_mode='cvae'"
        fused = self.encode(src, context, return_diagnostics=False)
        return self.decoder.predict_interval(fused, n_samples=n_samples, alpha=alpha)


def build_transformer_encoder(
    input_dim: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    context_dim: int,
    use_causal_mask: bool,
    pe_mode: str,
):
    return ICPETransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        context_dim=context_dim,
        use_causal_mask=use_causal_mask,
        pe_mode=pe_mode,
    )
