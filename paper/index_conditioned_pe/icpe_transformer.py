from __future__ import annotations

import math

import torch
import torch.nn as nn


class ICPEPositionalEncoding(nn.Module):
    """
    Index-conditioned positional encoding.

    modes:
      - static: sinusoidal PE only
      - concat_a: state is concatenated before projection; PE remains static
      - film_a: state modulates projected tokens via FiLM; PE remains static
      - xip_a: state enters via explicit low-rank interaction projection; PE remains static
      - cycle_pe: sinusoidal PE + intensity projection
      - cycle_pe_intensity: fair intensity-only cycle PE with bias-free linear projection
      - cycle_pe_intensity_embed: fair intensity-only cycle PE with binned embedding
      - cycle_pe_full: sinusoidal PE + intensity + position projection
      - flow_pe: context-driven coordinate warping before sinusoidal evaluation
    """

    INTENSITY_IDX = 1
    POSITION_IDX = 0

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        context_dim: int = 0,
        dropout: float = 0.1,
        mode: str = "cycle_pe",
    ):
        super().__init__()
        assert mode in (
            "static",
            "concat_a",
            "film_a",
            "xip_a",
            "cycle_pe",
            "cycle_pe_intensity",
            "cycle_pe_intensity_embed",
            "cycle_pe_full",
            "flow_pe",
        )
        self.mode = mode
        self.context_dim = context_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.register_buffer("div_term", div)

        if mode in ("cycle_pe", "cycle_pe_intensity", "cycle_pe_full") and context_dim > 0:
            use_bias = mode != "cycle_pe_intensity"
            self.intensity_proj = nn.Linear(1, d_model, bias=use_bias)
        if mode == "cycle_pe_intensity_embed" and context_dim > 0:
            self.intensity_bins = 32
            self.intensity_embed = nn.Embedding(self.intensity_bins, d_model)
        if mode == "cycle_pe_full" and context_dim > 0:
            self.position_proj = nn.Linear(1, d_model)
        if mode == "flow_pe" and context_dim > 0:
            self.flow_mlp = nn.Sequential(
                nn.Linear(context_dim, max(8, d_model // 4)),
                nn.GELU(),
                nn.Linear(max(8, d_model // 4), 1),
                nn.Tanh(),
            )
            self.flow_scale = nn.Parameter(torch.tensor(0.25))

    def _build_flow_pe(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        step_adjust = self.flow_mlp(context)
        scale = 0.5 * torch.sigmoid(self.flow_scale)
        step = 1.0 + scale * step_adjust
        warped_pos = torch.cumsum(step, dim=1) - step[:, :1, :]

        pe = torch.zeros_like(x)
        div = self.div_term.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        pe[..., 0::2] = torch.sin(warped_pos * div)
        pe[..., 1::2] = torch.cos(warped_pos * div)
        return pe

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "flow_pe" and context is not None:
            x = x + self._build_flow_pe(x, context)
        else:
            x = x + self.pe[:, : x.size(1), :]
        if self.mode in ("cycle_pe", "cycle_pe_full") and context is not None:
            intensity = context[:, :, self.INTENSITY_IDX : self.INTENSITY_IDX + 1]
            x = x + self.intensity_proj(intensity)
        if self.mode == "cycle_pe_intensity" and context is not None:
            intensity = context[:, :, 0:1]
            x = x + self.intensity_proj(intensity)
        if self.mode == "cycle_pe_intensity_embed" and context is not None:
            intensity = context[:, :, 0].clamp(0.0, 1.0)
            idx = torch.clamp((intensity * self.intensity_bins).long(), max=self.intensity_bins - 1)
            x = x + self.intensity_embed(idx)
        if self.mode == "cycle_pe_full" and context is not None:
            position = context[:, :, self.POSITION_IDX : self.POSITION_IDX + 1]
            x = x + self.position_proj(position)
        return self.dropout(x)


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, return_attention: bool = False, attn_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask, need_weights=return_attention)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        if return_attention:
            return src, attn
        return src


class ICPETransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        context_dim: int = 0,
        use_causal_mask: bool = False,
        pe_mode: str = "cycle_pe",
    ):
        super().__init__()
        self.d_model = d_model
        self.pe_mode = pe_mode
        self.use_causal_mask = use_causal_mask

        proj_input_dim = input_dim + context_dim if pe_mode == "concat_a" else input_dim
        self.input_proj = nn.Linear(proj_input_dim, d_model)
        if pe_mode == "film_a" and context_dim > 0:
            self.film_gamma = nn.Linear(context_dim, d_model, bias=False)
            self.film_beta = nn.Linear(context_dim, d_model, bias=False)
        if pe_mode == "xip_a" and context_dim > 0:
            interaction_rank = 4
            self.state_proj = nn.Linear(context_dim, d_model, bias=False)
            self.x_inter_proj = nn.Linear(input_dim, interaction_rank, bias=False)
            self.s_inter_proj = nn.Linear(context_dim, interaction_rank, bias=False)
            self.inter_out_proj = nn.Linear(interaction_rank, d_model, bias=False)
        self.pos_encoder = ICPEPositionalEncoding(
            d_model=d_model,
            context_dim=context_dim,
            dropout=dropout,
            mode="static" if pe_mode in ("film_a", "xip_a") else pe_mode,
        )
        self.layers = nn.ModuleList(
            [
                AttentionEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def _causal_mask(self, seq_len: int, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        src: torch.Tensor,
        context: torch.Tensor | None = None,
        return_attention: bool = False,
        disable_xip_interaction: bool = False,
    ):
        raw_src = src
        if self.pe_mode == "concat_a" and context is not None:
            src = torch.cat([src, context], dim=-1)

        src = self.input_proj(src) * math.sqrt(self.d_model)
        xip_diag = None
        if self.pe_mode == "film_a" and context is not None:
            gamma = torch.tanh(self.film_gamma(context))
            beta = self.film_beta(context)
            src = src * (1.0 + gamma) + beta
        if self.pe_mode == "xip_a" and context is not None:
            state_main = self.state_proj(context)
            if disable_xip_interaction:
                interaction = torch.zeros_like(state_main)
            else:
                interaction = self.inter_out_proj(self.x_inter_proj(raw_src) * self.s_inter_proj(context))
            src = src + state_main + interaction
            xip_diag = {
                "xip_state_main": state_main,
                "xip_interaction": interaction,
            }
        src = self.pos_encoder(src, context=context)
        attn_mask = self._causal_mask(src.size(1), src.device) if self.use_causal_mask else None

        attn_weights = []
        for layer in self.layers:
            if return_attention:
                src, attn = layer(src, return_attention=True, attn_mask=attn_mask)
                attn_weights.append(attn)
            else:
                src = layer(src, return_attention=False, attn_mask=attn_mask)
        if return_attention:
            if xip_diag is not None:
                return src, attn_weights, xip_diag
            return src, attn_weights
        return src
