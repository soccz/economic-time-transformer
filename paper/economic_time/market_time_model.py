from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGatedFusion(nn.Module):
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


class AttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        relative_bias_mode: str = "none",
        relative_bias_gamma: float = 1.0,
        fixed_bias_slopes: bool = False,
        tau_conditioning: bool = True,
    ):
        super().__init__()
        assert relative_bias_mode in ("none", "abs", "relu", "signed")
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n_heads = n_heads
        self.relative_bias_mode = relative_bias_mode
        self.relative_bias_gamma = relative_bias_gamma
        self.fixed_bias_slopes = fixed_bias_slopes
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau_to_q_gamma = nn.Linear(d_model, d_model)
        self.tau_to_q_beta = nn.Linear(d_model, d_model)
        self.tau_to_k_gamma = nn.Linear(d_model, d_model)
        self.tau_to_k_beta = nn.Linear(d_model, d_model)
        tau_freq = torch.linspace(0.25, 2.0, steps=d_model // 2, dtype=torch.float32)
        self.register_buffer("tau_freq", tau_freq)
        if relative_bias_mode != "none":
            if fixed_bias_slopes:
                slopes = torch.tensor([2.0 ** (-i) for i in range(n_heads)], dtype=torch.float32)
                self.register_buffer("fixed_slopes", slopes)
            else:
                self.log_bias_slopes = nn.Parameter(torch.full((n_heads,), -1.5))
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.tau_conditioning = tau_conditioning
        self.activation = nn.GELU()

    def _build_relative_bias(self, tau: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.relative_bias_mode == "none":
            return None, None

        delta = tau.unsqueeze(2) - tau.unsqueeze(1)
        if self.relative_bias_mode == "abs":
            base_bias = -delta.abs()
        elif self.relative_bias_mode == "relu":
            base_bias = -F.relu(delta)
        else:
            base_bias = -delta

        if self.fixed_bias_slopes:
            slopes = self.fixed_slopes.view(1, self.n_heads, 1, 1).to(device=tau.device, dtype=tau.dtype)
        else:
            slopes = F.softplus(self.log_bias_slopes).view(1, self.n_heads, 1, 1)
        head_bias = self.relative_bias_gamma * slopes * base_bias.unsqueeze(1)
        attn_mask = head_bias.reshape(-1, tau.size(1), tau.size(1))
        mean_bias = head_bias.mean(dim=1)
        return attn_mask, mean_bias

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, src: torch.Tensor, tau: torch.Tensor | None = None, return_attention: bool = False):
        attn_bias, mean_bias = self._build_relative_bias(tau) if tau is not None else (None, None)
        q_base = self.q_proj(src)
        k_base = self.k_proj(src)
        v = self.v_proj(src)

        tau_embed = torch.zeros_like(src)
        if tau is not None and self.tau_conditioning:
            tau_scaled = tau.unsqueeze(-1) * self.tau_freq.view(1, 1, -1).to(device=src.device, dtype=src.dtype)
            tau_embed[..., 0::2] = torch.sin(tau_scaled)
            tau_embed[..., 1::2] = torch.cos(tau_scaled)

        if self.tau_conditioning:
            q = q_base * (1.0 + torch.tanh(self.tau_to_q_gamma(tau_embed))) + self.tau_to_q_beta(tau_embed)
            k = k_base * (1.0 + torch.tanh(self.tau_to_k_gamma(tau_embed))) + self.tau_to_k_beta(tau_embed)
        else:
            q, k = q_base, k_base

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)
        logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qk_scores = logits.mean(dim=1)
        if attn_bias is not None:
            logits = logits + attn_bias.view(src.size(0), self.n_heads, src.size(1), src.size(1))
        attn_weights = torch.softmax(logits, dim=-1)
        src2 = torch.matmul(attn_weights, vh).transpose(1, 2).contiguous().view(src.size(0), src.size(1), -1)
        src2 = self.out_proj(src2)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        if return_attention:
            return src, attn_weights.mean(dim=1), mean_bias, qk_scores
        return src


class MarketTrajectoryEncoder(nn.Module):
    """
    Encode recent broad-market trajectory into:
      - a global market summary
      - a monotone economic-time coordinate tau_t
      - market-path importance weights over the sequence
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.step_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.explicit_alpha = nn.Parameter(torch.tensor([1.5, -1.0, 1.0, 0.5, 0.75, 0.5], dtype=torch.float32))
        self.explicit_scale = nn.Parameter(torch.tensor(1.0))
        self.learned_scale = nn.Parameter(torch.tensor(0.25))
        self.base_log_step = nn.Parameter(torch.tensor(0.54132485))  # softplus^-1(1.0)

    def _explicit_market_features(self, market_seq: torch.Tensor) -> torch.Tensor:
        ret1 = market_seq[..., 0]
        position = market_seq[..., 2]
        intensity = market_seq[..., 3]

        steps = torch.arange(1, market_seq.size(1) + 1, device=market_seq.device, dtype=market_seq.dtype).view(1, -1)
        cum_ret = torch.cumsum(ret1, dim=1)
        running_vol = torch.sqrt(torch.cumsum(ret1.pow(2), dim=1) / steps.clamp_min(1.0))
        drawdown = cum_ret - torch.cummax(cum_ret, dim=1).values

        sign = torch.sign(ret1)
        turn = torch.zeros_like(ret1)
        turn[:, 1:] = (sign[:, 1:] * sign[:, :-1] < 0).float()
        turn_rate = torch.cumsum(turn, dim=1) / steps.clamp_min(1.0)

        return torch.stack(
            [
                cum_ret,
                drawdown,
                running_vol,
                turn_rate,
                position,
                intensity,
            ],
            dim=-1,
        )

    def forward(self, market_seq: torch.Tensor):
        path_hidden = self.input_proj(market_seq)
        path_hidden = path_hidden + self.temporal(path_hidden.permute(0, 2, 1)).permute(0, 2, 1)

        summary = path_hidden.mean(dim=1)
        summary_seq = summary.unsqueeze(1).expand(-1, path_hidden.size(1), -1)
        explicit_stats = self._explicit_market_features(market_seq)
        explicit_term = (explicit_stats * self.explicit_alpha.view(1, 1, -1)).sum(dim=-1, keepdim=True)
        learned_residual = self.step_residual(torch.cat([path_hidden, summary_seq], dim=-1))
        step_raw = (
            self.base_log_step
            + self.explicit_scale * explicit_term
            + self.learned_scale * learned_residual
        )
        step = F.softplus(step_raw)
        tau = torch.cumsum(step, dim=1) - step[:, :1, :]

        similarity = torch.cosine_similarity(path_hidden, summary_seq, dim=-1)
        market_weights = torch.softmax(similarity, dim=-1)
        return path_hidden, summary, tau, market_weights, {
            "explicit_stats": explicit_stats,
            "explicit_alpha": self.explicit_alpha.detach(),
            "explicit_term": explicit_term.squeeze(-1),
            "learned_residual": learned_residual.squeeze(-1),
            "step": step.squeeze(-1),
        }


class EconomicTimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Start nearly closed so Q/K conditioning can learn first.
        self.pe_scale_logit = nn.Parameter(torch.tensor([-2.1972246], dtype=torch.float32))
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        pe = torch.zeros_like(x)
        div = self.div_term.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        pe[..., 0::2] = torch.sin(tau * div)
        pe[..., 1::2] = torch.cos(tau * div)
        scale = self.current_scale(device=x.device, dtype=x.dtype)
        return self.dropout(x + scale * pe)

    def current_scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.sigmoid(self.pe_scale_logit).to(device=device, dtype=dtype)


class StaticPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        return self.dropout(x + self.pe[:, : x.size(1)].to(x.device, x.dtype))

    def current_scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(float("nan"), device=device, dtype=dtype)


class MarketTimeTransformer(nn.Module):
    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        relative_bias_mode: str = "relu",
        relative_bias_gamma: float = 1.0,
        fixed_bias_slopes: bool = False,
        pe_tau: bool = True,
        tau_conditioning: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pe_tau = pe_tau
        self.asset_proj = nn.Linear(asset_input_dim, d_model)
        self.market_encoder = MarketTrajectoryEncoder(market_input_dim, d_model)
        self.market_bias = nn.Linear(d_model, d_model)
        self.pos_encoder = (
            EconomicTimePositionalEncoding(d_model=d_model, dropout=dropout)
            if pe_tau
            else StaticPositionalEncoding(d_model=d_model, dropout=dropout)
        )
        self.layers = nn.ModuleList(
            [
                AttentionEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    relative_bias_mode=relative_bias_mode,
                    relative_bias_gamma=relative_bias_gamma,
                    fixed_bias_slopes=fixed_bias_slopes,
                    tau_conditioning=tau_conditioning,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_attention: bool = False):
        path_hidden, market_summary, tau, market_weights, step_diag = self.market_encoder(market_seq)
        src = self.asset_proj(asset_seq) * math.sqrt(self.d_model)
        src = src + self.market_bias(market_summary).unsqueeze(1)
        src = self.pos_encoder(src, tau=tau if self.pe_tau else None)

        attn_weights = []
        bias_matrices = []
        qk_matrices = []
        for layer in self.layers:
            if return_attention:
                src, attn, bias, qk = layer(src, tau=tau.squeeze(-1), return_attention=True)
                attn_weights.append(attn)
                bias_matrices.append(bias)
                qk_matrices.append(qk)
            else:
                src = layer(src, tau=tau.squeeze(-1), return_attention=False)
        if return_attention:
            return src, attn_weights, {
                "tau": tau.squeeze(-1),
                "market_weights": market_weights,
                "market_summary": market_summary,
                "path_hidden": path_hidden,
                "bias_matrices": bias_matrices,
                "qk_matrices": qk_matrices,
                "pe_scale": self.pos_encoder.current_scale(device=src.device, dtype=src.dtype),
                **step_diag,
            }
        return src


class MarketTimeHybrid(nn.Module):
    """
    Market-path-conditioned economic-time hybrid.

    ablation_mode controls which tau components are active:
      'pe_qk'  : PE(tau) + Q/K tau-conditioning  (full model)
      'pe_only': PE(tau) only, Q/K conditioning disabled
      'qk_only': static PE, Q/K tau-conditioning only
    """

    ABLATION_MODES = ("pe_qk", "pe_only", "qk_only")

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        dropout: float = 0.1,
        output_dim: int = 1,
        relative_bias_mode: str = "relu",
        relative_bias_gamma: float = 1.0,
        fixed_bias_slopes: bool = False,
        ablation_mode: str = "pe_qk",
    ):
        super().__init__()
        assert ablation_mode in self.ABLATION_MODES, f"ablation_mode must be one of {self.ABLATION_MODES}"
        self.ablation_mode = ablation_mode
        pe_tau = ablation_mode in ("pe_qk", "pe_only")
        tau_conditioning = ablation_mode in ("pe_qk", "qk_only")
        self.transformer = MarketTimeTransformer(
            asset_input_dim=asset_input_dim,
            market_input_dim=market_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            relative_bias_mode=relative_bias_mode,
            relative_bias_gamma=relative_bias_gamma,
            fixed_bias_slopes=fixed_bias_slopes,
            pe_tau=pe_tau,
            tau_conditioning=tau_conditioning,
        )

        local_input_dim = asset_input_dim + market_input_dim + 2
        self.local_context_gate = nn.Sequential(
            nn.Linear(asset_input_dim * 2 + 2, asset_input_dim),
            nn.GELU(),
            nn.Linear(asset_input_dim, 1),
            nn.Sigmoid(),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(local_input_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 48, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(48, local_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fusion = SimpleGatedFusion(d_model=d_model, local_dim=local_dim)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def encode(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        encoded, attn_weights, market_diag = self.transformer(
            asset_seq,
            market_seq,
            return_attention=True,
        )
        global_feat = encoded[:, -1, :]
        attention_importance = torch.softmax(attn_weights[-1][:, -1, :], dim=-1)
        joint_routing = torch.softmax(0.5 * (attention_importance + market_diag["market_weights"]), dim=-1)
        attn_context = torch.matmul(attn_weights[-1], asset_seq)
        tau = market_diag["tau"]
        tau_norm = tau / tau[:, -1:].clamp_min(1e-6)

        routed_asset = asset_seq * joint_routing.unsqueeze(-1)
        context_gate = self.local_context_gate(
            torch.cat(
                [
                    routed_asset,
                    attn_context,
                    joint_routing.unsqueeze(-1),
                    tau_norm.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        blended_asset = routed_asset + context_gate * (attn_context - routed_asset)
        local_input = torch.cat(
            [
                blended_asset,
                market_seq,
                joint_routing.unsqueeze(-1),
                tau_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        local_feat = self.local_encoder(local_input.permute(0, 2, 1)).squeeze(-1)
        fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            return fused, {
                "attention_importance": attention_importance,
                "attention_map": attn_weights[-1],
                "attention_bias": market_diag["bias_matrices"][-1],
                "qk_scores": market_diag["qk_matrices"][-1],
                "market_weights": market_diag["market_weights"],
                "joint_routing": joint_routing,
                "attn_context": attn_context,
                "context_gate": context_gate.squeeze(-1),
                "blended_asset": blended_asset,
                "tau": market_diag["tau"],
                "pe_scale": market_diag["pe_scale"],
                "step": market_diag["step"],
                "explicit_stats": market_diag["explicit_stats"],
                "explicit_alpha": market_diag["explicit_alpha"],
                "explicit_term": market_diag["explicit_term"],
                "learned_residual": market_diag["learned_residual"],
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
        return fused

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        fused, diag = self.encode(asset_seq, market_seq, return_diagnostics=True)
        pred = self.decoder(fused)
        if return_diagnostics:
            diag["fused"] = fused
            return pred, diag
        return pred


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, tau: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    angles = tau.unsqueeze(-1) * inv_freq.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class RuleTauBuilder(nn.Module):
    """Rule-based economic time: time flows faster when market intensity is high."""

    INTENSITY_IDX = 3

    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([init_alpha], dtype=torch.float32))

    def forward(self, market_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intensity = market_seq[..., self.INTENSITY_IDX]
        step = F.softplus(self.alpha.view(1, 1) * intensity)
        tau = torch.cumsum(step, dim=1) - step[:, :1]
        return tau, step


class StaticTauBuilder(nn.Module):
    """Physical-time baseline: tau_t = t."""

    def forward(self, market_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del market_seq
        raise NotImplementedError("StaticTauBuilder is not used directly; use StaticTauRoPETransformer.")


class LearnedTauBuilder(nn.Module):
    """Learn a monotone economic-time step process from the full market path."""

    def __init__(
        self,
        market_input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(market_input_dim)
        self.encoder = nn.GRU(
            input_size=market_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.step_head = nn.Linear(hidden_dim, 1)

        init_bias = math.log(math.expm1(1.0))
        nn.init.zeros_(self.step_head.weight)
        nn.init.constant_(self.step_head.bias, init_bias)

    def forward(self, market_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(self.input_norm(market_seq))
        step = F.softplus(self.step_head(encoded).squeeze(-1))
        tau = torch.cumsum(step, dim=1) - step[:, :1]
        return tau, step


class TauRoPEAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float, linear_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"
        self.linear_attention = linear_attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, src: torch.Tensor, tau: torch.Tensor, return_attention: bool = False):
        q = self._reshape_heads(self.q_proj(src))
        k = self._reshape_heads(self.k_proj(src))
        v = self._reshape_heads(self.v_proj(src))
        q = _apply_rope(q, tau, self.inv_freq)
        k = _apply_rope(k, tau, self.inv_freq)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qk_scores = logits.mean(dim=1)
        if self.linear_attention:
            # ELU+1 feature-map linear attention: removes softmax compression
            q_prime = F.elu(q, alpha=1.0) + 1.0  # (B, H, T, D)
            k_prime = F.elu(k, alpha=1.0) + 1.0
            # Numerator: Q' @ (K'^T @ V) — but for diagnostics we compute the full matrix
            attn_raw = torch.matmul(q_prime, k_prime.transpose(-2, -1))  # (B, H, T, T)
            attn_weights = attn_raw / attn_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            src2 = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(src.size(0), src.size(1), -1)
        else:
            attn_weights = torch.softmax(logits, dim=-1)
            src2 = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(src.size(0), src.size(1), -1)
        src2 = self.out_proj(src2)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        if return_attention:
            return src, attn_weights.mean(dim=1), qk_scores
        return src


class RuleTauRoPETransformer(nn.Module):
    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        linear_attention: bool = False,
    ):
        super().__init__()
        del market_input_dim  # the rule-based tau only needs the intensity channel at runtime
        self.d_model = d_model
        self.asset_proj = nn.Linear(asset_input_dim, d_model)
        self.tau_builder = RuleTauBuilder()
        self.layers = nn.ModuleList(
            [
                TauRoPEAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    linear_attention=linear_attention,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_attention: bool = False):
        tau, step = self.tau_builder(market_seq)
        src = self.asset_proj(asset_seq) * math.sqrt(self.d_model)

        attn_weights = []
        qk_matrices = []
        for layer in self.layers:
            if return_attention:
                src, attn, qk = layer(src, tau=tau, return_attention=True)
                attn_weights.append(attn)
                qk_matrices.append(qk)
            else:
                src = layer(src, tau=tau, return_attention=False)
        if return_attention:
            return src, attn_weights, {
                "tau": tau,
                "step": step,
                "intensity": market_seq[..., RuleTauBuilder.INTENSITY_IDX],
                "qk_matrices": qk_matrices,
                "alpha": self.tau_builder.alpha.detach(),
            }
        return src


class StaticTauRoPETransformer(nn.Module):
    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.asset_proj = nn.Linear(asset_input_dim, d_model)
        self.market_input_dim = market_input_dim
        self.layers = nn.ModuleList(
            [
                TauRoPEAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_attention: bool = False):
        batch, seq_len = asset_seq.size(0), asset_seq.size(1)
        tau = torch.arange(seq_len, device=asset_seq.device, dtype=asset_seq.dtype).view(1, -1).expand(batch, -1)
        step = torch.ones(batch, seq_len, device=asset_seq.device, dtype=asset_seq.dtype)
        src = self.asset_proj(asset_seq) * math.sqrt(self.asset_proj.out_features)

        attn_weights = []
        qk_matrices = []
        for layer in self.layers:
            if return_attention:
                src, attn, qk = layer(src, tau=tau, return_attention=True)
                attn_weights.append(attn)
                qk_matrices.append(qk)
            else:
                src = layer(src, tau=tau, return_attention=False)
        if return_attention:
            return src, attn_weights, {
                "tau": tau,
                "step": step,
                "intensity": market_seq[..., RuleTauBuilder.INTENSITY_IDX],
                "qk_matrices": qk_matrices,
            }
        return src


class RuleTauRoPEHybrid(nn.Module):
    """Minimal hybrid for rule-based tau-RoPE sanity checks."""

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        dropout: float = 0.1,
        output_dim: int = 1,
        linear_attention: bool = False,
    ):
        super().__init__()
        self.transformer = RuleTauRoPETransformer(
            asset_input_dim=asset_input_dim,
            market_input_dim=market_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            linear_attention=linear_attention,
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(asset_input_dim, 32, kernel_size=3, padding=2, dilation=1),
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
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def encode(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        encoded, attn_weights, tau_diag = self.transformer(asset_seq, market_seq, return_attention=True)
        global_feat = encoded[:, -1, :]
        attention_importance = torch.softmax(attn_weights[-1][:, -1, :], dim=-1)
        guided_src = asset_seq * attention_importance.unsqueeze(-1)
        local_feat = self.local_encoder(guided_src.permute(0, 2, 1)).squeeze(-1)
        fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            nan_vec = torch.full_like(attention_importance, float("nan"))
            return fused, {
                "attention_importance": attention_importance,
                "joint_routing": attention_importance,
                "attention_map": attn_weights[-1],
                "attention_bias": None,
                "qk_scores": tau_diag["qk_matrices"][-1],
                "tau": tau_diag["tau"],
                "step": tau_diag["step"],
                "intensity": tau_diag["intensity"],
                "alpha": tau_diag["alpha"],
                "pe_scale": torch.tensor(float("nan"), device=asset_seq.device),
                "context_gate": nan_vec,
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
        return fused

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        fused, diag = self.encode(asset_seq, market_seq, return_diagnostics=True)
        pred = self.decoder(fused)
        if return_diagnostics:
            diag["fused"] = fused
            return pred, diag
        return pred


class StaticTauRoPEHybrid(nn.Module):
    """Hybrid/global-only baseline with physical-time RoPE."""

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        dropout: float = 0.1,
        output_dim: int = 1,
        fusion_mode: str = "hybrid",
    ):
        super().__init__()
        assert fusion_mode in ("hybrid", "global_only"), "fusion_mode must be 'hybrid' or 'global_only'"
        self.fusion_mode = fusion_mode
        self.transformer = StaticTauRoPETransformer(
            asset_input_dim=asset_input_dim,
            market_input_dim=market_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(asset_input_dim, 32, kernel_size=3, padding=2, dilation=1),
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
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def encode(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        encoded, attn_weights, tau_diag = self.transformer(asset_seq, market_seq, return_attention=True)
        global_feat = encoded[:, -1, :]
        attention_importance = torch.softmax(attn_weights[-1][:, -1, :], dim=-1)
        if self.fusion_mode == "global_only":
            local_feat = torch.zeros(
                global_feat.size(0),
                self.fusion.local_proj.in_features,
                device=global_feat.device,
                dtype=global_feat.dtype,
            )
            fused = global_feat
            gate = torch.ones(global_feat.size(0), device=global_feat.device, dtype=global_feat.dtype)
        else:
            guided_src = asset_seq * attention_importance.unsqueeze(-1)
            local_feat = self.local_encoder(guided_src.permute(0, 2, 1)).squeeze(-1)
            fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            nan_vec = torch.full_like(attention_importance, float("nan"))
            return fused, {
                "attention_importance": attention_importance,
                "joint_routing": attention_importance,
                "attention_map": attn_weights[-1],
                "attention_bias": None,
                "qk_scores": tau_diag["qk_matrices"][-1],
                "tau": tau_diag["tau"],
                "step": tau_diag["step"],
                "intensity": tau_diag["intensity"],
                "alpha": torch.tensor(float("nan"), device=asset_seq.device),
                "pe_scale": torch.tensor(float("nan"), device=asset_seq.device),
                "context_gate": nan_vec,
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
        return fused

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        fused, diag = self.encode(asset_seq, market_seq, return_diagnostics=True)
        pred = self.decoder(fused)
        if return_diagnostics:
            diag["fused"] = fused
            return pred, diag
        return pred


class LearnedTauRoPETransformer(nn.Module):
    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tau_hidden_dim: int = 32,
        tau_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.asset_proj = nn.Linear(asset_input_dim, d_model)
        self.tau_builder = LearnedTauBuilder(
            market_input_dim=market_input_dim,
            hidden_dim=tau_hidden_dim,
            num_layers=tau_layers,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                TauRoPEAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_attention: bool = False):
        tau, step = self.tau_builder(market_seq)
        src = self.asset_proj(asset_seq) * math.sqrt(self.d_model)

        attn_weights = []
        qk_matrices = []
        for layer in self.layers:
            if return_attention:
                src, attn, qk = layer(src, tau=tau, return_attention=True)
                attn_weights.append(attn)
                qk_matrices.append(qk)
            else:
                src = layer(src, tau=tau, return_attention=False)
        if return_attention:
            return src, attn_weights, {
                "tau": tau,
                "step": step,
                "intensity": market_seq[..., RuleTauBuilder.INTENSITY_IDX],
                "qk_matrices": qk_matrices,
            }
        return src


class LearnedTauRoPEHybrid(nn.Module):
    """Hybrid model with a causal learned economic-time generator and tau-RoPE attention."""

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        tau_hidden_dim: int = 32,
        tau_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 1,
        fusion_mode: str = "hybrid",
    ):
        super().__init__()
        assert fusion_mode in ("hybrid", "global_only"), "fusion_mode must be 'hybrid' or 'global_only'"
        self.fusion_mode = fusion_mode
        self.transformer = LearnedTauRoPETransformer(
            asset_input_dim=asset_input_dim,
            market_input_dim=market_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tau_hidden_dim=tau_hidden_dim,
            tau_layers=tau_layers,
            dropout=dropout,
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(asset_input_dim, 32, kernel_size=3, padding=2, dilation=1),
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
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def encode(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        encoded, attn_weights, tau_diag = self.transformer(asset_seq, market_seq, return_attention=True)
        global_feat = encoded[:, -1, :]
        attention_importance = torch.softmax(attn_weights[-1][:, -1, :], dim=-1)
        if self.fusion_mode == "global_only":
            local_feat = torch.zeros(
                global_feat.size(0),
                self.fusion.local_proj.in_features,
                device=global_feat.device,
                dtype=global_feat.dtype,
            )
            fused = global_feat
            gate = torch.ones(global_feat.size(0), device=global_feat.device, dtype=global_feat.dtype)
        else:
            guided_src = asset_seq * attention_importance.unsqueeze(-1)
            local_feat = self.local_encoder(guided_src.permute(0, 2, 1)).squeeze(-1)
            fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            nan_vec = torch.full_like(attention_importance, float("nan"))
            return fused, {
                "attention_importance": attention_importance,
                "joint_routing": attention_importance,
                "attention_map": attn_weights[-1],
                "attention_bias": None,
                "qk_scores": tau_diag["qk_matrices"][-1],
                "tau": tau_diag["tau"],
                "step": tau_diag["step"],
                "intensity": tau_diag["intensity"],
                "alpha": torch.tensor(float("nan"), device=asset_seq.device),
                "pe_scale": torch.tensor(float("nan"), device=asset_seq.device),
                "context_gate": nan_vec,
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
        return fused

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, return_diagnostics: bool = False):
        fused, diag = self.encode(asset_seq, market_seq, return_diagnostics=True)
        pred = self.decoder(fused)
        if return_diagnostics:
            diag["fused"] = fused
            return pred, diag
        return pred


class TauRoPeConcatTransformer(nn.Module):
    """tau-RoPE attention with market-state concatenation to input.

    Combines two mechanisms:
      1. tau-RoPE: economic-time position encoding via rotary embeddings
      2. concat: market context (position+intensity) concatenated to asset input,
         giving the transformer direct access to cross-feature interactions.
    """

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        context_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # Project asset + context (concat) into d_model
        self.asset_proj = nn.Linear(asset_input_dim + context_dim, d_model)
        self.tau_builder = RuleTauBuilder()
        self.layers = nn.ModuleList(
            [
                TauRoPEAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        asset_seq: torch.Tensor,
        market_seq: torch.Tensor,
        context: torch.Tensor,
        return_attention: bool = False,
    ):
        tau, step = self.tau_builder(market_seq)
        # Concat context to asset input for feature-interaction access
        src_input = torch.cat([asset_seq, context], dim=-1)
        src = self.asset_proj(src_input) * math.sqrt(self.d_model)

        attn_weights = []
        qk_matrices = []
        for layer in self.layers:
            if return_attention:
                src, attn, qk = layer(src, tau=tau, return_attention=True)
                attn_weights.append(attn)
                qk_matrices.append(qk)
            else:
                src = layer(src, tau=tau, return_attention=False)
        if return_attention:
            return src, attn_weights, {
                "tau": tau,
                "step": step,
                "intensity": market_seq[..., RuleTauBuilder.INTENSITY_IDX],
                "qk_matrices": qk_matrices,
            }
        return src


class TauRoPeConcatHybrid(nn.Module):
    """Hybrid combining tau-RoPE (economic time geometry) with context concatenation
    (feature interaction access).

    Hypothesis: tau-RoPE alone fails because it can't access feature interactions.
    Concat alone works because it CAN access interactions.
    This model does both: tau-RoPE changes temporal geometry AND concat provides
    interaction access.
    """

    def __init__(
        self,
        asset_input_dim: int,
        market_input_dim: int,
        context_dim: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        local_dim: int = 64,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.transformer = TauRoPeConcatTransformer(
            asset_input_dim=asset_input_dim,
            market_input_dim=market_input_dim,
            context_dim=context_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.local_encoder = nn.Sequential(
            nn.Conv1d(asset_input_dim, 32, kernel_size=3, padding=2, dilation=1),
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
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def encode(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, context: torch.Tensor, return_diagnostics: bool = False):
        encoded, attn_weights, tau_diag = self.transformer(
            asset_seq, market_seq, context, return_attention=True,
        )
        global_feat = encoded[:, -1, :]
        attention_importance = torch.softmax(attn_weights[-1][:, -1, :], dim=-1)
        guided_src = asset_seq * attention_importance.unsqueeze(-1)
        local_feat = self.local_encoder(guided_src.permute(0, 2, 1)).squeeze(-1)
        fused, gate = self.fusion(global_feat, local_feat)

        if return_diagnostics:
            nan_vec = torch.full_like(attention_importance, float("nan"))
            return fused, {
                "attention_importance": attention_importance,
                "joint_routing": attention_importance,
                "attention_map": attn_weights[-1],
                "attention_bias": None,
                "qk_scores": tau_diag["qk_matrices"][-1],
                "tau": tau_diag["tau"],
                "step": tau_diag["step"],
                "intensity": tau_diag["intensity"],
                "alpha": self.transformer.tau_builder.alpha.detach(),
                "pe_scale": torch.tensor(float("nan"), device=asset_seq.device),
                "context_gate": nan_vec,
                "gate": gate,
                "global_feat": global_feat,
                "local_feat": local_feat,
            }
        return fused

    def forward(self, asset_seq: torch.Tensor, market_seq: torch.Tensor, context: torch.Tensor, return_diagnostics: bool = False):
        fused, diag = self.encode(asset_seq, market_seq, context, return_diagnostics=True)
        pred = self.decoder(fused)
        if return_diagnostics:
            diag["fused"] = fused
            return pred, diag
        return pred
