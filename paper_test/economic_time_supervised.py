"""
Minimal supervised runner for testing the stronger economic-time thesis.

Compares:
  - static baseline
  - concat state baseline
  - market-path-conditioned economic-time hybrid (ablation_mode: pe_qk / pe_only / qk_only)
  - rule-based tau-RoPE hybrid
  - learned tau-RoPE hybrid

model_kinds syntax:
  static, concat_a, concat_a:no_intensity, concat_a:intensity_only,
  film_a:intensity_only, film_a:indexret_only, film_a:intensity_indexret,
  xip_a:intensity_only, xip_a:indexret_only, xip_a:intensity_indexret,
  cycle_pe:intensity_only, cycle_pe:intensity_embed,
  concat_a:binned_intensity_only, concat_a:shuffled_intensity,
  concat_a:position_only, concat_a:indexret_only, concat_a:intensity_indexret,
  econ_time, econ_time:pe_only, econ_time:qk_only, tau_rope, learned_tau_rope,
  tau_rope_concat, tau_rope_linear
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper.economic_time.market_time_model import (  # noqa: E402
    LearnedTauRoPEHybrid,
    MarketTimeHybrid,
    RuleTauRoPEHybrid,
    StaticTauRoPEHybrid,
    TauRoPeConcatHybrid,
)
from paper.economic_time.window_signature_model import WindowSignatureHybrid  # noqa: E402
from paper.index_conditioned_pe.icpe_hybrid_model import PaperICPEHybrid  # noqa: E402
from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_features,
    build_state,
    build_target,
    load_data,
    set_seed,
)


@dataclass
class EconSplitData:
    asset_src: np.ndarray
    market_seq: np.ndarray
    context: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    regime: np.ndarray


class EconDataset(Dataset):
    def __init__(self, split: EconSplitData):
        self.asset_src = torch.from_numpy(split.asset_src).float()
        self.market_seq = torch.from_numpy(split.market_seq).float()
        self.context = torch.from_numpy(split.context).float()
        self.y = torch.from_numpy(split.y).float().unsqueeze(-1)
        self.dates = split.dates
        self.regime = split.regime

    def __len__(self) -> int:
        return self.asset_src.size(0)

    def __getitem__(self, idx: int):
        return (
            self.asset_src[idx],
            self.market_seq[idx],
            self.context[idx],
            self.y[idx],
            self.dates[idx],
            self.regime[idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economic-time supervised comparison")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--relative-bias-mode", choices=("none", "abs", "relu", "signed"), default="relu")
    parser.add_argument("--relative-bias-gamma", type=float, default=1.0)
    parser.add_argument("--fixed-bias-slopes", action="store_true")
    parser.add_argument("--tau-align-lambda", type=float, default=0.1)
    parser.add_argument("--tau-geom-lambda", type=float, default=0.0)
    parser.add_argument("--tau-geom-warmup-epochs", type=int, default=1)
    parser.add_argument("--tau-ord-lambda", type=float, default=0.0)
    parser.add_argument("--tau-ord-margin", type=float, default=1e-3)
    parser.add_argument("--tau-ord-sigmas", default="0.1,0.2,0.4")
    parser.add_argument(
        "--model-kinds",
        default="static,concat_a,econ_time,econ_time:pe_only,econ_time:qk_only",
        help="comma-separated; econ_time variants: econ_time, econ_time:pe_only, econ_time:qk_only, tau_rope, learned_tau_rope",
    )
    parser.add_argument("--output-dir", default="paper/economic_time/results")
    return parser.parse_args()


def build_market_features(index_close: pd.Series, position: pd.Series, intensity: pd.Series) -> np.ndarray:
    log_ret = np.log(index_close / index_close.shift(1)).fillna(0.0)
    ret_mean5 = log_ret.rolling(5).mean().fillna(0.0)
    market_df = pd.DataFrame(
        {
            "ret1": log_ret,
            "ret5": ret_mean5,
            "position": position.fillna(0.0),
            "intensity": intensity.fillna(0.0),
        },
        index=index_close.index,
    )
    return market_df.values.astype(np.float32)


def build_split(
    dates: pd.Index,
    source: pd.DataFrame,
    target: pd.DataFrame,
    position: pd.Series,
    intensity: pd.Series,
    regime: pd.Series,
    index_close: pd.Series,
    seq_len: int,
) -> tuple[EconSplitData, EconSplitData, EconSplitData]:
    raw, mean5, std5, xs_rank = build_features(source)
    market_arr = build_market_features(index_close, position, intensity)
    pos_arr = position.fillna(0.0).values
    int_arr = intensity.fillna(0.0).values
    reg_arr = regime.values

    asset_list, market_list, ctx_list, y_list, date_list, regime_list = [], [], [], [], [], []

    for t in range(seq_len, len(dates)):
        date = dates[t]
        for asset_idx in range(source.shape[1]):
            y_true = target.iloc[t, asset_idx]
            if pd.isna(y_true):
                continue
            asset_seq = np.column_stack([
                raw[t - seq_len:t, asset_idx],
                mean5[t - seq_len:t, asset_idx],
                std5[t - seq_len:t, asset_idx],
                xs_rank[t - seq_len:t, asset_idx],
            ])
            mkt_seq = market_arr[t - seq_len:t]
            ctx_seq = np.column_stack([pos_arr[t - seq_len:t], int_arr[t - seq_len:t]])
            if np.isnan(asset_seq).any() or np.isnan(mkt_seq).any():
                continue
            asset_list.append(asset_seq.astype(np.float32))
            market_list.append(mkt_seq.astype(np.float32))
            ctx_list.append(ctx_seq.astype(np.float32))
            y_list.append(float(y_true))
            date_list.append(date)
            regime_list.append(int(reg_arr[t - 1]) if not pd.isna(reg_arr[t - 1]) else -1)

    split = EconSplitData(
        asset_src=np.stack(asset_list),
        market_seq=np.stack(market_list),
        context=np.stack(ctx_list),
        y=np.array(y_list, dtype=np.float32),
        dates=np.array(date_list, dtype="datetime64[ns]").astype("int64"),
        regime=np.array(regime_list, dtype=np.int64),
    )

    unique_dates = np.array(sorted(pd.unique(split.dates)))
    train_cut = int(len(unique_dates) * 0.7)
    val_cut = int(len(unique_dates) * 0.85)
    train_mask = np.isin(split.dates, unique_dates[:train_cut])
    val_mask = np.isin(split.dates, unique_dates[train_cut:val_cut])
    test_mask = np.isin(split.dates, unique_dates[val_cut:])
    return filter_split(split, train_mask), filter_split(split, val_mask), filter_split(split, test_mask)


def filter_split(split: EconSplitData, mask: np.ndarray) -> EconSplitData:
    return EconSplitData(
        asset_src=split.asset_src[mask],
        market_seq=split.market_seq[mask],
        context=split.context[mask],
        y=split.y[mask],
        dates=split.dates[mask],
        regime=split.regime[mask],
    )


def build_loader(split: EconSplitData, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(EconDataset(split), batch_size=batch_size, shuffle=shuffle)


def _is_econ(model_kind: str) -> bool:
    return model_kind.startswith("econ_time")


def _is_tau_rope_linear(model_kind: str) -> bool:
    return model_kind.split(":", 1)[0] == "tau_rope_linear"


def _is_tau_rope(model_kind: str) -> bool:
    base = model_kind.split(":", 1)[0]
    return base in ("tau_rope", "learned_tau_rope", "static_tau_rope", "tau_rope_linear")


def _is_tau_rope_concat(model_kind: str) -> bool:
    return model_kind.split(":", 1)[0] == "tau_rope_concat"


def _is_learned_tau_rope(model_kind: str) -> bool:
    return model_kind.split(":", 1)[0] == "learned_tau_rope"


def _is_window_sig(model_kind: str) -> bool:
    base = model_kind.split(":", 1)[0]
    return base in ("simple_summary_token", "shape_signature_token")


def _base_model_kind(model_kind: str) -> str:
    return model_kind.split(":", 1)[0]


def _model_variant(model_kind: str) -> str:
    return model_kind.split(":", 1)[1] if ":" in model_kind else ""


def _uses_market_model(model_kind: str) -> bool:
    return _is_econ(model_kind) or _is_tau_rope(model_kind) or _is_tau_rope_concat(model_kind) or _is_window_sig(model_kind)


def _prepare_context(model_kind: str, context: torch.Tensor, market_seq: torch.Tensor | None = None) -> torch.Tensor:
    base = _base_model_kind(model_kind)
    variant = _model_variant(model_kind)
    if base in ("concat_a", "cycle_pe", "film_a", "xip_a") and variant in (
        "intensity_only",
        "intensity_embed",
        "binned_intensity_only",
        "binned_all",
        "shuffled_intensity",
        "position_only",
        "indexret_only",
        "intensity_indexret",
    ):
        if variant == "position_only":
            context = context[..., :1].clone()
        elif variant == "indexret_only":
            assert market_seq is not None
            context = market_seq[..., 0:1].clone()
        elif variant == "intensity_indexret":
            assert market_seq is not None
            context = torch.cat([market_seq[..., 0:1], context[..., -1:].clone()], dim=-1)
        elif variant == "binned_all":
            context = context.clone()  # keep all channels, will be discretized below
        else:
            context = context[..., -1:].clone()
    if base in ("concat_a", "film_a", "xip_a") and variant == "no_intensity":
        context = context.clone()
        context[..., -1] = 0.0
    if base in ("concat_a", "film_a", "xip_a") and variant == "binned_intensity_only":
        bins = 16
        intensity = context[..., 0].clamp(0.0, 1.0)
        bin_idx = torch.clamp((intensity * bins).long(), max=bins - 1)
        context = ((bin_idx.float() + 0.5) / bins).unsqueeze(-1)
    if base in ("concat_a", "film_a", "xip_a") and variant == "binned_all":
        # Discretize ALL conditioning channels into 4 bins each (high-SNR regime labels)
        bins = 4
        out_channels = []
        for ch in range(context.shape[-1]):
            signal = context[..., ch].clamp(0.0, 1.0)
            bin_idx = torch.clamp((signal * bins).long(), max=bins - 1)
            out_channels.append(((bin_idx.float() + 0.5) / bins).unsqueeze(-1))
        context = torch.cat(out_channels, dim=-1)
    if base in ("concat_a", "film_a", "xip_a") and variant == "shuffled_intensity":
        intensity = context[..., 0:1]
        order = torch.argsort(torch.rand(intensity.shape[:2], device=intensity.device, dtype=intensity.dtype), dim=1)
        context = intensity.gather(1, order.unsqueeze(-1))
    return context


def build_model(args: argparse.Namespace, model_kind: str, train_split: EconSplitData):
    if _is_econ(model_kind):
        ablation = model_kind.split(":", 1)[1] if ":" in model_kind else "pe_qk"
        return MarketTimeHybrid(
            asset_input_dim=train_split.asset_src.shape[-1],
            market_input_dim=train_split.market_seq.shape[-1],
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            relative_bias_mode=args.relative_bias_mode,
            relative_bias_gamma=args.relative_bias_gamma,
            fixed_bias_slopes=args.fixed_bias_slopes,
            ablation_mode=ablation,
        ).to(args.device)
    if _is_tau_rope_concat(model_kind):
        return TauRoPeConcatHybrid(
            asset_input_dim=train_split.asset_src.shape[-1],
            market_input_dim=train_split.market_seq.shape[-1],
            context_dim=train_split.context.shape[-1],
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
        ).to(args.device)
    if _is_tau_rope(model_kind):
        base, variant = (model_kind.split(":", 1) + [""])[:2]
        if base == "learned_tau_rope":
            rope_cls = LearnedTauRoPEHybrid
        elif base == "static_tau_rope":
            rope_cls = StaticTauRoPEHybrid
        else:
            rope_cls = RuleTauRoPEHybrid
        kwargs = dict(
            asset_input_dim=train_split.asset_src.shape[-1],
            market_input_dim=train_split.market_seq.shape[-1],
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
        )
        if base in ("learned_tau_rope", "static_tau_rope"):
            kwargs["fusion_mode"] = variant or "hybrid"
        if base == "tau_rope_linear":
            kwargs["linear_attention"] = True
        return rope_cls(**kwargs).to(args.device)

    if _is_window_sig(model_kind):
        base, variant = (model_kind.split(":", 1) + [""])[:2]
        sig_mode = "simple" if base == "simple_summary_token" else "shape"
        return WindowSignatureHybrid(
            asset_input_dim=train_split.asset_src.shape[-1],
            market_input_dim=train_split.market_seq.shape[-1],
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            d_sig=16,
            signature_mode=sig_mode,
            fusion_mode=variant or "hybrid",
        ).to(args.device)

    base = _base_model_kind(model_kind)
    variant = _model_variant(model_kind)
    if base == "static":
        pe_mode = "static"
        context_dim = train_split.context.shape[-1]
    elif base == "concat_a":
        pe_mode = "concat_a"
        context_dim = (
            1
            if variant in (
                "intensity_only",
                "binned_intensity_only",
                "shuffled_intensity",
                "position_only",
                "indexret_only",
            )
            else train_split.context.shape[-1]
        )
        if variant == "intensity_indexret":
            context_dim = 2
    elif base == "film_a":
        pe_mode = "film_a"
        context_dim = (
            1
            if variant in (
                "intensity_only",
                "binned_intensity_only",
                "shuffled_intensity",
                "position_only",
                "indexret_only",
            )
            else train_split.context.shape[-1]
        )
        if variant == "intensity_indexret":
            context_dim = 2
    elif base == "xip_a":
        pe_mode = "xip_a"
        context_dim = (
            1
            if variant in (
                "intensity_only",
                "binned_intensity_only",
                "shuffled_intensity",
                "position_only",
                "indexret_only",
            )
            else train_split.context.shape[-1]
        )
        if variant == "intensity_indexret":
            context_dim = 2
    elif base == "cycle_pe":
        if variant == "intensity_only":
            pe_mode = "cycle_pe_intensity"
        elif variant == "intensity_embed":
            pe_mode = "cycle_pe_intensity_embed"
        else:
            pe_mode = "cycle_pe"
        context_dim = 1 if variant == "intensity_only" else train_split.context.shape[-1]
        if variant == "intensity_embed":
            context_dim = 1
    else:
        pe_mode = "concat_a"
        context_dim = train_split.context.shape[-1]
    return PaperICPEHybrid(
        input_dim=train_split.asset_src.shape[-1],
        context_dim=context_dim,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        pe_mode=pe_mode,
        decoder_mode="point",
    ).to(args.device)


def forward_model(model, model_kind: str, asset_src: torch.Tensor, market_seq: torch.Tensor, context: torch.Tensor):
    if _is_tau_rope_concat(model_kind):
        return model(asset_src, market_seq, context)
    if _uses_market_model(model_kind):
        return model(asset_src, market_seq)
    context = _prepare_context(model_kind, context, market_seq)
    pred, _ = model(asset_src, context)
    return pred


def compute_tau_metrics(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = torch.arange(tau.size(1), device=tau.device, dtype=tau.dtype).view(1, -1)
    tau_dev_std = (tau - ref).std(dim=1)
    step = tau[:, 1:] - tau[:, :-1]
    step_dev_std = (step - 1.0).std(dim=1)
    ref_centered = ref - ref.mean(dim=1, keepdim=True)
    tau_centered = tau - tau.mean(dim=1, keepdim=True)
    corr_num = (tau_centered * ref_centered).sum(dim=1)
    corr_den = torch.sqrt((tau_centered.pow(2).sum(dim=1) * ref_centered.pow(2).sum(dim=1)).clamp_min(1e-8))
    return tau_dev_std, step_dev_std, corr_num / corr_den


def compute_step_intensity_spearman(step: torch.Tensor, intensity: torch.Tensor) -> np.ndarray:
    step_np = step.detach().cpu().numpy()
    intensity_np = intensity.detach().cpu().numpy()
    corr = np.full(step_np.shape[0], np.nan, dtype=np.float64)
    for idx in range(step_np.shape[0]):
        if np.std(step_np[idx]) == 0 or np.std(intensity_np[idx]) == 0:
            continue
        corr[idx] = float(stats.spearmanr(step_np[idx], intensity_np[idx])[0])
    return corr


def _attn_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) averaged over query positions, per sample. Shape: (B,)"""
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1).mean(dim=-1)


def batch_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    denom = torch.sqrt(x.pow(2).sum(dim=1).clamp_min(eps) * y.pow(2).sum(dim=1).clamp_min(eps))
    return (x * y).sum(dim=1) / denom


def batch_cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    cos = F.cosine_similarity(x_flat, y_flat, dim=1, eps=eps)
    return 1.0 - cos


def batch_normalized_l2_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    x_norm = (x_flat - x_flat.mean(dim=1, keepdim=True)) / (x_flat.std(dim=1, keepdim=True).clamp_min(eps))
    y_norm = (y_flat - y_flat.mean(dim=1, keepdim=True)) / (y_flat.std(dim=1, keepdim=True).clamp_min(eps))
    return (x_norm - y_norm).pow(2).mean(dim=1)


def batch_l2_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    return (x_flat - y_flat).pow(2).mean(dim=1).clamp_min(eps).sqrt()


def build_ordered_market_perturbations(
    market_seq: torch.Tensor,
    sigmas: list[float],
    intensity_idx: int = 3,
) -> list[torch.Tensor]:
    base_noise = torch.randn_like(market_seq)
    channel_scale = market_seq.std(dim=1, keepdim=True).clamp_min(1e-4)
    outputs = []
    for sigma in sigmas:
        perturbed = market_seq + sigma * channel_scale * base_noise
        perturbed = perturbed.clone()
        perturbed[..., intensity_idx] = perturbed[..., intensity_idx].clamp(0.0, 1.0)
        outputs.append(perturbed)
    return outputs


def ordinal_margin_loss(
    base: torch.Tensor,
    variants: list[torch.Tensor],
    margin: float,
    distance_fn=batch_cosine_distance,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    distances = [distance_fn(base, variant) for variant in variants]
    if len(distances) < 2:
        return torch.tensor(0.0, device=base.device), distances
    loss = torch.tensor(0.0, device=base.device)
    for idx in range(len(distances) - 1):
        loss = loss + F.relu(margin - (distances[idx + 1] - distances[idx])).mean()
    return loss, distances


def summarize_ordinal_distances(
    distances: list[torch.Tensor],
    margin: float,
) -> dict[str, float]:
    summary: dict[str, float] = {
        "d1": np.nan,
        "d2": np.nan,
        "d3": np.nan,
        "gap12": np.nan,
        "gap23": np.nan,
        "order_rate": np.nan,
        "margin_rate": np.nan,
        "active_frac": np.nan,
    }
    if not distances:
        return summary

    distance_means = [float(d.mean().item()) for d in distances[:3]]
    for idx, value in enumerate(distance_means, start=1):
        summary[f"d{idx}"] = value

    if len(distances) < 2:
        return summary

    gap12 = distances[1] - distances[0]
    summary["gap12"] = float(gap12.mean().item())

    if len(distances) == 2:
        summary["order_rate"] = float((gap12 >= 0).float().mean().item())
        summary["margin_rate"] = float((gap12 >= margin).float().mean().item())
        summary["active_frac"] = float((gap12 < margin).float().mean().item())
        return summary

    gap23 = distances[2] - distances[1]
    summary["gap23"] = float(gap23.mean().item())
    order_ok = (gap12 >= 0) & (gap23 >= 0)
    margin_ok = (gap12 >= margin) & (gap23 >= margin)
    active = torch.stack([(gap12 < margin).float(), (gap23 < margin).float()], dim=0)
    summary["order_rate"] = float(order_ok.float().mean().item())
    summary["margin_rate"] = float(margin_ok.float().mean().item())
    summary["active_frac"] = float(active.mean().item())
    return summary


def parameter_grad_norm(module: nn.Module) -> float:
    sq_sum = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_sq = float(param.grad.detach().pow(2).sum().item())
        sq_sum += grad_sq
        found = True
    if not found:
        return float("nan")
    return float(sq_sum ** 0.5)


def vector_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.detach().reshape(-1).float()
    y = y.detach().reshape(-1).float()
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.pow(2).sum().clamp_min(eps) * y.pow(2).sum().clamp_min(eps))
    if float(denom.item()) <= eps:
        return float("nan")
    return float((x * y).sum().item() / float(denom.item()))


def train_model(args: argparse.Namespace, model_kind: str, train_split: EconSplitData, val_split: EconSplitData):
    model = build_model(args, model_kind, train_split)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_loader(train_split, args.batch_size, shuffle=True)
    val_loader = build_loader(val_split, args.batch_size, shuffle=False)
    tau_ord_sigmas = [float(s.strip()) for s in args.tau_ord_sigmas.split(",") if s.strip()]

    best_state = None
    best_score = -np.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        total_align_corr = 0.0
        total_align_n = 0
        total_geom_qk = 0.0
        total_geom_attn = 0.0
        total_geom_n = 0
        total_ord_loss = 0.0
        total_ord_n = 0
        total_ord_d1 = 0.0
        total_ord_d2 = 0.0
        total_ord_d3 = 0.0
        total_ord_qk_d1 = 0.0
        total_ord_qk_d2 = 0.0
        total_ord_qk_d3 = 0.0
        total_ord_gap12 = 0.0
        total_ord_gap23 = 0.0
        total_ord_qk_gap12 = 0.0
        total_ord_qk_gap23 = 0.0
        total_ord_order_rate = 0.0
        total_ord_margin_rate = 0.0
        total_ord_active_frac = 0.0
        total_ord_qk_order_rate = 0.0
        total_ord_qk_margin_rate = 0.0
        total_ord_qk_active_frac = 0.0
        total_ord_perturb_l2_1 = 0.0
        total_ord_perturb_l2_2 = 0.0
        total_ord_perturb_l2_3 = 0.0
        total_ord_pred_d1 = 0.0
        total_ord_pred_d2 = 0.0
        total_ord_pred_d3 = 0.0
        total_ord_fused_d1 = 0.0
        total_ord_fused_d2 = 0.0
        total_ord_fused_d3 = 0.0
        total_ord_global_d1 = 0.0
        total_ord_global_d2 = 0.0
        total_ord_global_d3 = 0.0
        total_ord_local_d1 = 0.0
        total_ord_local_d2 = 0.0
        total_ord_local_d3 = 0.0
        total_ord_qk_pred_corr = 0.0
        total_ord_qk_fused_corr = 0.0
        total_ord_qk_global_corr = 0.0
        total_ord_qk_local_corr = 0.0
        total_ord_corr_n = 0
        total_tau_grad_norm = 0.0
        total_tau_grad_n = 0
        geom_lambda = args.tau_geom_lambda if epoch > args.tau_geom_warmup_epochs else 0.0
        ord_lambda = args.tau_ord_lambda if epoch > args.tau_geom_warmup_epochs else 0.0
        for asset_src, market_seq, context, y, _, _ in train_loader:
            asset_src = asset_src.to(args.device)
            market_seq = market_seq.to(args.device)
            context = context.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            if _is_learned_tau_rope(model_kind):
                pred, diag = model(asset_src, market_seq, return_diagnostics=True)
                pred_loss = F.mse_loss(pred, y)
                align_corr = batch_pearson_corr(diag["step"], diag["intensity"]).mean()
                geom_qk = torch.tensor(0.0, device=args.device)
                geom_attn = torch.tensor(0.0, device=args.device)
                ord_loss = torch.tensor(0.0, device=args.device)
                if geom_lambda > 0:
                    swap_mkt = torch.roll(market_seq, shifts=1, dims=0) if market_seq.size(0) > 1 else market_seq.flip(dims=[1])
                    was_training = model.training
                    model.eval()
                    _, geom_diag = model(asset_src, market_seq, return_diagnostics=True)
                    _, swap_diag = model(asset_src, swap_mkt, return_diagnostics=True)
                    if was_training:
                        model.train()
                    geom_qk = batch_cosine_distance(geom_diag["qk_scores"], swap_diag["qk_scores"]).mean()
                    geom_attn = batch_cosine_distance(geom_diag["attention_map"], swap_diag["attention_map"]).mean()
                    total_geom_qk += float(geom_qk.item()) * asset_src.size(0)
                    total_geom_attn += float(geom_attn.item()) * asset_src.size(0)
                    total_geom_n += asset_src.size(0)
                if ord_lambda > 0:
                    was_training = model.training
                    model.eval()
                    base_pred_eval, base_diag = model(asset_src, market_seq, return_diagnostics=True)
                    ordered_markets = build_ordered_market_perturbations(market_seq, tau_ord_sigmas)
                    ordered_attn = []
                    ordered_qk = []
                    ordered_pred = []
                    ordered_fused = []
                    ordered_global = []
                    ordered_local = []
                    perturb_l2 = []
                    for perturbed_market in ordered_markets:
                        ord_pred_eval, ord_diag = model(asset_src, perturbed_market, return_diagnostics=True)
                        ordered_attn.append(ord_diag["attention_map"])
                        ordered_qk.append(ord_diag["qk_scores"])
                        ordered_pred.append(ord_pred_eval)
                        ordered_fused.append(ord_diag["fused"])
                        ordered_global.append(ord_diag["global_feat"])
                        ordered_local.append(ord_diag["local_feat"])
                        perturb_l2.append(
                            (perturbed_market - market_seq).pow(2).mean(dim=(1, 2)).sqrt()
                        )
                    if was_training:
                        model.train()
                    _, ord_distances = ordinal_margin_loss(
                        base_diag["attention_map"],
                        ordered_attn,
                        margin=args.tau_ord_margin,
                    )
                    ord_summary = summarize_ordinal_distances(ord_distances, margin=args.tau_ord_margin)
                    ord_loss, ord_qk_distances = ordinal_margin_loss(
                        base_diag["qk_scores"],
                        ordered_qk,
                        margin=args.tau_ord_margin,
                        distance_fn=batch_normalized_l2_distance,
                    )
                    ord_qk_summary = summarize_ordinal_distances(ord_qk_distances, margin=args.tau_ord_margin)
                    pred_distances = [batch_l2_distance(base_pred_eval, pred_eval) for pred_eval in ordered_pred]
                    fused_distances = [batch_l2_distance(base_diag["fused"], fused) for fused in ordered_fused]
                    global_distances = [batch_l2_distance(base_diag["global_feat"], global_feat) for global_feat in ordered_global]
                    local_distances = [batch_l2_distance(base_diag["local_feat"], local_feat) for local_feat in ordered_local]
                    pred_summary = summarize_ordinal_distances(pred_distances, margin=0.0)
                    fused_summary = summarize_ordinal_distances(fused_distances, margin=0.0)
                    global_summary = summarize_ordinal_distances(global_distances, margin=0.0)
                    local_summary = summarize_ordinal_distances(local_distances, margin=0.0)
                    total_ord_loss += float(ord_loss.item()) * asset_src.size(0)
                    total_ord_d1 += ord_summary["d1"] * asset_src.size(0)
                    total_ord_d2 += ord_summary["d2"] * asset_src.size(0)
                    total_ord_d3 += ord_summary["d3"] * asset_src.size(0)
                    total_ord_qk_d1 += ord_qk_summary["d1"] * asset_src.size(0)
                    total_ord_qk_d2 += ord_qk_summary["d2"] * asset_src.size(0)
                    total_ord_qk_d3 += ord_qk_summary["d3"] * asset_src.size(0)
                    total_ord_gap12 += ord_summary["gap12"] * asset_src.size(0)
                    total_ord_gap23 += ord_summary["gap23"] * asset_src.size(0)
                    total_ord_qk_gap12 += ord_qk_summary["gap12"] * asset_src.size(0)
                    total_ord_qk_gap23 += ord_qk_summary["gap23"] * asset_src.size(0)
                    total_ord_order_rate += ord_summary["order_rate"] * asset_src.size(0)
                    total_ord_margin_rate += ord_summary["margin_rate"] * asset_src.size(0)
                    total_ord_active_frac += ord_summary["active_frac"] * asset_src.size(0)
                    total_ord_qk_order_rate += ord_qk_summary["order_rate"] * asset_src.size(0)
                    total_ord_qk_margin_rate += ord_qk_summary["margin_rate"] * asset_src.size(0)
                    total_ord_qk_active_frac += ord_qk_summary["active_frac"] * asset_src.size(0)
                    total_ord_pred_d1 += pred_summary["d1"] * asset_src.size(0)
                    total_ord_pred_d2 += pred_summary["d2"] * asset_src.size(0)
                    total_ord_pred_d3 += pred_summary["d3"] * asset_src.size(0)
                    total_ord_fused_d1 += fused_summary["d1"] * asset_src.size(0)
                    total_ord_fused_d2 += fused_summary["d2"] * asset_src.size(0)
                    total_ord_fused_d3 += fused_summary["d3"] * asset_src.size(0)
                    total_ord_global_d1 += global_summary["d1"] * asset_src.size(0)
                    total_ord_global_d2 += global_summary["d2"] * asset_src.size(0)
                    total_ord_global_d3 += global_summary["d3"] * asset_src.size(0)
                    total_ord_local_d1 += local_summary["d1"] * asset_src.size(0)
                    total_ord_local_d2 += local_summary["d2"] * asset_src.size(0)
                    total_ord_local_d3 += local_summary["d3"] * asset_src.size(0)
                    qk_pred_corrs = [vector_pearson_corr(qk_d, pred_d) for qk_d, pred_d in zip(ord_qk_distances, pred_distances)]
                    qk_fused_corrs = [vector_pearson_corr(qk_d, fused_d) for qk_d, fused_d in zip(ord_qk_distances, fused_distances)]
                    qk_global_corrs = [vector_pearson_corr(qk_d, global_d) for qk_d, global_d in zip(ord_qk_distances, global_distances)]
                    qk_local_corrs = [vector_pearson_corr(qk_d, local_d) for qk_d, local_d in zip(ord_qk_distances, local_distances)]
                    for corr_list, total_ref in (
                        (qk_pred_corrs, "pred"),
                        (qk_fused_corrs, "fused"),
                        (qk_global_corrs, "global"),
                        (qk_local_corrs, "local"),
                    ):
                        valid = [c for c in corr_list if not np.isnan(c)]
                        if not valid:
                            continue
                        mean_corr = float(np.mean(valid)) * asset_src.size(0)
                        if total_ref == "pred":
                            total_ord_qk_pred_corr += mean_corr
                        elif total_ref == "fused":
                            total_ord_qk_fused_corr += mean_corr
                        elif total_ref == "global":
                            total_ord_qk_global_corr += mean_corr
                        else:
                            total_ord_qk_local_corr += mean_corr
                    total_ord_corr_n += asset_src.size(0)
                    for idx, l2_tensor in enumerate(perturb_l2[:3], start=1):
                        mean_l2 = float(l2_tensor.mean().item()) * asset_src.size(0)
                        if idx == 1:
                            total_ord_perturb_l2_1 += mean_l2
                        elif idx == 2:
                            total_ord_perturb_l2_2 += mean_l2
                        else:
                            total_ord_perturb_l2_3 += mean_l2
                    total_ord_n += asset_src.size(0)
                loss = pred_loss - args.tau_align_lambda * align_corr - geom_lambda * (geom_qk + geom_attn) + ord_lambda * ord_loss
                total_align_corr += float(align_corr.item()) * asset_src.size(0)
                total_align_n += asset_src.size(0)
            else:
                pred = forward_model(model, model_kind, asset_src, market_seq, context)
                pred_loss = F.mse_loss(pred, y)
                loss = pred_loss
            loss.backward()
            if _is_learned_tau_rope(model_kind):
                tau_grad_norm = parameter_grad_norm(model.transformer.tau_builder)
                if not np.isnan(tau_grad_norm):
                    total_tau_grad_norm += tau_grad_norm * asset_src.size(0)
                    total_tau_grad_n += asset_src.size(0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * asset_src.size(0)
            total_n += asset_src.size(0)

        val_metrics = evaluate_model(model, model_kind, val_loader, args.device)[0]
        align_mean = total_align_corr / max(total_align_n, 1) if total_align_n > 0 else np.nan
        geom_qk_mean = total_geom_qk / max(total_geom_n, 1) if total_geom_n > 0 else np.nan
        geom_attn_mean = total_geom_attn / max(total_geom_n, 1) if total_geom_n > 0 else np.nan
        ord_loss_mean = total_ord_loss / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_d1_mean = total_ord_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_d2_mean = total_ord_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_d3_mean = total_ord_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_d1_mean = total_ord_qk_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_d2_mean = total_ord_qk_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_d3_mean = total_ord_qk_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_gap12_mean = total_ord_gap12 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_gap23_mean = total_ord_gap23 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_gap12_mean = total_ord_qk_gap12 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_gap23_mean = total_ord_qk_gap23 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_order_rate_mean = total_ord_order_rate / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_margin_rate_mean = total_ord_margin_rate / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_active_frac_mean = total_ord_active_frac / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_order_rate_mean = total_ord_qk_order_rate / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_margin_rate_mean = total_ord_qk_margin_rate / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_active_frac_mean = total_ord_qk_active_frac / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_perturb_l2_1_mean = total_ord_perturb_l2_1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_perturb_l2_2_mean = total_ord_perturb_l2_2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_perturb_l2_3_mean = total_ord_perturb_l2_3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_pred_d1_mean = total_ord_pred_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_pred_d2_mean = total_ord_pred_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_pred_d3_mean = total_ord_pred_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_fused_d1_mean = total_ord_fused_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_fused_d2_mean = total_ord_fused_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_fused_d3_mean = total_ord_fused_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_global_d1_mean = total_ord_global_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_global_d2_mean = total_ord_global_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_global_d3_mean = total_ord_global_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_local_d1_mean = total_ord_local_d1 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_local_d2_mean = total_ord_local_d2 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_local_d3_mean = total_ord_local_d3 / max(total_ord_n, 1) if total_ord_n > 0 else np.nan
        ord_qk_pred_corr_mean = total_ord_qk_pred_corr / max(total_ord_corr_n, 1) if total_ord_corr_n > 0 else np.nan
        ord_qk_fused_corr_mean = total_ord_qk_fused_corr / max(total_ord_corr_n, 1) if total_ord_corr_n > 0 else np.nan
        ord_qk_global_corr_mean = total_ord_qk_global_corr / max(total_ord_corr_n, 1) if total_ord_corr_n > 0 else np.nan
        ord_qk_local_corr_mean = total_ord_qk_local_corr / max(total_ord_corr_n, 1) if total_ord_corr_n > 0 else np.nan
        tau_grad_norm_mean = total_tau_grad_norm / max(total_tau_grad_n, 1) if total_tau_grad_n > 0 else np.nan
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_n, 1),
                "train_align_corr": align_mean,
                "train_geom_lambda": geom_lambda,
                "train_geom_qk": geom_qk_mean,
                "train_geom_attn": geom_attn_mean,
                "train_ord_lambda": ord_lambda,
                "train_ord_loss": ord_loss_mean,
                "train_ord_d1": ord_d1_mean,
                "train_ord_d2": ord_d2_mean,
                "train_ord_d3": ord_d3_mean,
                "train_ord_qk_d1": ord_qk_d1_mean,
                "train_ord_qk_d2": ord_qk_d2_mean,
                "train_ord_qk_d3": ord_qk_d3_mean,
                "train_ord_gap12": ord_gap12_mean,
                "train_ord_gap23": ord_gap23_mean,
                "train_ord_qk_gap12": ord_qk_gap12_mean,
                "train_ord_qk_gap23": ord_qk_gap23_mean,
                "train_ord_order_rate": ord_order_rate_mean,
                "train_ord_margin_rate": ord_margin_rate_mean,
                "train_ord_active_frac": ord_active_frac_mean,
                "train_ord_qk_order_rate": ord_qk_order_rate_mean,
                "train_ord_qk_margin_rate": ord_qk_margin_rate_mean,
                "train_ord_qk_active_frac": ord_qk_active_frac_mean,
                "train_ord_perturb_l2_1": ord_perturb_l2_1_mean,
                "train_ord_perturb_l2_2": ord_perturb_l2_2_mean,
                "train_ord_perturb_l2_3": ord_perturb_l2_3_mean,
                "train_ord_pred_d1": ord_pred_d1_mean,
                "train_ord_pred_d2": ord_pred_d2_mean,
                "train_ord_pred_d3": ord_pred_d3_mean,
                "train_ord_fused_d1": ord_fused_d1_mean,
                "train_ord_fused_d2": ord_fused_d2_mean,
                "train_ord_fused_d3": ord_fused_d3_mean,
                "train_ord_global_d1": ord_global_d1_mean,
                "train_ord_global_d2": ord_global_d2_mean,
                "train_ord_global_d3": ord_global_d3_mean,
                "train_ord_local_d1": ord_local_d1_mean,
                "train_ord_local_d2": ord_local_d2_mean,
                "train_ord_local_d3": ord_local_d3_mean,
                "train_ord_qk_pred_corr": ord_qk_pred_corr_mean,
                "train_ord_qk_fused_corr": ord_qk_fused_corr_mean,
                "train_ord_qk_global_corr": ord_qk_global_corr_mean,
                "train_ord_qk_local_corr": ord_qk_local_corr_mean,
                "train_tau_grad_norm": tau_grad_norm_mean,
                "val_ic": val_metrics["ic"],
                "val_icir": val_metrics["icir"],
                "val_mae": val_metrics["mae"],
            }
        )
        score = val_metrics["ic"] if not np.isnan(val_metrics["ic"]) else -np.inf
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"[train] kind={model_kind:16s} epoch={epoch:02d} "
            f"loss={total_loss / max(total_n, 1):.6f} val_ic={val_metrics['ic']:.4f} "
            f"val_mae={val_metrics['mae']:.6f}"
            + (f" train_align={align_mean:.4f}" if total_align_n > 0 else "")
            + (f" geom_lambda={geom_lambda:.4f}" if _is_learned_tau_rope(model_kind) else "")
            + (f" train_geom_qk={geom_qk_mean:.4f} train_geom_attn={geom_attn_mean:.4f}" if total_geom_n > 0 else "")
            + (
                f" ord_lambda={ord_lambda:.4f} train_ord={ord_loss_mean:.4f}"
                f" d=({ord_d1_mean:.2e},{ord_d2_mean:.2e},{ord_d3_mean:.2e})"
                f" qk=({ord_qk_d1_mean:.2e},{ord_qk_d2_mean:.2e},{ord_qk_d3_mean:.2e})"
                f" pred=({ord_pred_d1_mean:.2e},{ord_pred_d2_mean:.2e},{ord_pred_d3_mean:.2e})"
                f" ord_rate={ord_order_rate_mean:.3f} margin_rate={ord_margin_rate_mean:.3f}"
                f" qk_ord={ord_qk_order_rate_mean:.3f}"
                f" qk_pred_corr={ord_qk_pred_corr_mean:.3f}"
                f" active={ord_active_frac_mean:.3f}"
                f" tau_grad={tau_grad_norm_mean:.2e}"
                if total_ord_n > 0
                else ""
            ),
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history = history
    return model


@torch.no_grad()
def evaluate_model(model, model_kind: str, loader: DataLoader, device: str):
    model.eval()
    param_count = int(sum(p.numel() for p in model.parameters()))
    rows = []
    xip_off_rows = []
    for asset_src, market_seq, context, y, dates, regimes in loader:
        asset_src = asset_src.to(device)
        market_seq = market_seq.to(device)
        context = context.to(device)
        y = y.to(device)

        if _uses_market_model(model_kind):
            if _is_tau_rope_concat(model_kind):
                pred, diag = model(asset_src, market_seq, context, return_diagnostics=True)
            else:
                pred, diag = model(asset_src, market_seq, return_diagnostics=True)
            tau_last = diag["tau"][:, -1].cpu().numpy()
            market_shift = (diag["joint_routing"] - diag["attention_importance"]).abs().mean(dim=1).cpu().numpy()
            pe_scale = np.full(pred.size(0), float(diag["pe_scale"].item()))
            alpha_mean = np.full(pred.size(0), float(diag["alpha"].mean().item())) if "alpha" in diag else np.full(pred.size(0), np.nan)
            fusion_gate = diag["gate"].cpu().numpy()
            context_gate_mean = diag["context_gate"].mean(dim=1).cpu().numpy()
            bias_abs_mean = (
                diag["attention_bias"].abs().mean(dim=(1, 2)).cpu().numpy()
                if diag["attention_bias"] is not None
                else np.full(pred.size(0), np.nan)
            )
            qk_abs_mean = diag["qk_scores"].abs().mean(dim=(1, 2)).cpu().numpy()
            tau_dev_std, step_dev_std, tau_corr = compute_tau_metrics(diag["tau"])
            step_intensity_spearman = (
                compute_step_intensity_spearman(diag["step"], diag["intensity"])
                if "step" in diag and "intensity" in diag
                else np.full(pred.size(0), np.nan)
            )

            # adjacent-roll swap (structured perturbation)
            swap_mkt = torch.roll(market_seq, shifts=1, dims=0) if market_seq.size(0) > 1 else market_seq.flip(dims=[1])
            if _is_tau_rope_concat(model_kind):
                swap_pred, swap_diag = model(asset_src, swap_mkt, context, return_diagnostics=True)
            else:
                swap_pred, swap_diag = model(asset_src, swap_mkt, return_diagnostics=True)

            # random-permutation swap (null baseline)
            rand_idx = torch.randperm(market_seq.size(0), device=market_seq.device)
            if _is_tau_rope_concat(model_kind):
                _, rand_diag = model(asset_src, market_seq[rand_idx], context, return_diagnostics=True)
            else:
                _, rand_diag = model(asset_src, market_seq[rand_idx], return_diagnostics=True)

            tau_swap_delta = (swap_diag["tau"] - diag["tau"]).abs().mean(dim=1).cpu().numpy()
            pred_swap_delta = (swap_pred.squeeze(-1) - pred.squeeze(-1)).abs().cpu().numpy()
            bias_swap_delta = (
                (swap_diag["attention_bias"] - diag["attention_bias"]).abs().mean(dim=(1, 2)).cpu().numpy()
                if diag["attention_bias"] is not None and swap_diag["attention_bias"] is not None
                else np.full(pred.size(0), np.nan)
            )
            attn_swap_delta = (swap_diag["attention_map"] - diag["attention_map"]).abs().mean(dim=(1, 2)).cpu().numpy()
            qk_swap_delta = (swap_diag["qk_scores"] - diag["qk_scores"]).abs().mean(dim=(1, 2)).cpu().numpy()

            # null deltas
            rand_attn_swap_delta = (rand_diag["attention_map"] - diag["attention_map"]).abs().mean(dim=(1, 2)).cpu().numpy()
            rand_qk_swap_delta = (rand_diag["qk_scores"] - diag["qk_scores"]).abs().mean(dim=(1, 2)).cpu().numpy()

            # top-k attention edge delta (k = 5% of T*T, min 1)
            seq_len = diag["attention_map"].size(-1)
            k = max(1, seq_len * seq_len // 20)
            attn_flat = diag["attention_map"].reshape(pred.size(0), -1)
            swap_attn_flat = swap_diag["attention_map"].reshape(pred.size(0), -1)
            topk_idx = attn_flat.topk(k, dim=-1).indices
            attn_topk_delta = (swap_attn_flat.gather(1, topk_idx) - attn_flat.gather(1, topk_idx)).abs().mean(dim=-1).cpu().numpy()

            # KL(orig || swap), mean over query positions
            attn_kl = _attn_kl(diag["attention_map"], swap_diag["attention_map"]).cpu().numpy()

            tau_dev_std = tau_dev_std.cpu().numpy()
            step_dev_std = step_dev_std.cpu().numpy()
            tau_corr = tau_corr.cpu().numpy()
            xip_h_int_norm = xip_h_int_ratio = xip_pred_delta = np.full(pred.size(0), np.nan)
        else:
            prepared_context = _prepare_context(model_kind, context, market_seq)
            pred, diag = model(asset_src, prepared_context, return_diagnostics=True)
            n = pred.size(0)
            tau_last = market_shift = pe_scale = alpha_mean = fusion_gate = context_gate_mean = bias_abs_mean = qk_abs_mean = np.full(n, np.nan)
            tau_dev_std = step_dev_std = tau_corr = step_intensity_spearman = np.full(n, np.nan)
            tau_swap_delta = pred_swap_delta = bias_swap_delta = np.full(n, np.nan)
            attn_swap_delta = qk_swap_delta = np.full(n, np.nan)
            rand_attn_swap_delta = rand_qk_swap_delta = np.full(n, np.nan)
            attn_topk_delta = attn_kl = np.full(n, np.nan)
            xip_h_int_norm = xip_h_int_ratio = xip_pred_delta = np.full(n, np.nan)

            if _base_model_kind(model_kind) == "xip_a" and "xip_interaction" in diag:
                h_int = diag["xip_interaction"]
                h_state = diag["xip_state_main"]
                xip_h_int_norm = h_int.norm(dim=-1).mean(dim=1).cpu().numpy()
                state_norm = h_state.norm(dim=-1).mean(dim=1).clamp_min(1e-8)
                xip_h_int_ratio = (h_int.norm(dim=-1).mean(dim=1) / state_norm).cpu().numpy()

                pred_off, _ = model(
                    asset_src,
                    prepared_context,
                    return_diagnostics=True,
                    disable_interaction=True,
                )
                xip_pred_delta = (pred_off.squeeze(-1) - pred.squeeze(-1)).abs().cpu().numpy()
                pred_off_np = pred_off.squeeze(-1).cpu().numpy()
                y_np_off = y.squeeze(-1).cpu().numpy()
                for idx in range(len(pred_off_np)):
                    xip_off_rows.append(
                        {
                            "date": pd.to_datetime(int(dates[idx])),
                            "y_true": float(y_np_off[idx]),
                            "pred": float(pred_off_np[idx]),
                        }
                    )

        pred_np = pred.squeeze(-1).cpu().numpy()
        y_np = y.squeeze(-1).cpu().numpy()
        for idx in range(len(pred_np)):
            rows.append({
                "date": pd.to_datetime(int(dates[idx])),
                "y_true": float(y_np[idx]),
                "pred": float(pred_np[idx]),
                "regime": int(regimes[idx]),
                "tau_last": float(tau_last[idx]) if not np.isnan(tau_last[idx]) else np.nan,
                "market_shift": float(market_shift[idx]) if not np.isnan(market_shift[idx]) else np.nan,
                "pe_scale": float(pe_scale[idx]) if not np.isnan(pe_scale[idx]) else np.nan,
                "alpha_mean": float(alpha_mean[idx]) if not np.isnan(alpha_mean[idx]) else np.nan,
                "fusion_gate": float(fusion_gate[idx]) if not np.isnan(fusion_gate[idx]) else np.nan,
                "context_gate_mean": float(context_gate_mean[idx]) if not np.isnan(context_gate_mean[idx]) else np.nan,
                "bias_abs_mean": float(bias_abs_mean[idx]) if not np.isnan(bias_abs_mean[idx]) else np.nan,
                "qk_abs_mean": float(qk_abs_mean[idx]) if not np.isnan(qk_abs_mean[idx]) else np.nan,
                "tau_dev_std": float(tau_dev_std[idx]) if not np.isnan(tau_dev_std[idx]) else np.nan,
                "step_dev_std": float(step_dev_std[idx]) if not np.isnan(step_dev_std[idx]) else np.nan,
                "tau_corr": float(tau_corr[idx]) if not np.isnan(tau_corr[idx]) else np.nan,
                "step_intensity_spearman": float(step_intensity_spearman[idx]) if not np.isnan(step_intensity_spearman[idx]) else np.nan,
                "tau_swap_delta": float(tau_swap_delta[idx]) if not np.isnan(tau_swap_delta[idx]) else np.nan,
                "pred_swap_delta": float(pred_swap_delta[idx]) if not np.isnan(pred_swap_delta[idx]) else np.nan,
                "bias_swap_delta": float(bias_swap_delta[idx]) if not np.isnan(bias_swap_delta[idx]) else np.nan,
                "attn_swap_delta": float(attn_swap_delta[idx]) if not np.isnan(attn_swap_delta[idx]) else np.nan,
                "qk_swap_delta": float(qk_swap_delta[idx]) if not np.isnan(qk_swap_delta[idx]) else np.nan,
                "rand_attn_swap_delta": float(rand_attn_swap_delta[idx]) if not np.isnan(rand_attn_swap_delta[idx]) else np.nan,
                "rand_qk_swap_delta": float(rand_qk_swap_delta[idx]) if not np.isnan(rand_qk_swap_delta[idx]) else np.nan,
                "attn_topk_delta": float(attn_topk_delta[idx]) if not np.isnan(attn_topk_delta[idx]) else np.nan,
                "attn_kl": float(attn_kl[idx]) if not np.isnan(attn_kl[idx]) else np.nan,
                "xip_h_int_norm": float(xip_h_int_norm[idx]) if not np.isnan(xip_h_int_norm[idx]) else np.nan,
                "xip_h_int_ratio": float(xip_h_int_ratio[idx]) if not np.isnan(xip_h_int_ratio[idx]) else np.nan,
                "xip_pred_delta": float(xip_pred_delta[idx]) if not np.isnan(xip_pred_delta[idx]) else np.nan,
            })

    df = pd.DataFrame(rows)
    daily_ic = df.groupby("date").apply(
        lambda g: stats.spearmanr(g["pred"], g["y_true"])[0], include_groups=False
    ).dropna()
    metrics = {
        "ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
        "icir": float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) and daily_ic.std() > 0 else np.nan,
        "mae": float(np.mean(np.abs(df["pred"] - df["y_true"]))),
        "param_count": param_count,
        "tau_last_mean": float(df["tau_last"].mean()),
        "market_shift_mean": float(df["market_shift"].mean()),
        "pe_scale_mean": float(df["pe_scale"].mean()),
        "alpha_mean": float(df["alpha_mean"].mean()),
        "fusion_gate_mean": float(df["fusion_gate"].mean()),
        "context_gate_mean": float(df["context_gate_mean"].mean()),
        "bias_abs_mean": float(df["bias_abs_mean"].mean()),
        "qk_abs_mean": float(df["qk_abs_mean"].mean()),
        "bias_to_qk_ratio": float((df["bias_abs_mean"] / df["qk_abs_mean"].replace(0, np.nan)).mean()),
        "tau_dev_std_mean": float(df["tau_dev_std"].mean()),
        "step_dev_std_mean": float(df["step_dev_std"].mean()),
        "tau_corr_mean": float(df["tau_corr"].mean()),
        "step_intensity_spearman_mean": float(df["step_intensity_spearman"].mean()),
        "tau_swap_delta_mean": float(df["tau_swap_delta"].mean()),
        "pred_swap_delta_mean": float(df["pred_swap_delta"].mean()),
        "bias_swap_delta_mean": float(df["bias_swap_delta"].mean()),
        "attn_swap_delta_mean": float(df["attn_swap_delta"].mean()),
        "qk_swap_delta_mean": float(df["qk_swap_delta"].mean()),
        "rand_attn_swap_delta_mean": float(df["rand_attn_swap_delta"].mean()),
        "rand_qk_swap_delta_mean": float(df["rand_qk_swap_delta"].mean()),
        "attn_topk_delta_mean": float(df["attn_topk_delta"].mean()),
        "attn_kl_mean": float(df["attn_kl"].mean()),
        "xip_h_int_norm_mean": float(df["xip_h_int_norm"].mean()),
        "xip_h_int_ratio_mean": float(df["xip_h_int_ratio"].mean()),
        "xip_pred_delta_mean": float(df["xip_pred_delta"].mean()),
    }
    if xip_off_rows:
        xip_off_df = pd.DataFrame(xip_off_rows)
        xip_off_daily_ic = _compute_daily_ic(xip_off_df)
        metrics["xip_ic_off"] = float(xip_off_daily_ic.mean()) if len(xip_off_daily_ic) else np.nan
        metrics["xip_ic_drop"] = metrics["ic"] - metrics["xip_ic_off"] if not np.isnan(metrics["ic"]) else np.nan
    else:
        metrics["xip_ic_off"] = np.nan
        metrics["xip_ic_drop"] = np.nan
    return metrics, df


def _compute_daily_ic(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date").apply(
        lambda g: stats.spearmanr(g["pred"], g["y_true"])[0], include_groups=False
    ).dropna()


def _paired_ttest(ic_a: pd.Series, ic_b: pd.Series) -> dict:
    """Paired t-test on daily IC series aligned by date. ic_a - ic_b."""
    common = ic_a.index.intersection(ic_b.index)
    if len(common) < 5:
        return {"t": np.nan, "p": np.nan, "n": len(common)}
    diff = (ic_a.loc[common] - ic_b.loc[common]).dropna()
    t, p = stats.ttest_1samp(diff, popmean=0.0)
    return {"t": float(t), "p": float(p), "n": int(len(diff))}


def main():
    args = parse_args()
    set_seed(args.seed)

    port25, factors, index_close = load_data(args.start, args.end, args.index_symbol)
    source, target = build_target(port25, factors, args.target, args.roll_beta, args.horizon)
    position, intensity, regime = build_state(index_close)
    train_split, val_split, test_split = build_split(
        dates=source.index,
        source=source,
        target=target,
        position=position,
        intensity=intensity,
        regime=regime,
        index_close=index_close,
        seq_len=args.seq_len,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pred_dfs: dict[str, pd.DataFrame] = {}
    model_kinds = [k.strip() for k in args.model_kinds.split(",") if k.strip()]
    test_loader = build_loader(test_split, args.batch_size, shuffle=False)

    for model_kind in model_kinds:
        set_seed(args.seed)
        print(f"\n[run] model_kind={model_kind}", flush=True)
        model = train_model(args, model_kind, train_split, val_split)
        metrics, pred_df = evaluate_model(model, model_kind, test_loader, args.device)
        metrics["model_kind"] = model_kind
        results.append(metrics)
        pred_dfs[model_kind] = pred_df
        stem = (
            f"economic_time_{args.index_symbol.replace('^', '').lower()}_"
            f"{args.target}_{args.start[:4]}_{args.end[:4]}_{args.relative_bias_mode}_"
            f"g{str(args.relative_bias_gamma).replace('.', 'p')}_"
            f"{'fixed' if args.fixed_bias_slopes else 'learned'}_{model_kind.replace(':', '_')}"
        )
        pred_df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
        print(
            f"[test] kind={model_kind:16s} ic={metrics['ic']:.4f} "
            f"icir={metrics['icir']:.4f} mae={metrics['mae']:.6f}",
            flush=True,
        )

    result_df = pd.DataFrame(results).sort_values("model_kind").reset_index(drop=True)
    summary_stem = (
        f"economic_time_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_{args.start[:4]}_{args.end[:4]}_{args.relative_bias_mode}_"
        f"g{str(args.relative_bias_gamma).replace('.', 'p')}_"
        f"{'fixed' if args.fixed_bias_slopes else 'learned'}_summary"
    )
    result_df.to_csv(out_dir / f"{summary_stem}.csv", index=False)
    print("\n[summary]", flush=True)
    print(result_df.to_string(index=False), flush=True)

    # paired t-test: all models vs concat_a
    if "concat_a" in pred_dfs:
        ic_concat = _compute_daily_ic(pred_dfs["concat_a"])
        ttest_rows = []
        for mk in model_kinds:
            if mk == "concat_a" or mk not in pred_dfs:
                continue
            res = _paired_ttest(_compute_daily_ic(pred_dfs[mk]), ic_concat)
            ttest_rows.append({"model_kind": mk, "vs": "concat_a", **res})
        if ttest_rows:
            ttest_df = pd.DataFrame(ttest_rows)
            ttest_path = out_dir / f"{summary_stem}_ttest.csv"
            ttest_df.to_csv(ttest_path, index=False)
            print("\n[paired t-test vs concat_a]", flush=True)
            print(ttest_df.to_string(index=False), flush=True)
            print(f"[save] {ttest_path}", flush=True)

    print(f"\n[save] {out_dir / f'{summary_stem}.csv'}", flush=True)


if __name__ == "__main__":
    main()
