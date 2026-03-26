"""
TTPA (Test-Time Positional Adaptation) Prototype Experiment

Core idea: Train a model with FIXED positional encoding (standard sinusoidal
at integer positions [0, 1, ..., T-1]).  At TEST TIME, replace the PE positions
with economic time tau = cumsum(softplus(alpha * intensity)), where alpha is a
fixed hyperparameter (not learned).

Hypothesis: If market intensity carries meaningful temporal-scaling information,
then tau-warped positions should yield better test IC than the default integer
grid — even though the model never saw tau during training.

Comparison:
  (a) static (fixed PE, integer positions) — the baseline
  (b) static + TTPA at various alpha values — test-time adapted PE

The experiment trains ONE static model, then evaluates it multiple times with
different alpha sweeps.  No re-training is needed per alpha.
"""

from __future__ import annotations

import argparse
import math
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

from paper.index_conditioned_pe.icpe_hybrid_model import PaperICPEHybrid  # noqa: E402
from paper.index_conditioned_pe.icpe_transformer import ICPEPositionalEncoding  # noqa: E402
from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_features,
    build_state,
    build_target,
    load_data,
    set_seed,
)
from paper_test.economic_time_supervised import (  # noqa: E402
    EconSplitData,
    EconDataset,
    build_market_features,
    build_split,
    build_loader,
    forward_model,
)


# ---------------------------------------------------------------------------
# TTPA core: monkey-patch the PE forward at test time
# ---------------------------------------------------------------------------

def _make_tau_from_intensity(
    market_seq: torch.Tensor,
    alpha: float,
    intensity_idx: int = 3,
) -> torch.Tensor:
    """
    Compute economic-time positions from market intensity.

    tau_t = cumsum(softplus(alpha * intensity_t))

    Returns shape (B, T, 1) for compatibility with sinusoidal PE computation.
    """
    intensity = market_seq[..., intensity_idx]  # (B, T)
    step = F.softplus(alpha * intensity)  # (B, T)
    tau = torch.cumsum(step, dim=1) - step[:, :1]  # start at 0
    return tau.unsqueeze(-1)  # (B, T, 1)


def _sinusoidal_pe_at_positions(
    positions: torch.Tensor,
    div_term: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    """
    Compute sinusoidal PE at arbitrary (non-integer) positions.

    positions: (B, T, 1)
    div_term: (d_model//2,)
    returns: (B, T, d_model)
    """
    div = div_term.to(device=positions.device, dtype=positions.dtype).view(1, 1, -1)
    pe = torch.zeros(positions.size(0), positions.size(1), d_model, device=positions.device, dtype=positions.dtype)
    pe[..., 0::2] = torch.sin(positions * div)
    pe[..., 1::2] = torch.cos(positions * div)
    return pe


class TTPAForwardHook:
    """
    Context manager that patches a PaperICPEHybrid's PE to use tau positions.

    Usage:
        with TTPAForwardHook(model, alpha=1.0, market_seq_holder=holder):
            pred = model(src, context)
    """

    def __init__(self, model: PaperICPEHybrid, alpha: float, market_seq_ref: list):
        self.model = model
        self.alpha = alpha
        self.market_seq_ref = market_seq_ref  # mutable container to pass current batch's market_seq
        self.pe_module: ICPEPositionalEncoding = model.transformer.pos_encoder
        self._original_forward = None

    def __enter__(self):
        pe_mod = self.pe_module
        alpha = self.alpha
        market_seq_ref = self.market_seq_ref

        original_forward = pe_mod.forward

        def patched_forward(x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
            market_seq = market_seq_ref[0]
            if market_seq is not None and alpha != 0.0:
                tau = _make_tau_from_intensity(market_seq, alpha)
                pe = _sinusoidal_pe_at_positions(tau, pe_mod.div_term, x.size(-1))
                return pe_mod.dropout(x + pe)
            else:
                # fallback to original static PE
                return original_forward(x, context)

        self._original_forward = original_forward
        pe_mod.forward = patched_forward
        return self

    def __exit__(self, *args):
        self.pe_module.forward = self._original_forward


# ---------------------------------------------------------------------------
# Training: standard static model
# ---------------------------------------------------------------------------

def train_static_model(
    args: argparse.Namespace,
    train_split: EconSplitData,
    val_split: EconSplitData,
) -> PaperICPEHybrid:
    """Train a static PE baseline (no market conditioning at all)."""
    model = PaperICPEHybrid(
        input_dim=train_split.asset_src.shape[-1],
        context_dim=train_split.context.shape[-1],
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        pe_mode="static",
        decoder_mode="point",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_loader(train_split, args.batch_size, shuffle=True)
    val_loader = build_loader(val_split, args.batch_size, shuffle=False)

    best_val_ic = -np.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for asset_src, market_seq, context, y, _, _ in train_loader:
            asset_src = asset_src.to(args.device)
            context = context.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            # Fast training path: bypass PaperICPEHybrid.forward which
            # always computes attention weights (very slow on CPU).
            # Instead, manually run transformer -> pool -> decode.
            encoded = model.transformer(asset_src, context=context, return_attention=False)
            global_feat = encoded[:, -1, :]
            local_feat = model.local_encoder(asset_src.permute(0, 2, 1)).squeeze(-1)
            fused, _ = model.fusion(global_feat, local_feat)
            pred = model.decoder(fused)
            loss = F.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * asset_src.size(0)
            total_n += asset_src.size(0)

        # Validation
        val_ic = _eval_ic(model, val_loader, args.device, mode="static")
        print(
            f"  [train] epoch={epoch} loss={total_loss / total_n:.6f} val_ic={val_ic:.4f}",
            flush=True,
        )
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_ic(
    model: PaperICPEHybrid,
    loader: DataLoader,
    device: str,
    mode: str = "static",
    alpha: float = 0.0,
) -> float:
    """Compute mean daily Spearman IC."""
    model.eval()
    rows = []
    market_seq_ref = [None]

    if mode == "ttpa" and alpha != 0.0:
        ctx_mgr = TTPAForwardHook(model, alpha, market_seq_ref)
    else:
        ctx_mgr = None

    if ctx_mgr is not None:
        ctx_mgr.__enter__()

    try:
        with torch.no_grad():
            for asset_src, market_seq, context, y, dates, _ in loader:
                asset_src = asset_src.to(device)
                market_seq = market_seq.to(device)
                context = context.to(device)
                y = y.to(device)
                market_seq_ref[0] = market_seq
                # Fast eval: no attention computation
                encoded = model.transformer(asset_src, context=context, return_attention=False)
                global_feat = encoded[:, -1, :]
                local_feat = model.local_encoder(asset_src.permute(0, 2, 1)).squeeze(-1)
                fused, _ = model.fusion(global_feat, local_feat)
                pred = model.decoder(fused)
                pred_np = pred.squeeze(-1).cpu().numpy()
                y_np = y.squeeze(-1).cpu().numpy()
                for i in range(len(pred_np)):
                    rows.append({
                        "date": int(dates[i]),
                        "y_true": float(y_np[i]),
                        "pred": float(pred_np[i]),
                    })
    finally:
        if ctx_mgr is not None:
            ctx_mgr.__exit__(None, None, None)

    if not rows:
        return float("nan")
    df = pd.DataFrame(rows)
    daily_ic = df.groupby("date").apply(
        lambda g: stats.spearmanr(g["pred"], g["y_true"])[0] if len(g) > 2 else np.nan,
        include_groups=False,
    ).dropna()
    return float(daily_ic.mean()) if len(daily_ic) else float("nan")


def _eval_full(
    model: PaperICPEHybrid,
    loader: DataLoader,
    device: str,
    mode: str = "static",
    alpha: float = 0.0,
) -> dict:
    """Full evaluation: IC, ICIR, MAE, daily IC series."""
    model.eval()
    rows = []
    market_seq_ref = [None]

    if mode == "ttpa" and alpha != 0.0:
        ctx_mgr = TTPAForwardHook(model, alpha, market_seq_ref)
    else:
        ctx_mgr = None

    if ctx_mgr is not None:
        ctx_mgr.__enter__()

    try:
        with torch.no_grad():
            for asset_src, market_seq, context, y, dates, regimes in loader:
                asset_src = asset_src.to(device)
                market_seq = market_seq.to(device)
                context = context.to(device)
                y = y.to(device)
                market_seq_ref[0] = market_seq
                # Fast eval: no attention computation
                encoded = model.transformer(asset_src, context=context, return_attention=False)
                global_feat = encoded[:, -1, :]
                local_feat = model.local_encoder(asset_src.permute(0, 2, 1)).squeeze(-1)
                fused, _ = model.fusion(global_feat, local_feat)
                pred = model.decoder(fused)
                pred_np = pred.squeeze(-1).cpu().numpy()
                y_np = y.squeeze(-1).cpu().numpy()
                for i in range(len(pred_np)):
                    rows.append({
                        "date": int(dates[i]),
                        "y_true": float(y_np[i]),
                        "pred": float(pred_np[i]),
                        "regime": int(regimes[i]),
                    })
    finally:
        if ctx_mgr is not None:
            ctx_mgr.__exit__(None, None, None)

    if not rows:
        return {"ic": float("nan"), "icir": float("nan"), "mae": float("nan")}
    df = pd.DataFrame(rows)
    daily_ic = df.groupby("date").apply(
        lambda g: stats.spearmanr(g["pred"], g["y_true"])[0] if len(g) > 2 else np.nan,
        include_groups=False,
    ).dropna()

    ic = float(daily_ic.mean()) if len(daily_ic) else float("nan")
    icir = float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) and daily_ic.std() > 0 else float("nan")
    mae = float(np.mean(np.abs(df["pred"] - df["y_true"])))

    # Regime-conditional IC
    regime_ics = {}
    for reg_val in sorted(df["regime"].unique()):
        sub = df[df["regime"] == reg_val]
        reg_daily = sub.groupby("date").apply(
            lambda g: stats.spearmanr(g["pred"], g["y_true"])[0] if len(g) > 2 else np.nan,
            include_groups=False,
        ).dropna()
        regime_ics[f"ic_regime_{reg_val}"] = float(reg_daily.mean()) if len(reg_daily) else float("nan")

    return {"ic": ic, "icir": icir, "mae": mae, "daily_ic": daily_ic, **regime_ics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTPA Prototype Experiment")
    parser.add_argument("--start", default="2022-01-01")
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
    parser.add_argument(
        "--alphas",
        default="-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0,3.0,5.0",
        help="Comma-separated alpha values for TTPA sweep (0.0 = static baseline)",
    )
    parser.add_argument("--output-dir", default="experiments/ttpa_results")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    print(f"[TTPA] Loading data: {args.index_symbol} {args.start} to {args.end}", flush=True)
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

    print(
        f"[TTPA] Splits: train={len(train_split.y)}, val={len(val_split.y)}, test={len(test_split.y)}",
        flush=True,
    )

    # Phase 1: Train static baseline
    print("\n=== Phase 1: Training static baseline ===", flush=True)
    set_seed(args.seed)
    model = train_static_model(args, train_split, val_split)

    # Phase 2: Evaluate static baseline on test
    test_loader = build_loader(test_split, args.batch_size, shuffle=False)
    baseline = _eval_full(model, test_loader, args.device, mode="static", alpha=0.0)
    print(
        f"\n[baseline] static PE: ic={baseline['ic']:.4f} icir={baseline['icir']:.4f} mae={baseline['mae']:.6f}",
        flush=True,
    )

    # Phase 3: TTPA sweep
    print("\n=== Phase 2: TTPA alpha sweep (test-time only) ===", flush=True)
    results = []
    baseline_daily_ic = baseline.pop("daily_ic", pd.Series(dtype=float))
    results.append({"alpha": 0.0, "mode": "static", **baseline})

    for alpha in alphas:
        if alpha == 0.0:
            continue  # already have baseline
        metrics = _eval_full(model, test_loader, args.device, mode="ttpa", alpha=alpha)
        daily_ic = metrics.pop("daily_ic", pd.Series(dtype=float))

        # Paired t-test vs baseline
        common = baseline_daily_ic.index.intersection(daily_ic.index)
        if len(common) >= 5:
            diff = (daily_ic.loc[common] - baseline_daily_ic.loc[common]).dropna()
            t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0)
        else:
            t_stat, p_val = float("nan"), float("nan")

        metrics["t_vs_static"] = float(t_stat)
        metrics["p_vs_static"] = float(p_val)
        ic_delta = metrics["ic"] - baseline["ic"] if not np.isnan(metrics["ic"]) else float("nan")
        metrics["ic_delta"] = ic_delta

        results.append({"alpha": alpha, "mode": "ttpa", **metrics})
        sign = "+" if ic_delta > 0 else ""
        print(
            f"  alpha={alpha:+5.1f}  ic={metrics['ic']:.4f} ({sign}{ic_delta:.4f})  "
            f"icir={metrics['icir']:.4f}  p={p_val:.4f}",
            flush=True,
        )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)

    # Display columns
    display_cols = ["alpha", "mode", "ic", "icir", "mae"]
    extra = [c for c in result_df.columns if c.startswith("ic_regime_")]
    if "ic_delta" in result_df.columns:
        display_cols.append("ic_delta")
    if "t_vs_static" in result_df.columns:
        display_cols.append("t_vs_static")
    if "p_vs_static" in result_df.columns:
        display_cols.append("p_vs_static")
    display_cols.extend(extra)

    available = [c for c in display_cols if c in result_df.columns]
    stem = (
        f"ttpa_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_{args.start[:4]}_{args.end[:4]}"
    )
    result_df.to_csv(out_dir / f"{stem}_sweep.csv", index=False)

    print("\n=== TTPA Results Summary ===", flush=True)
    print(result_df[available].to_string(index=False, float_format="%.4f"), flush=True)

    # Best alpha
    best_row = result_df.loc[result_df["ic"].idxmax()]
    print(f"\n[best] alpha={best_row['alpha']:.1f}  ic={best_row['ic']:.4f}", flush=True)
    if best_row["alpha"] != 0.0:
        print(
            f"  TTPA improves over static by {best_row.get('ic_delta', 0.0):.4f} IC",
            flush=True,
        )
    else:
        print("  Static baseline is best — TTPA does not improve.", flush=True)

    print(f"\n[save] {out_dir / f'{stem}_sweep.csv'}", flush=True)


if __name__ == "__main__":
    main()
