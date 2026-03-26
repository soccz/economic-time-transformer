"""
Routing-masking diagnostic for the stock-only IC-PE hybrid path.

Idea:
  if attention-guided routing is meaningful, masking top-importance timesteps
  should damage performance more than masking a random set of timesteps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_loader,
    build_state,
    build_target,
    load_data,
    make_splits,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-PE routing masking diagnostic")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--decoder", choices=("point", "cvae"), default="point")
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
    parser.add_argument("--mask-frac", type=float, default=0.2)
    parser.add_argument("--pe-modes", default="static,concat_a,flow_pe")
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
    )
    return parser.parse_args()


def _predict_point(model, src: torch.Tensor, context: torch.Tensor, decoder_mode: str):
    pred, diag = model(src, context, return_diagnostics=True)
    pred_point = pred.squeeze(-1)
    if decoder_mode == "cvae":
        pred_point, _, _ = model.predict_interval(src, context, n_samples=50, alpha=0.80)
        pred_point = pred_point.squeeze(-1)
    return pred_point, diag


def _mask_from_indices(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    masked = src.clone()
    masked.scatter_(
        1,
        indices.unsqueeze(-1).expand(-1, -1, src.size(-1)),
        0.0,
    )
    return masked


@torch.no_grad()
def evaluate_masking(
    model,
    loader,
    device: str,
    decoder_mode: str,
    mask_frac: float,
    seed: int,
):
    model.eval()
    seq_len = loader.dataset.src.size(1)
    k = max(1, int(round(seq_len * mask_frac)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    rows = []
    for src, context, y, dates, regimes in loader:
        src = src.to(device)
        context = context.to(device)
        y = y.to(device).squeeze(-1)

        base_pred, diag = _predict_point(model, src, context, decoder_mode)
        importance = diag["attention_importance"]
        top_idx = importance.topk(k, dim=1).indices
        top_pred, _ = _predict_point(model, _mask_from_indices(src, top_idx), context, decoder_mode)

        rand_idx = torch.stack(
            [torch.randperm(seq_len, generator=generator)[:k] for _ in range(src.size(0))],
            dim=0,
        ).to(device)
        rand_pred, _ = _predict_point(model, _mask_from_indices(src, rand_idx), context, decoder_mode)

        for i in range(src.size(0)):
            rows.append(
                {
                    "date": pd.to_datetime(int(dates[i])),
                    "regime": int(regimes[i]),
                    "y_true": float(y[i].item()),
                    "base_pred": float(base_pred[i].item()),
                    "top_pred": float(top_pred[i].item()),
                    "rand_pred": float(rand_pred[i].item()),
                    "top_shift": float(torch.abs(top_pred[i] - base_pred[i]).item()),
                    "rand_shift": float(torch.abs(rand_pred[i] - base_pred[i]).item()),
                }
            )

    df = pd.DataFrame(rows)

    def daily_ic(pred_col: str) -> float:
        series = df.groupby("date").apply(
            lambda g: stats.spearmanr(g[pred_col], g["y_true"])[0],
            include_groups=False,
        ).dropna()
        return float(series.mean()) if len(series) else np.nan

    def mae(pred_col: str) -> float:
        return float(np.mean(np.abs(df[pred_col] - df["y_true"])))

    base_ic = daily_ic("base_pred")
    top_ic = daily_ic("top_pred")
    rand_ic = daily_ic("rand_pred")
    base_mae = mae("base_pred")
    top_mae = mae("top_pred")
    rand_mae = mae("rand_pred")

    summary = {
        "base_ic": base_ic,
        "top_mask_ic": top_ic,
        "rand_mask_ic": rand_ic,
        "ic_drop_top": float(base_ic - top_ic) if not np.isnan(base_ic) and not np.isnan(top_ic) else np.nan,
        "ic_drop_rand": float(base_ic - rand_ic) if not np.isnan(base_ic) and not np.isnan(rand_ic) else np.nan,
        "base_mae": base_mae,
        "top_mask_mae": top_mae,
        "rand_mask_mae": rand_mae,
        "mae_up_top": float(top_mae - base_mae),
        "mae_up_rand": float(rand_mae - base_mae),
        "pred_shift_top": float(df["top_shift"].mean()),
        "pred_shift_rand": float(df["rand_shift"].mean()),
        "mask_k": k,
    }
    return summary, df


def main():
    args = parse_args()
    set_seed(args.seed)

    port25, factors, index_close = load_data(args.start, args.end, args.index_symbol)
    source, target = build_target(port25, factors, args.target, args.roll_beta, args.horizon)
    position, intensity, regime = build_state(index_close)
    train_split, val_split, test_split = make_splits(
        dates=source.index,
        source=source,
        target=target,
        position=position,
        intensity=intensity,
        regime=regime,
        seq_len=args.seq_len,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = f"{args.start[:4]}_{args.end[:4]}"
    pe_modes = [mode.strip() for mode in args.pe_modes.split(",") if mode.strip()]
    test_loader = build_loader(test_split, args.batch_size, shuffle=False)

    rows = []
    for pe_mode in pe_modes:
        print(f"\n[mask] pe_mode={pe_mode}", flush=True)
        run_args = SimpleNamespace(
            device=args.device,
            d_model=args.d_model,
            heads=args.heads,
            layers=args.layers,
            decoder=args.decoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        model = train_model(run_args, pe_mode, train_split, val_split)
        metrics, detail_df = evaluate_masking(
            model=model,
            loader=test_loader,
            device=args.device,
            decoder_mode=args.decoder,
            mask_frac=args.mask_frac,
            seed=args.seed + 1000,
        )
        metrics["pe_mode"] = pe_mode
        rows.append(metrics)
        stem = (
            f"hybrid_mask_{args.index_symbol.replace('^', '').lower()}_"
            f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_{pe_mode}"
        )
        detail_df.to_csv(out_dir / f"{stem}_details.csv", index=False)
        print(
            f"[mask-test] mode={pe_mode:8s} "
            f"ic_drop_top={metrics['ic_drop_top']:.4f} ic_drop_rand={metrics['ic_drop_rand']:.4f} "
            f"mae_up_top={metrics['mae_up_top']:.6f} mae_up_rand={metrics['mae_up_rand']:.6f}",
            flush=True,
        )

    result_df = pd.DataFrame(rows).sort_values("pe_mode").reset_index(drop=True)
    stem = (
        f"hybrid_mask_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_summary.csv"
    )
    result_path = out_dir / stem
    result_df.to_csv(result_path, index=False)
    print("\n[mask summary]", flush=True)
    print(result_df.to_string(index=False), flush=True)
    print(f"\n[save] {result_path}", flush=True)


if __name__ == "__main__":
    main()
