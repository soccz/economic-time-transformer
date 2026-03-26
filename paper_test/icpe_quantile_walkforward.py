"""
Walk-forward quantile evaluation for the stock-only IC-PE hybrid path.

Goal:
  evaluate distribution quality under mixed market regimes using an actual
  neural quantile head rather than a trailing holdout split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_full_split,
    build_loader,
    build_state,
    build_target,
    evaluate,
    filter_split,
    load_data,
    set_seed,
    train_model,
)


REGIME_LABELS = {
    0: "Bear/quiet",
    1: "Bear/volatile",
    2: "Bull/quiet",
    3: "Bull/volatile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-PE walk-forward quantile evaluation")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--val-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pe-modes", default="static,concat_a,flow_pe")
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
    )
    return parser.parse_args()


def _date_mask(split_dates: np.ndarray, allowed_dates: np.ndarray) -> np.ndarray:
    return np.isin(split_dates, allowed_dates)


def _fold_dates(unique_dates: np.ndarray, eval_start: pd.Timestamp, eval_end: pd.Timestamp, step_days: int):
    eval_dates = unique_dates[(unique_dates >= eval_start.value) & (unique_dates <= eval_end.value)]
    return [eval_dates[i : i + step_days] for i in range(0, len(eval_dates), step_days) if len(eval_dates[i : i + step_days])]


def main():
    args = parse_args()
    set_seed(args.seed)

    port25, factors, index_close = load_data(args.start, args.end, args.index_symbol)
    source, target = build_target(port25, factors, args.target, args.roll_beta, args.horizon)
    position, intensity, regime = build_state(index_close)

    full_split = build_full_split(
        dates=source.index,
        source=source,
        target=target,
        position=position,
        intensity=intensity,
        regime=regime,
        seq_len=args.seq_len,
    )

    unique_dates = np.array(sorted(pd.unique(full_split.dates)))
    eval_folds = _fold_dates(
        unique_dates=unique_dates,
        eval_start=pd.Timestamp(args.eval_start),
        eval_end=pd.Timestamp(args.eval_end),
        step_days=args.step_days,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pe_modes = [mode.strip() for mode in args.pe_modes.split(",") if mode.strip()]

    all_metrics = []
    all_predictions = []

    for fold_idx, test_dates in enumerate(eval_folds):
        test_start = test_dates[0]
        train_pool = unique_dates[unique_dates < test_start]
        if len(train_pool) < args.train_days:
            continue

        train_dates = train_pool[-args.train_days : -args.val_days]
        val_dates = train_pool[-args.val_days :]
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue

        train_split = filter_split(full_split, _date_mask(full_split.dates, train_dates))
        val_split = filter_split(full_split, _date_mask(full_split.dates, val_dates))
        test_split = filter_split(full_split, _date_mask(full_split.dates, test_dates))
        if len(test_split.y) == 0:
            continue

        test_loader = build_loader(test_split, args.batch_size, shuffle=False)

        print(
            f"\n[fold {fold_idx:02d}] test={pd.to_datetime(test_dates[0]).date()} -> "
            f"{pd.to_datetime(test_dates[-1]).date()} | "
            f"train_dates={len(train_dates)} val_dates={len(val_dates)} test_dates={len(test_dates)}",
            flush=True,
        )

        for pe_mode in pe_modes:
            set_seed(args.seed + fold_idx)
            run_args = SimpleNamespace(
                device=args.device,
                d_model=args.d_model,
                heads=args.heads,
                layers=args.layers,
                decoder="quantile",
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            model = train_model(run_args, pe_mode, train_split, val_split)
            metrics, pred_df = evaluate(model, test_loader, args.device, "quantile")
            metrics.update(
                {
                    "fold": fold_idx,
                    "pe_mode": pe_mode,
                    "index_symbol": args.index_symbol,
                    "test_start": pd.to_datetime(test_dates[0]),
                    "test_end": pd.to_datetime(test_dates[-1]),
                }
            )
            pred_df["fold"] = fold_idx
            pred_df["pe_mode"] = pe_mode
            all_metrics.append(metrics)
            all_predictions.append(pred_df)
            print(
                f"[fold-test] pe={pe_mode:8s} ic={metrics['ic']:.4f} crps={metrics['crps']:.6f} "
                f"pi80={metrics['pi80']:.3f} qce={metrics['qce']:.4f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_predictions, ignore_index=True)

    stem = (
        f"hybrid_walkforward_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_quantile_{args.eval_start[:4]}_{args.eval_end[:4]}_e{args.epochs}"
    )

    metrics_df.to_csv(out_dir / f"{stem}_fold_metrics.csv", index=False)
    preds_df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)

    overall = (
        metrics_df.groupby("pe_mode", as_index=False)
        .agg(
            ic_mean=("ic", "mean"),
            icir_mean=("icir", "mean"),
            mae_mean=("mae", "mean"),
            crps_mean=("crps", "mean"),
            pi80_mean=("pi80", "mean"),
            pi80_width_mean=("pi80_width", "mean"),
            qce_mean=("qce", "mean"),
        )
        .sort_values("pe_mode")
        .reset_index(drop=True)
    )
    overall.to_csv(out_dir / f"{stem}_summary.csv", index=False)

    regime_summary = (
        preds_df[preds_df["regime"] >= 0]
        .assign(regime_label=lambda df: df["regime"].map(REGIME_LABELS))
        .groupby(["pe_mode", "regime_label"], as_index=False)
        .agg(
            n=("pred", "size"),
            crps_mean=("crps", "mean"),
            pi80_mean=("pi80", "mean"),
            pi80_width_mean=("pi80_width", "mean"),
            mae_mean=("pred", lambda s: np.nan),  # placeholder, filled below
        )
    )

    mae_by_regime = (
        preds_df[preds_df["regime"] >= 0]
        .assign(regime_label=lambda df: df["regime"].map(REGIME_LABELS))
        .assign(abs_err=lambda df: (df["pred"] - df["y_true"]).abs())
        .groupby(["pe_mode", "regime_label"], as_index=False)
        .agg(mae_mean=("abs_err", "mean"))
    )
    regime_summary = regime_summary.drop(columns=["mae_mean"]).merge(
        mae_by_regime,
        on=["pe_mode", "regime_label"],
        how="left",
    )
    regime_summary.to_csv(out_dir / f"{stem}_regime_summary.csv", index=False)

    print("\n[walk-forward summary]", flush=True)
    print(overall.to_string(index=False), flush=True)
    print("\n[walk-forward regime summary]", flush=True)
    print(regime_summary.to_string(index=False), flush=True)
    print(f"\n[save] {out_dir / f'{stem}_summary.csv'}", flush=True)
    print(f"[save] {out_dir / f'{stem}_regime_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
