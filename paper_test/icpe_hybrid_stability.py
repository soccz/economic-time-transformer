"""
Stability evaluation for the stock-only IC-PE hybrid path.

Goal:
  quantify how often `cycle_pe`, `concat_a`, and `static` win across
  seeds, anchors, and date windows under a fixed training budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_loader,
    build_state,
    build_target,
    evaluate,
    load_data,
    make_splits,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-PE hybrid stability evaluation")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--decoder", choices=("point", "quantile", "cvae"), default="point")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--pe-modes", default="static,concat_a,cycle_pe")
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
    )
    return parser.parse_args()


def run_one_config(base_args: argparse.Namespace, index_symbol: str, start: str, end: str, seed: int):
    set_seed(seed)
    port25, factors, index_close = load_data(start, end, index_symbol)
    source, target = build_target(port25, factors, base_args.target, base_args.roll_beta, base_args.horizon)
    position, intensity, regime = build_state(index_close)
    train_split, val_split, test_split = make_splits(
        dates=source.index,
        source=source,
        target=target,
        position=position,
        intensity=intensity,
        regime=regime,
        seq_len=base_args.seq_len,
    )

    test_loader = build_loader(test_split, base_args.batch_size, shuffle=False)
    rows = []
    pe_modes = [mode.strip() for mode in base_args.pe_modes.split(",") if mode.strip()]
    for pe_mode in pe_modes:
        set_seed(seed)
        run_args = SimpleNamespace(
            device=base_args.device,
            d_model=base_args.d_model,
            heads=base_args.heads,
            layers=base_args.layers,
            decoder=base_args.decoder,
            lr=base_args.lr,
            weight_decay=base_args.weight_decay,
            batch_size=base_args.batch_size,
            epochs=base_args.epochs,
        )
        model = train_model(run_args, pe_mode, train_split, val_split)
        metrics, _ = evaluate(model, test_loader, base_args.device, base_args.decoder)
        metrics.update(
            {
                "pe_mode": pe_mode,
                "index_symbol": index_symbol,
                "start": start,
                "end": end,
                "seed": seed,
            }
        )
        rows.append(metrics)
    return rows


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("^GSPC", "2020-01-01", "2024-12-31"),
        ("^GSPC", "2022-01-01", "2024-12-31"),
        ("^IXIC", "2020-01-01", "2024-12-31"),
    ]

    all_rows = []
    for index_symbol, start, end in configs:
        for seed in seeds:
            print(f"\n[stability] index={index_symbol} window={start[:4]}-{end[:4]} seed={seed}", flush=True)
            all_rows.extend(run_one_config(args, index_symbol, start, end, seed))

    df = pd.DataFrame(all_rows)
    detailed_path = out_dir / f"hybrid_stability_{args.target}_{args.decoder}_details.csv"
    df.to_csv(detailed_path, index=False)

    agg = (
        df.groupby(["index_symbol", "start", "end", "pe_mode"], as_index=False)
        .agg(
            **{
                "ic_mean": ("ic", "mean"),
                "ic_std": ("ic", "std"),
                "icir_mean": ("icir", "mean"),
                "mae_mean": ("mae", "mean"),
                "swap_mean": ("swap_delta", "mean"),
                **({"crps_mean": ("crps", "mean")} if "crps" in df.columns else {}),
                **({"pi80_mean": ("pi80", "mean")} if "pi80" in df.columns else {}),
                **({"pi80_width_mean": ("pi80_width", "mean")} if "pi80_width" in df.columns else {}),
                **({"qce_mean": ("qce", "mean")} if "qce" in df.columns else {}),
            }
        )
    )
    agg_path = out_dir / f"hybrid_stability_{args.target}_{args.decoder}_summary.csv"
    agg.to_csv(agg_path, index=False)

    win_rows = []
    for (index_symbol, start, end), sub in df.groupby(["index_symbol", "start", "end"]):
        by_ic = sub.groupby("pe_mode")["ic"].mean().sort_values(ascending=False)
        by_mae = sub.groupby("pe_mode")["mae"].mean().sort_values()
        by_crps = sub.groupby("pe_mode")["crps"].mean().sort_values() if "crps" in sub.columns else None
        by_qce = sub.groupby("pe_mode")["qce"].mean().sort_values() if "qce" in sub.columns else None
        concat_ic = by_ic.get("concat_a", float("nan"))
        win_rows.append(
            {
                "index_symbol": index_symbol,
                "start": start,
                "end": end,
                "winner_ic": by_ic.index[0],
                "winner_mae": by_mae.index[0],
                "winner_crps": by_crps.index[0] if by_crps is not None else None,
                "winner_qce": by_qce.index[0] if by_qce is not None else None,
                "cycle_minus_concat_ic": by_ic.get("cycle_pe", float("nan")) - concat_ic,
                "cycle_full_minus_concat_ic": by_ic.get("cycle_pe_full", float("nan")) - concat_ic,
                "flow_minus_concat_ic": by_ic.get("flow_pe", float("nan")) - concat_ic,
            }
        )
    wins = pd.DataFrame(win_rows)
    wins_path = out_dir / f"hybrid_stability_{args.target}_{args.decoder}_winners.csv"
    wins.to_csv(wins_path, index=False)

    print("\n[stability summary]", flush=True)
    print(agg.to_string(index=False), flush=True)
    print("\n[winners]", flush=True)
    print(wins.to_string(index=False), flush=True)
    print(f"\n[save] {detailed_path}", flush=True)
    print(f"[save] {agg_path}", flush=True)
    print(f"[save] {wins_path}", flush=True)


if __name__ == "__main__":
    main()
