"""
Routing ablation for the stock-only IC-PE hybrid path.

Goal:
  test whether the hybrid's local branch benefits from attention-guided routing
  relative to a control routing prior under the same architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


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
    parser = argparse.ArgumentParser(description="IC-PE hybrid routing ablation")
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
    parser.add_argument("--pe-modes", default="concat_a,flow_pe")
    parser.add_argument("--routing-modes", default="attention,uniform")
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
    )
    return parser.parse_args()


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
    test_loader = build_loader(test_split, args.batch_size, shuffle=False)

    date_tag = f"{args.start[:4]}_{args.end[:4]}"
    pe_modes = [mode.strip() for mode in args.pe_modes.split(",") if mode.strip()]
    routing_modes = [mode.strip() for mode in args.routing_modes.split(",") if mode.strip()]

    rows = []
    for pe_mode in pe_modes:
        for routing_mode in routing_modes:
            print(f"\n[routing] pe_mode={pe_mode} routing_mode={routing_mode}", flush=True)
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
                routing_mode=routing_mode,
            )
            set_seed(args.seed)
            model = train_model(run_args, pe_mode, train_split, val_split)
            metrics, pred_df = evaluate(model, test_loader, args.device, args.decoder)
            metrics["pe_mode"] = pe_mode
            metrics["routing_mode"] = routing_mode
            rows.append(metrics)

            stem = (
                f"hybrid_routing_{args.index_symbol.replace('^', '').lower()}_"
                f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_"
                f"{pe_mode}_{routing_mode}"
            )
            pred_df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
            print(
                f"[routing-test] pe={pe_mode:8s} routing={routing_mode:9s} "
                f"ic={metrics['ic']:.4f} icir={metrics['icir']:.4f} "
                f"mae={metrics['mae']:.6f} swap={metrics['swap_delta']:.6f}",
                flush=True,
            )

    result_df = pd.DataFrame(rows).sort_values(["pe_mode", "routing_mode"]).reset_index(drop=True)

    attention_ref = (
        result_df[result_df["routing_mode"] == "attention"][["pe_mode", "ic", "mae"]]
        .rename(columns={"ic": "attention_ic", "mae": "attention_mae"})
    )
    result_df = result_df.merge(attention_ref, on="pe_mode", how="left")
    result_df["ic_minus_attention"] = result_df["ic"] - result_df["attention_ic"]
    result_df["mae_minus_attention"] = result_df["mae"] - result_df["attention_mae"]

    stem = (
        f"hybrid_routing_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_summary.csv"
    )
    result_path = out_dir / stem
    result_df.to_csv(result_path, index=False)

    print("\n[routing summary]", flush=True)
    print(result_df.to_string(index=False), flush=True)
    print(f"\n[save] {result_path}", flush=True)


if __name__ == "__main__":
    main()
