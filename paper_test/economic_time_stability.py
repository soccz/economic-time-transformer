"""
Repeated-seed stability evaluation for the economic-time path.

Focus:
  - compare static / concat_a / economic-time variants over multiple seeds
  - preserve paired daily-IC tests within each seed
  - flag sanity failures where economic-time falls below static
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_test.economic_time_supervised import (  # noqa: E402
    _compute_daily_ic,
    _paired_ttest,
    build_loader,
    build_split,
    build_target,
    build_state,
    evaluate_model,
    load_data,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economic-time repeated-seed stability evaluation")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--epoch-budgets",
        default="",
        help="optional comma-separated training budgets; if empty, uses --epochs",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="7,17,27")
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
        help="comma-separated; econ_time variants: econ_time, econ_time:pe_only, econ_time:qk_only",
    )
    parser.add_argument(
        "--reference-model",
        default="concat_a",
        help="model used as paired-test reference within each run",
    )
    parser.add_argument("--output-dir", default="paper/economic_time/results/stability")
    return parser.parse_args()


def build_run_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    return run_args


def _combine_pred_df(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    assert len(df_a) == len(df_b), "prediction lengths must match"
    assert np.array_equal(df_a["date"].values, df_b["date"].values), "prediction dates must align"
    assert np.allclose(df_a["y_true"].values, df_b["y_true"].values), "prediction targets must align"
    out = df_a.copy()
    out["pred"] = alpha * df_a["pred"].values + (1.0 - alpha) * df_b["pred"].values
    return out


def _metrics_from_pred_df(df: pd.DataFrame) -> dict[str, float]:
    daily_ic = _compute_daily_ic(df)
    return {
        "ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
        "icir": float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) and daily_ic.std() > 0 else np.nan,
        "mae": float(np.mean(np.abs(df["pred"] - df["y_true"]))),
    }


def _fit_linear_ensemble(val_df_a: pd.DataFrame, val_df_b: pd.DataFrame) -> tuple[float, dict[str, float]]:
    best_alpha = 0.5
    best_metrics = {"ic": -np.inf, "icir": np.nan, "mae": np.nan}
    for alpha in np.linspace(0.0, 1.0, 21):
        combo = _combine_pred_df(val_df_a, val_df_b, float(alpha))
        metrics = _metrics_from_pred_df(combo)
        score = metrics["ic"] if not np.isnan(metrics["ic"]) else -np.inf
        if score > best_metrics["ic"]:
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    epoch_budgets = [int(e.strip()) for e in args.epoch_budgets.split(",") if e.strip()] or [args.epochs]
    model_kinds = [k.strip() for k in args.model_kinds.split(",") if k.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    val_loader = build_loader(val_split, args.batch_size, shuffle=False)
    test_loader = build_loader(test_split, args.batch_size, shuffle=False)

    detail_rows = []
    ttest_rows = []
    sanity_rows = []
    history_rows = []
    pooled_ic_rows = []

    for epoch_budget in epoch_budgets:
        for seed in seeds:
            print(f"\n[stability] epochs={epoch_budget} seed={seed}", flush=True)
            run_args = build_run_args(args, seed)
            run_args.epochs = epoch_budget
            val_pred_dfs: dict[str, pd.DataFrame] = {}
            pred_dfs: dict[str, pd.DataFrame] = {}
            seed_metrics: dict[str, dict] = {}

            for model_kind in model_kinds:
                set_seed(seed)
                print(f"[run] epochs={epoch_budget} seed={seed} model_kind={model_kind}", flush=True)
                model = train_model(run_args, model_kind, train_split, val_split)
                for row in getattr(model, "training_history", []):
                    history_rows.append(
                        {
                            "epochs": epoch_budget,
                            "seed": seed,
                            "model_kind": model_kind,
                            "index_symbol": args.index_symbol,
                            "start": args.start,
                            "end": args.end,
                            **row,
                        }
                    )
                _, val_pred_df = evaluate_model(model, model_kind, val_loader, args.device)
                metrics, pred_df = evaluate_model(model, model_kind, test_loader, args.device)
                metrics.update(
                    {
                        "epochs": epoch_budget,
                        "seed": seed,
                        "model_kind": model_kind,
                        "index_symbol": args.index_symbol,
                        "start": args.start,
                        "end": args.end,
                    }
                )
                detail_rows.append(metrics)
                val_pred_dfs[model_kind] = val_pred_df
                pred_dfs[model_kind] = pred_df
                seed_metrics[model_kind] = metrics
                daily_ic = _compute_daily_ic(pred_df)
                for date, ic_val in daily_ic.items():
                    pooled_ic_rows.append(
                        {
                            "epochs": epoch_budget,
                            "seed": seed,
                            "model_kind": model_kind,
                            "date": pd.to_datetime(date),
                            "daily_ic": float(ic_val),
                        }
                    )
                print(
                    f"[test] epochs={epoch_budget} seed={seed} kind={model_kind:16s} ic={metrics['ic']:.4f} "
                    f"icir={metrics['icir']:.4f} mae={metrics['mae']:.6f}",
                    flush=True,
                )

            interaction_specs = [
                ("interaction:intensity+position", "concat_a:intensity_only", "concat_a:position_only", "concat_a"),
                ("interaction:intensity+indexret", "concat_a:intensity_only", "concat_a:indexret_only", "concat_a:intensity_indexret"),
            ]
            for interaction_name, mk_a, mk_b, mk_pair in interaction_specs:
                if not all(mk in val_pred_dfs for mk in (mk_a, mk_b, mk_pair)):
                    continue
                alpha, val_metrics = _fit_linear_ensemble(val_pred_dfs[mk_a], val_pred_dfs[mk_b])
                combo_test_df = _combine_pred_df(pred_dfs[mk_a], pred_dfs[mk_b], alpha)
                combo_test_metrics = _metrics_from_pred_df(combo_test_df)
                detail_rows.append(
                    {
                        "epochs": epoch_budget,
                        "seed": seed,
                        "model_kind": interaction_name,
                        "index_symbol": args.index_symbol,
                        "start": args.start,
                        "end": args.end,
                        "ic": combo_test_metrics["ic"],
                        "icir": combo_test_metrics["icir"],
                        "mae": combo_test_metrics["mae"],
                        "param_count": np.nan,
                        "ensemble_alpha": alpha,
                        "ensemble_val_ic": val_metrics["ic"],
                        "pair_model": mk_pair,
                        "single_a": mk_a,
                        "single_b": mk_b,
                        "interaction_gain_max": combo_test_metrics["ic"] - max(seed_metrics[mk_a]["ic"], seed_metrics[mk_b]["ic"]),
                        "interaction_gap_to_pair": seed_metrics[mk_pair]["ic"] - combo_test_metrics["ic"],
                    }
                )

            ic_reference = _compute_daily_ic(pred_dfs[args.reference_model]) if args.reference_model in pred_dfs else None
            ic_static = _compute_daily_ic(pred_dfs["static"]) if "static" in pred_dfs else None
            static_ic = seed_metrics["static"]["ic"] if "static" in seed_metrics else None

            for model_kind in model_kinds:
                if model_kind == args.reference_model or ic_reference is None:
                    continue
                res = _paired_ttest(_compute_daily_ic(pred_dfs[model_kind]), ic_reference)
                ttest_rows.append(
                    {
                        "epochs": epoch_budget,
                        "seed": seed,
                        "model_kind": model_kind,
                        "vs": args.reference_model,
                        **res,
                    }
                )
                if model_kind == "static" or ic_static is None:
                    continue
                res_static = _paired_ttest(_compute_daily_ic(pred_dfs[model_kind]), ic_static)
                ttest_rows.append(
                    {
                        "epochs": epoch_budget,
                        "seed": seed,
                        "model_kind": model_kind,
                        "vs": "static",
                        **res_static,
                    }
                )
                sanity_rows.append(
                    {
                        "epochs": epoch_budget,
                        "seed": seed,
                        "model_kind": model_kind,
                        "ic": seed_metrics[model_kind]["ic"],
                        "static_ic": static_ic,
                        "below_static": bool(seed_metrics[model_kind]["ic"] < static_ic),
                        "concat_ic": seed_metrics[args.reference_model]["ic"] if args.reference_model in seed_metrics else None,
                        "below_concat": bool(
                            seed_metrics[model_kind]["ic"] < seed_metrics[args.reference_model]["ic"]
                        )
                        if args.reference_model in seed_metrics
                        else None,
                        "pe_scale_mean": seed_metrics[model_kind].get("pe_scale_mean"),
                    }
                )

    detail_df = pd.DataFrame(detail_rows).sort_values(["epochs", "seed", "model_kind"]).reset_index(drop=True)
    detail_path = out_dir / "economic_time_stability_details.csv"
    detail_df.to_csv(detail_path, index=False)

    pooled_ic_df = pd.DataFrame(pooled_ic_rows)
    pooled_ic_path = out_dir / "economic_time_stability_pooled_daily_ic.csv"
    pooled_ic_df.to_csv(pooled_ic_path, index=False)

    summary_df = (
        detail_df.groupby(["epochs", "model_kind"], as_index=False)
        .agg(
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            icir_mean=("icir", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            param_count=("param_count", "mean"),
            step_intensity_spearman_mean=("step_intensity_spearman_mean", "mean"),
            tau_corr_mean=("tau_corr_mean", "mean"),
            pe_scale_mean=("pe_scale_mean", "mean"),
            pe_scale_std=("pe_scale_mean", "std"),
            fusion_gate_mean=("fusion_gate_mean", "mean"),
            context_gate_mean=("context_gate_mean", "mean"),
            attn_swap_delta_mean=("attn_swap_delta_mean", "mean"),
            qk_swap_delta_mean=("qk_swap_delta_mean", "mean"),
            xip_h_int_norm_mean=("xip_h_int_norm_mean", "mean"),
            xip_h_int_ratio_mean=("xip_h_int_ratio_mean", "mean"),
            xip_pred_delta_mean=("xip_pred_delta_mean", "mean"),
            xip_ic_off_mean=("xip_ic_off", "mean"),
            xip_ic_drop_mean=("xip_ic_drop", "mean"),
        )
        .sort_values(["epochs", "model_kind"])
        .reset_index(drop=True)
    )
    summary_path = out_dir / "economic_time_stability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if ttest_rows:
        ttest_df = pd.DataFrame(ttest_rows).sort_values(["epochs", "vs", "seed", "model_kind"]).reset_index(drop=True)
    else:
        ttest_df = pd.DataFrame(columns=["epochs", "seed", "model_kind", "vs"])
    ttest_path = out_dir / "economic_time_stability_ttests.csv"
    ttest_df.to_csv(ttest_path, index=False)

    pooled_ttest_rows = []
    if not pooled_ic_df.empty:
        for epoch_budget in sorted(pooled_ic_df["epochs"].unique()):
            epoch_df = pooled_ic_df[pooled_ic_df["epochs"] == epoch_budget]
            ref_df = epoch_df[epoch_df["model_kind"] == args.reference_model]
            if ref_df.empty:
                continue
            for model_kind in sorted(epoch_df["model_kind"].unique()):
                if model_kind == args.reference_model:
                    continue
                model_df = epoch_df[epoch_df["model_kind"] == model_kind]
                merged = model_df.merge(
                    ref_df,
                    on=["epochs", "seed", "date"],
                    how="inner",
                    suffixes=("_model", "_ref"),
                )
                if len(merged) < 5:
                    pooled_ttest_rows.append(
                        {
                            "epochs": epoch_budget,
                            "model_kind": model_kind,
                            "vs": args.reference_model,
                            "t": np.nan,
                            "p": np.nan,
                            "n": int(len(merged)),
                        }
                    )
                    continue
                res = _paired_ttest(
                    pd.Series(merged["daily_ic_model"].values, index=np.arange(len(merged))),
                    pd.Series(merged["daily_ic_ref"].values, index=np.arange(len(merged))),
                )
                pooled_ttest_rows.append(
                    {
                        "epochs": epoch_budget,
                        "model_kind": model_kind,
                        "vs": args.reference_model,
                        "t": res["t"],
                        "p": res["p"],
                        "n": res["n"],
                    }
                )
    pooled_ttest_df = pd.DataFrame(pooled_ttest_rows)
    pooled_ttest_path = out_dir / "economic_time_stability_pooled_ttests.csv"
    pooled_ttest_df.to_csv(pooled_ttest_path, index=False)

    if sanity_rows:
        sanity_df = pd.DataFrame(sanity_rows).sort_values(["epochs", "seed", "model_kind"]).reset_index(drop=True)
    else:
        sanity_df = pd.DataFrame(columns=["epochs", "seed", "model_kind"])
    sanity_path = out_dir / "economic_time_stability_sanity.csv"
    sanity_df.to_csv(sanity_path, index=False)

    if not sanity_df.empty:
        flag_df = (
            sanity_df.groupby(["epochs", "model_kind"], as_index=False)
            .agg(
                below_static_count=("below_static", "sum"),
                below_concat_count=("below_concat", "sum"),
                seed_count=("seed", "count"),
            )
            .sort_values(["epochs", "model_kind"])
            .reset_index(drop=True)
        )
    else:
        flag_df = pd.DataFrame(columns=["epochs", "model_kind", "below_static_count", "below_concat_count", "seed_count"])
    flag_path = out_dir / "economic_time_stability_flags.csv"
    flag_df.to_csv(flag_path, index=False)

    if history_rows:
        history_df = pd.DataFrame(history_rows).sort_values(["epochs", "seed", "model_kind", "epoch"]).reset_index(drop=True)
        history_path = out_dir / "economic_time_stability_train_history.csv"
        history_df.to_csv(history_path, index=False)
    else:
        history_path = None

    print("\n[stability summary]", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("\n[sanity flags]", flush=True)
    print(flag_df.to_string(index=False), flush=True)
    print(f"\n[save] {detail_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    print(f"[save] {ttest_path}", flush=True)
    print(f"[save] {pooled_ic_path}", flush=True)
    print(f"[save] {pooled_ttest_path}", flush=True)
    print(f"[save] {sanity_path}", flush=True)
    print(f"[save] {flag_path}", flush=True)
    if history_path is not None:
        print(f"[save] {history_path}", flush=True)


if __name__ == "__main__":
    main()
