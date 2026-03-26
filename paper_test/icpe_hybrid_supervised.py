"""
Paper-specific supervised hybrid evaluation for IC-PE.

This is the first actual neural path for:
  static vs concat_a vs cycle_pe

It is intentionally independent from the legacy GAN trainer.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pandas_datareader.data as web
except TypeError:
    # pandas 3.x removed deprecate_kwarg; patch it for pandas_datareader compat
    import pandas.util._decorators as _pd_dec
    _pd_dec.deprecate_kwarg = lambda old, new=None, mapping=None, stacklevel=2: (lambda f: f)
    import pandas_datareader.data as web
import torch
import torch.nn.functional as F
import yfinance as yf
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset


AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for path in (AAA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper.index_conditioned_pe.icpe_hybrid_model import PaperICPEHybrid  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
Q_MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-PE supervised hybrid evaluation")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--decoder", choices=("point", "quantile", "cvae"), default="point")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pe-modes", default="static,concat_a,cycle_pe")
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitData:
    src: np.ndarray
    context: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    regime: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, split: SplitData):
        self.src = torch.from_numpy(split.src).float()
        self.context = torch.from_numpy(split.context).float()
        self.y = torch.from_numpy(split.y).float().unsqueeze(-1)
        self.dates = split.dates
        self.regime = split.regime

    def __len__(self) -> int:
        return self.src.size(0)

    def __getitem__(self, idx: int):
        return self.src[idx], self.context[idx], self.y[idx], self.dates[idx], self.regime[idx]


def load_data(start: str, end: str, index_symbol: str):
    print(f"[load] Ken French 25 portfolios + FF3 + {index_symbol}", flush=True)
    cache_dir = AAA_ROOT / "paper" / "economic_time" / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    symbol_tag = index_symbol.replace("^", "").replace("/", "_").lower()
    port25_path = cache_dir / f"port25_{start}_{end}.csv"
    factors_path = cache_dir / f"ff3_{start}_{end}.csv"
    index_path = cache_dir / f"index_{symbol_tag}_{start}_{end}.csv"

    def _load_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        port25_cached = pd.read_csv(port25_path, index_col=0, parse_dates=True)
        factors_cached = pd.read_csv(factors_path, index_col=0, parse_dates=True)
        index_cached = pd.read_csv(index_path, index_col=0, parse_dates=True).iloc[:, 0]
        index_cached.name = "Close"
        return port25_cached, factors_cached, index_cached

    try:
        port25 = web.DataReader("25_Portfolios_5x5_daily", "famafrench", start, end)[0] / 100
        factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0] / 100
        index_close = pd.Series(dtype=float)
        last_error = None
        for attempt in range(1, 4):
            try:
                downloaded = yf.download(index_symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
                if isinstance(downloaded, pd.Series) and not downloaded.empty:
                    index_close = downloaded
                    break
                last_error = RuntimeError(f"empty download on attempt {attempt}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if attempt < 3:
                print(f"[load] retry {attempt}/3 for {index_symbol}", flush=True)
                time.sleep(2.0)

        if index_close.empty:
            raise RuntimeError(f"Failed to download {index_symbol} from Yahoo Finance") from last_error

        port25.to_csv(port25_path)
        factors.to_csv(factors_path)
        index_close.to_frame(name="Close").to_csv(index_path)
    except Exception as exc:  # noqa: BLE001
        if port25_path.exists() and factors_path.exists() and index_path.exists():
            print(f"[load] network fetch failed, using cached data for {index_symbol}", flush=True)
            port25, factors, index_close = _load_cached()
        else:
            raise exc

    common = port25.index.intersection(factors.index).intersection(index_close.index)
    port25 = port25.reindex(common)
    factors = factors.reindex(common)
    index_close = index_close.reindex(common).ffill()
    print(f"[load] {common[0].date()} ~ {common[-1].date()} | T={len(common)}", flush=True)
    return port25, factors, index_close


def build_target(port25, factors, target_kind: str, roll_beta: int, horizon: int):
    rf = factors["RF"]
    excess = port25.subtract(rf, axis=0)
    if target_kind == "raw":
        source = excess.copy()
        target = excess.rolling(horizon).sum().shift(-horizon)
        return source, target

    factor_cols = pd.DataFrame(
        {
            "MKT": factors["Mkt-RF"],
            "SMB": factors["SMB"],
            "HML": factors["HML"],
        },
        index=port25.index,
    )
    resid = pd.DataFrame(index=port25.index, columns=port25.columns, dtype=float)
    print(f"[target] rolling FF3 residual target | beta window={roll_beta}", flush=True)
    for col in port25.columns:
        y = excess[col]
        fitted = pd.Series(index=port25.index, dtype=float)
        for end_idx in range(roll_beta, len(port25.index)):
            sl = slice(end_idx - roll_beta, end_idx)
            x_win = factor_cols.iloc[sl].copy()
            x_win.insert(0, "const", 1.0)
            y_win = y.iloc[sl]
            if y_win.isna().any() or x_win.isna().any().any():
                continue
            beta = np.linalg.lstsq(x_win.values, y_win.values, rcond=None)[0]
            x_t = np.array([1.0, factor_cols.iloc[end_idx]["MKT"], factor_cols.iloc[end_idx]["SMB"], factor_cols.iloc[end_idx]["HML"]])
            fitted.iloc[end_idx] = float(np.dot(x_t, beta))
        resid[col] = y - fitted
    target = resid.rolling(horizon).sum().shift(-horizon)
    return resid, target


def build_state(index_close: pd.Series):
    ma200 = index_close.rolling(200).mean()
    position = (index_close - ma200) / ma200
    rv30 = np.log(index_close / index_close.shift(1)).rolling(30).std() * np.sqrt(252)
    intensity = rv30.rolling(252).rank(pct=True)
    regime = 2 * (position > 0).astype(int) + (intensity > 0.5).astype(int)
    return position, intensity, regime


def build_features(source: pd.DataFrame):
    values = source.values
    mean5 = source.rolling(5).mean().values
    std5 = source.rolling(5).std().values

    centered = source.sub(source.mean(axis=1), axis=0)
    xs_rank = centered.rank(axis=1, pct=True).sub(0.5).values
    return values, mean5, std5, xs_rank


def make_splits(
    dates: pd.Index,
    source: pd.DataFrame,
    target: pd.DataFrame,
    position: pd.Series,
    intensity: pd.Series,
    regime: pd.Series,
    seq_len: int,
) -> tuple[SplitData, SplitData, SplitData]:
    full_split = build_full_split(
        dates=dates,
        source=source,
        target=target,
        position=position,
        intensity=intensity,
        regime=regime,
        seq_len=seq_len,
    )

    unique_dates = np.array(sorted(pd.unique(full_split.dates)))
    n_dates = len(unique_dates)
    train_cut = int(n_dates * 0.7)
    val_cut = int(n_dates * 0.85)

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])

    return (
        filter_split(full_split, np.isin(full_split.dates, list(train_dates))),
        filter_split(full_split, np.isin(full_split.dates, list(val_dates))),
        filter_split(full_split, np.isin(full_split.dates, list(test_dates))),
    )


def build_full_split(
    dates: pd.Index,
    source: pd.DataFrame,
    target: pd.DataFrame,
    position: pd.Series,
    intensity: pd.Series,
    regime: pd.Series,
    seq_len: int,
) -> SplitData:
    raw, mean5, std5, xs_rank = build_features(source)
    pos_arr = position.values
    int_arr = intensity.values
    reg_arr = regime.values

    src_list = []
    ctx_list = []
    y_list = []
    date_list = []
    regime_list = []

    for t in range(seq_len, len(dates)):
        date = dates[t]
        for asset_idx, col in enumerate(source.columns):
            y_true = target.iloc[t, asset_idx]
            if pd.isna(y_true):
                continue

            src_seq = np.column_stack(
                [
                    raw[t - seq_len:t, asset_idx],
                    mean5[t - seq_len:t, asset_idx],
                    std5[t - seq_len:t, asset_idx],
                    xs_rank[t - seq_len:t, asset_idx],
                ]
            )
            ctx_seq = np.column_stack(
                [
                    pos_arr[t - seq_len:t],
                    int_arr[t - seq_len:t],
                ]
            )
            if np.isnan(src_seq).any() or np.isnan(ctx_seq).any():
                continue

            src_list.append(src_seq.astype(np.float32))
            ctx_list.append(ctx_seq.astype(np.float32))
            y_list.append(float(y_true))
            date_list.append(date)
            regime_list.append(int(reg_arr[t - 1]) if not pd.isna(reg_arr[t - 1]) else -1)

    src = np.stack(src_list)
    ctx = np.stack(ctx_list)
    y = np.array(y_list, dtype=np.float32)
    dts = np.array(date_list, dtype="datetime64[ns]").astype("int64")
    reg = np.array(regime_list, dtype=np.int64)
    return SplitData(src=src, context=ctx, y=y, dates=dts, regime=reg)


def filter_split(split: SplitData, mask: np.ndarray) -> SplitData:
    return SplitData(
        src=split.src[mask],
        context=split.context[mask],
        y=split.y[mask],
        dates=split.dates[mask],
        regime=split.regime[mask],
    )


def build_loader(split: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SequenceDataset(split), batch_size=batch_size, shuffle=shuffle)


def sample_crps(samples: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # samples: (B, S, 1), y_true: (B, 1)
    term1 = torch.mean(torch.abs(samples - y_true.unsqueeze(1)), dim=1).squeeze(-1)
    pairwise = torch.abs(samples.unsqueeze(2) - samples.unsqueeze(1))
    term2 = 0.5 * torch.mean(pairwise, dim=(1, 2, 3))
    return term1 - term2


def quantile_pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    taus = pred.new_tensor(QUANTILE_LEVELS).view(1, -1)
    errors = target - pred
    return torch.maximum((taus - 1.0) * errors, taus * errors).mean()


def quantile_crps(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    taus = pred.new_tensor(QUANTILE_LEVELS).view(1, -1)
    errors = target - pred
    pinball = torch.maximum((taus - 1.0) * errors, taus * errors)
    return pinball.mean(dim=-1)


def train_one_epoch(model, loader, optimizer, device: str, decoder_mode: str):
    model.train()
    total = 0.0
    total_n = 0
    for src, context, y, _, _ in loader:
        src = src.to(device)
        context = context.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, kl = model(src, context, y_true=y if decoder_mode == "cvae" else None)
        if decoder_mode == "quantile":
            loss = quantile_pinball_loss(pred, y)
        else:
            loss = F.mse_loss(pred, y)
        if decoder_mode == "cvae":
            loss = loss + 1e-3 * kl
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item()) * src.size(0)
        total_n += src.size(0)
    return total / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device: str, decoder_mode: str):
    model.eval()
    rows = []
    for src, context, y, dates, regimes in loader:
        src = src.to(device)
        context = context.to(device)
        y = y.to(device)

        pred, diag = model(src, context, return_diagnostics=True)
        quantile_values = None

        if decoder_mode == "cvae":
            pred_point = pred.squeeze(-1)
            mean_pred, pi_low, pi_high = model.predict_interval(src, context, n_samples=50, alpha=0.80)
            samples = model.decoder.sample(diag["fused"], n_samples=50)
            crps = sample_crps(samples, y).cpu().numpy()
            pi80 = ((y >= pi_low) & (y <= pi_high)).float().squeeze(-1).cpu().numpy()
            qce_components = np.full((pred_point.size(0), len(QUANTILE_LEVELS)), np.nan)
            interval_width = (pi_high - pi_low).squeeze(-1).cpu().numpy()
            pred_used = mean_pred.squeeze(-1).cpu().numpy()
        elif decoder_mode == "quantile":
            quantile_values = pred
            pred_point = quantile_values[:, Q_MEDIAN_INDEX]
            crps = quantile_crps(quantile_values, y).cpu().numpy()
            pi_low = quantile_values[:, 0]
            pi_high = quantile_values[:, -1]
            pi80 = ((y.squeeze(-1) >= pi_low) & (y.squeeze(-1) <= pi_high)).float().cpu().numpy()
            qce_components = np.stack(
                [
                    (y.squeeze(-1) <= quantile_values[:, idx]).float().cpu().numpy()
                    for idx in range(len(QUANTILE_LEVELS))
                ],
                axis=1,
            )
            interval_width = (pi_high - pi_low).cpu().numpy()
            pred_used = pred_point.cpu().numpy()
        else:
            pred_point = pred.squeeze(-1)
            crps = np.full(pred.size(0), np.nan)
            pi80 = np.full(pred.size(0), np.nan)
            qce_components = np.full((pred_point.size(0), len(QUANTILE_LEVELS)), np.nan)
            interval_width = np.full(pred_point.size(0), np.nan)
            pred_used = pred_point.cpu().numpy()

        swap_context = context.clone()
        swap_context[..., 0] = -swap_context[..., 0]
        swap_context[..., 1] = 1.0 - swap_context[..., 1]
        swap_pred, _ = model(src, swap_context)
        if decoder_mode == "quantile":
            swap_point = swap_pred[:, Q_MEDIAN_INDEX]
        else:
            swap_point = swap_pred.squeeze(-1)
        swap_delta = (swap_point - pred_point).abs().cpu().numpy()

        gate = diag["gate"].cpu().numpy()
        attn = diag["attention_importance"].cpu().numpy()
        attn_entropy = -(attn * np.log(np.clip(attn, 1e-8, 1.0))).sum(axis=1)

        for idx in range(len(pred_used)):
            row = {
                "date": pd.to_datetime(int(dates[idx])),
                "y_true": float(y[idx].item()),
                "pred": float(pred_used[idx]),
                "regime": int(regimes[idx]),
                "gate": float(gate[idx]),
                "attn_entropy": float(attn_entropy[idx]),
                "state_swap_delta": float(swap_delta[idx]),
                "crps": float(crps[idx]) if not math.isnan(crps[idx]) else np.nan,
                "pi80": float(pi80[idx]) if not math.isnan(pi80[idx]) else np.nan,
                "pi80_width": float(interval_width[idx]) if not math.isnan(interval_width[idx]) else np.nan,
            }
            for q_idx, tau in enumerate(QUANTILE_LEVELS):
                row[f"qce_hit_{int(tau * 100):02d}"] = float(qce_components[idx, q_idx]) if not math.isnan(qce_components[idx, q_idx]) else np.nan
                if quantile_values is not None:
                    row[f"q_{int(tau * 100):02d}"] = float(quantile_values[idx, q_idx].item())
            rows.append(row)
    df = pd.DataFrame(rows)
    daily_ic = df.groupby("date").apply(lambda g: stats.spearmanr(g["pred"], g["y_true"])[0], include_groups=False).dropna()
    metrics = {
        "ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
        "icir": float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) and daily_ic.std() > 0 else np.nan,
        "mae": float(np.mean(np.abs(df["pred"] - df["y_true"]))),
        "swap_delta": float(df["state_swap_delta"].mean()),
        "gate_mean": float(df["gate"].mean()),
        "attn_entropy": float(df["attn_entropy"].mean()),
    }
    if decoder_mode in ("cvae", "quantile"):
        metrics["crps"] = float(df["crps"].mean())
        metrics["pi80"] = float(df["pi80"].mean())
        metrics["pi80_width"] = float(df["pi80_width"].mean())
    if decoder_mode == "quantile":
        qce = []
        for tau in QUANTILE_LEVELS:
            hit_col = f"qce_hit_{int(tau * 100):02d}"
            qce.append(abs(float(df[hit_col].mean()) - tau))
        metrics["qce"] = float(np.mean(qce))
    return metrics, df


def train_model(args, pe_mode: str, train_split: SplitData, val_split: SplitData):
    device = args.device
    output_dim = len(QUANTILE_LEVELS) if args.decoder == "quantile" else 1
    model = PaperICPEHybrid(
        input_dim=train_split.src.shape[-1],
        context_dim=train_split.context.shape[-1],
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        pe_mode=pe_mode,
        routing_mode=getattr(args, "routing_mode", "attention"),
        decoder_mode=args.decoder,
        output_dim=output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = build_loader(train_split, args.batch_size, shuffle=True)
    val_loader = build_loader(val_split, args.batch_size, shuffle=False)

    best_state = None
    best_score = -np.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.decoder)
        val_metrics, _ = evaluate(model, val_loader, device, args.decoder)
        score = val_metrics["ic"] if not np.isnan(val_metrics["ic"]) else -np.inf
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"[train] mode={pe_mode:8s} epoch={epoch:02d} "
            f"loss={train_loss:.6f} val_ic={val_metrics['ic']:.4f} val_mae={val_metrics['mae']:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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
    results = []
    for pe_mode in pe_modes:
        print(f"\n[run] pe_mode={pe_mode}", flush=True)
        model = train_model(args, pe_mode, train_split, val_split)
        test_loader = build_loader(test_split, args.batch_size, shuffle=False)
        metrics, pred_df = evaluate(model, test_loader, args.device, args.decoder)
        metrics["pe_mode"] = pe_mode
        results.append(metrics)
        stem = (
            f"hybrid_{args.index_symbol.replace('^', '').lower()}_"
            f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_{pe_mode}"
        )
        pred_df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
        print(
            f"[test] mode={pe_mode:8s} ic={metrics['ic']:.4f} icir={metrics['icir']:.4f} "
            f"mae={metrics['mae']:.6f} swap={metrics['swap_delta']:.6f}",
            flush=True,
        )

    result_df = pd.DataFrame(results).sort_values("pe_mode").reset_index(drop=True)
    stem = (
        f"hybrid_{args.index_symbol.replace('^', '').lower()}_"
        f"{args.target}_{args.decoder}_{date_tag}_e{args.epochs}_summary.csv"
    )
    result_path = out_dir / stem
    result_df.to_csv(result_path, index=False)
    print("\n[summary]", flush=True)
    print(result_df.to_string(index=False), flush=True)
    print(f"\n[save] {result_path}", flush=True)


if __name__ == "__main__":
    main()
