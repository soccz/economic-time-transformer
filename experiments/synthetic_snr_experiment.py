"""
Synthetic SNR Experiment — Paper 1 Core Theory Validation

Tests whether FiLM conditioning outperforms concat conditioning
as the signal-to-noise ratio (SNR) of the conditioning signal varies.

Hypothesis: FiLM's multiplicative gating isolates the conditioning
signal better than concat under low-SNR regimes, because concat
forces the backbone to learn to "find" the signal in a mixed input,
while FiLM directly modulates activations.

Data generation:
  x:     (N, T=30, d_x=4) — random sequence features
  c_true:(N, 3) — true conditioning channels
  y:     c1*c2 * mean(x[:,-5:,0]) + noise — interaction target
  c_obs: c_true + N(0, 1/SNR) — noisy observation
"""

import csv
import math
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ──────────────────────────────────────────────
N = 5000
T = 30
D_X = 4
D_C = 3
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
EPOCHS = 20
BATCH = 256
LR = 1e-3
SNRS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
SEEDS = [0, 1, 2, 3, 4]
NOISE_SCALE_Y = 0.1  # irreducible noise on target


# ── Data Generation ──────────────────────────────────────────────
def generate_data(snr: float, seed: int):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(N, T, D_X, generator=rng)
    c_true = torch.randn(N, D_C, generator=rng)

    # Target: interaction between c1*c2 and recent x values
    tail_mean = x[:, -5:, 0].mean(dim=1)  # (N,)
    y = c_true[:, 0] * c_true[:, 1] * tail_mean
    y = y + NOISE_SCALE_Y * torch.randn(N, generator=rng)
    y = y.unsqueeze(1)  # (N,1)

    # Noisy conditioning observation
    eps_std = 1.0 / math.sqrt(snr) if snr > 0 else 1e6
    c_obs = c_true + eps_std * torch.randn(N, D_C, generator=rng)

    # Train/val/test split: 60/20/20
    n_train = int(0.6 * N)
    n_val = int(0.8 * N)

    def split(t):
        return t[:n_train], t[n_train:n_val], t[n_val:]

    x_tr, x_va, x_te = split(x)
    c_tr, c_va, c_te = split(c_obs)
    y_tr, y_va, y_te = split(y)
    return (x_tr, c_tr, y_tr), (x_va, c_va, y_va), (x_te, c_te, y_te)


# ── Models ───────────────────────────────────────────────────────
class TransformerBackbone(nn.Module):
    """Shared 2-layer Transformer encoder → mean-pool → head."""
    def __init__(self, d_in: int):
        super().__init__()
        self.proj = nn.Linear(d_in, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        """x: (B, T, d_in) → (B, 1)"""
        h = self.proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)  # mean-pool over time
        return self.head(h)


class ConcatModel(nn.Module):
    """Concat conditioning: broadcast c across time and concatenate with x."""
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBackbone(d_in=D_X + D_C)

    def forward(self, x, c):
        c_broad = c.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_C)
        xc = torch.cat([x, c_broad], dim=-1)         # (B, T, D_X+D_C)
        return self.backbone(xc)


class FiLMModel(nn.Module):
    """FiLM conditioning: gamma(c) * proj(x) + beta(c)."""
    def __init__(self):
        super().__init__()
        self.x_proj = nn.Linear(D_X, D_MODEL)
        # FiLM generators
        self.gamma_net = nn.Sequential(
            nn.Linear(D_C, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(D_C, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x, c):
        h = self.x_proj(x)                           # (B, T, D_MODEL)
        gamma = self.gamma_net(c).unsqueeze(1)        # (B, 1, D_MODEL)
        beta = self.beta_net(c).unsqueeze(1)          # (B, 1, D_MODEL)
        h = gamma * h + beta                          # FiLM modulation
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


# ── Training ─────────────────────────────────────────────────────
def train_and_eval(model, train_data, val_data, test_data):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    x_tr, c_tr, y_tr = [t.to(DEVICE) for t in train_data]
    x_va, c_va, y_va = [t.to(DEVICE) for t in val_data]
    x_te, c_te, y_te = [t.to(DEVICE) for t in test_data]

    train_ds = TensorDataset(x_tr, c_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, cb, yb in train_dl:
            pred = model(xb, cb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(x_va, c_va)
            val_mse = F.mse_loss(val_pred, y_va).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_te, c_te)
        test_mse = F.mse_loss(test_pred, y_te).item()
    return test_mse


# ── Main Sweep ───────────────────────────────────────────────────
def main():
    results = []
    out_path = Path(__file__).parent / "synthetic_snr_results.csv"

    total_runs = len(SNRS) * len(SEEDS) * 2
    run_idx = 0
    t0 = time.time()

    for snr in SNRS:
        for seed in SEEDS:
            train_d, val_d, test_d = generate_data(snr, seed)

            for method_name, ModelClass in [("concat", ConcatModel), ("film", FiLMModel)]:
                run_idx += 1
                torch.manual_seed(seed + 1000)
                torch.cuda.manual_seed_all(seed + 1000)

                model = ModelClass()
                mse = train_and_eval(model, train_d, val_d, test_d)
                results.append({
                    "snr": snr, "seed": seed, "method": method_name, "test_mse": mse
                })
                elapsed = time.time() - t0
                eta = elapsed / run_idx * (total_runs - run_idx)
                print(f"[{run_idx}/{total_runs}] SNR={snr:<6} seed={seed} "
                      f"{method_name:<7} MSE={mse:.5f}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Save CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr", "seed", "method", "test_mse"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")

    # ── Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'SNR':>6}  {'Concat MSE':>12} {'FiLM MSE':>12} {'Δ (C-F)':>10} {'Winner':>8}")
    print("-" * 70)
    for snr in SNRS:
        concat_vals = [r["test_mse"] for r in results
                       if r["snr"] == snr and r["method"] == "concat"]
        film_vals = [r["test_mse"] for r in results
                     if r["snr"] == snr and r["method"] == "film"]
        c_mean = sum(concat_vals) / len(concat_vals)
        f_mean = sum(film_vals) / len(film_vals)
        delta = c_mean - f_mean
        winner = "FiLM" if delta > 0 else "Concat"
        print(f"{snr:>6.1f}  {c_mean:>12.5f} {f_mean:>12.5f} {delta:>+10.5f} {winner:>8}")
    print("=" * 70)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
