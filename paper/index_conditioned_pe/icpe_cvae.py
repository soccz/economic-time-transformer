from __future__ import annotations

import torch
import torch.nn as nn


class ICPECVAE(nn.Module):
    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.enc = nn.Sequential(
            nn.Linear(context_dim + output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _encode(self, y: torch.Tensor, c: torch.Tensor):
        h = self.enc(torch.cat([y, c], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.dec(torch.cat([z, c], dim=-1))

    def forward(self, context: torch.Tensor, y: torch.Tensor | None = None):
        if y is not None:
            mu, logvar = self._encode(y, context)
            z = self._reparameterize(mu, logvar)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            return self._decode(z, context), kl
        z = torch.randn(context.size(0), self.latent_dim, device=context.device)
        return self._decode(z, context)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, n_samples: int = 100):
        batch = context.size(0)
        ctx = context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        z = torch.randn(batch * n_samples, self.latent_dim, device=context.device)
        out = self._decode(z, ctx)
        return out.view(batch, n_samples, self.output_dim)

    @torch.no_grad()
    def predict_interval(self, context: torch.Tensor, n_samples: int = 100, alpha: float = 0.80):
        samples = self.sample(context, n_samples=n_samples)
        lo = (1 - alpha) / 2
        pi_low = samples.quantile(lo, dim=1)
        pi_high = samples.quantile(1 - lo, dim=1)
        return samples.mean(dim=1), pi_low, pi_high
