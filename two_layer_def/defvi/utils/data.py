from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def load_count_csv(csv_path: str | Path, device: str = "cpu") -> Tuple[torch.Tensor, pd.DataFrame]:
    """Load a count matrix from CSV.

    Rows are samples. Columns are observed dimensions.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] != df.shape[1]:
        dropped = set(df.columns) - set(numeric_df.columns)
        raise ValueError(
            f"CSV contains non-numeric columns that were not allowed: {sorted(dropped)}"
        )

    values = numeric_df.to_numpy(dtype=np.float32)

    if np.any(values < 0):
        raise ValueError("All counts must be nonnegative.")

    x = torch.tensor(values, dtype=torch.float32, device=device)
    return x, numeric_df


def make_synthetic_counts(
    n_samples: int = 500,
    obs_dim: int = 64,
    latent_dim_1: int = 16,
    latent_dim_2: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a synthetic count matrix from a two-layer Gamma-Poisson DEF."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    w2 = torch.rand(latent_dim_1, latent_dim_2, generator=g, device=device) * 0.8 + 0.2
    w1 = torch.rand(obs_dim, latent_dim_1, generator=g, device=device) * 0.8 + 0.2

    z2 = torch.distributions.Gamma(
        concentration=torch.full((n_samples, latent_dim_2), 1.2, device=device),
        rate=torch.full((n_samples, latent_dim_2), 1.0, device=device),
    ).sample()

    z1_mean = z2 @ w2.T
    z1_rate = 1.5 / (z1_mean + 1e-6)
    z1 = torch.distributions.Gamma(
        concentration=torch.full((n_samples, latent_dim_1), 1.5, device=device),
        rate=z1_rate,
    ).sample()

    lam = z1 @ w1.T + 1e-4
    x = torch.distributions.Poisson(rate=lam).sample()
    return x
