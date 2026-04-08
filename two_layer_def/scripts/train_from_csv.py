from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from defvi.inference import fit_two_layer_def
from defvi.utils import load_count_csv, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-layer Gamma-Poisson DEF on a CSV count matrix.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--latent_dim_1", type=int, default=32)
    parser.add_argument("--latent_dim_2", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr_model", type=float, default=5e-3)
    parser.add_argument("--lr_variational", type=float, default=1e-2)
    parser.add_argument("--n_samples_mc", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs/csv_run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    x, df = load_count_csv(args.csv_path, device=args.device)
    print(f"Loaded CSV with shape: {df.shape}")

    result = fit_two_layer_def(
        x=x,
        latent_dim_1=args.latent_dim_1,
        latent_dim_2=args.latent_dim_2,
        epochs=args.epochs,
        lr_model=args.lr_model,
        lr_variational=args.lr_variational,
        n_samples_mc=args.n_samples_mc,
        save_dir=args.save_dir,
        device=args.device,
    )

    print("Training finished.")
    print(result["metrics"])


if __name__ == "__main__":
    main()
