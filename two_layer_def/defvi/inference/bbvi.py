from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from defvi.models import TwoLayerGammaPoissonDEF


@dataclass
class TrainingConfig:
    epochs: int = 300
    lr_model: float = 5e-3
    lr_variational: float = 1e-2
    n_samples_mc: int = 3
    weight_decay: float = 1e-6
    log_every: int = 25
    device: str = "cpu"
    save_dir: str = "outputs/run"


class VariationalParameters(nn.Module):
    """Per-sample mean-field Gamma parameters for z1 and z2."""

    def __init__(self, n_samples: int, latent_dim_1: int, latent_dim_2: int, init_conc: float = 1.0) -> None:
        super().__init__()
        # raw -> softplus to keep positivity
        self.raw_alpha1 = nn.Parameter(torch.full((n_samples, latent_dim_1), init_conc))
        self.raw_beta1 = nn.Parameter(torch.full((n_samples, latent_dim_1), init_conc))
        self.raw_alpha2 = nn.Parameter(torch.full((n_samples, latent_dim_2), init_conc))
        self.raw_beta2 = nn.Parameter(torch.full((n_samples, latent_dim_2), init_conc))

    def positive_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "q_alpha1": torch.nn.functional.softplus(self.raw_alpha1) + 1e-4,
            "q_beta1": torch.nn.functional.softplus(self.raw_beta1) + 1e-4,
            "q_alpha2": torch.nn.functional.softplus(self.raw_alpha2) + 1e-4,
            "q_beta2": torch.nn.functional.softplus(self.raw_beta2) + 1e-4,
        }

    @torch.no_grad()
    def posterior_means(self) -> Dict[str, torch.Tensor]:
        params = self.positive_parameters()
        return {
            "z1_mean": (params["q_alpha1"] / params["q_beta1"]).detach().cpu(),
            "z2_mean": (params["q_alpha2"] / params["q_beta2"]).detach().cpu(),
        }


@torch.no_grad()
def reconstruct_rate(model: TwoLayerGammaPoissonDEF, var_params: VariationalParameters) -> torch.Tensor:
    means = var_params.posterior_means()
    z1_mean = means["z1_mean"].to(model.device)
    lam = z1_mean @ model.get_decoder_weights()["w1"].to(model.device).T
    return lam.cpu()


def fit_two_layer_def(
    x: torch.Tensor,
    latent_dim_1: int,
    latent_dim_2: int,
    epochs: int = 300,
    lr_model: float = 5e-3,
    lr_variational: float = 1e-2,
    n_samples_mc: int = 3,
    weight_decay: float = 1e-6,
    log_every: int = 25,
    save_dir: str = "outputs/run",
    device: str = "cpu",
    alpha2_prior: float = 1.0,
    beta2_prior: float = 1.0,
    alpha1_prior: float = 1.5,
) -> Dict[str, object]:
    x = x.to(device)
    n_samples, obs_dim = x.shape
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = TwoLayerGammaPoissonDEF(
        obs_dim=obs_dim,
        latent_dim_1=latent_dim_1,
        latent_dim_2=latent_dim_2,
        alpha2_prior=alpha2_prior,
        beta2_prior=beta2_prior,
        alpha1_prior=alpha1_prior,
    ).to(device)

    var_params = VariationalParameters(
        n_samples=n_samples,
        latent_dim_1=latent_dim_1,
        latent_dim_2=latent_dim_2,
    ).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr_model, weight_decay=weight_decay)
    opt_var = torch.optim.Adam(var_params.parameters(), lr=lr_variational, weight_decay=weight_decay)

    elbo_history = []
    best_elbo = float("-inf")
    best_state: Optional[Dict[str, object]] = None

    for epoch in range(1, epochs + 1):
        opt_model.zero_grad()
        opt_var.zero_grad()

        params = var_params.positive_parameters()
        elbo, diagnostics = model.elbo_mc(
            x=x,
            q_alpha1=params["q_alpha1"],
            q_beta1=params["q_beta1"],
            q_alpha2=params["q_alpha2"],
            q_beta2=params["q_beta2"],
            n_samples_mc=n_samples_mc,
        )

        loss = -elbo
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(var_params.parameters(), max_norm=10.0)

        opt_model.step()
        opt_var.step()

        current_elbo = float(elbo.detach().cpu())
        elbo_history.append(current_elbo)

        if current_elbo > best_elbo:
            best_elbo = current_elbo
            best_state = {
                "model": model.state_dict(),
                "var_params": var_params.state_dict(),
                "epoch": epoch,
                "elbo": best_elbo,
            }

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[Epoch {epoch:04d}] ELBO={current_elbo:.4f} | "
                f"log_joint={diagnostics['avg_log_joint']:.4f} | "
                f"log_q={diagnostics['avg_log_q']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        var_params.load_state_dict(best_state["var_params"])

    # Save training artifacts
    torch.save(model.state_dict(), save_path / "model.pt")
    torch.save(var_params.state_dict(), save_path / "variational_params.pt")

    posterior_means = var_params.posterior_means()
    torch.save(posterior_means, save_path / "posterior_means.pt")
    torch.save(model.get_decoder_weights(), save_path / "decoder_weights.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(elbo_history)
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("Training ELBO")
    plt.tight_layout()
    plt.savefig(save_path / "elbo_curve.png", dpi=160)
    plt.close()

    config = TrainingConfig(
        epochs=epochs,
        lr_model=lr_model,
        lr_variational=lr_variational,
        n_samples_mc=n_samples_mc,
        weight_decay=weight_decay,
        log_every=log_every,
        device=device,
        save_dir=str(save_path),
    )
    with open(save_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    metrics = {
        "best_elbo": best_elbo,
        "final_elbo": elbo_history[-1],
        "best_epoch": best_state["epoch"] if best_state is not None else None,
        "n_samples": int(n_samples),
        "obs_dim": int(obs_dim),
        "latent_dim_1": int(latent_dim_1),
        "latent_dim_2": int(latent_dim_2),
    }
    with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model": model,
        "variational_parameters": var_params,
        "elbo_history": elbo_history,
        "metrics": metrics,
        "posterior_means": posterior_means,
    }
