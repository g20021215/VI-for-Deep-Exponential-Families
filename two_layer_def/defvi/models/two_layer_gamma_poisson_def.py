from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    obs_dim: int
    latent_dim_1: int
    latent_dim_2: int
    alpha2_prior: float = 1.0
    beta2_prior: float = 1.0
    alpha1_prior: float = 1.5
    min_rate: float = 1e-5
    min_concentration: float = 1e-4


class TwoLayerGammaPoissonDEF(nn.Module):
    """Two-layer Gamma-Poisson DEF with global positive decoder weights.

    Layer 2 latent: z2
    Layer 1 latent: z1
    Observation: x
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim_1: int,
        latent_dim_2: int,
        alpha2_prior: float = 1.0,
        beta2_prior: float = 1.0,
        alpha1_prior: float = 1.5,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            obs_dim=obs_dim,
            latent_dim_1=latent_dim_1,
            latent_dim_2=latent_dim_2,
            alpha2_prior=alpha2_prior,
            beta2_prior=beta2_prior,
            alpha1_prior=alpha1_prior,
        )

        self.raw_w2 = nn.Parameter(torch.randn(latent_dim_1, latent_dim_2) * init_scale)
        self.raw_w1 = nn.Parameter(torch.randn(obs_dim, latent_dim_1) * init_scale)

    @property
    def device(self) -> torch.device:
        return self.raw_w1.device

    def positive_w1(self) -> torch.Tensor:
        return F.softplus(self.raw_w1) + 1e-6

    def positive_w2(self) -> torch.Tensor:
        return F.softplus(self.raw_w2) + 1e-6

    def sample_latents(
        self,
        q_alpha1: torch.Tensor,
        q_beta1: torch.Tensor,
        q_alpha2: torch.Tensor,
        q_beta2: torch.Tensor,
        n_samples_mc: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample reparameterized Gamma latent variables.

        Returns:
            z1: [mc, batch, latent_dim_1]
            z2: [mc, batch, latent_dim_2]
        """
        q_z2 = torch.distributions.Gamma(concentration=q_alpha2, rate=q_beta2)
        q_z1 = torch.distributions.Gamma(concentration=q_alpha1, rate=q_beta1)

        z2 = q_z2.rsample((n_samples_mc,))
        z1 = q_z1.rsample((n_samples_mc,))
        return z1, z2

    def log_joint(self, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute log p(x, z1, z2) for MC samples.

        Args:
            x: [batch, obs_dim]
            z1: [mc, batch, latent_dim_1]
            z2: [mc, batch, latent_dim_2]

        Returns:
            log_joint: [mc, batch]
        """
        cfg = self.config
        w1 = self.positive_w1()
        w2 = self.positive_w2()

        x_expanded = x.unsqueeze(0).expand(z1.shape[0], -1, -1)

        # p(z2)
        p_z2 = torch.distributions.Gamma(
            concentration=torch.full_like(z2, cfg.alpha2_prior),
            rate=torch.full_like(z2, cfg.beta2_prior),
        )
        log_p_z2 = p_z2.log_prob(z2).sum(dim=-1)

        # p(z1 | z2), parameterized so E[z1 | z2] = W2 z2
        z1_mean = torch.matmul(z2, w2.T).clamp_min(cfg.min_rate)
        z1_rate = (cfg.alpha1_prior / z1_mean).clamp_min(cfg.min_rate)
        p_z1 = torch.distributions.Gamma(
            concentration=torch.full_like(z1, cfg.alpha1_prior),
            rate=z1_rate,
        )
        log_p_z1_given_z2 = p_z1.log_prob(z1).sum(dim=-1)

        # p(x | z1)
        lam = torch.matmul(z1, w1.T).clamp_min(cfg.min_rate)
        p_x = torch.distributions.Poisson(rate=lam)
        log_p_x_given_z1 = p_x.log_prob(x_expanded).sum(dim=-1)

        return log_p_z2 + log_p_z1_given_z2 + log_p_x_given_z1

    @staticmethod
    def log_q(
        z1: torch.Tensor,
        z2: torch.Tensor,
        q_alpha1: torch.Tensor,
        q_beta1: torch.Tensor,
        q_alpha2: torch.Tensor,
        q_beta2: torch.Tensor,
    ) -> torch.Tensor:
        q_z1 = torch.distributions.Gamma(concentration=q_alpha1, rate=q_beta1)
        q_z2 = torch.distributions.Gamma(concentration=q_alpha2, rate=q_beta2)
        log_q_z1 = q_z1.log_prob(z1).sum(dim=-1)
        log_q_z2 = q_z2.log_prob(z2).sum(dim=-1)
        return log_q_z1 + log_q_z2

    def elbo_mc(
        self,
        x: torch.Tensor,
        q_alpha1: torch.Tensor,
        q_beta1: torch.Tensor,
        q_alpha2: torch.Tensor,
        q_beta2: torch.Tensor,
        n_samples_mc: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        z1, z2 = self.sample_latents(
            q_alpha1=q_alpha1,
            q_beta1=q_beta1,
            q_alpha2=q_alpha2,
            q_beta2=q_beta2,
            n_samples_mc=n_samples_mc,
        )
        log_joint = self.log_joint(x=x, z1=z1, z2=z2)
        log_q = self.log_q(
            z1=z1,
            z2=z2,
            q_alpha1=q_alpha1,
            q_beta1=q_beta1,
            q_alpha2=q_alpha2,
            q_beta2=q_beta2,
        )
        elbo_per_sample = (log_joint - log_q).mean(dim=0)
        elbo = elbo_per_sample.mean()

        diagnostics = {
            "avg_log_joint": float(log_joint.mean().detach().cpu()),
            "avg_log_q": float(log_q.mean().detach().cpu()),
            "avg_elbo": float(elbo.detach().cpu()),
        }
        return elbo, diagnostics

    def get_decoder_weights(self) -> Dict[str, torch.Tensor]:
        return {
            "w1": self.positive_w1().detach().cpu(),
            "w2": self.positive_w2().detach().cpu(),
        }
