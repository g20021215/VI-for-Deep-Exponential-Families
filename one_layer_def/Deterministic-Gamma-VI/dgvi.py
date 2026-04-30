"""
dgvi.py

Deterministic full-batch gradient-based variational inference for one-layer
Sparse Gamma DEF.

This file is intentionally compatible with the original notebook usage:

    from dgvi import OneLayerSparseGammaDEF_DeterministicGradientVI

    dgvi_model = OneLayerSparseGammaDEF_DeterministicGradientVI(
        N=X.shape[0],
        V=X.shape[1],
        K=K,
        alpha=0.3,
        beta=0.3,
        a_w=0.3,
        b_w=0.3,
        lr=LR_DGVI,
        max_iter=MAX_ITER_DGVI
    )

    dgvi_model.fit(X, verbose=True)

The public constructor parameters are kept consistent with the original code:
    N, V, K, alpha, beta, a_w, b_w, lr, max_iter, device, eps
"""

import time
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, sp.spmatrix]


class OneLayerSparseGammaDEF_DeterministicGradientVI(nn.Module):
    """
    Deterministic full-batch gradient-based VI for one-layer Sparse Gamma DEF.

    Model:
        z_nk ~ Gamma(alpha, beta)
        w_ik ~ Gamma(a_w, b_w)
        x_ni ~ Poisson(sum_k z_nk w_ik)

    Variational family:
        q(z_nk) = Gamma(a_z[n,k], b_z[n,k])
        q(w_ik) = Gamma(a_w_var[i,k], b_w_var[i,k])
    """

    def __init__(
        self,
        N,
        V,
        K,
        alpha=0.3,
        beta=0.3,
        a_w=0.3,
        b_w=0.3,
        lr=1e-2,
        max_iter=150,
        device=None,
        eps=1e-8,
    ):
        super().__init__()

        if N <= 0:
            raise ValueError("N must be positive.")
        if V <= 0:
            raise ValueError("V must be positive.")
        if K <= 0:
            raise ValueError("K must be positive.")

        self.N = int(N)
        self.V = int(V)
        self.K = int(K)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.a_w_prior = float(a_w)
        self.b_w_prior = float(b_w)

        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.eps = float(eps)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        init_shape = 0.5
        init_rate = 0.5

        raw_init_shape = np.log(np.exp(init_shape) - 1.0)
        raw_init_rate = np.log(np.exp(init_rate) - 1.0)

        self.raw_az = nn.Parameter(
            torch.full(
                (self.N, self.K),
                raw_init_shape,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.raw_bz = nn.Parameter(
            torch.full(
                (self.N, self.K),
                raw_init_rate,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.raw_aw = nn.Parameter(
            torch.full(
                (self.V, self.K),
                raw_init_shape,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.raw_bw = nn.Parameter(
            torch.full(
                (self.V, self.K),
                raw_init_rate,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.history_ = []
        self.iter_times_ = []
        self.cumulative_times_ = []

    def positive_params(self):
        az = F.softplus(self.raw_az) + self.eps
        bz = F.softplus(self.raw_bz) + self.eps
        aw = F.softplus(self.raw_aw) + self.eps
        bw = F.softplus(self.raw_bw) + self.eps
        return az, bz, aw, bw

    @staticmethod
    def gamma_expected_log_q(shape, rate):
        elogx = torch.digamma(shape) - torch.log(rate)
        ex = shape / rate

        return (
            (shape - 1.0) * elogx
            - rate * ex
            + shape * torch.log(rate)
            - torch.lgamma(shape)
        )

    def gamma_expected_log_prior(self, shape_q, rate_q, shape_prior, rate_prior):
        elogx = torch.digamma(shape_q) - torch.log(rate_q)
        ex = shape_q / rate_q

        shape_prior_t = torch.tensor(
            shape_prior,
            dtype=torch.float32,
            device=self.device,
        )

        rate_prior_t = torch.tensor(
            rate_prior,
            dtype=torch.float32,
            device=self.device,
        )

        return (
            (shape_prior_t - 1.0) * elogx
            - rate_prior_t * ex
            + shape_prior_t * torch.log(rate_prior_t)
            - torch.lgamma(shape_prior_t)
        )

    def full_objective(self, rows, cols, vals):
        az, bz, aw, bw = self.positive_params()

        Ez = az / bz
        Ew = aw / bw

        Elogz = torch.digamma(az) - torch.log(bz)
        Elogw = torch.digamma(aw) - torch.log(bw)

        prior_z = self.gamma_expected_log_prior(
            az,
            bz,
            self.alpha,
            self.beta,
        ).sum()

        prior_w = self.gamma_expected_log_prior(
            aw,
            bw,
            self.a_w_prior,
            self.b_w_prior,
        ).sum()

        entropy_z = -self.gamma_expected_log_q(az, bz).sum()
        entropy_w = -self.gamma_expected_log_q(aw, bw).sum()

        rate_term = -torch.sum(Ez.sum(dim=0) * Ew.sum(dim=0))

        if vals.numel() == 0:
            obs_term = torch.zeros((), dtype=torch.float32, device=self.device)
        else:
            e_log_z_n = Elogz[rows]
            e_log_w_i = Elogw[cols]

            logits = e_log_z_n + e_log_w_i
            r = torch.softmax(logits, dim=1)

            obs_term = torch.sum(
                vals[:, None] * r * (logits - torch.log(r + self.eps))
            )

        objective = prior_z + prior_w + entropy_z + entropy_w + rate_term + obs_term

        return objective

    def _to_csr(self, X: ArrayLike) -> sp.csr_matrix:
        if sp.issparse(X):
            X_csr = X.tocsr()
        else:
            X_csr = sp.csr_matrix(X)

        if X_csr.shape != (self.N, self.V):
            raise ValueError(
                f"X shape mismatch. Expected {(self.N, self.V)}, got {X_csr.shape}."
            )

        if X_csr.nnz > 0 and np.min(X_csr.data) < 0:
            raise ValueError("X must contain nonnegative count data.")

        return X_csr.astype(np.float32)

    def fit(self, X, verbose=True):
        X_csr = self._to_csr(X)

        rows, cols = X_csr.nonzero()
        vals = X_csr[rows, cols].A1.astype(np.float32)

        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.history_ = []
        self.iter_times_ = []
        self.cumulative_times_ = []

        start_time = time.time()

        for it in range(self.max_iter):
            iter_start = time.time()
            iter_display = it + 1

            optimizer.zero_grad()

            objective = self.full_objective(rows_t, cols_t, vals_t)

            loss = -objective
            loss.backward()
            optimizer.step()

            obj_value = objective.detach().cpu().item()
            self.history_.append(obj_value)

            iter_time = time.time() - iter_start
            self.iter_times_.append(iter_time)
            self.cumulative_times_.append(time.time() - start_time)

            if verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(
                    f"[Deterministic Gradient VI] Iter {iter_display:03d} | "
                    f"Objective proxy = {obj_value:.4f} | "
                    f"Iter time = {iter_time:.2f}s"
                )

        return self

    def get_variational_params(self):
        self.eval()
        with torch.no_grad():
            az, bz, aw, bw = self.positive_params()

        return {
            "az": az.detach().cpu().numpy(),
            "bz": bz.detach().cpu().numpy(),
            "aw": aw.detach().cpu().numpy(),
            "bw": bw.detach().cpu().numpy(),
        }

    def get_variational_means(self):
        self.eval()
        with torch.no_grad():
            az, bz, aw, bw = self.positive_params()
            Ez = az / bz
            Ew = aw / bw

        return {
            "Ez": Ez.detach().cpu().numpy(),
            "Ew": Ew.detach().cpu().numpy(),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    X_demo = sp.random(
        100,
        50,
        density=0.05,
        format="csr",
        random_state=123,
        data_rvs=lambda n: rng.poisson(2.0, size=n).astype(np.float32) + 1.0,
    )

    model = OneLayerSparseGammaDEF_DeterministicGradientVI(
        N=X_demo.shape[0],
        V=X_demo.shape[1],
        K=5,
        alpha=0.3,
        beta=0.3,
        a_w=0.3,
        b_w=0.3,
        lr=1e-2,
        max_iter=20,
    )

    start = time.time()
    model.fit(X_demo, verbose=True)
    total_time = time.time() - start

    print("Deterministic Gradient VI total runtime:", total_time)
    print("Deterministic Gradient VI final objective:", model.history_[-1])
