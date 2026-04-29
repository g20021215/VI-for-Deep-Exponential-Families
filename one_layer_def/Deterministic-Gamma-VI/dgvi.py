"""
dgvi.py

Deterministic full-batch gradient-based variational inference for a one-layer
Sparse Gamma Deep Exponential Family.

Model:
    z_nk ~ Gamma(alpha, beta)
    w_vk ~ Gamma(a_w, b_w)
    x_nv ~ Poisson(sum_k z_nk w_vk)

Variational family:
    q(z_nk) = Gamma(a_z[n, k], b_z[n, k])
    q(w_vk) = Gamma(a_w_var[v, k], b_w_var[v, k])

The objective implemented here is an ELBO proxy using a deterministic
full-batch auxiliary-responsibility approximation for the expected
Poisson log-rate term.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, sp.spmatrix]


@dataclass(frozen=True)
class DGVIConfig:
    """Configuration for deterministic gradient-based VI."""

    K: int
    alpha: float = 0.3
    beta: float = 0.3
    a_w: float = 0.3
    b_w: float = 0.3
    lr: float = 1e-2
    max_iter: int = 150
    init_shape: float = 0.5
    init_rate: float = 0.5
    eps: float = 1e-8
    seed: Optional[int] = 123
    device: Optional[str] = None
    dtype: str = "float32"
    log_every: int = 10
    grad_clip_norm: Optional[float] = None


@dataclass
class DGVIResult:
    """Container returned by run_dgvi."""

    total_time: float
    final_objective: float
    history: np.ndarray
    iter_times: np.ndarray
    cumulative_times: np.ndarray
    config: Dict[str, object]
    device: str

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save optimization traces to a compressed npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            total_time=np.array(self.total_time, dtype=np.float64),
            final_objective=np.array(self.final_objective, dtype=np.float64),
            history=self.history,
            iter_times=self.iter_times,
            cumulative_times=self.cumulative_times,
            device=np.array(self.device),
            config_json=np.array(json.dumps(self.config, ensure_ascii=False)),
        )

    def save_csv(self, path: Union[str, Path]) -> None:
        """Save per-iteration traces to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack(
            [
                np.arange(1, len(self.history) + 1),
                self.history,
                self.iter_times,
                self.cumulative_times,
            ]
        )
        header = "iteration,objective,iter_time,cumulative_time"
        np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")


def set_global_seed(seed: Optional[int]) -> None:
    """Set random seeds for reproducible initialization and runs."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower().strip()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be either 'float32' or 'float64'.")


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device

    return "cuda" if torch.cuda.is_available() else "cpu"


def _inverse_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("Softplus initialization value must be positive.")

    return math.log(math.expm1(x))


def _to_csr_float_matrix(X: ArrayLike) -> sp.csr_matrix:
    """Convert dense or sparse input to CSR float32 matrix."""
    if sp.issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = sp.csr_matrix(np.asarray(X))

    if X_csr.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    if X_csr.shape[0] == 0 or X_csr.shape[1] == 0:
        raise ValueError("X must have positive number of rows and columns.")

    if X_csr.nnz > 0 and X_csr.data.min() < 0:
        raise ValueError("X must contain nonnegative count data.")

    return X_csr.astype(np.float32)


class OneLayerSparseGammaDEFDeterministicGradientVI(nn.Module):
    """
    Deterministic full-batch gradient-based VI for one-layer Sparse Gamma DEF.

    This class is designed as a reusable GitHub module. The recommended public
    entry point is run_dgvi, but the class can also be used directly.
    """

    def __init__(
        self,
        N: int,
        V: int,
        config: DGVIConfig,
    ) -> None:
        super().__init__()

        if N <= 0:
            raise ValueError("N must be positive.")
        if V <= 0:
            raise ValueError("V must be positive.")
        if config.K <= 0:
            raise ValueError("K must be positive.")

        self.N = int(N)
        self.V = int(V)
        self.K = int(config.K)
        self.config = config

        self.alpha = float(config.alpha)
        self.beta = float(config.beta)
        self.a_w_prior = float(config.a_w)
        self.b_w_prior = float(config.b_w)
        self.lr = float(config.lr)
        self.max_iter = int(config.max_iter)
        self.eps = float(config.eps)

        self.device_name = _resolve_device(config.device)
        self.torch_dtype = _resolve_dtype(config.dtype)

        raw_init_shape = _inverse_softplus(config.init_shape)
        raw_init_rate = _inverse_softplus(config.init_rate)

        self.raw_az = nn.Parameter(
            torch.full(
                (self.N, self.K),
                raw_init_shape,
                dtype=self.torch_dtype,
                device=self.device_name,
            )
        )
        self.raw_bz = nn.Parameter(
            torch.full(
                (self.N, self.K),
                raw_init_rate,
                dtype=self.torch_dtype,
                device=self.device_name,
            )
        )
        self.raw_aw = nn.Parameter(
            torch.full(
                (self.V, self.K),
                raw_init_shape,
                dtype=self.torch_dtype,
                device=self.device_name,
            )
        )
        self.raw_bw = nn.Parameter(
            torch.full(
                (self.V, self.K),
                raw_init_rate,
                dtype=self.torch_dtype,
                device=self.device_name,
            )
        )

        self.history_: list[float] = []
        self.iter_times_: list[float] = []
        self.cumulative_times_: list[float] = []

    def positive_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return positive Gamma variational parameters."""
        az = F.softplus(self.raw_az) + self.eps
        bz = F.softplus(self.raw_bz) + self.eps
        aw = F.softplus(self.raw_aw) + self.eps
        bw = F.softplus(self.raw_bw) + self.eps
        return az, bz, aw, bw

    @staticmethod
    def gamma_expected_log_q(shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """
        Compute E_q[log q(X)] for Gamma(shape, rate).

        Gamma density:
            log q(x) = (a - 1) log x - b x + a log b - log Gamma(a)
        """
        elogx = torch.digamma(shape) - torch.log(rate)
        ex = shape / rate
        return (shape - 1.0) * elogx - rate * ex + shape * torch.log(rate) - torch.lgamma(shape)

    def gamma_expected_log_prior(
        self,
        shape_q: torch.Tensor,
        rate_q: torch.Tensor,
        shape_prior: float,
        rate_prior: float,
    ) -> torch.Tensor:
        """Compute E_q[log p(X)] where p is Gamma(shape_prior, rate_prior)."""
        elogx = torch.digamma(shape_q) - torch.log(rate_q)
        ex = shape_q / rate_q

        shape_prior_t = torch.as_tensor(
            shape_prior,
            dtype=self.torch_dtype,
            device=self.device_name,
        )
        rate_prior_t = torch.as_tensor(
            rate_prior,
            dtype=self.torch_dtype,
            device=self.device_name,
        )

        return (
            (shape_prior_t - 1.0) * elogx
            - rate_prior_t * ex
            + shape_prior_t * torch.log(rate_prior_t)
            - torch.lgamma(shape_prior_t)
        )

    def full_objective(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute deterministic full-batch ELBO proxy.

        The Poisson log-factorial constant is omitted because it does not affect
        gradient optimization or convergence comparison.
        """
        az, bz, aw, bw = self.positive_params()

        ez = az / bz
        ew = aw / bw

        elogz = torch.digamma(az) - torch.log(bz)
        elogw = torch.digamma(aw) - torch.log(bw)

        prior_z = self.gamma_expected_log_prior(az, bz, self.alpha, self.beta).sum()
        prior_w = self.gamma_expected_log_prior(aw, bw, self.a_w_prior, self.b_w_prior).sum()

        entropy_z = -self.gamma_expected_log_q(az, bz).sum()
        entropy_w = -self.gamma_expected_log_q(aw, bw).sum()

        rate_term = -torch.sum(ez.sum(dim=0) * ew.sum(dim=0))

        if vals.numel() == 0:
            obs_term = torch.zeros((), dtype=self.torch_dtype, device=self.device_name)
        else:
            logits = elogz[rows] + elogw[cols]
            responsibilities = torch.softmax(logits, dim=1)
            obs_term = torch.sum(
                vals[:, None]
                * responsibilities
                * (logits - torch.log(responsibilities + self.eps))
            )

        return prior_z + prior_w + entropy_z + entropy_w + rate_term + obs_term

    def _prepare_sparse_tensors(
        self,
        X: ArrayLike,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_csr = _to_csr_float_matrix(X)

        if X_csr.shape != (self.N, self.V):
            raise ValueError(
                f"X shape mismatch. Expected {(self.N, self.V)}, got {X_csr.shape}."
            )

        rows, cols = X_csr.nonzero()
        vals = np.asarray(X_csr[rows, cols]).reshape(-1).astype(np.float32)

        rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.device_name)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=self.device_name)
        vals_t = torch.as_tensor(vals, dtype=self.torch_dtype, device=self.device_name)

        return rows_t, cols_t, vals_t

    def fit(self, X: ArrayLike, verbose: bool = True) -> "OneLayerSparseGammaDEFDeterministicGradientVI":
        """Fit the model on a dense or sparse count matrix X."""
        rows_t, cols_t, vals_t = self._prepare_sparse_tensors(X)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.history_.clear()
        self.iter_times_.clear()
        self.cumulative_times_.clear()

        start_time = time.perf_counter()

        for it in range(self.max_iter):
            iter_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            objective = self.full_objective(rows_t, cols_t, vals_t)
            loss = -objective
            loss.backward()

            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip_norm)

            optimizer.step()

            obj_value = float(objective.detach().cpu().item())
            iter_time = time.perf_counter() - iter_start
            cum_time = time.perf_counter() - start_time

            self.history_.append(obj_value)
            self.iter_times_.append(iter_time)
            self.cumulative_times_.append(cum_time)

            should_log = (
                verbose
                and self.config.log_every > 0
                and ((it + 1) == 1 or (it + 1) % self.config.log_every == 0 or (it + 1) == self.max_iter)
            )

            if should_log:
                print(
                    f"[DGVI] Iter {it + 1:04d}/{self.max_iter} | "
                    f"Objective = {obj_value:.6f} | "
                    f"Iter time = {iter_time:.3f}s | "
                    f"Cum time = {cum_time:.3f}s"
                )

        return self

    @torch.no_grad()
    def get_variational_params(self) -> Dict[str, np.ndarray]:
        """Return variational Gamma parameters as NumPy arrays."""
        az, bz, aw, bw = self.positive_params()
        return {
            "az": az.detach().cpu().numpy(),
            "bz": bz.detach().cpu().numpy(),
            "aw": aw.detach().cpu().numpy(),
            "bw": bw.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def get_variational_means(self) -> Dict[str, np.ndarray]:
        """Return E_q[z] and E_q[w] as NumPy arrays."""
        az, bz, aw, bw = self.positive_params()
        ez = az / bz
        ew = aw / bw
        return {
            "Ez": ez.detach().cpu().numpy(),
            "Ew": ew.detach().cpu().numpy(),
        }

    def result(self, total_time: Optional[float] = None) -> DGVIResult:
        """Build a DGVIResult object from the fitted model traces."""
        if len(self.history_) == 0:
            raise RuntimeError("No optimization history found. Call fit(X) first.")

        if total_time is None:
            total_time = float(self.cumulative_times_[-1])

        return DGVIResult(
            total_time=float(total_time),
            final_objective=float(self.history_[-1]),
            history=np.asarray(self.history_, dtype=np.float64),
            iter_times=np.asarray(self.iter_times_, dtype=np.float64),
            cumulative_times=np.asarray(self.cumulative_times_, dtype=np.float64),
            config=asdict(self.config),
            device=self.device_name,
        )


def run_dgvi(
    X: ArrayLike,
    K: int,
    alpha: float = 0.3,
    beta: float = 0.3,
    a_w: float = 0.3,
    b_w: float = 0.3,
    lr: float = 1e-2,
    max_iter: int = 150,
    init_shape: float = 0.5,
    init_rate: float = 0.5,
    eps: float = 1e-8,
    seed: Optional[int] = 123,
    device: Optional[str] = None,
    dtype: str = "float32",
    log_every: int = 10,
    grad_clip_norm: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[OneLayerSparseGammaDEFDeterministicGradientVI, DGVIResult]:
    """
    One-call runner for deterministic full-batch gradient VI.

    Parameters
    ----------
    X:
        Dense NumPy array or SciPy sparse matrix with nonnegative counts.
    K:
        Number of latent factors.
    verbose:
        Whether to print optimization progress.

    Returns
    -------
    model:
        Fitted OneLayerSparseGammaDEFDeterministicGradientVI model.
    result:
        DGVIResult containing runtime and optimization traces.
    """
    X_csr = _to_csr_float_matrix(X)
    N, V = X_csr.shape

    config = DGVIConfig(
        K=K,
        alpha=alpha,
        beta=beta,
        a_w=a_w,
        b_w=b_w,
        lr=lr,
        max_iter=max_iter,
        init_shape=init_shape,
        init_rate=init_rate,
        eps=eps,
        seed=seed,
        device=device,
        dtype=dtype,
        log_every=log_every,
        grad_clip_norm=grad_clip_norm,
    )

    set_global_seed(seed)

    model = OneLayerSparseGammaDEFDeterministicGradientVI(N=N, V=V, config=config)

    start = time.perf_counter()
    model.fit(X_csr, verbose=verbose)
    total_time = time.perf_counter() - start

    result = model.result(total_time=total_time)

    return model, result


# Backward-compatible alias for your original class name.
OneLayerSparseGammaDEF_DeterministicGradientVI = OneLayerSparseGammaDEFDeterministicGradientVI


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

    model_demo, result_demo = run_dgvi(
        X_demo,
        K=5,
        lr=1e-2,
        max_iter=20,
        seed=123,
        verbose=True,
    )

    print("DGVI total runtime:", result_demo.total_time)
    print("DGVI final objective:", result_demo.final_objective)

