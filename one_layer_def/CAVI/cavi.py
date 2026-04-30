"""
cavi.py

CAVI implementation for one-layer Sparse Gamma DEF.

This file is a pure importable module. It does not run experiments by itself.
"""

import time
import numpy as np
from scipy.special import digamma, gammaln


class OneLayerSparseGammaDEF_CAVI_SparseTimed:
    def __init__(
        self,
        K=50,
        alpha=0.3,
        beta=0.3,
        a_w=0.3,
        b_w=0.3,
        max_iter=100,
        tol=1e-5,
        random_state=123
    ):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.a_w = a_w
        self.b_w = b_w
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.history_ = []
        self.iter_times_ = []
        self.cumulative_times_ = []

    def _initialize(self, X):
        rng = np.random.default_rng(self.random_state)
        N, V = X.shape
        K = self.K

        self.z_shape = self.alpha + rng.random((N, K))
        self.z_rate = self.beta + rng.random((N, K))
        self.w_shape = self.a_w + rng.random((V, K))
        self.w_rate = self.b_w + rng.random((V, K))

    def _expectations(self):
        Ez = self.z_shape / self.z_rate
        Ew = self.w_shape / self.w_rate
        Elogz = digamma(self.z_shape) - np.log(self.z_rate)
        Elogw = digamma(self.w_shape) - np.log(self.w_rate)
        return Ez, Ew, Elogz, Elogw

    def _compute_objective_proxy(self, X, Ez, Ew):
        Lambda = Ez @ Ew.T
        Lambda = np.maximum(Lambda, 1e-12)
        return np.sum(X * np.log(Lambda) - Lambda - gammaln(X + 1))

    def fit(self, X, verbose=True):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=float)
        N, V = X.shape
        K = self.K

        self._initialize(X)

        self.history_ = []
        self.iter_times_ = []
        self.cumulative_times_ = []

        prev_obj = -np.inf
        total_start = time.time()

        for it in range(self.max_iter):
            iter_start = time.time()

            Ez, Ew, Elogz, Elogw = self._expectations()

            z_shape_new = np.full((N, K), self.alpha, dtype=float)
            w_shape_add = np.zeros((V, K), dtype=float)
            z_rate_new = self.beta + Ew.sum(axis=0)[None, :]

            for n in range(N):
                row = X[n]
                nz_idx = np.where(row > 0)[0]

                if len(nz_idx) == 0:
                    continue

                counts = row[nz_idx]

                log_r = Elogz[n][None, :] + Elogw[nz_idx, :]
                log_r = log_r - np.max(log_r, axis=1, keepdims=True)

                r = np.exp(log_r)
                r /= np.sum(r, axis=1, keepdims=True)

                xr = counts[:, None] * r

                z_shape_new[n, :] += xr.sum(axis=0)
                w_shape_add[nz_idx, :] += xr

            self.z_shape = z_shape_new
            self.z_rate = np.broadcast_to(z_rate_new, (N, K)).copy()

            Ez, Ew, Elogz, Elogw = self._expectations()

            self.w_shape = self.a_w + w_shape_add
            self.w_rate = self.b_w + Ez.sum(axis=0)[None, :]
            self.w_rate = np.broadcast_to(self.w_rate, (V, K)).copy()

            Ez, Ew, Elogz, Elogw = self._expectations()

            obj = self._compute_objective_proxy(X, Ez, Ew)
            self.history_.append(obj)

            iter_time = time.time() - iter_start
            self.iter_times_.append(iter_time)
            self.cumulative_times_.append(time.time() - total_start)

            if verbose:
                print(
                    f"[Sparse CAVI Iter {it + 1:4d}] "
                    f"objective_proxy={obj:.4f} "
                    f"iter_time={iter_time:.4f}s"
                )

            if it > 0:
                improvement = abs(obj - prev_obj)

                if improvement < self.tol:
                    if verbose:
                        print(f"Sparse CAVI converged at iteration {it + 1}")
                    break

            prev_obj = obj

        return self


def safe_last(arr, default=np.nan):
    arr = np.asarray(arr)

    if arr.size == 0:
        return default

    return arr[-1]


def find_convergence_iter(history, rel_tol=1e-4, patience=3):
    history = np.asarray(history, dtype=float)

    if history.size < patience + 1:
        return None

    stable_count = 0

    for i in range(1, history.size):
        prev = history[i - 1]
        curr = history[i]

        denom = max(abs(prev), 1e-12)
        rel_change = abs(curr - prev) / denom

        if rel_change < rel_tol:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= patience:
            return i + 1

    return None


if __name__ == "__main__":
    pass
