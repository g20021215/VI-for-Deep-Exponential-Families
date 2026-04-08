# def_gamma_poisson.py
# Comments in English as requested.

import os
import math
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_vocab(vocab_path: str) -> List[str]:
    vocab = []
    with open(vocab_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vocab.append(line.strip())
    return vocab


def read_cpp_dat(file_path: str, vocab_size: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    """
    Read Blei-style sparse document file.

    Each line is usually:
    <num_unique_terms> term_id:count term_id:count ...

    Output:
        X: [num_docs, vocab_size] dense matrix
    """
    rows = []
    max_term_id = -1

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                rows.append({})
                continue

            parts = line.split()
            term_dict = {}

            for item in parts[1:]:
                if ":" not in item or item == "":
                    print("BAD:", item)
                    continue
                term_id_str, cnt_str = item.split(":")
                term_id = int(term_id_str)
                cnt = float(cnt_str)
                term_dict[term_id] = cnt
                if term_id > max_term_id:
                    max_term_id = term_id

            rows.append(term_dict)

    if vocab_size is None:
        vocab_size = max_term_id + 1

    X = np.zeros((len(rows), vocab_size), dtype=dtype)
    for d, term_dict in enumerate(rows):
        for term_id, cnt in term_dict.items():
            if 0 <= term_id < vocab_size:
                X[d, term_id] = cnt

    return X


class GammaPoissonDEF(nn.Module):
    """
    A simplified one-layer Gamma-Poisson DEF.

    Generative view:
        theta_d ~ Gamma(a_theta, b_theta)
        beta_kv ~ Gamma(a_beta, b_beta)
        x_dv ~ Poisson(sum_k theta_dk * beta_kv)

    Optimization view here:
        We directly optimize positive latent parameters via softplus transforms.
    """

    def __init__(
        self,
        num_docs: int,
        vocab_size: int,
        num_topics: int,
        init_scale: float = 0.02,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_docs = num_docs
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.device = device

        # Unconstrained parameters for document-topic matrix theta and topic-word matrix beta
        self.theta_unconstrained = nn.Parameter(
            init_scale * torch.randn(num_docs, num_topics, device=device)
        )
        self.beta_unconstrained = nn.Parameter(
            init_scale * torch.randn(num_topics, vocab_size, device=device)
        )

    def theta(self) -> torch.Tensor:
        return F.softplus(self.theta_unconstrained) + 1e-6

    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_unconstrained) + 1e-6

    def rate(self) -> torch.Tensor:
        # [D, K] @ [K, V] -> [D, V]
        return self.theta() @ self.beta()

    def poisson_nll(self, X: torch.Tensor) -> torch.Tensor:
        """
        Negative log likelihood up to a constant:
            sum(lambda - x * log lambda)
        """
        lam = self.rate().clamp_min(1e-8)
        return torch.sum(lam - X * torch.log(lam))

    def gamma_like_penalty(
        self,
        a_theta: float = 0.3,
        b_theta: float = 0.3,
        a_beta: float = 0.3,
        b_beta: float = 0.3,
    ) -> torch.Tensor:
        """
        Negative log prior up to constants for Gamma priors.

        Gamma(shape=a, rate=b):
            log p(z) = (a-1)log z - b z + const

        So negative log prior up to constants:
            -(a-1)log z + b z
        """
        theta = self.theta().clamp_min(1e-8)
        beta = self.beta().clamp_min(1e-8)

        theta_penalty = torch.sum(-(a_theta - 1.0) * torch.log(theta) + b_theta * theta)
        beta_penalty = torch.sum(-(a_beta - 1.0) * torch.log(beta) + b_beta * beta)

        return theta_penalty + beta_penalty

    def objective(
        self,
        X: torch.Tensor,
        a_theta: float = 0.3,
        b_theta: float = 0.3,
        a_beta: float = 0.3,
        b_beta: float = 0.3,
        prior_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = self.poisson_nll(X)
        prior = self.gamma_like_penalty(a_theta, b_theta, a_beta, b_beta)
        loss = recon + prior_weight * prior
        return loss, recon, prior


def top_words_from_beta(beta: np.ndarray, vocab: List[str], top_k: int = 20) -> List[List[Tuple[str, float]]]:
    topics = []
    for k in range(beta.shape[0]):
        idx = np.argsort(beta[k])[::-1][:top_k]
        topics.append([(vocab[i], float(beta[k, i])) for i in idx])
    return topics


def save_top_words(beta: np.ndarray, vocab: List[str], output_path: str, top_k: int = 20) -> None:
    topics = top_words_from_beta(beta, vocab, top_k)
    with open(output_path, "w", encoding="utf-8") as f:
        for k, topic_words in enumerate(topics):
            f.write(f"Topic {k}\n")
            for word, weight in topic_words:
                f.write(f"{word}\t{weight:.6f}\n")
            f.write("\n")


def compute_perplexity(X: np.ndarray, rate: np.ndarray) -> float:
    rate = np.clip(rate, 1e-8, None)
    nll = np.sum(rate - X * np.log(rate))
    token_count = np.sum(X)
    if token_count <= 0:
        return float("nan")
    return float(np.exp(nll / token_count))


def train_model(
    train_X: np.ndarray,
    valid_X: Optional[np.ndarray],
    test_X: Optional[np.ndarray],
    vocab: List[str],
    num_topics: int,
    output_dir: str,
    epochs: int = 500,
    lr: float = 0.03,
    a_theta: float = 0.3,
    b_theta: float = 0.3,
    a_beta: float = 0.3,
    b_beta: float = 0.3,
    prior_weight: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    set_seed(seed)
    ensure_dir(output_dir)

    train_tensor = torch.tensor(train_X, dtype=torch.float32, device=device)
    D, V = train_X.shape

    model = GammaPoissonDEF(
        num_docs=D,
        vocab_size=V,
        num_topics=num_topics,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss, recon, prior = model.objective(
            train_tensor,
            a_theta=a_theta,
            b_theta=b_theta,
            a_beta=a_beta,
            b_beta=b_beta,
            prior_weight=prior_weight,
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_rate = model.rate().detach().cpu().numpy()
            train_perp = compute_perplexity(train_X, train_rate)

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "recon": float(recon.item()),
            "prior": float(prior.item()),
            "train_perplexity": train_perp,
        }

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_state = {
                "theta_unconstrained": model.theta_unconstrained.detach().cpu().clone(),
                "beta_unconstrained": model.beta_unconstrained.detach().cpu().clone(),
            }

        history.append(row)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[Epoch {epoch:4d}] "
                f"loss={row['loss']:.4f} "
                f"recon={row['recon']:.4f} "
                f"prior={row['prior']:.4f} "
                f"train_perp={row['train_perplexity']:.4f}"
            )

    if best_state is not None:
        model.theta_unconstrained.data.copy_(best_state["theta_unconstrained"].to(device))
        model.beta_unconstrained.data.copy_(best_state["beta_unconstrained"].to(device))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)

    with torch.no_grad():
        theta_train = model.theta().detach().cpu()
        beta = model.beta().detach().cpu()
        train_rate = (theta_train @ beta).numpy()

    torch.save(theta_train, os.path.join(output_dir, "theta_train.pt"))
    torch.save(beta, os.path.join(output_dir, "beta.pt"))

    train_perp = compute_perplexity(train_X, train_rate)

    metrics = {
        "train_perplexity": train_perp,
        "num_topics": num_topics,
        "epochs": epochs,
        "lr": lr,
        "a_theta": a_theta,
        "b_theta": b_theta,
        "a_beta": a_beta,
        "b_beta": b_beta,
        "prior_weight": prior_weight,
    }

    # Simple fold-in for valid/test:
    # We optimize theta for new docs while fixing beta.
    if valid_X is not None:
        theta_valid, valid_rate, valid_perp = infer_theta_for_new_docs(
            X=valid_X,
            beta=beta.numpy(),
            epochs=300,
            lr=0.05,
            a_theta=a_theta,
            b_theta=b_theta,
            device=device,
        )
        torch.save(torch.tensor(theta_valid), os.path.join(output_dir, "theta_valid.pt"))
        metrics["valid_perplexity"] = valid_perp

    if test_X is not None:
        theta_test, test_rate, test_perp = infer_theta_for_new_docs(
            X=test_X,
            beta=beta.numpy(),
            epochs=300,
            lr=0.05,
            a_theta=a_theta,
            b_theta=b_theta,
            device=device,
        )
        torch.save(torch.tensor(theta_test), os.path.join(output_dir, "theta_test.pt"))
        metrics["test_perplexity"] = test_perp

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_top_words(
        beta=beta.numpy(),
        vocab=vocab,
        output_path=os.path.join(output_dir, "top_words.txt"),
        top_k=20,
    )

    print("\nTraining finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def infer_theta_for_new_docs(
    X: np.ndarray,
    beta: np.ndarray,
    epochs: int = 300,
    lr: float = 0.05,
    a_theta: float = 0.3,
    b_theta: float = 0.3,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Infer theta for new documents with fixed beta.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    beta_tensor = torch.tensor(beta, dtype=torch.float32, device=device)

    D, V = X.shape
    K = beta.shape[0]

    theta_unconstrained = nn.Parameter(0.02 * torch.randn(D, K, device=device))
    optimizer = torch.optim.Adam([theta_unconstrained], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        theta = F.softplus(theta_unconstrained) + 1e-6
        lam = (theta @ beta_tensor).clamp_min(1e-8)

        recon = torch.sum(lam - X_tensor * torch.log(lam))
        prior = torch.sum(-(a_theta - 1.0) * torch.log(theta) + b_theta * theta)
        loss = recon + prior

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        theta = (F.softplus(theta_unconstrained) + 1e-6).cpu().numpy()
        rate = theta @ beta
        perp = compute_perplexity(X, rate)

    return theta, rate, perp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)

    parser.add_argument("--a_theta", type=float, default=0.3)
    parser.add_argument("--b_theta", type=float, default=0.3)
    parser.add_argument("--a_beta", type=float, default=0.3)
    parser.add_argument("--b_beta", type=float, default=0.3)
    parser.add_argument("--prior_weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    vocab = read_vocab(args.vocab_path)
    vocab_size = len(vocab)

    print("Loading data...")
    train_X = read_cpp_dat(args.train_path, vocab_size=vocab_size)
    valid_X = read_cpp_dat(args.valid_path, vocab_size=vocab_size) if args.valid_path else None
    test_X = read_cpp_dat(args.test_path, vocab_size=vocab_size) if args.test_path else None

    print(f"Train shape: {train_X.shape}")
    if valid_X is not None:
        print(f"Valid shape: {valid_X.shape}")
    if test_X is not None:
        print(f"Test shape: {test_X.shape}")

    train_model(
        train_X=train_X,
        valid_X=valid_X,
        test_X=test_X,
        vocab=vocab,
        num_topics=args.num_topics,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        a_theta=args.a_theta,
        b_theta=args.b_theta,
        a_beta=args.a_beta,
        b_beta=args.b_beta,
        prior_weight=args.prior_weight,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
