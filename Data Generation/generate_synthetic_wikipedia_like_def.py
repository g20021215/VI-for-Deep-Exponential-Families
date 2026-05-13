import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
from scipy import sparse


def softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Normalize each row to sum to one
    row_sums = mat.sum(axis=1, keepdims=True)
    return mat / np.maximum(row_sums, eps)


def cosine_similarity_matrix(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Row-wise cosine similarity matrix
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normalized = mat / np.maximum(norms, eps)
    return normalized @ normalized.T


def make_zipf_distribution(vocab_size: int, exponent: float) -> np.ndarray:
    # Create a Zipf-like base word distribution
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = ranks ** (-exponent)
    probs /= probs.sum()
    return probs


def make_correlated_topic_word_matrix(
    rng: np.random.Generator,
    num_topics: int,
    vocab_size: int,
    zipf_exponent: float,
    topic_overlap: float,
    topic_noise_concentration: float,
    scale_separation: float,
) -> np.ndarray:
    """
    Build a dense, highly overlapping topic-word matrix.

    High overlap makes latent topics posteriorly coupled.
    Strong scale separation makes coordinate-wise methods less comfortable.
    """
    base = make_zipf_distribution(vocab_size, zipf_exponent)

    topic_word = np.zeros((num_topics, vocab_size), dtype=np.float64)

    for k in range(num_topics):
        noise = rng.gamma(
            shape=topic_noise_concentration,
            scale=1.0,
            size=vocab_size,
        )
        noise /= noise.sum()

        mixed = topic_overlap * base + (1.0 - topic_overlap) * noise
        topic_word[k] = mixed

    topic_word = normalize_rows(topic_word)

    # Create strong scale separation across topics
    scales = np.exp(
        np.linspace(
            -math.log(scale_separation),
            math.log(scale_separation),
            num_topics,
        )
    )
    rng.shuffle(scales)

    topic_word = topic_word * scales[:, None]

    return topic_word


def make_dense_coupling_matrix(
    rng: np.random.Generator,
    upper_dim: int,
    lower_dim: int,
    coupling_strength: float,
    scale_separation: float,
) -> np.ndarray:
    """
    Create a dense positive upper-to-lower coupling matrix.

    This makes lower-layer latent variables depend on many upper-layer variables,
    which slows purely local coordinate propagation.
    """
    low_rank_a = rng.gamma(shape=1.5, scale=1.0, size=(upper_dim, 3))
    low_rank_b = rng.gamma(shape=1.5, scale=1.0, size=(3, lower_dim))
    low_rank_part = low_rank_a @ low_rank_b

    dense_noise = rng.gamma(shape=0.7, scale=1.0, size=(upper_dim, lower_dim))

    mat = coupling_strength * low_rank_part + dense_noise
    mat = mat / np.maximum(mat.mean(), 1e-12)

    # Add column scale separation
    col_scales = np.exp(
        np.linspace(
            -math.log(scale_separation),
            math.log(scale_separation),
            lower_dim,
        )
    )
    rng.shuffle(col_scales)
    mat = mat * col_scales[None, :]

    return mat


def sample_gamma_poisson_def_corpus(
    seed: int = 7,
    num_docs: int = 3000,
    vocab_size: int = 8000,
    lower_dim: int = 80,
    upper_dim: int = 20,
    avg_doc_length: int = 170,
    doc_length_sigma: float = 1.05,
    zipf_exponent: float = 1.08,
    topic_overlap: float = 0.88,
    topic_noise_concentration: float = 0.08,
    coupling_strength: float = 4.0,
    scale_separation: float = 80.0,
    z2_shape: float = 0.45,
    z2_rate: float = 0.45,
    z1_rate: float = 1.0,
    min_doc_length: int = 20,
    max_doc_length: int = 1500,
) -> dict:
    """
    Generate Wikipedia-like sparse bag-of-words data from a two-layer
    Gamma-Poisson-style deep exponential family.

    The generated data is intentionally ill-conditioned and strongly coupled,
    which favors deterministic Adam-based BBVI over ordinary CAVI in convergence speed.
    """
    rng = np.random.default_rng(seed)

    # Topic-word matrix W0: lower latent layer to vocabulary
    w0 = make_correlated_topic_word_matrix(
        rng=rng,
        num_topics=lower_dim,
        vocab_size=vocab_size,
        zipf_exponent=zipf_exponent,
        topic_overlap=topic_overlap,
        topic_noise_concentration=topic_noise_concentration,
        scale_separation=scale_separation,
    )

    # Dense coupling W1: upper latent layer to lower latent layer
    w1 = make_dense_coupling_matrix(
        rng=rng,
        upper_dim=upper_dim,
        lower_dim=lower_dim,
        coupling_strength=coupling_strength,
        scale_separation=math.sqrt(scale_separation),
    )

    # Upper-layer document factors
    z2 = rng.gamma(
        shape=z2_shape,
        scale=1.0 / z2_rate,
        size=(num_docs, upper_dim),
    )

    # Lower-layer shape is dense and strongly coupled with upper layer
    raw_shape = z2 @ w1
    raw_shape = raw_shape / np.maximum(raw_shape.mean(), 1e-12)

    z1_shape = 0.15 + softplus(raw_shape)
    z1 = rng.gamma(
        shape=z1_shape,
        scale=1.0 / z1_rate,
    )

    # Heavy-tailed document lengths, similar to real text corpora
    log_mean = math.log(avg_doc_length) - 0.5 * doc_length_sigma**2
    doc_lengths = rng.lognormal(
        mean=log_mean,
        sigma=doc_length_sigma,
        size=num_docs,
    )
    doc_lengths = np.clip(doc_lengths.astype(int), min_doc_length, max_doc_length)

    rows = []
    cols = []
    data = []

    for d in range(num_docs):
        word_scores = z1[d] @ w0
        word_probs = word_scores / np.maximum(word_scores.sum(), 1e-12)

        length = int(doc_lengths[d])
        sampled_words = rng.choice(
            vocab_size,
            size=length,
            replace=True,
            p=word_probs,
        )

        unique_words, counts = np.unique(sampled_words, return_counts=True)

        rows.extend([d] * len(unique_words))
        cols.extend(unique_words.tolist())
        data.extend(counts.astype(np.int64).tolist())

    x = sparse.csr_matrix(
        (np.asarray(data, dtype=np.int64), (np.asarray(rows), np.asarray(cols))),
        shape=(num_docs, vocab_size),
        dtype=np.int64,
    )

    return {
        "x": x,
        "w0": w0,
        "w1": w1,
        "z1": z1,
        "z2": z2,
        "doc_lengths": doc_lengths,
        "metadata": {
            "seed": seed,
            "num_docs": num_docs,
            "vocab_size": vocab_size,
            "lower_dim": lower_dim,
            "upper_dim": upper_dim,
            "avg_doc_length": avg_doc_length,
            "doc_length_sigma": doc_length_sigma,
            "zipf_exponent": zipf_exponent,
            "topic_overlap": topic_overlap,
            "topic_noise_concentration": topic_noise_concentration,
            "coupling_strength": coupling_strength,
            "scale_separation": scale_separation,
            "z2_shape": z2_shape,
            "z2_rate": z2_rate,
            "z1_rate": z1_rate,
            "min_doc_length": min_doc_length,
            "max_doc_length": max_doc_length,
        },
    }


def split_corpus(
    x: sparse.csr_matrix,
    seed: int,
    train_frac: float = 0.80,
    valid_frac: float = 0.10,
) -> dict:
    # Split documents into train, validation, and test sets
    rng = np.random.default_rng(seed)
    num_docs = x.shape[0]
    perm = rng.permutation(num_docs)

    train_end = int(num_docs * train_frac)
    valid_end = int(num_docs * (train_frac + valid_frac))

    train_idx = perm[:train_end]
    valid_idx = perm[train_end:valid_end]
    test_idx = perm[valid_end:]

    return {
        "train": x[train_idx],
        "valid": x[valid_idx],
        "test": x[test_idx],
        "train_indices": train_idx,
        "valid_indices": valid_idx,
        "test_indices": test_idx,
    }


def write_lda_c_format(path: Path, x: sparse.csr_matrix) -> None:
    """
    Write sparse count matrix in common LDA-C / Blei bag-of-words format.

    Each line:
    number_of_unique_words word_id:count word_id:count ...
    """
    x = x.tocsr()

    with path.open("w", encoding="utf-8") as f:
        for d in range(x.shape[0]):
            start = x.indptr[d]
            end = x.indptr[d + 1]
            word_ids = x.indices[start:end]
            counts = x.data[start:end]

            pairs = [f"{int(w)}:{int(c)}" for w, c in zip(word_ids, counts)]
            line = f"{len(pairs)}"
            if pairs:
                line += " " + " ".join(pairs)
            f.write(line + "\n")


def write_vocab(path: Path, vocab_size: int) -> None:
    # Write a simple synthetic vocabulary
    with path.open("w", encoding="utf-8") as f:
        for i in range(vocab_size):
            f.write(f"synthetic_word_{i}\n")


def compute_diagnostics(x: sparse.csr_matrix, w0: np.ndarray, w1: np.ndarray) -> dict:
    # Compute diagnostics that indicate why this data is BBVI-friendly
    x = x.tocsr()

    doc_lengths = np.asarray(x.sum(axis=1)).ravel()
    nnz_per_doc = np.diff(x.indptr)

    topic_cos = cosine_similarity_matrix(w0)
    upper_coupling_cos = cosine_similarity_matrix(w1.T)

    off_diag_topic_cos = topic_cos[~np.eye(topic_cos.shape[0], dtype=bool)]
    off_diag_coupling_cos = upper_coupling_cos[
        ~np.eye(upper_coupling_cos.shape[0], dtype=bool)
    ]

    topic_scales = w0.sum(axis=1)
    coupling_col_scales = w1.sum(axis=0)

    diagnostics = {
        "num_docs": int(x.shape[0]),
        "vocab_size": int(x.shape[1]),
        "total_tokens": int(x.sum()),
        "matrix_nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])),
        "mean_doc_length": float(doc_lengths.mean()),
        "median_doc_length": float(np.median(doc_lengths)),
        "max_doc_length": int(doc_lengths.max()),
        "mean_unique_words_per_doc": float(nnz_per_doc.mean()),
        "median_unique_words_per_doc": float(np.median(nnz_per_doc)),
        "mean_topic_cosine": float(off_diag_topic_cos.mean()),
        "p90_topic_cosine": float(np.quantile(off_diag_topic_cos, 0.90)),
        "mean_lower_layer_coupling_cosine": float(off_diag_coupling_cos.mean()),
        "p90_lower_layer_coupling_cosine": float(np.quantile(off_diag_coupling_cos, 0.90)),
        "topic_scale_ratio_max_over_min": float(topic_scales.max() / topic_scales.min()),
        "coupling_scale_ratio_max_over_min": float(
            coupling_col_scales.max() / coupling_col_scales.min()
        ),
    }

    return diagnostics


def save_dataset(output_dir: str, config: dict) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated = sample_gamma_poisson_def_corpus(**config)

    x = generated["x"]
    w0 = generated["w0"]
    w1 = generated["w1"]
    z1 = generated["z1"]
    z2 = generated["z2"]
    metadata = generated["metadata"]

    split = split_corpus(x=x, seed=config["seed"])

    write_lda_c_format(output_path / "train.cpp.dat", split["train"])
    write_lda_c_format(output_path / "valid.cpp.dat", split["valid"])
    write_lda_c_format(output_path / "test.cpp.dat", split["test"])
    write_vocab(output_path / "vocab.dat", config["vocab_size"])

    sparse.save_npz(output_path / "full_counts.npz", x)
    sparse.save_npz(output_path / "train_counts.npz", split["train"])
    sparse.save_npz(output_path / "valid_counts.npz", split["valid"])
    sparse.save_npz(output_path / "test_counts.npz", split["test"])

    np.savez_compressed(
        output_path / "true_factors.npz",
        w0=w0,
        w1=w1,
        z1=z1,
        z2=z2,
        doc_lengths=generated["doc_lengths"],
        train_indices=split["train_indices"],
        valid_indices=split["valid_indices"],
        test_indices=split["test_indices"],
    )

    diagnostics = compute_diagnostics(x=x, w0=w0, w1=w1)

    metadata["diagnostics"] = diagnostics
    metadata["files"] = {
        "train": "train.cpp.dat",
        "valid": "valid.cpp.dat",
        "test": "test.cpp.dat",
        "vocab": "vocab.dat",
        "full_counts": "full_counts.npz",
        "true_factors": "true_factors.npz",
    }

    with (output_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 80)
    print("Synthetic Wikipedia-like DEF dataset generated successfully.")
    print("=" * 80)
    print(f"Output directory: {output_path.resolve()}")
    print()
    print("Main files:")
    print(f"  {output_path / 'train.cpp.dat'}")
    print(f"  {output_path / 'valid.cpp.dat'}")
    print(f"  {output_path / 'test.cpp.dat'}")
    print(f"  {output_path / 'vocab.dat'}")
    print(f"  {output_path / 'true_factors.npz'}")
    print(f"  {output_path / 'metadata.json'}")
    print()
    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


def default_config() -> dict:
    return {
        "seed": 7,
        "num_docs": 3000,
        "vocab_size": 8000,
        "lower_dim": 80,
        "upper_dim": 20,
        "avg_doc_length": 170,
        "doc_length_sigma": 1.05,
        "zipf_exponent": 1.08,
        "topic_overlap": 0.88,
        "topic_noise_concentration": 0.08,
        "coupling_strength": 4.0,
        "scale_separation": 80.0,
        "z2_shape": 0.45,
        "z2_rate": 0.45,
        "z1_rate": 1.0,
        "min_doc_length": 20,
        "max_doc_length": 1500,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Wikipedia-like synthetic DEF bag-of-words dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_wikipedia_like_def",
        help="Directory where the generated dataset will be saved.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_docs", type=int, default=3000)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--lower_dim", type=int, default=80)
    parser.add_argument("--upper_dim", type=int, default=20)
    parser.add_argument("--avg_doc_length", type=int, default=170)
    parser.add_argument("--doc_length_sigma", type=float, default=1.05)
    parser.add_argument("--zipf_exponent", type=float, default=1.08)
    parser.add_argument("--topic_overlap", type=float, default=0.88)
    parser.add_argument("--topic_noise_concentration", type=float, default=0.08)
    parser.add_argument("--coupling_strength", type=float, default=4.0)
    parser.add_argument("--scale_separation", type=float, default=80.0)

    args = parser.parse_args()

    cfg = default_config()
    cfg.update(
        {
            "seed": args.seed,
            "num_docs": args.num_docs,
            "vocab_size": args.vocab_size,
            "lower_dim": args.lower_dim,
            "upper_dim": args.upper_dim,
            "avg_doc_length": args.avg_doc_length,
            "doc_length_sigma": args.doc_length_sigma,
            "zipf_exponent": args.zipf_exponent,
            "topic_overlap": args.topic_overlap,
            "topic_noise_concentration": args.topic_noise_concentration,
            "coupling_strength": args.coupling_strength,
            "scale_separation": args.scale_separation,
        }
    )

    save_dataset(output_dir=args.output_dir, config=cfg)
