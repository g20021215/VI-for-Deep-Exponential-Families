# Two-Layer Gamma-Poisson DEF

A compact, production-friendly PyTorch implementation of a two-layer Gamma-Poisson Deep Exponential Family for count data.

## Project structure

```text
.
├── defvi/
│   ├── models/
│   │   └── two_layer_gamma_poisson_def.py
│   ├── inference/
│   │   └── bbvi.py
│   └── utils/
│       ├── data.py
│       └── seed.py
├── scripts/
│   ├── train_synthetic.py
│   └── train_from_csv.py
├── requirements.txt
└── README.md
```

## Model

For each sample `n`:

- `z2_n ~ Gamma(alpha2_prior, beta2_prior)`
- `z1_n ~ Gamma(alpha1_prior, alpha1_prior / (W2 z2_n + eps))`
- `x_n ~ Poisson(W1 z1_n + eps)`

where:

- `W2` is a positive matrix mapping layer 2 to layer 1
- `W1` is a positive matrix mapping layer 1 to observed counts
- all positivity constraints are enforced with `softplus`

The variational family is mean-field Gamma:

- `q(z2_n) = Gamma(alpha2_q_n, beta2_q_n)`
- `q(z1_n) = Gamma(alpha1_q_n, beta1_q_n)`

Optimization is done with Monte Carlo ELBO maximization using PyTorch autograd.

## Install

```bash
pip install -r requirements.txt
```

## Run locally on synthetic data

```bash
python scripts/train_synthetic.py
```

## Run on your own count matrix from CSV

Expected CSV format:
- rows = samples/documents
- columns = observed count dimensions
- every entry must be a nonnegative integer or float count

```bash
python scripts/train_from_csv.py \
    --csv_path data/counts.csv \
    --latent_dim_1 32 \
    --latent_dim_2 8 \
    --epochs 300
```

## Colab quick start

Upload this whole repository to GitHub, then in Colab:

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -r requirements.txt
!python scripts/train_synthetic.py
```

Or for your own CSV already uploaded to Colab:

```python
!python scripts/train_from_csv.py --csv_path /content/your_counts.csv
```

## Outputs

Training scripts save:
- learned decoder matrices
- learned variational parameters
- latent posterior means
- training curve plot
- config and metrics json

## Notes

- This implementation is intentionally modular so you can later swap in CAVI approximations, minibatching, or more layers.
- For real sparse text/count datasets, start with standardized preprocessing and inspect very large counts before training.
