# von Mises-Fisher Parameter Estimation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of concentration parameter (κ) estimation for the
von Mises-Fisher (vMF) distribution, including mixture models and model
selection. Implements the MML estimators from Wallace & Dowe (1993) as well
as Maximum Likelihood and Schou's bias-corrected estimator.

---

## Installation

```bash
git clone https://github.com/Palikaris/vmf-mml.git
cd vmf-mml
pip install -e .
```

To run the real-data experiments (20 Newsgroups / digits datasets):

```bash
pip install -e ".[experiments]"
```

---

## Package overview

```
vmf_estimation/
├── core.py            # Bessel functions, A_d(κ), Fisher information
├── data.py            # vMF sample generation (Wood/Ulrich rejection sampling)
├── estimators.py      # ML, Schou, and MML estimators (h₁, h₂, h₃ priors)
├── priors.py          # MML prior densities
├── evaluation.py      # Monte Carlo evaluation framework
├── mixture.py         # EM-based mixture of vMF (MoVMF)
├── model_selection.py # AIC, BIC, MML message length for mixture selection
└── kl.py              # KL divergence (closed-form vMF + Monte Carlo mixture)
```

---

## Single-distribution estimators

```python
import numpy as np
from vmf_estimation.estimators import MLEstimator, SchouEstimator, MMLEstimator
from vmf_estimation.data import generate_vmf_samples

mu_true = np.array([1.0, 0.0])
data = generate_vmf_samples(mu_true, kappa=2.0, n_samples=32, random_state=0)

ml  = MLEstimator().estimate(data)
sch = SchouEstimator().estimate(data)
mml = MMLEstimator(prior='h3').estimate(data)   # h3 recommended
```

**Prior choices for MMLEstimator:**

| Prior | Formula | Notes |
|-------|---------|-------|
| `h1`  | κ⁻¹ | Jeffreys-like |
| `h2`  | 2/(π(1+κ²)) | Uniform on μ |
| `h3`  | κ/(1+κ²)^(3/2) | Wallace-Dowe recommended |

---

## Mixture models

```python
from vmf_estimation.mixture import MovMF
from vmf_estimation.model_selection import aic, bic, mml_message_length

X = ...  # (N, d) array of unit vectors

model = MovMF(n_clusters=5, n_init=3, random_state=0)
model.fit(X)

ll      = model.log_likelihood(X)
n_par   = model.n_params(d=X.shape[1])
aic_score = aic(ll, n_par)
bic_score = bic(ll, n_par, N=len(X))
mml_score = mml_message_length(model, X)   # lower = better for all three
```

### Automatic K selection

```python
from vmf_estimation.model_selection import select_k

best_k, results = select_k(X, k_range=range(1, 8), method='mml')
```

---

## Reproducing paper results

### Simulated data experiments (Table 1 / Figure 1)

```bash
python scripts/run_mixture_experiments.py
# → figures/model_selection_simulated.{pdf,png}
```

### 20 Newsgroups experiments (Table 2 / Figure 2)

```bash
python scripts/run_word_embedding_experiments.py --newsgroups --k-range 5 10 15 20 25
# → figures/model_selection_newsgroups.{pdf,png}
```

The 20 Newsgroups corpus is downloaded automatically by scikit-learn on first run.

---

## Validation against Wallace & Dowe (1993) Table 1

```bash
python -m pytest tests/test_table1.py -v
```

All 157 test cases pass with 0.00% error.

---

## References

- Wallace, C. S. & Dowe, D. L. (1993). *MML estimation of the von Mises
  concentration parameter.* Technical Report 93/193, Monash University.
- Schou, G. (1978). Estimation of the concentration parameter in von
  Mises-Fisher distributions. *Biometrika*, 65(2), 369–377.
- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on
  the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6,
  1345–1382.

---

## License

MIT — see [LICENSE](LICENSE).
