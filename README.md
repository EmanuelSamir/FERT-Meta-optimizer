# FRTE: First Return, Then Explore

**A Non-Markovian Meta-Optimizer for Deep Learning, Scientific Machine Learning, and Non-Convex Optimization**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Overview

Standard deep learning gradient optimizers (SGD-Momentum, Adam, AdamW, SAM) operate as **Markovian operators**: their update state $s_{t+1}$ depends strictly on the current parameter vector $\theta_t$, instantaneous stochastic gradient $g_t$, and exponentially discounted momentum buffers ($m_t, v_t$). When navigating non-convex objective surfaces $\mathcal{L}: \mathbb{R}^D \to \mathbb{R}$, this memorylessness causes severe optimization failure modes:
1. **Deceptive Basin Entrapment:** Descending irreversibly into wide, high-residual local wells.
2. **Multi-Objective Pathology in Stiff PINNs:** Over-optimizing boundary conditions while the internal physics residual stalls on a flat attractor.
3. **Bottleneck Gradient Vanishing:** Premature stagnation in narrow latent representation corridors.

**FRTE** (*First Return, Then Explore*) is a principled, transparent **non-Markovian meta-optimizer** implemented in PyTorch that wraps any base gradient optimizer. It resolves non-convex stagnation via two coupled operators:
- **First Return ($\mathcal{R}$):** Stateless rollback to an archived functional checkpoint $\theta_{\mathrm{anc}}$ coupled with an instantaneous purge of corrupt momentum moments, rupturing the trapped Markov chain.
- **Then Explore ($\mathcal{E}$):** Extraction of the Grassmannian tangent subspace $\mathrm{span}(Q) \in \mathrm{Gr}(k, D)$ of the failed trajectory corridor via an $\mathcal{O}(w^2 D)$ Dual-Gram spectral decomposition ($w \ll D$), followed by orthogonal gradient projection $\tilde{g} = (I - Q Q^T) g$ for $\tau_{\mathrm{explore}}$ steps, strictly forbidding re-traversal of the failed descent path.

---

## 2. Directory Structure

```text
frte_project/
├── benchmarks/
│   ├── deceptive_landscape.py        # Benchmark 1: Multimodal 2D deceptive landscape
│   ├── pinn.py                       # Benchmark 2: Stiff reaction-diffusion PINN (lambda = 150)
│   ├── deep_autoencoder.py           # Benchmark 3: Fashion-MNIST 784-500-250-30 bottleneck (1.4M params)
│   ├── modular_addition.py           # Benchmark 4: Modular arithmetic (a + b) mod 97 memorization corridor
│   └── sam.py                        # Sharpness-Aware Minimization baseline wrapper
├── experiments/
│   └── run_suite.py                  # Master runner with multi-seed statistics (mu +/- sigma)
├── reports/
│   ├── figures/                      # High-resolution benchmark figures
│   │   ├── exp1_deceptive_landscape.png
│   │   ├── exp2_pinn.png
│   │   ├── exp3_deep_autoencoder.png
│   │   └── exp4_modular_addition.png
│   ├── benchmark_summary.json        # Consolidated quantitative metrics (mean +/- std)
│   └── SCIENTIFIC_REPORT.md          # Comprehensive theoretical foundations & empirical paper
├── src/
│   └── frte/
│       ├── __init__.py
│       ├── archive.py                # Stateless parameter checkpoint archive
│       ├── grassmann.py              # Dual-Gram O(w^2 D) Grassmannian manifold algebra
│       └── optimizer.py              # Transparent FRTE Meta-Optimizer class
├── tests/
│   └── test_frte.py                  # Orthonormality, invariance, and rollback unit tests
└── setup.py
```

---

## 3. Quickstart

### Installation
```bash
git clone https://github.com/emanuelsamir/frte.git
cd frte
pip install -e .
```

### Usage in PyTorch
```python
import torch
from frte import FRTE

model = MyNeuralNetwork()

# Wrap any base optimizer (e.g. Adam, AdamW, SGD)
optimizer = FRTE(
    model.parameters(),
    base_optimizer_cls=torch.optim.Adam,
    lr=1e-3,
    k_check=15,          # Health check frequency (iterations)
    patience=2,          # Stagnation checks before triggering First Return
    tau_explore=20,      # Exploration steps with orthogonal projection
    subspace_k=4         # Grassmannian subspace rank
)

for step, (inputs, targets) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()

    # Monitor loss and maintain functional archive
    optimizer.monitor_and_archive(step, loss.item())

    # Executes orthogonal projection if exploring, then steps base optimizer
    optimizer.step()
```

---

## 4. Benchmark Results (Strict Compute Match, Multi-Seed $\mu \pm \sigma$)

| Benchmark Experiment | Metric | Adam | AdamW | SGD-M | SAM | **FRTE (Ours)** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **1. Deceptive 2D Landscape** | Final Loss $\downarrow$ | $0.325 \pm 0.000$ | $0.325 \pm 0.000$ | $0.325 \pm 0.000$ | $0.325 \pm 0.000$ | **$-6.659 \pm 0.000$** |
| | Convergence State | Local Basin | Local Basin | Local Basin | Local Basin | **Global Minimum** |
| **2. Stiff PINN ($\lambda=150$)** | Composite Loss $\downarrow$ | $30.54 \pm 25.58$ | $30.55 \pm 25.59$ | $3155.7 \pm 851.9$ | $2165.0 \pm 162.8$ | **$6.47 \pm 4.86$ ($4.7\times$ Lower)** |
| | Solution $L_2$ Error $\downarrow$ | $0.00197 \pm 0.0013$ | $0.00197 \pm 0.0013$ | $0.1174 \pm 0.027$ | $0.0840 \pm 0.006$ | **$0.00044 \pm 0.0002$ ($4.5\times$ Lower)** |
| **3. Deep Autoencoder (1.4M)** | Test MSE Loss $\downarrow$ | $0.02035 \pm 0.00013$ | $0.02036 \pm 0.00004$ | $0.09945 \pm 0.00035$ | $0.02815 \pm 0.00095$ | **$0.01865 \pm 0.00018$** |
| | Train Loss $\downarrow$ | $0.02176 \pm 0.00014$ | $0.02190 \pm 0.00013$ | $0.09834 \pm 0.00024$ | $0.02572 \pm 0.00086$ | **$0.02013 \pm 0.00008$** |

For full formal proofs, lemmas, and mathematical foundations, see [SCIENTIFIC_REPORT.md](file:///private/tmp/frte_project/reports/SCIENTIFIC_REPORT.md).
