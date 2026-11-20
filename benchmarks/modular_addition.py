"""
Benchmark 4: Algorithmic Reasoning and Grokking in Modular Addition (a + b) mod 97.
Evaluates escape from catastrophic memorization attractor corridors under strict matched budget
comparing FRTE, Adam, AdamW, SGD-Momentum, and SAM.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frte.optimizer import FRTE
from benchmarks.sam import SAM


class ModularAdditionNet(nn.Module):
    """
    Two-layer MLP for modular arithmetic:
    Input: One-hot concatenation of (a, b) -> dimension 2*p = 194.
    Hidden layers: 128 -> 128 with ReLU activations.
    Output: Logits over p = 97 residue classes.
    """
    def __init__(self, p: int = 97, hidden_dim: int = 128):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def get_modular_dataset(p: int = 97, train_ratio: float = 0.5, seed: int = 42):
    """Generates the exhaustive Cayley table for (a + b) mod p and performs fixed random split."""
    a_vals = torch.arange(p).repeat_interleave(p)
    b_vals = torch.arange(p).repeat(p)
    targets = (a_vals + b_vals) % p

    X_a = F.one_hot(a_vals, num_classes=p).float()
    X_b = F.one_hot(b_vals, num_classes=p).float()
    X = torch.cat([X_a, X_b], dim=-1) # (p^2, 2*p)
    Y = targets

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(targets), generator=g)
    n_train = int(train_ratio * len(targets))
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    return X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx]


def run_modular_addition_suite(
    p: int = 97,
    budget: int = 350,
    seeds: list[int] = [42, 123, 456, 789, 1024]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs compute-matched modular addition evaluation across optimizers with multiple seeds."""
    methods = ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]
    loss_fn = nn.CrossEntropyLoss()

    all_seed_results = {
        m: {
            "final_train_losses": [],
            "final_val_losses": [],
            "train_accs": [],
            "val_accs": [],
            "train_loss_hists": [],
            "val_loss_hists": []
        }
        for m in methods
    }
    frte_backtracks = []
    primary_seed_results = None

    for seed in seeds:
        seed_res = {}
        # Fixed split for this seed
        X_tr, Y_tr, X_va, Y_va = get_modular_dataset(p=p, train_ratio=0.5, seed=seed)

        def create_fresh_model():
            torch.manual_seed(seed)
            return ModularAdditionNet(p=p, hidden_dim=128)

        # 1. Standard Adam
        m_adam = create_fresh_model()
        opt_adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
        tr_loss_adam, va_loss_adam = [], []
        for step in range(budget):
            opt_adam.zero_grad()
            logits = m_adam(X_tr)
            loss = loss_fn(logits, Y_tr)
            loss.backward()
            opt_adam.step()
            tr_loss_adam.append(loss.item())
            if (step + 1) % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    va_loss_adam.append(loss_fn(m_adam(X_va), Y_va).item())

        with torch.no_grad():
            tr_acc_adam = (m_adam(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc_adam = (m_adam(X_va).argmax(-1) == Y_va).float().mean().item()
            final_va_loss_adam = loss_fn(m_adam(X_va), Y_va).item()

        seed_res["Adam"] = {
            "final_train_loss": tr_loss_adam[-1],
            "final_val_loss": final_va_loss_adam,
            "train_acc": tr_acc_adam,
            "val_acc": va_acc_adam,
            "train_loss_hist": tr_loss_adam,
            "val_loss_hist": va_loss_adam
        }

        # 2. AdamW
        m_adamw = create_fresh_model()
        opt_adamw = torch.optim.AdamW(m_adamw.parameters(), lr=1e-3, weight_decay=1e-2)
        tr_loss_adamw, va_loss_adamw = [], []
        for step in range(budget):
            opt_adamw.zero_grad()
            logits = m_adamw(X_tr)
            loss = loss_fn(logits, Y_tr)
            loss.backward()
            opt_adamw.step()
            tr_loss_adamw.append(loss.item())
            if (step + 1) % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    va_loss_adamw.append(loss_fn(m_adamw(X_va), Y_va).item())

        with torch.no_grad():
            tr_acc_adamw = (m_adamw(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc_adamw = (m_adamw(X_va).argmax(-1) == Y_va).float().mean().item()
            final_va_loss_adamw = loss_fn(m_adamw(X_va), Y_va).item()

        seed_res["AdamW"] = {
            "final_train_loss": tr_loss_adamw[-1],
            "final_val_loss": final_va_loss_adamw,
            "train_acc": tr_acc_adamw,
            "val_acc": va_acc_adamw,
            "train_loss_hist": tr_loss_adamw,
            "val_loss_hist": va_loss_adamw
        }

        # 3. SGD with Momentum
        m_sgdm = create_fresh_model()
        opt_sgdm = torch.optim.SGD(m_sgdm.parameters(), lr=0.01, momentum=0.9)
        tr_loss_sgdm, va_loss_sgdm = [], []
        for step in range(budget):
            opt_sgdm.zero_grad()
            logits = m_sgdm(X_tr)
            loss = loss_fn(logits, Y_tr)
            loss.backward()
            opt_sgdm.step()
            tr_loss_sgdm.append(loss.item())
            if (step + 1) % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    va_loss_sgdm.append(loss_fn(m_sgdm(X_va), Y_va).item())

        with torch.no_grad():
            tr_acc_sgdm = (m_sgdm(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc_sgdm = (m_sgdm(X_va).argmax(-1) == Y_va).float().mean().item()
            final_va_loss_sgdm = loss_fn(m_sgdm(X_va), Y_va).item()

        seed_res["SGD-M"] = {
            "final_train_loss": tr_loss_sgdm[-1],
            "final_val_loss": final_va_loss_sgdm,
            "train_acc": tr_acc_sgdm,
            "val_acc": va_acc_sgdm,
            "train_loss_hist": tr_loss_sgdm,
            "val_loss_hist": va_loss_sgdm
        }

        # 4. SAM (Sharpness-Aware Minimization)
        m_sam = create_fresh_model()
        opt_sam = SAM(m_sam.parameters(), torch.optim.Adam, rho=0.05, lr=1e-3)
        tr_loss_sam, va_loss_sam = [], []
        for step in range(budget):
            loss = loss_fn(m_sam(X_tr), Y_tr)
            loss.backward()
            opt_sam.first_step(zero_grad=True)
            loss_2 = loss_fn(m_sam(X_tr), Y_tr)
            loss_2.backward()
            opt_sam.second_step(zero_grad=True)
            tr_loss_sam.append(loss.item())
            if (step + 1) % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    va_loss_sam.append(loss_fn(m_sam(X_va), Y_va).item())

        with torch.no_grad():
            tr_acc_sam = (m_sam(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc_sam = (m_sam(X_va).argmax(-1) == Y_va).float().mean().item()
            final_va_loss_sam = loss_fn(m_sam(X_va), Y_va).item()

        seed_res["SAM"] = {
            "final_train_loss": tr_loss_sam[-1],
            "final_val_loss": final_va_loss_sam,
            "train_acc": tr_acc_sam,
            "val_acc": va_acc_sam,
            "train_loss_hist": tr_loss_sam,
            "val_loss_hist": va_loss_sam
        }

        # 5. FRTE Meta-Optimizer
        m_frte = create_fresh_model()
        opt_frte = FRTE(
            m_frte.parameters(),
            base_optimizer_cls=torch.optim.Adam,
            lr=1e-3,
            k_check=15,
            patience=2,
            tau_explore=15,
            subspace_k=4,
            stagnation_threshold=0.01
        )
        tr_loss_frte, va_loss_frte = [], []
        for step in range(budget):
            opt_frte.zero_grad()
            logits = m_frte(X_tr)
            loss = loss_fn(logits, Y_tr)
            loss.backward()
            val = loss.item()
            tr_loss_frte.append(val)

            opt_frte.monitor_and_archive(
                step, val,
                allow_archive=(step == 0 or val < opt_frte.best_loss),
                ancestor_strategy="best"
            )
            opt_frte.step()

            if (step + 1) % 10 == 0 or step == budget - 1:
                with torch.no_grad():
                    va_loss_frte.append(loss_fn(m_frte(X_va), Y_va).item())

        with torch.no_grad():
            tr_acc_frte = (m_frte(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc_frte = (m_frte(X_va).argmax(-1) == Y_va).float().mean().item()
            final_va_loss_frte = loss_fn(m_frte(X_va), Y_va).item()

        seed_res["FRTE"] = {
            "final_train_loss": tr_loss_frte[-1],
            "final_val_loss": final_va_loss_frte,
            "train_acc": tr_acc_frte,
            "val_acc": va_acc_frte,
            "train_loss_hist": tr_loss_frte,
            "val_loss_hist": va_loss_frte,
            "backtracks": opt_frte.total_backtracks
        }
        frte_backtracks.append(opt_frte.total_backtracks)

        for m in methods:
            all_seed_results[m]["final_train_losses"].append(seed_res[m]["final_train_loss"])
            all_seed_results[m]["final_val_losses"].append(seed_res[m]["final_val_loss"])
            all_seed_results[m]["train_accs"].append(seed_res[m]["train_acc"])
            all_seed_results[m]["val_accs"].append(seed_res[m]["val_acc"])
            all_seed_results[m]["train_loss_hists"].append(seed_res[m]["train_loss_hist"])
            all_seed_results[m]["val_loss_hists"].append(seed_res[m]["val_loss_hist"])

        if seed == seeds[0]:
            primary_seed_results = seed_res

    # Compile statistics across seeds
    summary_stats = {}
    for m in methods:
        tr_losses = np.array(all_seed_results[m]["final_train_losses"])
        va_losses = np.array(all_seed_results[m]["final_val_losses"])
        tr_accs = np.array(all_seed_results[m]["train_accs"])
        va_accs = np.array(all_seed_results[m]["val_accs"])
        tr_hists = np.array(all_seed_results[m]["train_loss_hists"])
        va_hists = np.array(all_seed_results[m]["val_loss_hists"])

        summary_stats[m] = {
            "mean_train_loss": float(np.mean(tr_losses)),
            "std_train_loss": float(np.std(tr_losses)),
            "mean_val_loss": float(np.mean(va_losses)),
            "std_val_loss": float(np.std(va_losses)),
            "mean_train_acc": float(np.mean(tr_accs)),
            "std_train_acc": float(np.std(tr_accs)),
            "mean_val_acc": float(np.mean(va_accs)),
            "std_val_acc": float(np.std(va_accs)),
            "mean_train_hist": np.mean(tr_hists, axis=0),
            "std_train_hist": np.std(tr_hists, axis=0),
            "mean_val_hist": np.mean(va_hists, axis=0),
            "std_val_hist": np.std(va_hists, axis=0)
        }

    return {
        "stats": summary_stats,
        "primary_results": primary_seed_results,
        "frte_mean_backtracks": float(np.mean(frte_backtracks))
    }, {
        "X_va": X_va,
        "Y_va": Y_va,
        "models": {
            "Adam": m_adam,
            "AdamW": m_adamw,
            "SAM": m_sam,
            "FRTE": m_frte
        }
    }


def plot_modular_addition(
    suite_output: Dict[str, Any],
    eval_data: Dict[str, Any],
    budget: int = 350,
    save_path: str = "/private/tmp/frte_project/reports/figures/exp4_modular_addition.png"
):
    """Renders multi-panel scientific visualization for Benchmark 4."""
    stats = suite_output["stats"]
    primary_res = suite_output["primary_results"]
    p = 97

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    # Row 1: Attention / Representation Cayley Grid Prediction Matrix (Adam vs SAM vs FRTE)
    # Reconstruct 2D residue matrix mod 97 for primary models
    models = eval_data["models"]
    X_a_full = F.one_hot(torch.arange(p).repeat_interleave(p), num_classes=p).float()
    X_b_full = F.one_hot(torch.arange(p).repeat(p), num_classes=p).float()
    X_full = torch.cat([X_a_full, X_b_full], dim=-1)
    Y_full = (torch.arange(p).repeat_interleave(p) + torch.arange(p).repeat(p)) % p

    with torch.no_grad():
        preds_adam = models["Adam"](X_full).argmax(-1).reshape(p, p).cpu().numpy()
        preds_sam = models["SAM"](X_full).argmax(-1).reshape(p, p).cpu().numpy()
        preds_frte = models["FRTE"](X_full).argmax(-1).reshape(p, p).cpu().numpy()
        ground_truth_mat = Y_full.reshape(p, p).cpu().numpy()

    # Panel 1: Ground Truth Cayley table
    ax_gt = fig.add_subplot(gs[0, 0])
    im0 = ax_gt.imshow(ground_truth_mat, cmap="viridis", origin="lower")
    ax_gt.set_title("Ground Truth\n(a + b) mod 97", fontsize=10, fontweight="bold")
    ax_gt.set_xlabel("$b$", fontsize=9)
    ax_gt.set_ylabel("$a$", fontsize=9)

    # Panel 2: Adam Prediction Matrix
    ax_ad = fig.add_subplot(gs[0, 1])
    im1 = ax_ad.imshow(preds_adam, cmap="viridis", origin="lower")
    ax_ad.set_title(f"Adam (Memorization Trap)\nVal Loss: {stats['Adam']['mean_val_loss']:.2f}", fontsize=10, fontweight="bold", color="darkred")
    ax_ad.set_xlabel("$b$", fontsize=9)

    # Panel 3: SAM Prediction Matrix
    ax_sam = fig.add_subplot(gs[0, 2])
    im2 = ax_sam.imshow(preds_sam, cmap="viridis", origin="lower")
    ax_sam.set_title(f"SAM (Overfitting Plateau)\nVal Loss: {stats['SAM']['mean_val_loss']:.2f}", fontsize=10, fontweight="bold", color="green")
    ax_sam.set_xlabel("$b$", fontsize=9)

    # Panel 4: FRTE Transverse Exploration Matrix
    ax_frte = fig.add_subplot(gs[0, 3])
    im3 = ax_frte.imshow(preds_frte, cmap="viridis", origin="lower")
    ax_frte.set_title(f"FRTE (Ours, Non-Markovian)\nVal Loss: {stats['FRTE']['mean_val_loss']:.2f} ({suite_output['frte_mean_backtracks']:.1f} Backtracks)", fontsize=10, fontweight="bold", color="teal")
    ax_frte.set_xlabel("$b$", fontsize=9)

    # Row 2, Left: Training Loss & Validation Divergence (Memorization Pathology)
    ax_loss = fig.add_subplot(gs[1, :2])
    steps = np.arange(1, budget + 1)
    ax_loss.plot(steps, stats["Adam"]["mean_train_hist"], "r--", label="Adam (Train Loss)")
    ax_loss.plot(steps, stats["AdamW"]["mean_train_hist"], "m:", label="AdamW (Train Loss)")
    ax_loss.plot(steps, stats["SGD-M"]["mean_train_hist"], "orange", linestyle="-.", label="SGD-M (Train Loss)")
    ax_loss.plot(steps, stats["SAM"]["mean_train_hist"], "g--", label="SAM (Train Loss)")
    ax_loss.plot(steps, stats["FRTE"]["mean_train_hist"], "teal", linewidth=2.4, label="FRTE (Train Loss, Ours)")

    ax_loss.set_xlabel(f"Gradient Iterations (Matched Budget: {budget})", fontsize=10)
    ax_loss.set_ylabel("Cross-Entropy Training Loss", fontsize=10)
    ax_loss.set_title("Training Loss Convergence under Strict Budget", fontsize=11, fontweight="bold")
    ax_loss.grid(True, ls="--", alpha=0.35)
    ax_loss.legend(loc="upper right", framealpha=0.92, fontsize=8.8)

    # Row 2, Right: Validation Loss Divergence Bar Chart (Generalization Gap / Overfitting)
    ax_bar = fig.add_subplot(gs[1, 2:])
    opts = ["SGD-M", "FRTE", "SAM", "AdamW", "Adam"]
    mean_val_losses = [stats[m]["mean_val_loss"] for m in opts]
    std_val_losses = [stats[m]["std_val_loss"] for m in opts]
    colors = ["orange", "teal", "green", "magenta", "red"]

    bars = ax_bar.bar(opts, mean_val_losses, yerr=std_val_losses, capsize=4, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_bar.set_ylabel("Validation Loss (Lower = Less Overfitting)", fontsize=10)
    ax_bar.set_title("Validation Divergence Penalty ($\\mu \\pm \\sigma$ Across 5 Seeds)", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="y", ls="--", alpha=0.35)
    for bar, m_val, s_val in zip(bars, mean_val_losses, std_val_losses):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, m_val + s_val + 0.2, f"{m_val:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Benchmark 4: Modular Addition (a + b) mod 97 Non-Markovian Exploration Suite", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
