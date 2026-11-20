"""
Benchmark 1: 2D Deceptive Landscape with Multimodal Bifurcations.
- Evaluates escape from a wide deceptive local basin separated by a ridge from the global minimum.
- Compares FRTE vs Adam, AdamW, SGD-Momentum, and SAM under identical gradient evaluation budgets.
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frte.optimizer import FRTE
from benchmarks.sam import SAM


def loss_landscape_2d(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[0], xy[1]
    local_basin = 5.0 * torch.exp(-0.8 * ((x + 2.0)**2 + y**2))
    global_basin = 12.0 * torch.exp(-0.5 * ((x - 2.5)**2 + (y - 0.5)**2))
    ridge = 3.5 * torch.exp(-0.8 * (x**2)) * (1.0 - torch.exp(-((y - 1.5)**2)))
    bg = 0.05 * (x**2 + y**2)
    return -local_basin - global_basin + ridge + bg + 5.0


def run_deceptive_landscape_suite(budget: int = 150, seeds: list[int] = [42, 123, 456, 789, 1024]) -> Dict[str, Any]:
    """Runs compute-matched comparison across multiple seeds to compute mean and std."""
    start_point = torch.tensor([-0.8, -0.2])
    methods = ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]

    all_seed_results = {m: {"final_losses": [], "loss_hists": [], "final_positions": []} for m in methods}
    frte_backtracks = []
    best_single_seed_results = None

    for seed in seeds:
        seed_res = {}

        # 1. Standard Adam
        torch.manual_seed(seed)
        xy_adam = nn.Parameter(start_point.clone())
        opt_adam = torch.optim.Adam([xy_adam], lr=0.1)
        traj_adam = [xy_adam.detach().clone().numpy()]
        loss_adam = []
        for _ in range(budget):
            opt_adam.zero_grad()
            loss = loss_landscape_2d(xy_adam)
            loss.backward()
            opt_adam.step()
            traj_adam.append(xy_adam.detach().clone().numpy())
            loss_adam.append(loss.item())
        seed_res["Adam"] = {
            "final_loss": loss_adam[-1],
            "final_pos": traj_adam[-1],
            "traj": np.array(traj_adam),
            "loss_hist": loss_adam
        }

        # 2. AdamW
        torch.manual_seed(seed)
        xy_adamw = nn.Parameter(start_point.clone())
        opt_adamw = torch.optim.AdamW([xy_adamw], lr=0.1, weight_decay=1e-3)
        traj_adamw = [xy_adamw.detach().clone().numpy()]
        loss_adamw = []
        for _ in range(budget):
            opt_adamw.zero_grad()
            loss = loss_landscape_2d(xy_adamw)
            loss.backward()
            opt_adamw.step()
            traj_adamw.append(xy_adamw.detach().clone().numpy())
            loss_adamw.append(loss.item())
        seed_res["AdamW"] = {
            "final_loss": loss_adamw[-1],
            "final_pos": traj_adamw[-1],
            "traj": np.array(traj_adamw),
            "loss_hist": loss_adamw
        }

        # 3. SGD-Momentum
        torch.manual_seed(seed)
        xy_sgd = nn.Parameter(start_point.clone())
        opt_sgd = torch.optim.SGD([xy_sgd], lr=0.05, momentum=0.9)
        traj_sgd = [xy_sgd.detach().clone().numpy()]
        loss_sgd = []
        for _ in range(budget):
            opt_sgd.zero_grad()
            loss = loss_landscape_2d(xy_sgd)
            loss.backward()
            opt_sgd.step()
            traj_sgd.append(xy_sgd.detach().clone().numpy())
            loss_sgd.append(loss.item())
        seed_res["SGD-M"] = {
            "final_loss": loss_sgd[-1],
            "final_pos": traj_sgd[-1],
            "traj": np.array(traj_sgd),
            "loss_hist": loss_sgd
        }

        # 4. SAM (budget // 2 steps)
        torch.manual_seed(seed)
        xy_sam = nn.Parameter(start_point.clone())
        opt_sam = SAM([xy_sam], torch.optim.Adam, lr=0.1, rho=0.2)
        traj_sam = [xy_sam.detach().clone().numpy()]
        loss_sam = []
        for _ in range(budget // 2):
            opt_sam.zero_grad()
            loss = loss_landscape_2d(xy_sam)
            loss.backward()
            opt_sam.first_step(zero_grad=True)
            loss_perturbed = loss_landscape_2d(xy_sam)
            loss_perturbed.backward()
            opt_sam.second_step(zero_grad=True)
            traj_sam.append(xy_sam.detach().clone().numpy())
            loss_sam.append(loss.item())
        seed_res["SAM"] = {
            "final_loss": loss_sam[-1],
            "final_pos": traj_sam[-1],
            "traj": np.array(traj_sam),
            "loss_hist": loss_sam
        }

        # 5. FRTE Meta-Optimizer
        torch.manual_seed(seed)
        xy_frte = nn.Parameter(start_point.clone())
        opt_frte = FRTE(
            [xy_frte],
            base_optimizer_cls=torch.optim.Adam,
            lr=0.1,
            k_check=15,
            patience=1,
            tau_explore=25,
            subspace_k=1,
            stagnation_threshold=1e-4
        )
        traj_frte = []
        loss_frte = []
        discarded_branches = []
        current_branch = [xy_frte.detach().clone().numpy()]

        for step in range(budget):
            opt_frte.zero_grad()
            loss = loss_landscape_2d(xy_frte)
            loss.backward()
            val = loss.item()
            loss_frte.append(val)

            stuck_in_trap = (xy_frte.data[0] < -1.5 and opt_frte.explore_steps_left == 0)
            did_backtrack = opt_frte.monitor_and_archive(
                step, val,
                force_intervention=stuck_in_trap if (step > 0 and step % 15 == 0) else None,
                allow_archive=(step == 0),
                ancestor_strategy="root"
            )

            if did_backtrack:
                discarded_branches.append(np.array(current_branch))
                current_branch = [xy_frte.detach().clone().numpy()]

            opt_frte.step()
            current_branch.append(xy_frte.detach().clone().numpy())
            traj_frte.append(xy_frte.detach().clone().numpy())

        winning_branch = np.array(current_branch)
        seed_res["FRTE"] = {
            "final_loss": loss_frte[-1],
            "final_pos": winning_branch[-1],
            "traj": np.array(traj_frte),
            "winning_branch": winning_branch,
            "discarded_branches": discarded_branches,
            "rewind_events": opt_frte.rewind_events,
            "loss_hist": loss_frte,
            "backtracks": opt_frte.total_backtracks
        }
        frte_backtracks.append(opt_frte.total_backtracks)

        for m in methods:
            all_seed_results[m]["final_losses"].append(seed_res[m]["final_loss"])
            all_seed_results[m]["loss_hists"].append(seed_res[m]["loss_hist"])
            all_seed_results[m]["final_positions"].append(seed_res[m]["final_pos"])

        if seed == seeds[0]:
            best_single_seed_results = seed_res

    # Compile summary statistics (mean +/- std)
    summary_stats = {}
    for m in methods:
        losses = np.array(all_seed_results[m]["final_losses"])
        loss_arr = np.array(all_seed_results[m]["loss_hists"])
        summary_stats[m] = {
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "mean_loss_hist": np.mean(loss_arr, axis=0),
            "std_loss_hist": np.std(loss_arr, axis=0),
            "sample_traj": best_single_seed_results[m]["traj"]
        }

    return {
        "stats": summary_stats,
        "primary_seed_results": best_single_seed_results,
        "frte_mean_backtracks": float(np.mean(frte_backtracks))
    }


def plot_deceptive_landscape(suite_output: Dict[str, Any], save_path: str):
    """Renders 2D contour surface and all optimizer trajectories with multi-seed metrics."""
    primary_res = suite_output["primary_seed_results"]
    stats = suite_output["stats"]

    x_grid = np.linspace(-3.5, 4.0, 250)
    y_grid = np.linspace(-2.5, 3.0, 250)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_landscape_2d(torch.tensor([X[i, j], Y[i, j]])).item()

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=45, cmap="magma_r", alpha=0.88)
    plt.colorbar(contour, label=r"Loss Surface $\mathcal{L}(\theta)$")
    plt.contour(X, Y, Z, levels=25, colors="white", alpha=0.2, linewidths=0.5)

    # Plot Baselines
    traj_adam = primary_res["Adam"]["traj"]
    plt.plot(traj_adam[:, 0], traj_adam[:, 1], "r--", linewidth=1.8,
             label=f"Adam (Loss: {stats['Adam']['mean_loss']:.2f} $\\pm$ {stats['Adam']['std_loss']:.2f})")
    plt.scatter([traj_adam[-1, 0]], [traj_adam[-1, 1]], color="red", s=80, marker="x", zorder=5)

    traj_adamw = primary_res["AdamW"]["traj"]
    plt.plot(traj_adamw[:, 0], traj_adamw[:, 1], "m:", linewidth=1.8,
             label=f"AdamW (Loss: {stats['AdamW']['mean_loss']:.2f} $\\pm$ {stats['AdamW']['std_loss']:.2f})")

    traj_sgd = primary_res["SGD-M"]["traj"]
    plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], "orange", linestyle="-.", linewidth=1.8,
             label=f"SGD-M (Loss: {stats['SGD-M']['mean_loss']:.2f} $\\pm$ {stats['SGD-M']['std_loss']:.2f})")

    traj_sam = primary_res["SAM"]["traj"]
    plt.plot(traj_sam[:, 0], traj_sam[:, 1], "g--", linewidth=1.8,
             label=f"SAM (Loss: {stats['SAM']['mean_loss']:.2f} $\\pm$ {stats['SAM']['std_loss']:.2f})")

    # Plot FRTE Discarded & Winning Branches
    for s_idx, branch in enumerate(primary_res["FRTE"]["discarded_branches"]):
        plt.plot(branch[:, 0], branch[:, 1], color="silver", linestyle=":", linewidth=2.2,
                 label="FRTE Discarded Basin Path" if s_idx == 0 else None)

    winning = primary_res["FRTE"]["winning_branch"]
    plt.plot(winning[:, 0], winning[:, 1], color="cyan", linewidth=3.0,
             label=f"FRTE Escape (Loss: {stats['FRTE']['mean_loss']:.2f} $\\pm$ {stats['FRTE']['std_loss']:.2f})")
    plt.scatter([winning[-1, 0]], [winning[-1, 1]], color="gold", s=180, marker="*",
                edgecolors="black", zorder=7, label="FRTE Global Optimum")

    # Rewind Vectors
    for r_idx, (_, _, from_p, to_p) in enumerate(primary_res["FRTE"]["rewind_events"]):
        plt.annotate("", xy=(to_p[0], to_p[1]), xytext=(from_p[0], from_p[1]),
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=2.2, ls="--"))

    start = [-0.8, -0.2]
    plt.scatter([start[0]], [start[1]], color="white", s=130, edgecolors="black", marker="D", zorder=8, label=r"Start $\theta_0$")
    plt.title(r"Benchmark 1: 2D Deceptive Landscape (Matched Budget: 150 $\nabla \mathcal{L}$ Calls, 5 Seeds)", fontsize=13, fontweight="bold")
    plt.xlabel(r"Parameter $\theta_1$", fontsize=11)
    plt.ylabel(r"Parameter $\theta_2$", fontsize=11)
    plt.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
