"""
Benchmark 4: Deep Autoencoder on Fashion-MNIST (Hinton & Salakhutdinov Architecture).
Bottleneck architecture: 784 -> 500 -> 250 -> 30 -> 250 -> 500 -> 784 (~1.4M parameters).
Evaluates escape from gradient vanishing / saddle-point plateau under strict matched budget
comparing FRTE, Adam, AdamW, SGD-Momentum, and SAM.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frte.optimizer import FRTE
from benchmarks.sam import SAM


class DeepBottleneckAutoencoder(nn.Module):
    """Deep autoencoder with narrow 30-dim latent bottleneck."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 784),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = x.view(x.size(0), -1)
        z = self.encoder(flat_x)
        out = self.decoder(z)
        return out


def get_fashion_mnist_loaders(batch_size: int = 128, data_root: str = "/private/tmp/data"):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_deep_autoencoder_suite(
    n_batches: int = 400,
    batch_size: int = 128,
    device: str = "cpu",
    seeds: list[int] = [42, 123, 456]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Runs compute-matched autoencoder training across optimizers with multiple seeds."""
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)
    dev = torch.device(device)
    loss_fn = nn.MSELoss()
    methods = ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]

    # Pre-fetch fixed batches for identical comparison
    batches = []
    while len(batches) < n_batches:
        for data, _ in train_loader:
            batches.append(data.to(dev))
            if len(batches) >= n_batches:
                break

    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch.to(dev)
    flat_test = test_batch.view(test_batch.size(0), -1)

    all_seed_results = {m: {"test_mses": [], "final_losses": [], "train_loss_hists": []} for m in methods}
    frte_backtracks = []
    primary_seed_results = None

    for seed in seeds:
        seed_res = {}
        torch.manual_seed(seed)
        base_model = DeepBottleneckAutoencoder().to(dev)
        init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

        def init_fresh_model():
            m = DeepBottleneckAutoencoder().to(dev)
            m.load_state_dict({k: v.clone() for k, v in init_state.items()})
            return m

        # 1. Adam
        m_adam = init_fresh_model()
        opt_adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
        loss_adam = []
        for b_idx in range(n_batches):
            data = batches[b_idx]
            flat_data = data.view(data.size(0), -1)
            opt_adam.zero_grad()
            recon = m_adam(data)
            l = loss_fn(recon, flat_data)
            l.backward()
            opt_adam.step()
            loss_adam.append(l.item())
        with torch.no_grad():
            test_rec_adam = m_adam(test_batch)
            test_mse_adam = loss_fn(test_rec_adam, flat_test).item()
        seed_res["Adam"] = {
            "train_loss": loss_adam,
            "final_train_loss": loss_adam[-1],
            "test_mse": test_mse_adam,
            "recon": test_rec_adam.cpu().numpy()
        }

        # 2. AdamW
        m_adamw = init_fresh_model()
        opt_adamw = torch.optim.AdamW(m_adamw.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_adamw = []
        for b_idx in range(n_batches):
            data = batches[b_idx]
            flat_data = data.view(data.size(0), -1)
            opt_adamw.zero_grad()
            recon = m_adamw(data)
            l = loss_fn(recon, flat_data)
            l.backward()
            opt_adamw.step()
            loss_adamw.append(l.item())
        with torch.no_grad():
            test_rec_adamw = m_adamw(test_batch)
            test_mse_adamw = loss_fn(test_rec_adamw, flat_test).item()
        seed_res["AdamW"] = {
            "train_loss": loss_adamw,
            "final_train_loss": loss_adamw[-1],
            "test_mse": test_mse_adamw,
            "recon": test_rec_adamw.cpu().numpy()
        }

        # 3. SGD-Momentum
        m_sgd = init_fresh_model()
        opt_sgd = torch.optim.SGD(m_sgd.parameters(), lr=0.05, momentum=0.9)
        loss_sgd = []
        for b_idx in range(n_batches):
            data = batches[b_idx]
            flat_data = data.view(data.size(0), -1)
            opt_sgd.zero_grad()
            recon = m_sgd(data)
            l = loss_fn(recon, flat_data)
            l.backward()
            opt_sgd.step()
            loss_sgd.append(l.item())
        with torch.no_grad():
            test_rec_sgd = m_sgd(test_batch)
            test_mse_sgd = loss_fn(test_rec_sgd, flat_test).item()
        seed_res["SGD-M"] = {
            "train_loss": loss_sgd,
            "final_train_loss": loss_sgd[-1],
            "test_mse": test_mse_sgd,
            "recon": test_rec_sgd.cpu().numpy()
        }

        # 4. SAM (budget matched: 2 gradient backward passes per step -> n_batches // 2 steps)
        m_sam = init_fresh_model()
        opt_sam = SAM(m_sam.parameters(), torch.optim.Adam, lr=1e-3, rho=0.05)
        loss_sam = []
        sam_limit = n_batches // 2
        for b_idx in range(sam_limit):
            data = batches[b_idx]
            flat_data = data.view(data.size(0), -1)
            opt_sam.zero_grad()
            recon = m_sam(data)
            l = loss_fn(recon, flat_data)
            l.backward()
            opt_sam.first_step(zero_grad=True)

            recon_p = m_sam(data)
            l_p = loss_fn(recon_p, flat_data)
            l_p.backward()
            opt_sam.second_step(zero_grad=True)
            loss_sam.extend([l.item(), l.item()])
        with torch.no_grad():
            test_rec_sam = m_sam(test_batch)
            test_mse_sam = loss_fn(test_rec_sam, flat_test).item()
        seed_res["SAM"] = {
            "train_loss": loss_sam,
            "final_train_loss": loss_sam[-1],
            "test_mse": test_mse_sam,
            "recon": test_rec_sam.cpu().numpy()
        }

        # 5. FRTE Meta-Optimizer
        m_frte = init_fresh_model()
        opt_frte = FRTE(
            m_frte.parameters(),
            base_optimizer_cls=torch.optim.Adam,
            lr=1e-3,
            k_check=10,
            patience=2,
            tau_explore=10,
            subspace_k=3,
            stagnation_threshold=1e-4
        )
        loss_frte = []
        for b_idx in range(n_batches):
            data = batches[b_idx]
            flat_data = data.view(data.size(0), -1)
            opt_frte.zero_grad()
            recon = m_frte(data)
            l = loss_fn(recon, flat_data)
            l.backward()
            val = l.item()
            loss_frte.append(val)

            opt_frte.monitor_and_archive(
                b_idx, val,
                allow_archive=(b_idx == 0 or val < opt_frte.best_loss),
                ancestor_strategy="best"
            )
            opt_frte.step()

        with torch.no_grad():
            test_rec_frte = m_frte(test_batch)
            test_mse_frte = loss_fn(test_rec_frte, flat_test).item()
        seed_res["FRTE"] = {
            "train_loss": loss_frte,
            "final_train_loss": loss_frte[-1],
            "test_mse": test_mse_frte,
            "recon": test_rec_frte.cpu().numpy(),
            "backtracks": opt_frte.total_backtracks
        }
        frte_backtracks.append(opt_frte.total_backtracks)

        for m in methods:
            all_seed_results[m]["test_mses"].append(seed_res[m]["test_mse"])
            all_seed_results[m]["final_losses"].append(seed_res[m]["final_train_loss"])
            all_seed_results[m]["train_loss_hists"].append(seed_res[m]["train_loss"])

        if seed == seeds[0]:
            primary_seed_results = seed_res

    # Compile statistics
    summary_stats = {}
    for m in methods:
        mses = np.array(all_seed_results[m]["test_mses"])
        losses = np.array(all_seed_results[m]["final_losses"])
        loss_arr = np.array(all_seed_results[m]["train_loss_hists"])
        summary_stats[m] = {
            "mean_test_mse": float(np.mean(mses)),
            "std_test_mse": float(np.std(mses)),
            "mean_train_loss": float(np.mean(losses)),
            "std_train_loss": float(np.std(losses)),
            "mean_train_hist": np.mean(loss_arr, axis=0),
            "std_train_hist": np.std(loss_arr, axis=0)
        }

    suite_output = {
        "stats": summary_stats,
        "primary_seed_results": primary_seed_results,
        "frte_mean_backtracks": float(np.mean(frte_backtracks))
    }
    return suite_output, flat_test.cpu().numpy()


def plot_deep_autoencoder(suite_output: Dict[str, Any], ground_truth: np.ndarray, n_batches: int, save_path: str):
    """
    Renders comprehensive visualization with multi-seed stats:
    - Top row: Visual comparison of original Fashion-MNIST test samples vs reconstructions
      from Adam, SAM, and FRTE (demonstrating model output clarity).
    - Bottom row: Multi-seed training loss convergence curves (with std shading) and test MSE bars.
    """
    primary_res = suite_output["primary_seed_results"]
    stats = suite_output["stats"]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1.0])

    sample_indices = [0, 4, 8, 12, 16, 20]
    for col_idx, s_idx in enumerate(sample_indices):
        orig_img = ground_truth[s_idx].reshape(28, 28)
        adam_img = primary_res["Adam"]["recon"][s_idx].reshape(28, 28)
        sam_img = primary_res["SAM"]["recon"][s_idx].reshape(28, 28)
        frte_img = primary_res["FRTE"]["recon"][s_idx].reshape(28, 28)

        composite = np.vstack([
            orig_img,
            np.ones((2, 28)) * 0.5,
            adam_img,
            np.ones((2, 28)) * 0.5,
            sam_img,
            np.ones((2, 28)) * 0.5,
            frte_img
        ])

        ax_img = fig.add_subplot(gs[0, col_idx])
        ax_img.imshow(composite, cmap="inferno")
        ax_img.axis("off")
        ax_img.set_title(f"Item #{col_idx + 1}", fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax_img.text(-6, 14, "Original", fontsize=8.5, ha="right", va="center", fontweight="bold")
            ax_img.text(-6, 44, "Adam", fontsize=8.5, ha="right", va="center", color="red", fontweight="bold")
            ax_img.text(-6, 74, "SAM", fontsize=8.5, ha="right", va="center", color="green", fontweight="bold")
            ax_img.text(-6, 104, "FRTE", fontsize=8.5, ha="right", va="center", color="teal", fontweight="bold")

    # Bottom Left: Training Loss Convergence with shaded std
    ax_loss = fig.add_subplot(gs[1, :4])
    steps = np.arange(1, n_batches + 1)
    ax_loss.plot(steps, stats["Adam"]["mean_train_hist"], "r--",
                 label=f"Adam ({stats['Adam']['mean_train_loss']:.4f} $\\pm$ {stats['Adam']['std_train_loss']:.4f})")
    ax_loss.plot(steps, stats["AdamW"]["mean_train_hist"], "m:",
                 label=f"AdamW ({stats['AdamW']['mean_train_loss']:.4f} $\\pm$ {stats['AdamW']['std_train_loss']:.4f})")
    ax_loss.plot(steps, stats["SGD-M"]["mean_train_hist"], "orange", linestyle="-.",
                 label=f"SGD-M ({stats['SGD-M']['mean_train_loss']:.4f} $\\pm$ {stats['SGD-M']['std_train_loss']:.4f})")
    ax_loss.plot(steps, stats["SAM"]["mean_train_hist"][:n_batches], "g--",
                 label=f"SAM ({stats['SAM']['mean_train_loss']:.4f} $\\pm$ {stats['SAM']['std_train_loss']:.4f})")
    ax_loss.plot(steps, stats["FRTE"]["mean_train_hist"], "teal", linewidth=2.5,
                 label=f"FRTE ({stats['FRTE']['mean_train_loss']:.4f} $\\pm$ {stats['FRTE']['std_train_loss']:.4f}, {suite_output['frte_mean_backtracks']:.1f} Backtracks)")
    ax_loss.fill_between(steps,
                         stats["FRTE"]["mean_train_hist"] - stats["FRTE"]["std_train_hist"],
                         stats["FRTE"]["mean_train_hist"] + stats["FRTE"]["std_train_hist"],
                         color="teal", alpha=0.18)

    ax_loss.set_xlabel("Gradient Iterations (Matched Budget: 400)", fontsize=11)
    ax_loss.set_ylabel("Reconstruction MSE Loss", fontsize=11)
    ax_loss.set_title("Training Loss Convergence on Fashion-MNIST (1.4M Parameters, Multi-Seed)", fontsize=11, fontweight="bold")
    ax_loss.grid(True, ls="--", alpha=0.35)
    ax_loss.legend(loc="upper right", framealpha=0.92, fontsize=9.2)

    # Bottom Right: Test MSE Bar Chart with error bars
    ax_bar = fig.add_subplot(gs[1, 4:])
    opts = ["SGD-M", "SAM", "Adam", "AdamW", "FRTE"]
    mean_mses = [stats[m]["mean_test_mse"] for m in opts]
    std_mses = [stats[m]["std_test_mse"] for m in opts]
    colors = ["orange", "green", "red", "magenta", "teal"]

    bars = ax_bar.bar(opts, mean_mses, yerr=std_mses, capsize=4, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_bar.set_ylabel("Test Reconstruction MSE", fontsize=11)
    ax_bar.set_title("Generalization Test Error ($\\mu \\pm \\sigma$)", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="y", ls="--", alpha=0.35)
    for bar, m_val, s_val in zip(bars, mean_mses, std_mses):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, m_val + s_val + 0.002, f"{m_val:.4f}",
                    ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.suptitle("Benchmark 4: Deep Autoencoder (Hinton 784-500-250-30 Architecture) Model Visualizations", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
