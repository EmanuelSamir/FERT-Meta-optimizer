"""
Master Experiment Runner for FRTE Evaluation Suite:
Executes:
  1. Benchmark 1: Deceptive 2D Landscape (Multi-seed: 5 seeds, mu +/- sigma)
  2. Benchmark 2 (formerly 3): Stiff Reaction-Diffusion PINN (Multi-seed: 5 seeds, mu +/- sigma)
  3. Benchmark 3 (formerly 4): Deep Bottleneck Autoencoder (Fashion-MNIST, 400 steps, Multi-seed)

Saves all figures into reports/figures/ and produces a consolidated metrics summary.
"""

import os
import sys

# Ensure frte_project has strict top precedence over current working directory
PROJECT_ROOT = "/private/tmp/frte_project"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path = [PROJECT_ROOT, SRC_ROOT] + [p for p in sys.path if p not in ("", "/private/tmp/simple_experiment")]

import time
import json
import torch
import numpy as np

from benchmarks.deceptive_landscape import (
    run_deceptive_landscape_suite,
    plot_deceptive_landscape
)
from benchmarks.pinn import (
    run_pinn_suite,
    plot_pinn_results
)
from benchmarks.deep_autoencoder import (
    run_deep_autoencoder_suite,
    plot_deep_autoencoder
)
from benchmarks.modular_addition import (
    run_modular_addition_suite,
    plot_modular_addition
)


def run_full_suite():
    os.makedirs("/private/tmp/frte_project/reports/figures", exist_ok=True)
    summary = {}

    SEEDS = [42, 123, 456, 789, 1024]

    print("================================================================")
    print("STARTING FRTE COMPREHENSIVE MULTI-SEED BENCHMARK SUITE")
    print(f"Random Seeds: {SEEDS}")
    print("================================================================")

    # ------------------------------------------------------------------
    # 1. Deceptive Landscape (5 seeds)
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark 1: Deceptive 2D Landscape (5 Seeds) ---")
    t0 = time.time()
    res1 = run_deceptive_landscape_suite(budget=150, seeds=SEEDS)
    fig1_path = "/private/tmp/frte_project/reports/figures/exp1_deceptive_landscape.png"
    plot_deceptive_landscape(res1, fig1_path)
    dur1 = time.time() - t0
    print(f"Done in {dur1:.2f}s! Figure saved to {fig1_path}")
    st1 = res1["stats"]
    for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]:
        print(f"  {m:6s}: Mean Loss = {st1[m]['mean_loss']:7.3f} +/- {st1[m]['std_loss']:.3f}")
    print(f"  FRTE Mean Backtracks: {res1['frte_mean_backtracks']:.1f}")

    summary["exp1_deceptive"] = {
        m: {
            "mean_loss": st1[m]["mean_loss"],
            "std_loss": st1[m]["std_loss"]
        } for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]
    }
    summary["exp1_deceptive"]["frte_mean_backtracks"] = res1["frte_mean_backtracks"]

    # ------------------------------------------------------------------
    # 2. Physics-Informed Neural Network (PINN) (5 seeds)
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark 2: PINN Stiff Reaction-Diffusion (5 Seeds) ---")
    t0 = time.time()
    res2, x_test_pinn, u_exact_pinn = run_pinn_suite(budget=300, seeds=SEEDS)
    fig2_path = "/private/tmp/frte_project/reports/figures/exp2_pinn.png"
    plot_pinn_results(res2, x_test_pinn, u_exact_pinn, fig2_path)
    dur2 = time.time() - t0
    print(f"Done in {dur2:.2f}s! Figure saved to {fig2_path}")
    st2 = res2["stats"]
    for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]:
        print(f"  {m:6s}: Loss = {st2[m]['mean_loss']:7.2f} +/- {st2[m]['std_loss']:6.2f} | L2 Error = {st2[m]['mean_l2']:.6f} +/- {st2[m]['std_l2']:.6f}")
    print(f"  FRTE Mean Backtracks: {res2['frte_mean_backtracks']:.1f}")

    summary["exp2_pinn"] = {
        m: {
            "mean_loss": st2[m]["mean_loss"],
            "std_loss": st2[m]["std_loss"],
            "mean_l2": st2[m]["mean_l2"],
            "std_l2": st2[m]["std_l2"]
        } for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]
    }
    summary["exp2_pinn"]["frte_mean_backtracks"] = res2["frte_mean_backtracks"]

    # ------------------------------------------------------------------
    # 3. Deep Autoencoder (Fashion-MNIST, 400 steps, 3 seeds)
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark 3: Deep Autoencoder Bottleneck (Fashion-MNIST, 400 steps, 3 Seeds) ---")
    t0 = time.time()
    res3, ground_truth_test = run_deep_autoencoder_suite(n_batches=400, batch_size=128, device="cpu", seeds=[42, 123, 456])
    fig3_path = "/private/tmp/frte_project/reports/figures/exp3_deep_autoencoder.png"
    plot_deep_autoencoder(res3, ground_truth_test, n_batches=400, save_path=fig3_path)
    dur3 = time.time() - t0
    print(f"Done in {dur3:.2f}s! Figure saved to {fig3_path}")
    st3 = res3["stats"]
    for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]:
        print(f"  {m:6s}: Test MSE = {st3[m]['mean_test_mse']:.5f} +/- {st3[m]['std_test_mse']:.5f}")
    print(f"  FRTE Mean Backtracks: {res3['frte_mean_backtracks']:.1f}")

    summary["exp3_autoencoder"] = {
        m: {
            "mean_test_mse": st3[m]["mean_test_mse"],
            "std_test_mse": st3[m]["std_test_mse"],
            "mean_train_loss": st3[m]["mean_train_loss"],
            "std_train_loss": st3[m]["std_train_loss"]
        } for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]
    }
    summary["exp3_autoencoder"]["frte_mean_backtracks"] = res3["frte_mean_backtracks"]

    # ------------------------------------------------------------------
    # 4. Modular Addition (a + b) mod 97 (5 seeds)
    # ------------------------------------------------------------------
    print("\n--- Running Benchmark 4: Modular Addition (a + b) mod 97 (5 Seeds) ---")
    t0 = time.time()
    res4, eval_data4 = run_modular_addition_suite(p=97, budget=350, seeds=SEEDS)
    fig4_path = "/private/tmp/frte_project/reports/figures/exp4_modular_addition.png"
    plot_modular_addition(res4, eval_data4, budget=350, save_path=fig4_path)
    dur4 = time.time() - t0
    print(f"Done in {dur4:.2f}s! Figure saved to {fig4_path}")
    st4 = res4["stats"]
    for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]:
        print(f"  {m:6s}: Val Loss = {st4[m]['mean_val_loss']:.3f} +/- {st4[m]['std_val_loss']:.3f} | Train Acc = {st4[m]['mean_train_acc']*100:.1f}% | Val Acc = {st4[m]['mean_val_acc']*100:.1f}%")
    print(f"  FRTE Mean Backtracks: {res4['frte_mean_backtracks']:.1f}")

    summary["exp4_modular_addition"] = {
        m: {
            "mean_train_loss": st4[m]["mean_train_loss"],
            "std_train_loss": st4[m]["std_train_loss"],
            "mean_val_loss": st4[m]["mean_val_loss"],
            "std_val_loss": st4[m]["std_val_loss"],
            "mean_train_acc": st4[m]["mean_train_acc"],
            "std_train_acc": st4[m]["std_train_acc"],
            "mean_val_acc": st4[m]["mean_val_acc"],
            "std_val_acc": st4[m]["std_val_acc"]
        } for m in ["Adam", "AdamW", "SGD-M", "SAM", "FRTE"]
    }
    summary["exp4_modular_addition"]["frte_mean_backtracks"] = res4["frte_mean_backtracks"]

    with open("/private/tmp/frte_project/reports/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY WITH MULTI-SEED STATISTICS!")
    print("Metrics written to: /private/tmp/frte_project/reports/benchmark_summary.json")
    print("================================================================")


if __name__ == "__main__":
    run_full_suite()
