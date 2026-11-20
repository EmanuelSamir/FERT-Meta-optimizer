"""
Unit Tests for FRTE Core Components:
1. Dual-Gram Grassmannian Subspace decomposition
2. Orthogonal gradient projection and invariance
3. Functional Archive prioritized admission and retrieval
4. FRTE step, rollback, and momentum purge
"""

import unittest
import torch
import torch.nn as nn
from frte.grassmann import compute_grassmann_subspace, project_gradient_orthogonal
from frte.archive import FunctionalArchive
from frte.optimizer import FRTE


class SimpleNet(nn.Module):
    def __init__(self, in_features=10, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestFRTE(unittest.TestCase):
    def test_dual_gram_grassmann_orthonormality(self):
        """Verify that Dual-Gram SVD produces an orthonormal basis Q^T Q = I."""
        torch.manual_seed(42)
        D = 1000
        w = 10
        k = 4
        X = torch.randn(D, w)

        Q = compute_grassmann_subspace(X, k=k)
        self.assertIsNotNone(Q)
        self.assertEqual(Q.shape, (D, k))

        # Check Q^T Q == I_k
        QtQ = torch.matmul(Q.T, Q)
        I_k = torch.eye(k)
        self.assertTrue(torch.allclose(QtQ, I_k, atol=1e-5), f"Q is not orthonormal: {QtQ}")

    def test_gradient_orthogonal_projection(self):
        """Verify that g_perp is orthogonal to all columns in Q: Q^T g_perp = 0."""
        torch.manual_seed(42)
        model = SimpleNet(in_features=10, out_features=2)
        params = list(model.parameters())

        # Set dummy gradients
        for p in params:
            p.grad = torch.randn_like(p.data)

        D = sum(p.numel() for p in params)
        k = 3
        # Synthetic Q basis
        rand_X = torch.randn(D, 8)
        Q = compute_grassmann_subspace(rand_X, k=k)

        project_gradient_orthogonal(params, Q, preserve_norm=True)

        # Flatten projected gradients
        flat_g_perp = torch.cat([p.grad.data.view(-1) for p in params])
        overlap = torch.matmul(Q.T, flat_g_perp)

        self.assertTrue(torch.allclose(overlap, torch.zeros(k), atol=1e-5), f"Gradients not orthogonal: {overlap}")

    def test_functional_archive_priority(self):
        """Verify prioritized admission and anchor selection."""
        archive = FunctionalArchive(max_size=3)
        p = [torch.tensor([1.0, 2.0])]

        archive.add(p, loss=1.0, step=10)
        archive.add(p, loss=0.5, step=20)
        archive.add(p, loss=2.0, step=30)
        archive.add(p, loss=0.1, step=40)

        # Should retain best 3 (loss 0.1, 0.5, 1.0) and drop loss 2.0
        self.assertEqual(len(archive), 3)
        best_node = archive.select_ancestor(strategy="best")
        self.assertEqual(best_node.step, 40)
        self.assertEqual(best_node.loss, 0.1)

        last_node = archive.select_ancestor(strategy="last")
        self.assertIsNotNone(last_node)

    def test_frte_step_and_intervention(self):
        """Verify FRTE optimization step, tracking, and trigger_first_return."""
        torch.manual_seed(42)
        model = SimpleNet(in_features=5, out_features=1)
        optimizer = FRTE(
            model.parameters(),
            base_optimizer_cls=torch.optim.Adam,
            lr=0.01,
            k_check=5,
            patience=1,
            tau_explore=5
        )

        x = torch.randn(4, 5)
        y = torch.randn(4, 1)

        # Step 0: Save baseline
        loss_0 = nn.functional.mse_loss(model(x), y)
        optimizer.monitor_and_archive(0, loss_0.item())
        initial_w = [p.clone() for p in model.parameters()]

        # Run a few steps
        for step in range(1, 6):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.monitor_and_archive(step, loss.item())

        # Trigger First Return explicitly
        current_w = [p.clone() for p in model.parameters()]
        diff_before = sum((w1 - w2).abs().sum().item() for w1, w2 in zip(initial_w, current_w))
        self.assertGreater(diff_before, 1e-4)

        triggered = optimizer.trigger_first_return(6, 10.0, strategy="root")
        self.assertTrue(triggered)
        self.assertEqual(optimizer.explore_steps_left, 5)

        # Weights should be restored to initial_w
        restored_w = [p.clone() for p in model.parameters()]
        for w_init, w_rest in zip(initial_w, restored_w):
            self.assertTrue(torch.allclose(w_init, w_rest, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

