"""
Grassmann Manifold Algebra Module for FRTE.

Provides:
- compute_grassmann_subspace: Computes the principal orthonormal basis Q in Gr(k, D)
  via the Dual-Gram method in O(w^2 D) time and O(w^2) matrix memory (w << D).
- project_gradient_orthogonal: Projects gradient g onto the orthogonal complement of Q:
  g_perp = (I - Q Q^T) g in O(k D) time.
"""

from typing import List, Optional
import torch


def compute_grassmann_subspace(
    X_displacements: torch.Tensor,
    k: int = 4,
    eps: float = 1e-7
) -> Optional[torch.Tensor]:
    """
    Computes orthonormal basis Q in R^{D x k_eff} spanning the dominant subspace
    of failed trajectory displacements on the Grassmannian manifold Gr(k, D).

    Algorithm (Dual-Gram SVD):
      Instead of SVD on (D, w) matrix in O(D w^2) or O(D^2 w),
      we construct the Gram matrix K = X^T X in R^{w x w}.
      Spectral decomposition of K yields eigenvalues Lambda and eigenvectors V in R^{w x w} (O(w^3)).
      Then principal basis is recovered via Q = X V Lambda^{-1/2} in O(D w k).
      Overall complexity: O(w^2 D) with w << D (typically w in [5, 20], D in [1e4, 1e8]).

    Args:
        X_displacements: Tensor of shape (D, w) where w is trajectory window length
        k: Maximum subspace rank to block
        eps: Small eigenvalue threshold to discard near-zero directions

    Returns:
        Q: Orthonormal basis matrix of shape (D, k_eff) such that Q^T Q = I_{k_eff},
           or None if displacements carry zero energy.
    """
    if X_displacements.ndim != 2:
        raise ValueError(f"Expected 2D displacement matrix, got shape {X_displacements.shape}")

    D, w = X_displacements.shape
    if w == 0:
        return None

    # Enforce subspace dimension ceiling
    k_target = min(k, w, max(1, D - 1))

    # Compute Dual Gram matrix K = X^T X in R^{w x w}
    K = torch.matmul(X_displacements.T, X_displacements)

    # Compute spectral decomposition of small w x w Gram matrix
    try:
        eigenvalues, V = torch.linalg.eigh(K)
    except Exception:
        # Fallback to CPU if device linalg encounters issues
        eigenvalues, V = torch.linalg.eigh(K.cpu())
        eigenvalues = eigenvalues.to(K.device)
        V = V.to(K.device)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)[:k_target]
    top_eigenvalues = eigenvalues[idx]
    top_V = V[:, idx]

    # Filter numerical zero energy
    valid_mask = top_eigenvalues > eps
    if not torch.any(valid_mask):
        return None

    top_eigenvalues = top_eigenvalues[valid_mask]
    top_V = top_V[:, valid_mask]

    # Principal basis Q = X V Lambda^{-1/2} in R^{D x k_eff}
    inv_sigmas = 1.0 / torch.sqrt(top_eigenvalues)
    Q = torch.matmul(X_displacements, top_V) * inv_sigmas.unsqueeze(0)

    return Q


def project_gradient_orthogonal(
    params: List[torch.nn.Parameter],
    Q_basis: Optional[torch.Tensor],
    preserve_norm: bool = True
) -> None:
    """
    Projects flat gradient vector g onto the orthogonal complement of Q:
      g_perp = (I - Q Q^T) g in O(k D) time.

    Optionally rescales g_perp so that ||g_perp||_2 == ||g||_2, ensuring
    unconstrained progress along transverse exploratory directions.
    """
    if Q_basis is None:
        return

    device = Q_basis.device
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.data.view(-1).to(device))
        else:
            grads.append(torch.zeros_like(p.data).view(-1).to(device))

    flat_g = torch.cat(grads)
    orig_norm = torch.linalg.norm(flat_g)
    if orig_norm < 1e-12:
        return

    # alpha = Q^T g in R^{k_eff} (O(k D))
    alpha = torch.matmul(Q_basis.T, flat_g)
    # g_perp = g - Q alpha in R^D (O(k D))
    g_perp = flat_g - torch.matmul(Q_basis, alpha)
    perp_norm = torch.linalg.norm(g_perp)

    if preserve_norm and perp_norm > 1e-7:
        g_perp = g_perp * (orig_norm / perp_norm)

    # Reassign projected gradients back to parameters
    offset = 0
    for p in params:
        numel = p.numel()
        slice_g = g_perp[offset : offset + numel].view_as(p.data).to(p.data.device)
        if p.grad is None:
            p.grad = slice_g.clone()
        else:
            p.grad.data.copy_(slice_g)
        offset += numel
