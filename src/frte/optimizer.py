"""
FRTE: First Return, Then Explore Meta-Optimizer.

A non-Markovian meta-optimizer that wraps any standard PyTorch gradient optimizer
(e.g., Adam, AdamW, SGD-Momentum). It monitors the trajectory for failure modes
(stagnation, loss spike, or saddle entrapment), executes a First Return operator
(rolling back weights to an archive anchor and purging corrupt optimizer momentum),
extracts the Grassmannian subspace of the failed descent corridor, and enforces a
Then Explore operator by projecting gradients onto the orthogonal complement.
"""

from typing import Type, List, Optional, Tuple, Any
import math
import torch
from frte.grassmann import compute_grassmann_subspace, project_gradient_orthogonal
from frte.archive import FunctionalArchive


class FRTE(torch.optim.Optimizer):
    """
    FRTE (First Return, Then Explore) Meta-Optimizer.

    Wraps a base optimizer and implements:
      1. Non-Markovian state archiving (theta_anc, loss_anc).
      2. Trajectory history buffering in a sliding window w.
      3. First Return Operator (R): Stateless rollback to theta_anc and clean momentum wipe.
      4. Then Explore Operator (E): Grassmannian subspace projection g_perp = (I - Q Q^T) g
         for tau_explore steps, where Q spans the principal directions of failed descent.
    """
    def __init__(
        self,
        params,
        base_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        k_check: int = 10,
        patience: int = 2,
        tau_explore: int = 20,
        subspace_k: int = 4,
        stagnation_threshold: float = 1e-4,
        divergence_ratio: float = 1.5,
        archive_max_size: int = 50,
        **base_optimizer_kwargs
    ):
        self.param_groups_source = list(params)
        # Extract flat list of parameters
        if len(self.param_groups_source) > 0 and isinstance(self.param_groups_source[0], dict):
            self.param_list: List[torch.nn.Parameter] = []
            for g in self.param_groups_source:
                self.param_list.extend(g["params"])
        else:
            self.param_list: List[torch.nn.Parameter] = list(self.param_groups_source)

        self.lr = lr
        self.k_check = k_check
        self.patience = patience
        self.tau_explore = tau_explore
        self.subspace_k = subspace_k
        self.stagnation_threshold = stagnation_threshold
        self.divergence_ratio = divergence_ratio

        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = base_optimizer_kwargs

        # Initialize underlying base optimizer
        self.base_optimizer: torch.optim.Optimizer = self.base_optimizer_cls(
            self.param_groups_source, lr=lr, **base_optimizer_kwargs
        )

        # Non-Markovian Archive & Trajectory tracking
        self.archive = FunctionalArchive(max_size=archive_max_size)
        self.recent_trajectory: List[torch.Tensor] = []
        self.max_recent_history: int = k_check * 2

        # Exploration state
        self.explore_steps_left: int = 0
        self.Q_basis: Optional[torch.Tensor] = None

        # Health monitoring
        self.best_loss: float = float("inf")
        self.stagnant_checks: int = 0
        self.total_backtracks: int = 0

        # Detailed logging of intervention events
        self.rewind_events: List[Tuple[int, int, Any, Any]] = []

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _flatten_params(self, params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if params is None:
            params = [p.data for p in self.param_list]
        return torch.cat([p.detach().view(-1) for p in params])

    def _unflatten_to_params(self, flat_weights: torch.Tensor):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.data.copy_(flat_weights[offset : offset + numel].view_as(p.data))
            offset += numel

    def step(self, closure=None):
        """
        Performs a single optimization step.
        If in explore phase, orthogonally projects gradients to prevent re-entering
        the failed subspace before calling the base optimizer.
        """
        if self.explore_steps_left > 0 and self.Q_basis is not None:
            project_gradient_orthogonal(self.param_list, self.Q_basis, preserve_norm=True)
            self.explore_steps_left -= 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.base_optimizer.step()

        # Record trajectory snapshot for displacement matrix construction
        curr_flat = self._flatten_params()
        self.recent_trajectory.append(curr_flat)
        if len(self.recent_trajectory) > self.max_recent_history:
            self.recent_trajectory.pop(0)

        return loss

    def monitor_and_archive(
        self,
        step: int,
        loss_val: float,
        force_intervention: Optional[bool] = None,
        allow_archive: bool = True,
        ancestor_strategy: str = "last"
    ) -> bool:
        """
        Monitors health at regular intervals (k_check), updates the functional archive,
        and triggers First Return upon detecting stagnation or loss divergence.

        Args:
            step: Current iteration index
            loss_val: Current objective value
            force_intervention: Explicit trigger (True=force backtrack, False=skip)
            allow_archive: Whether to admit viable configurations into archive
            ancestor_strategy: 'last', 'best', or 'root' anchor selection

        Returns:
            True if First Return was triggered; False otherwise.
        """
        loss_val = float(loss_val)

        # 1. Archive viable configuration
        if allow_archive and not (math.isnan(loss_val) or math.isinf(loss_val)):
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.archive.add([p.data for p in self.param_list], loss_val, step)
            elif (len(self.archive) == 0) or (step % self.k_check == 0 and loss_val < self.best_loss * 1.2):
                self.archive.add([p.data for p in self.param_list], loss_val, step)

        # 2. Check for failure conditions
        if force_intervention is True:
            return self.trigger_first_return(step, loss_val, strategy=ancestor_strategy)

        if force_intervention is False:
            return False

        if step > 0 and step % self.k_check == 0:
            is_divergent = (loss_val > self.best_loss * self.divergence_ratio) and (self.best_loss < 1e5)
            progress = self.best_loss - loss_val
            is_stagnant = (abs(progress) < self.stagnation_threshold) and (loss_val > 1e-4)

            if is_divergent:
                return self.trigger_first_return(step, loss_val, strategy=ancestor_strategy)
            elif is_stagnant:
                self.stagnant_checks += 1
                if self.stagnant_checks >= self.patience:
                    return self.trigger_first_return(step, loss_val, strategy=ancestor_strategy)
            else:
                self.stagnant_checks = 0

        return False

    def trigger_first_return(self, current_step: int, current_loss: float, strategy: str = "last") -> bool:
        """
        Executes the First Return operator and prepares Then Explore:
          1. Selects anchor checkpoint theta_anc from FunctionalArchive.
          2. Assembles displacement matrix X_fail = [theta_{t-i} - theta_anc].
          3. Solves Dual-Gram SVD to obtain Grassmannian basis Q in Gr(k, D).
          4. Rewinds model weights: theta_{t+1} <- theta_anc.
          5. Re-initializes base optimizer (wiping momentum/second moments).
          6. Activates orthogonal projection for tau_explore steps.
        """
        if len(self.archive) == 0:
            return False

        anchor_node = self.archive.select_ancestor(strategy=strategy)
        if anchor_node is None:
            return False

        anchor_flat = self._flatten_params(anchor_node.weights)
        curr_flat = self._flatten_params()

        # Build trajectory displacement matrix X_fail in R^{D x w}
        primary_disp = curr_flat - anchor_flat
        displacements = [primary_disp]
        for past_weights in self.recent_trajectory:
            diff = past_weights - anchor_flat
            if torch.linalg.norm(diff) > 1e-6:
                displacements.append(diff)

        # Stack displacements along columns
        X_fail = torch.stack(displacements, dim=1)
        # Weight primary exit corridor
        if primary_disp.norm() > 1e-6:
            X_fail[:, 0] = primary_disp * 2.0

        # Extract Grassmannian basis Q spanning failed corridor
        self.Q_basis = compute_grassmann_subspace(X_fail, k=self.subspace_k)

        # 4. First Return: Rollback weights
        self._unflatten_to_params(anchor_flat)

        # 5. Purge optimizer momentum (Rupture the Markov chain)
        self.base_optimizer = self.base_optimizer_cls(
            self.param_groups_source, lr=self.lr, **self.base_optimizer_kwargs
        )

        # 6. Then Explore: set active exploration counter
        self.explore_steps_left = self.tau_explore
        self.stagnant_checks = 0
        self.recent_trajectory.clear()
        self.total_backtracks += 1

        self.rewind_events.append((
            current_step,
            anchor_node.step,
            curr_flat.detach().cpu().numpy(),
            anchor_flat.detach().cpu().numpy()
        ))
        return True
