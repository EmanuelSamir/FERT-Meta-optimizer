"""
Functional Archive Module for FRTE.

Maintains a bounded set of healthy, high-performing checkpoints (the 'Return' targets).
Crucially, checkpoints are purely structural/stateless (weights theta, objective score)
and intentionally do NOT preserve optimizer momentum states (m_t, v_t), ensuring
clean Markov chain rupture upon First Return.
"""

import math
from typing import List, Optional
import torch


class ArchiveNode:
    """Lightweight stateless checkpoint."""
    def __init__(self, weights: List[torch.Tensor], loss: float, step: int, score: float):
        self.weights = [w.detach().clone() for w in weights]
        self.loss = float(loss)
        self.step = int(step)
        self.score = float(score)
        self.visits = 0


class FunctionalArchive:
    """Bounded, prioritized archive of parameter checkpoints."""
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.nodes: List[ArchiveNode] = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add(self, weights: List[torch.Tensor], loss: float, step: int) -> bool:
        """
        Admit checkpoint if viable. Score is higher for lower loss.
        """
        if math.isnan(loss) or math.isinf(loss):
            return False

        score = -loss
        if len(self.nodes) == 0 or score > (max(n.score for n in self.nodes) - 1.0):
            node = ArchiveNode(weights, loss, step, score)
            self.nodes.append(node)
            if len(self.nodes) > self.max_size:
                self.nodes.sort(key=lambda n: n.score, reverse=True)
                self.nodes = self.nodes[:self.max_size]
            return True
        return False

    def select_ancestor(self, strategy: str = "last") -> Optional[ArchiveNode]:
        """
        Selects an ancestor node according to selection strategy:
        - 'last': most recently saved healthy checkpoint (local branching).
        - 'best': globally highest scoring checkpoint penalizing over-visited nodes.
        - 'root': initial viable configuration.
        """
        if not self.nodes:
            return None

        if strategy == "root":
            node = min(self.nodes, key=lambda n: n.step)
        elif strategy == "best":
            node = max(self.nodes, key=lambda n: n.score - 0.2 * n.visits)
        else:  # default "last"
            node = self.nodes[-1]

        node.visits += 1
        return node

    def clear(self):
        self.nodes.clear()
