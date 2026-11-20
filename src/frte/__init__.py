"""
FRTE: First Return, Then Explore.
A Non-Markovian Meta-Optimizer for Deep Learning and Non-Convex Optimization.
"""

from frte.optimizer import FRTE
from frte.archive import FunctionalArchive, ArchiveNode
from frte.grassmann import compute_grassmann_subspace, project_gradient_orthogonal

__all__ = [
    "FRTE",
    "FunctionalArchive",
    "ArchiveNode",
    "compute_grassmann_subspace",
    "project_gradient_orthogonal"
]

__version__ = "0.1.0"
