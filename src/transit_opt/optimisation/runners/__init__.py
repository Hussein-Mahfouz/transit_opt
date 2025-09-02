"""
Production optimization runners for transit optimization.

This module provides high-level optimization runners that integrate
the configuration system with optimization algorithms and problem
definitions to provide a complete optimization solution.
"""

from .pso_runner import PSORunner

__all__ = [
    'PSORunner'
]
