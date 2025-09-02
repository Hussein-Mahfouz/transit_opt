"""
Configuration management for transit optimization.

This module provides comprehensive configuration management for PSO-based
transit optimization, extending the existing constraint configuration system
with algorithm parameters, monitoring options, and multi-run capabilities.
"""

from .config_manager import (
    MonitoringConfig,
    MultiRunConfig,
    OptimizationConfigManager,
    PSOConfig,
    TerminationConfig,
)

# from .yaml_schemas import validate_optimization_config

__all__ = [
    "PSOConfig",
    "TerminationConfig",
    "MonitoringConfig",
    "MultiRunConfig",
    "OptimizationConfigManager",
    "validate_optimization_config",
]
