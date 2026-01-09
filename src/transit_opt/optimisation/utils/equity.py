"""
Equity metrics for spatial objectives.

Provides Atkinson Index and related inequality measures for
transit accessibility analysis.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_atkinson_index(
    waiting_times: np.ndarray,
    weights: np.ndarray | None = None,
    epsilon: float = 2.0,
    min_waiting_time: float = 1.0,
) -> float:
    """
    Calculate Atkinson Index from waiting times.

    Waiting times are inverted to accessibility scores (1/w) since
    Atkinson Index is designed for "goods" where more is better.

    Args:
        waiting_times: Array of waiting times per zone (minutes)
        weights: Population/demand weights per zone (None = equal weights)
        epsilon: Inequality aversion parameter (0=neutral, higher=more aversion)
                 Recommended: epsilon=2 focuses heavily on worst-off zones
        min_waiting_time: Minimum waiting time to avoid infinity (default 1.0 min)

    Returns:
        Atkinson Index value in [0, 1]. Lower = more equitable.
    """
    if len(waiting_times) == 0:
        return 0.0

    # Cap waiting times to avoid division by zero
    waiting_times_capped = np.maximum(waiting_times, min_waiting_time)

    # Handle finite check
    if not np.all(np.isfinite(waiting_times_capped)):
        return 1.0  # Max inequality if infinite waiting times exist

    # Convert to accessibility (more is better)
    accessibility = 1.0 / waiting_times_capped

    return _calculate_weighted_atkinson(accessibility, weights, epsilon)


def calculate_atkinson_index_from_vehicles(
    vehicles_per_zone: np.ndarray,
    weights: np.ndarray | None = None,
    epsilon: float = 2.0,
    min_vehicles: float = 0.1,
) -> float:
    """
    Calculate Atkinson Index directly from vehicle counts.

    For StopCoverageObjective where vehicles = accessibility (more is better).
    No inversion needed.

    Args:
        vehicles_per_zone: Array of vehicle counts per zone
        weights: Population/demand weights per zone
        epsilon: Inequality aversion parameter
        min_vehicles: Minimum vehicle count to avoid log(0)

    Returns:
        Atkinson Index value in [0, 1]
    """
    if len(vehicles_per_zone) == 0:
        return 0.0

    # Cap to minimum (vehicles are already "goods")
    vehicles_capped = np.maximum(vehicles_per_zone, min_vehicles)

    return _calculate_weighted_atkinson(vehicles_capped, weights, epsilon)


def _calculate_weighted_atkinson(
    values: np.ndarray,
    weights: np.ndarray | None,
    epsilon: float,
) -> float:
    """
    Internal calculation of weighted Atkinson Index.

    Formula: A = 1 - (x_ede / mu)

    Where:
      mu = arithmetic mean
      x_ede = general mean of order (1-epsilon)
    """
    # 1. Handle Weights
    if weights is None:
        p = np.ones(len(values))
    else:
        p = np.array(weights)
        # Ensure weights are non-negative
        p = np.maximum(p, 0.0)

    total_pop = np.sum(p)
    if total_pop <= 0:
        logger.warning("Total weight is 0, returning 0 inequality")
        return 0.0

    # Normalize weights
    w = p / total_pop

    # 2. Calculate Weighted Mean (mu)
    mu = np.sum(w * values)

    if mu <= 0:
        return 1.0  # If mean benefit is 0, inequality is max (or undefined)

    # 3. Calculate EDE (Equally Distributed Equivalent)
    if np.isclose(epsilon, 1.0):
        # Geometric mean case (epsilon = 1)
        # EDE = exp( sum(w_i * ln(x_i)) )
        # Use simple clipping to avoid log(0) although input should be capped
        log_vals = np.log(np.maximum(values, 1e-9))
        ede = np.exp(np.sum(w * log_vals))
    else:
        # General case (epsilon != 1)
        # EDE = [ sum(w_i * x_i^(1-e)) ] ^ (1 / (1-e))
        power = 1.0 - epsilon

        # Calculate generalized mean
        # Check for potential overflow with large negative powers
        val_pow = np.power(values, power)
        mean_pow = np.sum(w * val_pow)

        # Handle negative mean_pow possibilities if math goes weird (shouldn't for positive inputs)
        if mean_pow <= 0:
            return 1.0

        ede = np.power(mean_pow, 1.0 / power)

    # 4. Calculate Index
    atkinson = 1.0 - (ede / mu)

    # Clip for safety (floating point errors can cause slight <0 or >1)
    return float(np.clip(atkinson, 0.0, 1.0))
