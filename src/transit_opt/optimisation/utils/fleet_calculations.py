"""
Shared fleet calculation utilities for consistent fleet analysis.

This module provides standardized fleet calculation methods used by both
GTFSDataPreparator and optimization constraint handlers to ensure consistency.
"""

from typing import Any

import numpy as np


def calculate_fleet_requirements(
    headways_matrix: np.ndarray,
    round_trip_times: np.ndarray,
    operational_buffer: float = 1.15,
    no_service_threshold: float = 480,
    allowed_headways: np.ndarray | None = None,
    no_service_index: int | None = None
) -> dict[str, Any]:
    """
    Unified fleet calculation for both baseline analysis and optimization constraints.
    
    This function implements the standardized fleet calculation logic used across
    the transit optimization system, ensuring consistency between baseline analysis
    (GTFSDataPreparator) and optimization constraints (BaseConstraintHandler).
    
    Args:
        headways_matrix: Matrix of headway values OR choice indices (n_routes × n_intervals)
        round_trip_times: Round-trip times per route (n_routes,)
        operational_buffer: Buffer factor for vehicle scheduling (default: 1.15)
        no_service_threshold: Headways above this are no-service in minutes (default: 480)
        allowed_headways: For decoding solution indices (optimization context only)
        no_service_index: Index representing no-service (optimization context only)
    
    Returns:
        Dictionary containing:
        - 'fleet_per_route': Peak fleet per route (n_routes,)
        - 'fleet_per_interval': Total fleet per interval (n_intervals,)
        - 'total_peak_fleet': Maximum fleet across all intervals
        - 'route_fleet_matrix': Fleet by route and interval (n_routes × n_intervals)
        - 'operational_buffer': Buffer factor used in calculations
    """
    n_routes, n_intervals = headways_matrix.shape

    # Validate inputs
    if len(round_trip_times) != n_routes:
        raise ValueError(f"round_trip_times length ({len(round_trip_times)}) must match n_routes ({n_routes})")

    # Initialize output arrays
    route_fleet_matrix = np.zeros((n_routes, n_intervals), dtype=int)
    fleet_per_interval = np.zeros(n_intervals, dtype=int)

    # Process each route-interval combination
    for route_idx in range(n_routes):
        round_trip_time = round_trip_times[route_idx]

        for interval_idx in range(n_intervals):
            headway_value = headways_matrix[route_idx, interval_idx]

            # Decode headway value based on context
            if allowed_headways is not None and no_service_index is not None:
                # Optimization context: decode choice index to headway value
                if isinstance(headway_value, (int, np.integer)) and headway_value < len(allowed_headways):
                    if headway_value == no_service_index:
                        actual_headway = np.inf  # No service
                    else:
                        actual_headway = allowed_headways[headway_value]
                else:
                    actual_headway = headway_value
            else:
                # GTFSDataPreparator context: headway_value is already the actual headway
                actual_headway = headway_value

            # Calculate vehicles needed using standardized logic
            if (not np.isnan(actual_headway) and
                not np.isinf(actual_headway) and
                actual_headway < no_service_threshold):
                # Valid service headway - apply same formula as GTFSDataPreparator
                vehicles_needed = np.ceil((round_trip_time * operational_buffer) / actual_headway)
                vehicles_needed = max(1, int(vehicles_needed))  # At least 1 vehicle
            else:
                # No service or invalid headway
                vehicles_needed = 0

            # Store results
            route_fleet_matrix[route_idx, interval_idx] = vehicles_needed
            fleet_per_interval[interval_idx] += vehicles_needed

    # Calculate per-route peak fleet (max across intervals for each route)
    fleet_per_route = np.max(route_fleet_matrix, axis=1)

    # Total system peak (realistic concurrent total, not sum of route peaks)
    total_peak_fleet = int(np.max(fleet_per_interval))

    return {
        'fleet_per_route': fleet_per_route.astype(int),
        'fleet_per_interval': fleet_per_interval.astype(int),
        'total_peak_fleet': total_peak_fleet,
        'route_fleet_matrix': route_fleet_matrix.astype(int),
        'operational_buffer': operational_buffer
    }


def get_operational_parameters(opt_data: dict[str, Any]) -> dict[str, float]:
    """Extract operational parameters from optimization data."""
    fleet_analysis = opt_data.get("constraints", {}).get("fleet_analysis", {})

    return {
        'operational_buffer': fleet_analysis.get('operational_buffer', 1.15),
        'no_service_threshold': 480,  # Could be extracted if stored in opt_data
    }
