"""
Population weighting utilities for spatial objectives.

Provides shared functionality for interpolating population raster data
onto hexagonal zone grids and calculating population-weighted metrics.
"""

import logging
from typing import Any

import numpy as np
import rasterio
from rasterstats import zonal_stats

logger = logging.getLogger(__name__)


def interpolate_population_to_zones(spatial_system, population_layer: Any) -> np.ndarray:
    """
    Assign population from raster to hexagons using zonal_stats.

    Shared implementation for both service coverage and waiting time objectives.
    Uses raster population data from WorldPop or similar sources.

    Args:
        spatial_system: HexagonalZoneSystem with hex_grid attribute
        population_layer: Path to population raster file

    Returns:
        np.ndarray: Population count for each zone
    """
    hexgrid = spatial_system.hex_grid

    # Check crs compatibility
    with rasterio.open(population_layer) as src:
        raster_crs = src.crs
        vector_crs = hexgrid.crs

        if raster_crs != vector_crs:
            logger.warning(f"⚠️ CRS mismatch: Raster {raster_crs} != Vector{vector_crs}")
            logger.info(" Creating transformed hexgrid with raster CRS for zonal stats.")
            hexgrid_transformed = hexgrid.to_crs(raster_crs)
        else:
            hexgrid_transformed = hexgrid

    stats = zonal_stats(
        hexgrid_transformed,
        population_layer,
        stats=["sum"],
        nodata=0,
        geojson_out=False
    )

    pop_array = np.array([item['sum'] if item['sum'] is not None else 0 for item in stats])
    # Replace negative values with 0 (WorldPop can have negatives)
    pop_array = np.maximum(pop_array, 0)
    logger.info("📊 Population per zone: min=%d, max=%d, mean=%.2f",
                np.min(pop_array), np.max(pop_array), np.mean(pop_array))
    return pop_array


def calculate_population_weighted_variance(
    values: np.ndarray,
    population: np.ndarray,
    population_power: float = 1.0
) -> float:
    """
    Calculate population-weighted variance of values across zones.

    Generic implementation that works for both vehicles_per_zone and waiting_times.

    Args:
        values: Array of values per zone (vehicles, waiting times, etc.)
        population: Array of population per zone
        population_power: Exponent for population weighting

    Returns:
        float: Population-weighted variance
    """
    if population is None or len(population) != len(values):
        raise ValueError("Population data is not properly initialized or mismatched.")

    # Apply power transformation to population if needed
    pop_weighted = np.power(population, population_power)

    if np.sum(pop_weighted) == 0:
        # Fallback to unweighted variance
        return np.var(values)

    mean_weighted = np.sum(pop_weighted * values) / np.sum(pop_weighted)
    variance_weighted = np.sum(pop_weighted * (values - mean_weighted) ** 2) / np.sum(pop_weighted)

    return variance_weighted


def calculate_population_weighted_total(
    values: np.ndarray,
    population: np.ndarray,
    population_power: float = 1.0
) -> float:
    """
    Calculate population-weighted total of values across zones.

    Currently used by waiting time objective, available for future objectives.

    Args:
        values: Array of values per zone (waiting times, costs, etc.)
        population: Array of population per zone
        population_power: Exponent for population weighting

    Returns:
        float: Population-weighted total
    """

    logger.debug(f"""
        🔍 Pop-weighted total DEBUG:
        * Values: {values}
        * Population: {population}
        * Power: {population_power}
    """)

    if population is None:
        logger.error("Population data not available")
        raise ValueError("Population data not available")

    pop_weighted = np.power(population, population_power)
    logger.info("Population weights: %s", pop_weighted)

    # Handle infinite values (from waiting time objective)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float('inf')  # All zones have infinite values

    # Only sum for zones with finite values
    finite_values = values[finite_mask]
    finite_pop = pop_weighted[finite_mask]

    return np.sum(finite_pop * finite_values)


def validate_population_config(population_weighted: bool, population_layer: Any) -> None:
    """
    Validate population weighting configuration.

    Args:
        population_weighted: Whether population weighting is enabled
        population_layer: Population raster path or None

    Raises:
        ValueError: If configuration is invalid
    """
    if population_weighted and population_layer is None:
        logger.error("Population layer must be provided if population_weighted is True")
        raise ValueError("Population layer must be provided if population_weighted is True")
