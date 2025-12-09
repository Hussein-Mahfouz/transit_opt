"""
Travel demand weighting utilities for spatial objectives.

Provides functionality for calculating travel demand from trip OD data
and using it to weight objective functions.
"""

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_trip_data(trip_data_path: str, crs: str, min_distance_m: float | None = None) -> gpd.GeoDataFrame:
    """
    Load trip data from CSV with STRICT format requirements.

    Args:
        trip_data_path: Path to CSV file with trip data
        crs: CRS of input coordinates (e.g., "EPSG:3857"). Must be metric/projected.
        min_distance_m: Optional minimum trip distance filter (meters)

    Returns:
        GeoDataFrame with trip origins as Point geometries

    Required CSV columns:
        - origin_x: X coordinate in specified CRS (meters)
        - origin_y: Y coordinate in specified CRS (meters)
        - departure_time: Seconds since midnight (0-86400)
        - euclidean_distance: Trip distance in meters (optional)

    Raises:
        ValueError: If required columns missing or data invalid
        FileNotFoundError: If file doesn't exist
    """
    # 1. Check file exists
    if not Path(trip_data_path).exists():
        raise FileNotFoundError(f"Trip data file not found: {trip_data_path}")

    # 2. Read CSV only (no other formats)
    if not trip_data_path.endswith(".csv"):
        raise ValueError(f"Only CSV files supported. Got: {trip_data_path}")

    logger.info(f"📖 Reading CSV file: {trip_data_path}")
    trips_df = pd.read_csv(trip_data_path)
    logger.info(f"   ✅ CSV loaded: {len(trips_df):,} rows")

    # 3. Validate required columns exist
    required_cols = ["origin_x", "origin_y", "departure_time"]
    missing_cols = [col for col in required_cols if col not in trips_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\nRequired: {required_cols}\nFound: {list(trips_df.columns)}"
        )

    # 4. Validate departure_time is integer seconds (0-86400)
    if not pd.api.types.is_integer_dtype(trips_df["departure_time"]):
        raise ValueError(
            "departure_time must be integer type (seconds since midnight).\n"
            f"Got dtype: {trips_df['departure_time'].dtype}"
        )

    # Filter invalid departure times
    valid_time_mask = (trips_df["departure_time"] >= 0) & (trips_df["departure_time"] <= 86400)
    invalid_count = (~valid_time_mask).sum()

    if invalid_count > 0:
        logger.warning(f"⚠️  Filtering {invalid_count:,} trips with invalid departure times (outside 0-86400 seconds)")
        trips_df = trips_df[valid_time_mask].copy()

    if len(trips_df) == 0:
        raise ValueError("No trips remaining after filtering invalid departure times")

    # 5. Create geometry from projected coordinates
    logger.info(f"🔧 Converting DataFrame to GeoDataFrame with {len(trips_df):,} points...")
    trips_gdf = gpd.GeoDataFrame(
        trips_df, geometry=gpd.points_from_xy(trips_df["origin_x"], trips_df["origin_y"]), crs=crs
    )
    logger.info(f"   ✅ GeoDataFrame created with CRS: {crs}")

    logger.info(f"📊 Loaded {len(trips_gdf):,} trips from {trip_data_path}")
    logger.info(f"   CRS: {crs}")
    logger.info(f"   Departure time range: {trips_gdf['departure_time'].min()}-{trips_gdf['departure_time'].max()}s")

    # 6. Apply distance filter if specified
    if min_distance_m is not None:
        if "euclidean_distance" not in trips_gdf.columns:
            logger.warning(
                f"⚠️  min_distance_m={min_distance_m} specified but "
                "euclidean_distance column not found - skipping filter"
            )
        else:
            original_count = len(trips_gdf)
            trips_gdf = trips_gdf[trips_gdf["euclidean_distance"] >= min_distance_m].copy()

            filtered_count = original_count - len(trips_gdf)
            logger.info(
                f"🔍 Filtered {filtered_count:,} trips with distance < {min_distance_m}m "
                f"({100 * filtered_count / original_count:.1f}% of total)"
            )

            if len(trips_gdf) == 0:
                raise ValueError(f"No trips remaining after distance filter (min={min_distance_m}m)")

    return trips_gdf


def assign_trips_to_time_intervals(
    trips_gdf: gpd.GeoDataFrame, n_intervals: int, interval_hours: int
) -> gpd.GeoDataFrame:
    """
    Assign each trip to a time interval based on departure time.

    Args:
        trips_gdf: GeoDataFrame with trips and departure_time column (integer seconds)
        n_intervals: Number of time intervals per day
        interval_hours: Duration of each interval in hours

    Returns:
        GeoDataFrame with added 'interval_idx' column
    """
    trips_gdf = trips_gdf.copy()

    # Validate departure_time is integer (should already be validated in load_trip_data)
    if not pd.api.types.is_integer_dtype(trips_gdf["departure_time"]):
        raise ValueError(
            "departure_time must be integer type (seconds since midnight).\n"
            f"Got dtype: {trips_gdf['departure_time'].dtype}"
        )

    # Convert seconds to hours
    trips_gdf["departure_hour"] = (trips_gdf["departure_time"] / 3600).astype(int)
    logger.info("📍 Parsed departure times as integer seconds since midnight")

    # Assign to intervals
    trips_gdf["interval_idx"] = (trips_gdf["departure_hour"] // interval_hours).astype(int)

    # Validate interval assignments are within bounds
    trips_gdf = trips_gdf[trips_gdf["interval_idx"] < n_intervals].copy()

    logger.info(f"✅ Assigned trips to {n_intervals} intervals:")
    for i in range(n_intervals):
        count = np.sum(trips_gdf["interval_idx"] == i)
        hour_start = i * interval_hours
        hour_end = (i + 1) * interval_hours
        logger.info(f"   Interval {i} ({hour_start:02d}:00-{hour_end:02d}:00): {count:,} trips")

    return trips_gdf


def calculate_demand_per_zone_interval(spatial_system, trips_gdf: gpd.GeoDataFrame, n_intervals: int) -> np.ndarray:
    """
    Calculate travel demand (trip count) per zone per interval.

    Args:
        spatial_system: HexagonalZoneSystem with hex_grid attribute
        trips_gdf: GeoDataFrame with trips, geometry (origins), and interval_idx
        n_intervals: Number of time intervals

    Returns:
        Array of shape (n_zones, n_intervals) with trip counts
    """
    hex_grid = spatial_system.hex_grid
    n_zones = len(hex_grid)

    logger.info(f"📊 Starting demand calculation for {n_zones:,} zones across {n_intervals} intervals")

    # Reproject trips to match hex_grid CRS
    if trips_gdf.crs != hex_grid.crs:
        logger.info(f"⚙️  Reprojecting trips from {trips_gdf.crs} to {hex_grid.crs}")
        trips_gdf = trips_gdf.to_crs(hex_grid.crs)

    # Spatial join: assign trips to zones
    logger.info(f"🗺️  Starting spatial join ({len(trips_gdf):,} trips → {n_zones:,} zones)...")
    trips_with_zones = gpd.sjoin(trips_gdf, hex_grid[["zone_id", "geometry"]], how="left", predicate="within")

    # Initialize demand matrix
    logger.info(f"📈 Aggregating trips into {n_zones:,} × {n_intervals} demand matrix...")
    demand_matrix = np.zeros((n_zones, n_intervals), dtype=int)

    # Count trips per zone per interval
    for zone_idx, zone_id in enumerate(hex_grid["zone_id"]):
        zone_trips = trips_with_zones[trips_with_zones["zone_id"] == zone_id]

        for interval_idx in range(n_intervals):
            interval_trips = zone_trips[zone_trips["interval_idx"] == interval_idx]
            demand_matrix[zone_idx, interval_idx] = len(interval_trips)

    # Log statistics
    total_trips = np.sum(demand_matrix)
    zones_with_demand = np.sum(np.any(demand_matrix > 0, axis=1))

    logger.info(f"📊 Demand calculation complete:")
    logger.info(f"   Total trips assigned: {total_trips:,}")
    logger.info(f"   Zones with demand: {zones_with_demand}/{n_zones}")
    logger.info(f"   Mean trips per zone: {np.mean(np.sum(demand_matrix, axis=1)):.1f}")
    logger.info(f"   Max trips in any zone-interval: {np.max(demand_matrix):,}")

    return demand_matrix


def calculate_demand_weighted_variance(values: np.ndarray, demand: np.ndarray, demand_power: float = 1.0) -> float:
    """
    Calculate demand-weighted variance of values across zones.

    Generic implementation that works for both vehicles_per_zone and waiting_times.

    Args:
        values: Array of values per zone (vehicles, waiting times, etc.)
        demand: Array of travel demand per zone (trip counts)
        demand_power: Exponent for demand weighting

    Returns:
        float: Demand-weighted variance
    """
    if demand is None or len(demand) != len(values):
        raise ValueError("Demand data is not properly initialized or mismatched.")

    # Apply power transformation to demand if needed
    demand_weighted = np.power(demand, demand_power)

    if np.sum(demand_weighted) == 0:
        logger.warning("⚠️  Total demand is zero, falling back to unweighted variance")
        return np.var(values)

    # Weighted mean
    mean_weighted = np.sum(demand_weighted * values) / np.sum(demand_weighted)

    # Weighted variance
    variance_weighted = np.sum(demand_weighted * (values - mean_weighted) ** 2) / np.sum(demand_weighted)

    return variance_weighted


def calculate_demand_weighted_total(values: np.ndarray, demand: np.ndarray, demand_power: float = 1.0) -> float:
    """
    Calculate demand-weighted total of values across zones.

    Used by waiting time objective for demand-weighted total waiting time.

    Args:
        values: Array of values per zone (waiting times, costs, etc.)
        demand: Array of travel demand per zone (trip counts)
        demand_power: Exponent for demand weighting

    Returns:
        float: Demand-weighted total
    """
    logger.debug(f"""
        🔍 Demand-weighted total DEBUG:
        * Values: {values}
        * Demand: {demand}
        * Power: {demand_power}
    """)

    if demand is None:
        logger.error("Demand data not available")
        raise ValueError("Demand data not available")

    if values is None or len(values) == 0:
        logger.error("Values data is None or empty")
        raise ValueError("Values data is None or empty")


    if len(values) != len(demand):
        logger.error(f"Shape mismatch: values {len(values)} vs demand {len(demand)}")
        raise ValueError(f"Shape mismatch: values {len(values)} vs demand {len(demand)}")

    demand_weighted = np.power(demand, demand_power)
    logger.debug("Demand weights: %s", demand_weighted)

    # Handle infinite values (from waiting time objective)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float("inf")  # All zones have infinite values

    # Only sum for zones with finite values
    finite_values = values[finite_mask]
    finite_demand = demand_weighted[finite_mask]

    return np.sum(finite_demand * finite_values)


def validate_demand_config(demand_weighted: bool, trip_data_path: Any) -> None:
    """
    Validate demand weighting configuration.

    Args:
        demand_weighted: Whether demand weighting is enabled
        trip_data_path: Trip data file path or None

    Raises:
        ValueError: If configuration is invalid
    """
    if demand_weighted and trip_data_path is None:
        logger.error("Trip data path must be provided if demand_weighted is True")
        raise ValueError("Trip data path must be provided if demand_weighted is True")
