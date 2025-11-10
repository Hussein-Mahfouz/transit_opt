import logging
from typing import Any, Optional

import numpy as np

from ..spatial.boundaries import StudyAreaBoundary
from ..spatial.zoning import HexagonalZoneSystem
from ..utils.population import (
    calculate_population_weighted_total,
    calculate_population_weighted_variance,
    interpolate_population_to_zones,
)
from .base import BaseSpatialObjective

logger = logging.getLogger(__name__)


class WaitingTimeObjective(BaseSpatialObjective):
    """
    Waiting time objective using hexagonal zones with population weighting.

    This objective minimizes user waiting time by optimizing service frequency
    in zones. Waiting time is estimated as headway/2 for the top-N routes
    serving each zone.

    Features:
    - **Population weighting**: Weight by zone population for equity
    - **Multiple metrics**: Total waiting time vs variance in waiting time
    - **Time aggregation**: Average, peak, sum, or interval-specific analysis

    Args:
        optimization_data: Complete optimization data from preparator
        spatial_resolution_km: Hexagon size in kilometers
        crs: Coordinate reference system
        boundary: Geographic boundary filter
        time_aggregation: 'average', 'peak', 'sum', or 'intervals'
        metric: 'total' (minimize total) or 'variance' (minimize inequality)
        population_weighted: Enable population weighting
        population_layer: Population raster path
        population_power: Population weighting exponent
    """

    def __init__(
        self,
        optimization_data: dict[str, Any],
        spatial_resolution_km: float = 2.0,
        crs: str = "EPSG:3857",
        boundary: Optional["StudyAreaBoundary"] = None,
        time_aggregation: str = "average",
        metric: str = "total",  # 'total' or 'variance'
        population_weighted: bool = False,
        population_layer: Any | None = None,
        population_power: float = 1.0,
    ):
        # Validate parameters
        if metric not in ["total", "variance"]:
            raise ValueError("metric must be 'total' or 'variance'")
        if time_aggregation not in ["average", "peak", "intervals", "sum"]:
            raise ValueError("time_aggregation must be 'average', 'peak', 'sum' or 'intervals'")

        self.boundary = boundary
        self.time_aggregation = time_aggregation
        self.metric = metric
        self.population_weighted = population_weighted
        self.population_layer = population_layer
        self.population_power = population_power
        self.population_per_zone = None
        self._interval_length_minutes = None

        super().__init__(optimization_data, spatial_resolution_km, crs)

        if self.population_weighted:
            if self.population_layer is None:
                raise ValueError("Population layer must be provided if population_weighted is True")
            self.population_per_zone = interpolate_population_to_zones(self.spatial_system, self.population_layer)

    def _create_spatial_system(self):
        """Create hexagonal zoning system."""

        # Extract DRT config if available
        drt_config = None
        if self.opt_data.get("drt_enabled", False):
            drt_config = self.opt_data.get("drt_config")

        return HexagonalZoneSystem(
            gtfs_feed=self.gtfs_feed,
            hex_size_km=self.spatial_resolution,
            crs=self.crs,
            boundary=self.boundary,
            drt_config=drt_config,
        )

    def evaluate(self, solution_matrix: np.ndarray | dict) -> float:
        """
        Evaluate waiting time objective for given solution.

        Args:
            solution_matrix:
            - PT-only: Decision matrix (n_routes × n_intervals)
            - PT+DRT: Dict with 'pt' and 'drt' keys

        Returns:
            Objective value (lower is better)
        """

        # ALWAYS calculate per-interval waiting times first
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        logger.debug("   Vehicle data keys: %s", list(vehicles_data.keys()))

        # Calculate waiting times for each interval
        interval_waiting_times = []
        for interval_idx in range(vehicles_data["intervals"].shape[0]):  # Iterate over intervals
            vehicles_this_interval = vehicles_data["intervals"][interval_idx, :]  # Vehicles per zone for this interval
            waiting_times_this_interval = self._convert_vehicles_to_waiting_times_for_interval(
                vehicles_this_interval, solution_matrix, interval_idx
            )
            interval_waiting_times.append(waiting_times_this_interval)

        # Convert to numpy array for easier manipulation
        interval_waiting_times = np.array(interval_waiting_times)  # [intervals x zones]

        # Apply time aggregation
        if self.time_aggregation == "average":
            # Average waiting time across intervals for each zone
            # aggregated_waiting_times = np.mean(interval_waiting_times, axis=0)
            # Use pre-computed vehicle averages, not averaged waiting times
            vehicles_per_zone = vehicles_data["average"]
            interval_length = self._get_interval_length_minutes()
            aggregated_waiting_times = np.array(
                [self._convert_vehicle_count_to_waiting_time(v, interval_length) for v in vehicles_per_zone]
            )
        elif self.time_aggregation == "sum":
            # Sum waiting times across intervals for each zone
            aggregated_waiting_times = np.sum(interval_waiting_times, axis=0)
        elif self.time_aggregation == "peak":
            # Use waiting time from interval with most vehicles per zone
            aggregated_waiting_times = self._get_peak_interval_waiting_times(
                interval_waiting_times, vehicles_data["intervals"]
            )
        elif self.time_aggregation == "intervals":
            # Calculate objective for each interval, then average
            return self._evaluate_intervals_separately(interval_waiting_times)
        else:
            raise ValueError(f"Unknown time_aggregation: {self.time_aggregation}")

        # Apply metric calculation
        return self._calculate_final_objective(aggregated_waiting_times)

    def _convert_vehicles_to_waiting_times_for_interval(
        self, vehicles_per_zone: np.ndarray, solution_matrix: np.ndarray, interval_idx: int
    ) -> np.ndarray:
        """Calculate waiting times using vehicle counts."""
        # Get interval length ONCE per call, not once per zone
        interval_length_minutes = self._get_interval_length_minutes()

        # Vectorized conversion for all zones at once
        # Create result array
        waiting_times = np.zeros_like(vehicles_per_zone, dtype=float)

        # Handle zones with no vehicles (penalty)
        no_vehicle_mask = vehicles_per_zone <= 0
        waiting_times[no_vehicle_mask] = interval_length_minutes

        # Handle zones with vehicles (inverse relationship)
        has_vehicle_mask = vehicles_per_zone > 0
        vehicles_with_service = vehicles_per_zone[has_vehicle_mask]

        # Convert to vehicles per hour and then to waiting time
        vehicles_per_hour = vehicles_with_service * (60 / interval_length_minutes)
        effective_headway = 60 / vehicles_per_hour
        waiting_times[has_vehicle_mask] = effective_headway / 2.0

        return waiting_times

    def _convert_vehicle_count_to_waiting_time(self, vehicle_count: float, interval_length_minutes: float) -> float:
        """
        Convert vehicle count to waiting time for a zone.
        Simple inverse relationship: more vehicles = lower waiting time.

        Zones with no vehicles get a penalty equal to the full interval length. Ideally
        the waiting time should be infinite, but that causes problems for calculations.

        Args:
            vehicle_count: Number of vehicles serving this zone
            interval_length_minutes: Pre-computed interval length to avoid repeated calls

        Returns:
            Waiting time in minutes
        """
        if vehicle_count <= 0:
            # Use penalty equal to the full interval length
            return interval_length_minutes

        # Convert vehicle count to effective frequency
        # vehicle_count represents vehicles serving this zone in this interval
        # More vehicles = higher effective frequency = lower waiting time
        vehicles_per_hour = vehicle_count * (60 / interval_length_minutes)

        if vehicles_per_hour > 0:
            # Effective headway = 60 minutes / vehicles per hour
            effective_headway = 60 / vehicles_per_hour
            waiting_time = effective_headway / 2.0
            return waiting_time
        else:
            # Fallback to penalty
            return interval_length_minutes

    def _get_interval_length_minutes(self) -> float:
        """Calculate the length of each time interval in minutes (cached)."""

        # Return cached value if already computed
        if self._interval_length_minutes is not None:
            # print(f"⏱️ Interval length: {self._interval_length_minutes:.0f} minutes (cached)")
            return self._interval_length_minutes

        shape = self.opt_data.get("decision_matrix_shape")
        if not shape or len(shape) < 2:
            raise ValueError("Missing or invalid 'decision_matrix_shape' in opt_data.")

        n_intervals = shape[1]
        interval_length_hours = 24 / n_intervals
        self._interval_length_minutes = interval_length_hours * 60  # Convert to minutes

        logger.info(
            f"📊 Using {n_intervals} intervals, each {interval_length_hours:.1f} hours ({self._interval_length_minutes:.0f} minutes)"
        )

        return self._interval_length_minutes

    def _get_peak_interval_waiting_times(
        self, interval_waiting_times: np.ndarray, vehicles_intervals: np.ndarray
    ) -> np.ndarray:
        """
        Get waiting times from the system-wide peak interval.

        Peak interval = interval with most total vehicles across all zones.
        All zones use waiting times from this same interval.
        """
        # Sum vehicles across all zones for each interval
        total_vehicles_by_interval = np.sum(vehicles_intervals, axis=0)  # Sum across zones

        # Find interval with most total vehicles
        peak_interval_idx = np.argmax(total_vehicles_by_interval)

        # Return waiting times from that interval for ALL zones
        return interval_waiting_times[peak_interval_idx, :]

    def _evaluate_intervals_separately(self, interval_waiting_times: np.ndarray) -> float:
        """Calculate objective for each interval separately, then average."""
        interval_objectives = []

        for interval_idx in range(interval_waiting_times.shape[0]):
            waiting_times_this_interval = interval_waiting_times[interval_idx, :]
            interval_obj = self._calculate_final_objective(waiting_times_this_interval)
            interval_objectives.append(interval_obj)

        return np.mean(interval_objectives)

    def _calculate_final_objective(self, waiting_times: np.ndarray) -> float:
        """
        Calculate final objective value from zone waiting times.
        (no spatial lag - doesn't make sense for waiting times)
        """
        logger.debug(f"   Metric: {self.metric}")
        logger.debug(f"   Waiting times shape: {waiting_times.shape}")
        logger.debug(f"   Sample waiting times: {waiting_times[:10]}")
        logger.debug(f"   Min/Max waiting times: {waiting_times.min():.2f}/{waiting_times.max():.2f}")

        if self.population_weighted:
            logger.debug(f"   Population per zone: {self.population_per_zone}")
            logger.debug(f"   Population power: {self.population_power}")

            if self.metric == "total":
                result = calculate_population_weighted_total(
                    waiting_times, self.population_per_zone, self.population_power
                )
                logger.debug(f"   Population-weighted total result: {result}")
                return result

            else:  # variance
                result = calculate_population_weighted_variance(
                    waiting_times, self.population_per_zone, self.population_power
                )
                logger.debug(f"   Population-weighted variance result: {result}")
                return result
        else:
            # Unweighted calculations
            if self.metric == "total":
                result = np.sum(waiting_times)
                logger.debug(f"   Unweighted total result: {result}")
                return result
            else:  # variance
                result = np.var(waiting_times)
                logger.debug(f"   Unweighted variance result: {result}")
                return result

    def _get_spatial_summary(self) -> str:
        """Return summary string for logging."""
        features = [f"{len(self.spatial_system.hex_grid)} hexagonal zones"]
        if self.population_weighted:
            features.append("population weighted")
        features.append(f"metric: {self.metric}")
        features.append(f"time aggregation: {self.time_aggregation}")
        return ", ".join(features)

    def get_waiting_times_per_zone(self, solution_matrix: np.ndarray, aggregation: str = "average") -> np.ndarray:
        """
        Get waiting times per zone for visualization and analysis.

        This method converts vehicle counts to waiting times using the standard
        transit formula: waiting_time = headway / 2, where headway = interval_length / vehicles.

        Args:
            solution_matrix: Decision matrix (PT-only) or dict with 'pt'/'drt' keys
            aggregation: How to aggregate across time intervals:
                        - 'average': Mean vehicle count across intervals
                        - 'peak': Maximum vehicle count across intervals
                        - str(interval_idx): Specific interval by index

        Returns:
            Array of waiting times per zone in minutes. Higher values = worse service.
        """
        # Get vehicles per zone first
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        # Get interval length
        interval_length = self._get_interval_length_minutes()

        # Convert to waiting times using existing method
        if aggregation == "average":
            vehicle_counts = vehicles_data["average"]
        elif aggregation == "peak":
            vehicle_counts = vehicles_data["peak"]
        else:
            # Assume it's an interval index
            interval_idx = int(aggregation)
            vehicle_counts = vehicles_data["intervals"][interval_idx, :]

        # Use existing conversion method
        waiting_times = np.array(
            [self._convert_vehicle_count_to_waiting_time(v, interval_length) for v in vehicle_counts]
        )

        return waiting_times

    def visualize(
        self,
        solution_matrix: np.ndarray,
        aggregation: str = "average",
        interval_idx: int | None = None,
        figsize=(15, 12),
        show_stops=True,
        show_drt_zones=None,
        ax=None,
        vmin=None,
        vmax=None,
    ):
        """
        Visualize waiting times spatially.

        Args:
            solution_matrix: Decision matrix (PT-only) or dict with 'pt'/'drt' keys
            aggregation: Time aggregation method ('average', 'peak', 'intervals')
            interval_idx: Specific interval index (only used when aggregation='intervals')
            figsize: Figure size (only used if ax is None)
            show_stops: Whether to show transit stops as blue dots
            show_drt_zones: Whether to show DRT service areas (auto-detects if None)
            ax: Optional matplotlib axis to plot on
            vmin: Minimum value for color scale (auto-calculated if None)
            vmax: Maximum value for color scale (auto-calculated if None)

        Returns:
            Tuple of (figure, axis) objects
        """
        # Get waiting times using existing methods
        if aggregation == "intervals" and interval_idx is not None:
            waiting_times_per_zone = self.get_waiting_times_per_zone(solution_matrix, str(interval_idx))
            title_suffix = f"(Interval {interval_idx})"
        else:
            waiting_times_per_zone = self.get_waiting_times_per_zone(solution_matrix, aggregation)
            title_suffix = f"({aggregation.capitalize()} Waiting Times)"

        # Use the spatial system's visualization capabilities
        return self.spatial_system._visualize_with_data(
            data_per_zone=waiting_times_per_zone,
            data_column_name="waiting_time",
            data_label="Waiting Time (minutes)",
            colormap="RdYlBu_r",  # Red=high waiting, blue=low waiting
            optimization_data=self.opt_data,
            title_suffix=title_suffix,
            figsize=figsize,
            show_stops=show_stops,
            show_drt_zones=show_drt_zones,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
