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


class WaitingTimeObjective(BaseSpatialObjective):
    """
    Waiting time objective using hexagonal zones with population weighting.
    
    This objective minimizes user waiting time by optimizing service frequency
    in zones. Waiting time is estimated as headway/2 for the top-N routes 
    serving each zone.
    
    Features:
    - **Population weighting**: Weight by zone population for equity
    - **Multiple metrics**: Total waiting time vs variance in waiting time
    - **Time aggregation**: Average, peak, or interval-specific analysis
    
    Args:
        optimization_data: Complete optimization data from preparator
        spatial_resolution_km: Hexagon size in kilometers
        crs: Coordinate reference system
        boundary: Geographic boundary filter
        time_aggregation: 'average', 'peak', or 'intervals'
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
        population_power: float = 1.0
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
            self.population_per_zone = interpolate_population_to_zones(
                self.spatial_system, self.population_layer
            )

    def _create_spatial_system(self):
        """Create hexagonal zoning system."""
        return HexagonalZoneSystem(
            gtfs_feed=self.gtfs_feed,
            hex_size_km=self.spatial_resolution,
            crs=self.crs,
            boundary=self.boundary,
        )

    def evaluate(self, solution_matrix: np.ndarray) -> float:
        """Calculate waiting time objective - always compute per interval first."""
        # ALWAYS calculate per-interval waiting times first
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        # Calculate waiting times for each interval
        interval_waiting_times = []
        for interval_idx in range(vehicles_data["intervals"].shape[1]):
            vehicles_this_interval = vehicles_data["intervals"][:, interval_idx]
            waiting_times_this_interval = self._convert_vehicles_to_waiting_times_for_interval(
                vehicles_this_interval, solution_matrix, interval_idx
            )
            interval_waiting_times.append(waiting_times_this_interval)

        # Convert to numpy array for easier manipulation
        interval_waiting_times = np.array(interval_waiting_times)  # [intervals x zones]

        # Apply time aggregation
        if self.time_aggregation == "average":
            # Average waiting time across intervals for each zone
            aggregated_waiting_times = np.mean(interval_waiting_times, axis=0)
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
        """Calculate waiting times using vehicle counts (no route mapping needed)."""
        n_zones = len(vehicles_per_zone)
        waiting_times = np.full(n_zones, float('inf'))

        for zone_idx in range(n_zones):
            vehicle_count = vehicles_per_zone[zone_idx]

            if vehicle_count == 0:
                continue  # Keep as infinity - no service

            # Convert vehicle count to effective waiting time
            waiting_times[zone_idx] = self._convert_vehicle_count_to_waiting_time(
                vehicle_count
            )

        return waiting_times

    def _convert_vehicle_count_to_waiting_time(self, vehicle_count: float) -> float:
        """
        Convert vehicle count to waiting time for a zone.
        Simple inverse relationship: more vehicles = lower waiting time.
        """
        if vehicle_count <= 0:
            return float('inf')

        # Calculate interval length in minutes
        interval_length_minutes = self._get_interval_length_minutes()

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
            return float('inf')

    def _get_interval_length_minutes(self) -> float:
        """Calculate the length of each time interval in minutes (cached)."""

        # Return cached value if already computed
        if self._interval_length_minutes is not None:
            print(f"⏱️ Interval length: {self._interval_length_minutes:.0f} minutes (cached)")
            return self._interval_length_minutes

        shape = self.opt_data.get('decision_matrix_shape')
        if not shape or len(shape) < 2:
            raise ValueError("Missing or invalid 'decision_matrix_shape' in opt_data.")

        n_intervals = shape[1]
        interval_length_hours = 24 / n_intervals
        self._interval_length_minutes = interval_length_hours * 60  # Convert to minutes

        print(f"📊 Using {n_intervals} intervals, each {interval_length_hours:.1f} hours ({self._interval_length_minutes:.0f} minutes)")

        return self._interval_length_minutes

    def _get_peak_interval_waiting_times(self, interval_waiting_times: np.ndarray, vehicles_intervals: np.ndarray) -> np.ndarray:
        """Get waiting times from the peak interval (most vehicles) for each zone."""
        n_zones = vehicles_intervals.shape[0]
        peak_waiting_times = np.zeros(n_zones)

        for zone_idx in range(n_zones):
            # Find interval with most vehicles for this zone
            zone_vehicles_by_interval = vehicles_intervals[zone_idx, :]
            peak_interval_idx = np.argmax(zone_vehicles_by_interval)

            # Use waiting time from that interval
            peak_waiting_times[zone_idx] = interval_waiting_times[peak_interval_idx, zone_idx]

        return peak_waiting_times

    def _evaluate_intervals_separately(self, interval_waiting_times: np.ndarray) -> float:
        """Calculate objective for each interval separately, then average."""
        interval_objectives = []

        for interval_idx in range(interval_waiting_times.shape[0]):
            waiting_times_this_interval = interval_waiting_times[interval_idx, :]
            interval_obj = self._calculate_final_objective(waiting_times_this_interval)
            interval_objectives.append(interval_obj)

        return np.mean(interval_objectives)

    def _calculate_final_objective(self, waiting_times: np.ndarray) -> float:
        """Apply metric calculation (no spatial lag - doesn't make sense for waiting times)."""

        if self.metric == "total":
            if self.population_weighted:
                return calculate_population_weighted_total(
                    waiting_times, self.population_per_zone, self.population_power
                )
            else:
                return np.sum(waiting_times[np.isfinite(waiting_times)])

        else:  # variance
            if self.population_weighted:
                return calculate_population_weighted_variance(
                    waiting_times, self.population_per_zone, self.population_power
                )
            else:
                finite_waiting = waiting_times[np.isfinite(waiting_times)]
                return np.var(finite_waiting) if len(finite_waiting) > 0 else float('inf')

    def _get_spatial_summary(self) -> str:
        """Return summary string for logging."""
        features = [f"{len(self.spatial_system.hex_grid)} hexagonal zones"]
        if self.population_weighted:
            features.append("population weighted")
        features.append(f"metric: {self.metric}")
        features.append(f"time aggregation: {self.time_aggregation}")
        return ", ".join(features)
