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
        """
            Evaluate waiting time objective for given solution.
            
            Args:
                solution_matrix: Decision matrix (n_routes × n_intervals)
                
            Returns:
                Objective value (lower is better)
            """

        # ALWAYS calculate per-interval waiting times first
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        print(f"   Vehicle data keys: {list(vehicles_data.keys())}")

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
            # aggregated_waiting_times = np.mean(interval_waiting_times, axis=0)
            # Use pre-computed vehicle averages, not averaged waiting times
            vehicles_per_zone = vehicles_data["average"]
            interval_length = self._get_interval_length_minutes()
            aggregated_waiting_times = np.array([
                self._convert_vehicle_count_to_waiting_time(v, interval_length)
                for v in vehicles_per_zone
            ])
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
        # Get interval length ONCE per interval, not once per zone
        interval_length_minutes = self._get_interval_length_minutes()

        # Apply the same conversion to all zones - this handles zero vehicles correctly
        # Pass the interval length to avoid repeated calls
        return np.array([
            self._convert_vehicle_count_to_waiting_time(vehicle_count, interval_length_minutes)
            for vehicle_count in vehicles_per_zone
        ])

    def _convert_vehicle_count_to_waiting_time(
        self, vehicle_count: float, interval_length_minutes: float
    ) -> float:
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
        print(f"   Metric: {self.metric}")
        print(f"   Waiting times shape: {waiting_times.shape}")
        print(f"   Sample waiting times: {waiting_times[:10]}")
        print(f"   Min/Max waiting times: {waiting_times.min():.2f}/{waiting_times.max():.2f}")

        if self.population_weighted:
            print(f"   Population per zone: {self.population_per_zone}")
            print(f"   Population power: {self.population_power}")

            if self.metric == "total":
                result = calculate_population_weighted_total(
                    waiting_times, self.population_per_zone, self.population_power
                )
                print(f"   Population-weighted total result: {result}")
                return result

            else:  # variance
                result = calculate_population_weighted_variance(
                    waiting_times, self.population_per_zone, self.population_power
                )
                print(f"   Population-weighted variance result: {result}")
                return result
        else:
            # Unweighted calculations
            if self.metric == "total":
                result = np.sum(waiting_times)
                print(f"   Unweighted total result: {result}")
                return result
            else:  # variance
                result = np.var(waiting_times)
                print(f"   Unweighted variance result: {result}")
                return result

    def _get_spatial_summary(self) -> str:
        """Return summary string for logging."""
        features = [f"{len(self.spatial_system.hex_grid)} hexagonal zones"]
        if self.population_weighted:
            features.append("population weighted")
        features.append(f"metric: {self.metric}")
        features.append(f"time aggregation: {self.time_aggregation}")
        return ", ".join(features)
