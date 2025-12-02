import logging
from typing import Any, Optional

import numpy as np

from ..spatial.boundaries import StudyAreaBoundary
from ..spatial.zoning import HexagonalZoneSystem
from ..utils.demand import calculate_demand_weighted_variance, validate_demand_config
from ..utils.population import (
    calculate_population_weighted_variance,
    interpolate_population_to_zones,
    validate_population_config,
)
from .base import BaseSpatialObjective

logger = logging.getLogger(__name__)


class StopCoverageObjective(BaseSpatialObjective):
    """
    Spatial equity objective using hexagonal zones with optional spatial lag analysis.

    This objective minimizes the variance in vehicle distribution across hexagonal
    zones, promoting equitable spatial coverage of transit service. It supports
    several advanced features:

    - **Spatial lag analysis**: Incorporates neighbor accessibility effects
    - **Boundary filtering**: Restricts analysis to specific study areas
    - **Multiple time aggregations**: Average, peak, or interval-specific analysis
    - **Population weighting**: Planned feature for demographic equity (placeholder)

    The core objective minimizes:
    - Standard mode: variance(vehicles_per_zone)
    - Spatial lag mode: variance(accessibility_scores) where accessibility includes
      neighbor effects

    Attributes:
        boundary (Optional[StudyAreaBoundary]): Geographic boundary for filtering.
        time_aggregation (str): How to aggregate across time intervals.
        spatial_lag (bool): Whether to use spatial lag accessibility calculation.
        alpha (float): Spatial lag decay parameter (0=no neighbors, 1=equal weight).
        population_weighted (bool): Whether to use population weighting (placeholder).

    Args:
        optimization_data (Dict[str, Any]): Complete optimization data from preparator.
        spatial_resolution_km (float, optional): Hexagon size in kilometers.
                                               Defaults to 2.0.
        crs (str, optional): Coordinate reference system (metric).
                           Defaults to "EPSG:3857".
        boundary (Optional[StudyAreaBoundary], optional): Geographic boundary filter.
                                                        Defaults to None.
        time_aggregation (str, optional): Time aggregation method. Options:
                                        'average', 'peak', 'sum'. Defaults to 'average'.
        spatial_lag (bool, optional): Enable spatial lag accessibility.
                                    Defaults to False.
        alpha (float, optional): Spatial lag decay factor [0,1]. Higher values
                               give more weight to neighbors. Defaults to 0.1.
        population_weighted (bool, optional): Enable population weighting.
                                            Defaults to False.
        population_layer (Any, optional): Population raster layer for weighting.
                                        Required if population_weighted is True.
                                        Recommended: WorldPop data. Defaults to None.
        population_power (float, optional): Exponent for population weighting.
                                          Defaults to 1.0 (linear). Values <1.0
                                          dampen influence of high-pop areas.

    Mathematical Details:
        Standard variance: σ² = Σ(vᵢ - μ)² / n

        Spatial lag: accessibility_i = vehicles_i + α × Σ(w_ij × vehicles_j)
        where w_ij are spatial weights (1 if neighbors, 0 otherwise)

        Population weighting (planned): σ²_w = Σ(p_i × (x_i - μ_w)²) / Σ(p_i)

    Example:
        ```python
        # Standard equity objective
        equity_obj = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=3.0
        )

        # With spatial lag and boundary filtering
        spatial_equity_obj = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=2.0,
            boundary=study_boundary,
            spatial_lag=True,
            alpha=0.15  # 15% neighbor influence
        )
        # With population weighting
        pop_equity_obj = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=2.0,
            boundary=study_boundary,
            spatial_lag=True,
            alpha=0.1,
            population_weighted=True,
            population_layer="path/to/worldpop.tif"
        )
        ```
    """

    def __init__(
        self,
        optimization_data: dict[str, Any],
        spatial_resolution_km: float = 2.0,
        crs: str = "EPSG:3857",
        boundary: Optional["StudyAreaBoundary"] = None,
        time_aggregation: str = "average",
        spatial_lag: bool = False,
        alpha: float = 0.1,
        # ===== WEIGHTING OPTIONS =====
        # Population weighting
        population_weighted: bool = False,
        # Population data used if population_weighted is True
        population_layer: Any | None = None,
        population_power: float = 1.0,
        # Demand weighting
        demand_weighted: bool = False,
        trip_data_path: str | None = None,
        trip_data_crs: str | None = None,
        demand_power: float = 1.0,
        min_trip_distance_m: float | None = None,
    ):
        # Validate that only one weighting method is used
        if population_weighted and demand_weighted:
            raise ValueError("Cannot use both population_weighted and demand_weighted. Choose one weighting method.")
        # Validate demand weighting configuration
        if demand_weighted:
            if trip_data_path is None:
                raise ValueError("trip_data_path required when demand_weighted=True")
            if trip_data_crs is None:
                raise ValueError("trip_data_crs required when demand_weighted=True")

        self.boundary = boundary
        self.time_aggregation = time_aggregation
        self.spatial_lag = spatial_lag
        self.alpha = alpha  # Decay factor for neighbor influence
        # Population weighting
        self.population_weighted = population_weighted
        self.population_layer = population_layer
        self.population_per_zone = None
        self.population_power = population_power
        # Demand weighting
        self.demand_weighted = demand_weighted
        self.trip_data_path = trip_data_path
        self.trip_data_crs = (trip_data_crs,)
        self.demand_per_zone_interval = None  # Shape: (n_zones, n_intervals)
        self.demand_power = demand_power
        self.min_trip_distance_m = min_trip_distance_m

        super().__init__(optimization_data, spatial_resolution_km, crs)

        validate_population_config(population_weighted, population_layer)
        validate_demand_config(demand_weighted, trip_data_path)

        if self.population_weighted and self.population_layer is None:
            raise ValueError("Population layer must be provided if population_weighted is True.")
        if population_layer is not None:
            self.population_per_zone = interpolate_population_to_zones(self.spatial_system, self.population_layer)

        # Load and process demand data
        if self.demand_weighted:
            from ..utils.demand import (
                assign_trips_to_time_intervals,
                calculate_demand_per_zone_interval,
                load_trip_data,
            )

            # Load trip data
            trips_gdf = load_trip_data(
                trip_data_path=trip_data_path, crs=trip_data_crs, min_distance_m=min_trip_distance_m
            )

            # Assign trips to time intervals
            n_intervals = optimization_data["n_intervals"]
            interval_hours = 24 // n_intervals

            trips_gdf = assign_trips_to_time_intervals(trips_gdf, n_intervals, interval_hours)

            # Calculate demand per zone per interval
            self.demand_per_zone_interval = calculate_demand_per_zone_interval(
                self.spatial_system, trips_gdf, n_intervals
            )

            logger.info(
                f"✅ Demand data loaded: {np.sum(self.demand_per_zone_interval):,} trips across {n_intervals} intervals"
            )

    def _create_spatial_system(self):
        """Use your existing HexagonalZoneSystem."""

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

    def _get_spatial_summary(self) -> str:
        features = [f"{len(self.spatial_system.hex_grid)} hexagonal zones"]
        if self.spatial_lag:
            features.append(f"spatial lag (α={self.alpha})")
        if self.population_weighted:
            features.append("population weighted [PLACEHOLDER]")
        if self.demand_weighted:
            features.append("demand weighted [PLACEHOLDER]")
        return ", ".join(features)

    def evaluate(self, solution_matrix: np.ndarray | dict) -> float:
        """
        Minimize variance in vehicle distribution across hexagons.

        Args:
                solution_matrix:
                - PT-only: Decision matrix (n_routes × n_intervals)
                - PT+DRT: Dict with 'pt' and 'drt' keys

            Returns:
                Objective value (lower is better)
        """
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        # Get pre-calculated peak interval directly
        fleet_stats = self.opt_data["constraints"]["fleet_analysis"]["fleet_stats"]
        peak_interval_idx = fleet_stats["peak_interval"]

        # Get vehicles_per_zone based on time aggregation parameter:
        # TODO: implement 'intervals' option as done in waiting_times.py
        if self.time_aggregation == "average":
            # Use average vehicles across intervals
            vehicles_per_zone = vehicles_data["average"]

        elif self.time_aggregation == "peak":
            # Use pre-calculated peak interval
            vehicles_per_zone = vehicles_data["intervals"][peak_interval_idx, :]

        elif self.time_aggregation == "sum":
            # Handle demand vs population weighting differently
            if self.demand_weighted:
                # DEMAND WEIGHTING: Calculate variance across all intervals
                # Each interval contributes: variance(vehicles[i], demand[i])
                # Then average variances across intervals
                interval_variances = []

                for interval_idx in range(vehicles_data["intervals"].shape[0]):
                    vehicles_this_interval = vehicles_data["intervals"][interval_idx, :]
                    demand_this_interval = self.demand_per_zone_interval[:, interval_idx]

                    # Calculate variance for this interval
                    interval_variance = calculate_demand_weighted_variance(
                        vehicles_this_interval, demand_this_interval, self.demand_power
                    )
                    interval_variances.append(interval_variance)

                # Return average variance across intervals
                return float(np.mean(interval_variances))

            else:
                # POPULATION/UNWEIGHTED: Original logic (sum vehicles per zone)
                # Variance of total service across all time periods
                vehicles_per_zone = vehicles_data["sum"]

        elif self.time_aggregation == "intervals":
            return self._evaluate_intervals_separately(vehicles_data, peak_interval_idx, interval_idx_context=True)

        else:
            raise ValueError(
                f"Unknown time_aggregation: {self.time_aggregation}. Must be one of 'average', 'peak', 'sum', 'intervals'"
            )

        # Calculate variance (only reached if NOT demand_weighted + sum OR intervals)
        if len(vehicles_per_zone) > 1 and np.sum(vehicles_per_zone) > 0:
            # Choose calculation method based on parameters
            if self.demand_weighted and self.spatial_lag:
                variance = self._calculate_demand_weighted_spatial_variance(
                    vehicles_per_zone, vehicles_data, peak_interval_idx
                )
            elif self.demand_weighted:
                variance = self._calculate_demand_weighted_variance(vehicles_per_zone, vehicles_data, peak_interval_idx)
            elif self.population_weighted and self.spatial_lag:
                variance = self._calculate_population_weighted_spatial_variance(vehicles_per_zone)
            elif self.population_weighted:
                variance = self._calculate_population_weighted_variance(vehicles_per_zone)
            elif self.spatial_lag:
                variance = self._calculate_spatial_lag_variance(vehicles_per_zone)
            else:
                variance = np.var(vehicles_per_zone)  # Standard variance

            logger.debug(
                "📊 Vehicles per zone (%s): min=%d, max=%d, var=%.2f",
                self.time_aggregation,
                np.min(vehicles_per_zone),
                np.max(vehicles_per_zone),
                variance,
            )
            return float(variance)
        else:
            return 0.0

    def _calculate_spatial_lag_variance(self, vehicles_per_zone: np.ndarray) -> float:
        """Calculate variance using spatial lag (accessibility scores)."""
        accessibility_scores = self.spatial_system._calculate_accessibility_scores(vehicles_per_zone, self.alpha)
        return np.var(accessibility_scores)

    def _calculate_population_weighted_variance(self, vehicles_per_zone: np.ndarray) -> float:
        """Use shared population-weighted variance calculation."""
        return calculate_population_weighted_variance(
            vehicles_per_zone, self.population_per_zone, self.population_power
        )

    def _calculate_population_weighted_spatial_variance(self, vehicles_per_zone: np.ndarray) -> float:
        """Calculate population-weighted variance using spatial lag accessibility."""
        accessibility_scores = self.spatial_system._calculate_accessibility_scores(vehicles_per_zone, self.alpha)

        # Use shared variance calculation
        return calculate_population_weighted_variance(
            accessibility_scores, self.population_per_zone, self.population_power
        )

    def _calculate_demand_weighted_variance(self, vehicles_per_zone, vehicles_data, peak_interval_idx=None):
        demand_per_zone = self._aggregate_demand_for_weighting(peak_interval_idx)
        return calculate_demand_weighted_variance(vehicles_per_zone, demand_per_zone, self.demand_power)

    def _calculate_demand_weighted_spatial_variance(self, vehicles_per_zone, vehicles_data, peak_interval_idx=None):
        accessibility_scores = self.spatial_system._calculate_accessibility_scores(vehicles_per_zone, self.alpha)
        demand_per_zone = self._aggregate_demand_for_weighting(peak_interval_idx)
        return calculate_demand_weighted_variance(accessibility_scores, demand_per_zone, self.demand_power)

    def get_detailed_analysis(self, solution_matrix: np.ndarray) -> dict[str, Any]:
        """Get detailed spatial equity analysis."""
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        analysis = {
            "vehicles_per_zone_average": vehicles_data["average"],
            "vehicles_per_zone_peak": vehicles_data["peak"],
            "vehicles_per_zone_intervals": vehicles_data["intervals"],
            "interval_labels": vehicles_data["interval_labels"],
            "total_vehicles_average": np.sum(vehicles_data["average"]),
            "total_vehicles_peak": np.sum(vehicles_data["peak"]),
            "variance_average": np.var(vehicles_data["average"]),
            "variance_peak": np.var(vehicles_data["peak"]),
            "std_dev_average": np.std(vehicles_data["average"]),
            "std_dev_peak": np.std(vehicles_data["peak"]),
            "mean_vehicles_average": np.mean(vehicles_data["average"]),
            "mean_vehicles_peak": np.mean(vehicles_data["peak"]),
            "zones_with_service_average": np.sum(vehicles_data["average"] > 0),
            "zones_with_service_peak": np.sum(vehicles_data["peak"] > 0),
            "zones_without_service": len(vehicles_data["average"]) - np.sum(vehicles_data["average"] > 0),
            "coefficient_of_variation_average": (
                (np.std(vehicles_data["average"]) / np.mean(vehicles_data["average"]))
                if np.mean(vehicles_data["average"]) > 0
                else 0
            ),
            "coefficient_of_variation_peak": (
                (np.std(vehicles_data["peak"]) / np.mean(vehicles_data["peak"]))
                if np.mean(vehicles_data["peak"]) > 0
                else 0
            ),
        }

        # Add spatial lag analysis if enabled
        if self.spatial_lag:
            accessibility_avg = self.spatial_system._calculate_accessibility_scores(
                vehicles_data["average"], self.alpha
            )
            accessibility_peak = self.spatial_system._calculate_accessibility_scores(vehicles_data["peak"], self.alpha)

            analysis.update(
                {
                    "accessibility_scores_average": accessibility_avg,
                    "accessibility_scores_peak": accessibility_peak,
                    "variance_accessibility_average": np.var(accessibility_avg),
                    "variance_accessibility_peak": np.var(accessibility_peak),
                    "mean_accessibility_average": np.mean(accessibility_avg),
                    "mean_accessibility_peak": np.mean(accessibility_peak),
                    "spatial_lag_alpha": self.alpha,
                }
            )

        return analysis

    def _aggregate_demand_for_weighting(self, peak_interval_idx: int | None = None) -> np.ndarray:
        """
        Aggregate demand across time intervals to match time_aggregation setting.

        Args:
            peak_interval_idx: Index of peak interval (only used for peak aggregation)

        Returns:
            Demand per zone, aggregated across time (n_zones,)
        """
        if self.time_aggregation == "average":
            return np.mean(self.demand_per_zone_interval, axis=1)

        elif self.time_aggregation == "sum":
            # SAFETY CHECK: This shouldn't be called for demand + sum
            # Sum aggregation with demand weighting is handled directly in evaluate()
            raise ValueError(
                "demand + sum aggregation should be handled in evaluate(), not in _aggregate_demand_for_weighting()"
            )

        elif self.time_aggregation == "peak":
            if peak_interval_idx is None:
                fleet_stats = self.opt_data["constraints"]["fleet_analysis"]["fleet_stats"]
                peak_interval_idx = fleet_stats["peak_interval"]
            return self.demand_per_zone_interval[:, peak_interval_idx]

        else:
            raise ValueError(f"Unknown time_aggregation: {self.time_aggregation}")

    def _evaluate_intervals_separately(
        self, vehicles_data: dict, peak_interval_idx: int, interval_idx_context: bool = False
    ) -> float:
        """
        Calculate variance for each interval separately, then average.

        Args:
            vehicles_data: Dict with 'intervals' key containing (n_intervals, n_zones) array
            peak_interval_idx: Peak interval index (not used here, but kept for consistency)
            interval_idx_context: If True, use per-interval demand weighting

        Returns:
            Average variance across all intervals
        """
        interval_variances = []

        for interval_idx in range(vehicles_data["intervals"].shape[0]):
            vehicles_this_interval = vehicles_data["intervals"][interval_idx, :]

            # Handle demand weighting with interval context
            if self.demand_weighted and interval_idx_context:
                # Use demand from THIS specific interval
                demand_this_interval = self.demand_per_zone_interval[:, interval_idx]

                if self.spatial_lag:
                    # Calculate accessibility scores
                    accessibility_scores = self.spatial_system._calculate_accessibility_scores(
                        vehicles_this_interval, self.alpha
                    )

                    # Calculate demand-weighted variance of accessibility
                    interval_var = calculate_demand_weighted_variance(
                        accessibility_scores, demand_this_interval, self.demand_power
                    )
                else:
                    # Calculate demand-weighted variance directly
                    interval_var = calculate_demand_weighted_variance(
                        vehicles_this_interval, demand_this_interval, self.demand_power
                    )

            # Handle other weighting methods (population, spatial lag, unweighted)
            elif self.population_weighted and self.spatial_lag:
                accessibility_scores = self.spatial_system._calculate_accessibility_scores(
                    vehicles_this_interval, self.alpha
                )
                interval_var = calculate_population_weighted_variance(
                    accessibility_scores, self.population_per_zone, self.population_power
                )

            elif self.population_weighted:
                interval_var = calculate_population_weighted_variance(
                    vehicles_this_interval, self.population_per_zone, self.population_power
                )

            elif self.spatial_lag:
                accessibility_scores = self.spatial_system._calculate_accessibility_scores(
                    vehicles_this_interval, self.alpha
                )
                interval_var = np.var(accessibility_scores)

            else:
                # Standard unweighted variance
                interval_var = np.var(vehicles_this_interval)

            interval_variances.append(interval_var)

        # Return average variance across intervals
        return float(np.mean(interval_variances))

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
        Visualize spatial coverage (vehicle distribution).

        Args:
            solution_matrix: Decision matrix (PT-only) or dict with 'pt'/'drt' keys
            aggregation: 'average', 'peak', or 'intervals'
            interval_idx: Specific interval index (only used when aggregation='intervals')
            figsize: Figure size (only used if ax is None)
            show_stops: Whether to show transit stops
            show_drt_zones: Whether to show DRT service areas (auto-detects if None)
            ax: Optional matplotlib axis to plot on
            vmin: Minimum value for color scale (auto-calculated if None)
            vmax: Maximum value for color scale (auto-calculated if None)

        Returns:
            Tuple of (figure, axis) objects
        """
        # Calculate vehicles per zone using objective's data
        vehicles_data = self.spatial_system._vehicles_per_zone(solution_matrix, self.opt_data)

        if aggregation == "intervals" and interval_idx is not None:
            data_per_zone = vehicles_data["intervals"][interval_idx, :]
            interval_labels = vehicles_data["interval_labels"]
            title_suffix = f"(Interval {interval_idx}: {interval_labels[interval_idx]})"
        else:
            data_per_zone = vehicles_data[aggregation]
            title_suffix = f"({aggregation.capitalize()} Coverage)"

        # Use generic visualization infrastructure
        return self.spatial_system._visualize_with_data(
            data_per_zone=data_per_zone,
            data_column_name="vehicles",
            data_label="Vehicles per Zone",
            colormap="YlOrRd",
            optimization_data=self.opt_data,
            title_suffix=title_suffix,
            figsize=figsize,
            show_stops=show_stops,
            show_drt_zones=show_drt_zones,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )

    # Keep the old method for backward compatibility
    def visualize_spatial_coverage(self, *args, **kwargs):
        """Deprecated: Use visualize() instead."""
        import warnings

        warnings.warn(
            "visualize_spatial_coverage() is deprecated. Use visualize() instead.", DeprecationWarning, stacklevel=2
        )
        return self.visualize(*args, **kwargs)
