from typing import Any, Optional

import numpy as np

from ..spatial.boundaries import StudyAreaBoundary
from ..spatial.zoning import HexagonalZoneSystem
from .base import BaseSpatialObjective


class HexagonalCoverageObjective(BaseSpatialObjective):
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
                                        'average', 'peak'. Defaults to 'average'.
        spatial_lag (bool, optional): Enable spatial lag accessibility.
                                    Defaults to False.
        alpha (float, optional): Spatial lag decay factor [0,1]. Higher values
                               give more weight to neighbors. Defaults to 0.1.
        population_weighted (bool, optional): Enable population weighting.
                                           Currently placeholder. Defaults to False.

    Mathematical Details:
        Standard variance: σ² = Σ(vᵢ - μ)² / n

        Spatial lag: accessibility_i = vehicles_i + α × Σ(w_ij × vehicles_j)
        where w_ij are spatial weights (1 if neighbors, 0 otherwise)

        Population weighting (planned): σ²_w = Σ(p_i × (x_i - μ_w)²) / Σ(p_i)

    Example:
        ```python
        # Standard equity objective
        equity_obj = HexagonalCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=3.0
        )

        # With spatial lag and boundary filtering
        spatial_equity_obj = HexagonalCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=2.0,
            boundary=study_boundary,
            spatial_lag=True,
            alpha=0.15  # 15% neighbor influence
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
        # NEW SPATIAL LAG PARAMETERS:
        spatial_lag: bool = False,
        alpha: float = 0.1,
        # PLACEHOLDER POPULATION PARAMETERS:
        population_weighted: bool = False,
    ):
        self.boundary = boundary
        self.time_aggregation = time_aggregation
        self.spatial_lag = spatial_lag
        self.alpha = alpha  # Decay factor for neighbor influence
        self.population_weighted = population_weighted
        super().__init__(optimization_data, spatial_resolution_km, crs)

    def _create_spatial_system(self):
        """Use your existing HexagonalZoneSystem."""
        return HexagonalZoneSystem(
            gtfs_feed=self.gtfs_feed,
            hex_size_km=self.spatial_resolution,
            crs=self.crs,
            boundary=self.boundary,
        )

    def _get_spatial_summary(self) -> str:
        features = [f"{len(self.spatial_system.hex_grid)} hexagonal zones"]
        if self.spatial_lag:
            features.append(f"spatial lag (α={self.alpha})")
        if self.population_weighted:
            features.append("population weighted [PLACEHOLDER]")
        return ", ".join(features)

    def evaluate(self, solution_matrix: np.ndarray) -> float:
        """Minimize variance in vehicle distribution across hexagons."""
        vehicles_data = self.spatial_system._vehicles_per_zone(
            solution_matrix, self.opt_data
        )
        vehicles_per_zone = vehicles_data[self.time_aggregation]

        if len(vehicles_per_zone) > 1 and np.sum(vehicles_per_zone) > 0:
            # Choose calculation method based on parameters
            if self.population_weighted and self.spatial_lag:
                variance = self._calculate_population_weighted_spatial_variance(
                    vehicles_per_zone
                )
            elif self.population_weighted:
                variance = self._calculate_population_weighted_variance(
                    vehicles_per_zone
                )
            elif self.spatial_lag:
                variance = self._calculate_spatial_lag_variance(vehicles_per_zone)
            else:
                variance = np.var(vehicles_per_zone)  # Standard variance

            print(
                f"📊 Vehicles per zone ({self.time_aggregation}): min={np.min(vehicles_per_zone)}, "
                f"max={np.max(vehicles_per_zone)}, var={variance:.2f}"
            )
            return float(variance)
        else:
            return 0.0

    def _calculate_spatial_lag_variance(self, vehicles_per_zone: np.ndarray) -> float:
        """Calculate variance using spatial lag (accessibility scores)."""
        accessibility_scores = self.spatial_system._calculate_accessibility_scores(
            vehicles_per_zone, self.alpha
        )
        return np.var(accessibility_scores)

    def _calculate_population_weighted_variance(
        self, vehicles_per_zone: np.ndarray
    ) -> float:
        """
        Calculate population-weighted variance.

        PLACEHOLDER: Population data integration coming soon.
        Currently returns standard variance with warning.

        Formula:
        σ_w² = Σ(p_i * (x_i - μ_w)²) / Σ(p_i)
        where μ_w = Σ(p_i * x_i) / Σ(p_i)
        """
        print("⚠️  Population weighting requested but not yet implemented")
        print(
            "    Using standard variance. Population data integration coming in next update."
        )
        return np.var(vehicles_per_zone)

    def _calculate_population_weighted_spatial_variance(
        self, vehicles_per_zone: np.ndarray
    ) -> float:
        """
        Calculate population-weighted variance using spatial lag accessibility.

        PLACEHOLDER: Advanced combination of population weighting and spatial lag.
        Currently uses spatial lag only with warning.
        """
        print(
            "⚠️  Population-weighted spatial variance requested but population data not yet available"
        )
        print("    Using spatial lag variance only. Full implementation coming soon.")
        return self._calculate_spatial_lag_variance(vehicles_per_zone)

    def get_detailed_analysis(self, solution_matrix: np.ndarray) -> dict[str, Any]:
        """Get detailed spatial equity analysis."""
        vehicles_data = self.spatial_system._vehicles_per_zone(
            solution_matrix, self.opt_data
        )

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
            "zones_without_service": len(vehicles_data["average"])
            - np.sum(vehicles_data["average"] > 0),
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
            accessibility_peak = self.spatial_system._calculate_accessibility_scores(
                vehicles_data["peak"], self.alpha
            )

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
