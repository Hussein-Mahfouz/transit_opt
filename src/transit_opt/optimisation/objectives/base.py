# Add this new cell to test the objective classes
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class BaseObjective(ABC):
    """Base class for all optimization objectives."""

    def __init__(self, optimization_data: dict[str, Any]):
        self.opt_data = optimization_data
        self._baseline_value = None
        self.setup_data()

    @abstractmethod
    def setup_data(self):
        """Prepare any additional data structures needed."""
        pass

    @abstractmethod
    def evaluate(self, solution_matrix: np.ndarray) -> float:
        """Evaluate objective for a given solution matrix."""
        pass

    def evaluate_normalized(self, solution_matrix: np.ndarray) -> float:
        """Return normalized objective value (0-1 scale)."""
        raw_value = self.evaluate(solution_matrix)
        if self._baseline_value is None:
            self._baseline_value = self.evaluate(self.opt_data["initial_solution"])

        return (
            raw_value / self._baseline_value if self._baseline_value != 0 else raw_value
        )


class BaseSpatialObjective(BaseObjective):
    """
    Base class for optimization objectives requiring spatial analysis.

    This class provides common infrastructure for objectives that need to analyze
    spatial patterns of transit service

    Attributes:
        spatial_resolution (float): Spatial resolution in kilometers for analysis grid.
        crs (str): Coordinate Reference System (CRS) identifier. Should be metric
                   (e.g., "EPSG:3857" for Web Mercator).
        gtfs_feed: GTFS feed object from optimization data.
        spatial_system: Spatial analysis system (created by subclasses).

    Args:
        optimization_data (Dict[str, Any]): Complete optimization data structure
                                          from GTFSDataPreparator.
        spatial_resolution_km (float, optional): Grid resolution in kilometers.
                                               Defaults to 2.0.
        crs (str, optional): Coordinate reference system. Must be metric.
                           Defaults to "EPSG:3857" (Web Mercator).

    Note:
        The CRS should use metric units (meters) for accurate distance calculations.
        Geographic CRS like EPSG:4326 (lat/lon) will produce warnings and give
        inaccurate results.

    Example:
        ```python
        # Create a hexagonal coverage objective with 1.5km resolution
        spatial_obj = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=1.5,
            crs="EPSG:3857"
        )

        # Evaluate spatial equity for a solution
        equity_score = spatial_obj.evaluate(solution_matrix)
        ```
    """

    def __init__(
        self,
        optimization_data: dict[str, Any],
        spatial_resolution_km: float = 2.0,
        crs: str = "EPSG:4326",
    ):
        self.spatial_resolution = spatial_resolution_km
        self.crs = crs
        super().__init__(optimization_data)

    def setup_data(self):
        """Set up common spatial infrastructure."""
        logger.info(
            "🗺️ Setting up spatial analysis with %skm resolution",
            self.spatial_resolution
        )

        # Get GTFS feed from your opt_data structure
        self.gtfs_feed = self.opt_data["reconstruction"]["gtfs_feed"]

        # Create spatial system
        self.spatial_system = self._create_spatial_system()

        print(f"✅ Spatial system ready: {self._get_spatial_summary()}")

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
        **kwargs
    ):
        """
        Generic visualization method for all spatial objectives.

        This method should be overridden by each objective to provide
        objective-specific data conversion and visualization parameters.
        """
        raise NotImplementedError("Subclasses must implement visualize()")

    @abstractmethod
    def _create_spatial_system(self):
        """Create the specific spatial representation."""
        pass

    @abstractmethod
    def _get_spatial_summary(self) -> str:
        """Return summary string for logging."""
        pass
