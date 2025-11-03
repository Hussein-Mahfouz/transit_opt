"""
Study area boundary management for spatial transit optimization.

This module provides tools for defining, loading, and working with geographic
boundaries. It handles CRS validation, spatial
filtering, and coordinate transformations.
"""

import logging
from typing import Literal

import geopandas as gpd

logger = logging.getLogger(__name__)


class StudyAreaBoundary:
    """
    Manages study area boundaries

    This class handles loading, validating, and working with geographic boundaries
    that define the study area for transit optimization. It enforces the use of
    metric coordinate systems and provides methods for spatial filtering of
    transit network elements.

    Key features:
    - Automatic CRS validation (enforces metric systems)
    - Spatial filtering of points and grid cells
    - Buffer operations
    - Multiple loading options (file, bounds, center+radius)

    Attributes:
        target_crs (str): Target coordinate reference system (must be metric)
        buffer_km (float): Buffer distance in kilometers applied to boundary
        boundary_gdf (gpd.GeoDataFrame): The boundary geometry in target CRS

    Examples:
        Load boundary from file:

        >>> boundary = StudyAreaBoundary.from_file(
        ...     "study_area.geojson",
        ...     crs="EPSG:3857",
        ...     buffer_km=2.0
        ... )

        Filter transit stops to study area:

        >>> filtered_stops = boundary.filter_points(gtfs_stops_gdf)

        Create from bounding box:

        >>> boundary = StudyAreaBoundary.from_bounds(
        ...     minx=-79.0, miny=35.9,
        ...     maxx=-78.8, maxy=36.1,
        ...     crs="EPSG:3857"
        ... )
    """

    def __init__(
        self,
        boundary_gdf: gpd.GeoDataFrame | None = None,
        crs: str = "EPSG:3857",
        buffer_km: float = 0.0,
    ) -> None:
        """
        Initialize a StudyAreaBoundary instance.

        Args:
            boundary_gdf: GeoDataFrame containing boundary geometry.
                         Can be in any CRS - will be converted to target CRS.
                         If None, boundary must be set later using set_boundary()
                         or class methods.
            crs: Target coordinate reference system. Must be a metric CRS
                 (e.g., 'EPSG:3857', 'EPSG:32617'). Geographic CRS like
                 'EPSG:4326' are rejected.
            buffer_km: Buffer distance in kilometers to apply to the boundary.
                      Use 0.0 for no buffer. Must be >= 0.

        Raises:
            ValueError: If CRS is geographic (lat/lon) or uses non-metric units
            ValueError: If buffer_km is negative
            ImportError: If pyproj is not available for CRS validation

        Examples:
            Create with existing GeoDataFrame:

            >>> gdf = gpd.read_file("boundary.shp")
            >>> boundary = StudyAreaBoundary(gdf, crs="EPSG:3857", buffer_km=1.5)
        """
        if buffer_km < 0:
            raise ValueError("Buffer distance cannot be negative")

        self.target_crs = self._validate_metric_crs(crs)
        self.buffer_km = buffer_km
        self.boundary_gdf = None

        if boundary_gdf is not None:
            self.set_boundary(boundary_gdf)

    def _validate_metric_crs(self, crs: str) -> str:
        """
        Validate that CRS is metric and suitable for distance calculations.

        Args:
            crs: Coordinate reference system string (e.g., 'EPSG:3857')

        Returns:
            Validated CRS string

        Raises:
            ValueError: If CRS is geographic or uses non-metric units
            ImportError: If pyproj is not available

        Note:
            This method requires pyproj for CRS validation. Geographic CRS
            (latitude/longitude) are rejected because buffer and distance
            calculations require metric units.
        """
        try:
            import pyproj

            crs_info = pyproj.CRS(crs)

            if crs_info.is_geographic:
                raise ValueError(
                    f"CRS {crs} is geographic (lat/lon). Please use a metric CRS like EPSG:3857 or a local UTM zone."
                )

            # Check units are metric
            if hasattr(crs_info.axis_info[0], "unit_name"):
                unit = crs_info.axis_info[0].unit_name.lower()
                if "metre" not in unit and "meter" not in unit:
                    raise ValueError(
                        f"CRS {crs} uses non-metric units ({unit}). Please use a metric CRS."
                    )

            logger.info("✅ Validated metric CRS: %s", crs)
            return crs

        except ImportError:
            raise ImportError(
                "pyproj is required for CRS validation. Install with: pip install pyproj"
            )
        except Exception as e:
            if "geographic" in str(e) or "metric" in str(e):
                raise  # Re-raise our validation errors
            else:
                raise ValueError(f"Invalid CRS {crs}: {e}")

    def set_boundary(self, boundary_gdf: gpd.GeoDataFrame) -> None:
        """
        Set the boundary from a GeoDataFrame with automatic processing.

        This method handles:
        - CRS conversion to target CRS if needed
        - Buffer application if specified
        - Dissolving multiple polygons into a single geometry
        - Validation that boundary is not empty

        Args:
            boundary_gdf: GeoDataFrame containing boundary polygon(s).
                         Can be in any CRS - will be converted automatically.

        Raises:
            ValueError: If boundary_gdf is empty or contains no valid geometry

        Examples:
            Set boundary from existing GeoDataFrame:

            >>> boundary_gdf = gpd.read_file("study_area.geojson")
            >>> boundary_obj.set_boundary(boundary_gdf)
        """
        if len(boundary_gdf) == 0:
            raise ValueError("Boundary GeoDataFrame is empty")

        if boundary_gdf.geometry.isna().all():
            raise ValueError("Boundary contains no valid geometry")

        original_crs = boundary_gdf.crs

        # Convert to target CRS if different
        if original_crs != self.target_crs:
            logger.info("🔄 Converting boundary CRS: %s → %s", original_crs, self.target_crs)
            boundary_gdf = boundary_gdf.to_crs(self.target_crs)

        # Apply buffer if specified
        if self.buffer_km > 0:
            boundary_gdf = self._apply_buffer(boundary_gdf)

        # Dissolve multiple polygons into one if needed
        if len(boundary_gdf) > 1:
            boundary_gdf = boundary_gdf.dissolve().reset_index(drop=True)

        self.boundary_gdf = boundary_gdf
        logger.info("✅ Study area set: %d polygon(s) in %s", len(boundary_gdf), self.target_crs)

    @classmethod
    def from_file(
        cls, file_path: str, crs: str = "EPSG:3857", buffer_km: float = 0.0
    ) -> "StudyAreaBoundary":
        """
        Create StudyAreaBoundary from a spatial data file.

        Supports any format readable by GeoPandas (GeoJSON, Shapefile, etc.).
        The boundary will be automatically converted to the specified metric CRS.

        Args:
            file_path: Path to spatial data file (e.g., .geojson, .shp, .gpkg)
            crs: Target metric CRS for the boundary
            buffer_km: Buffer distance in kilometers to apply

        Returns:
            StudyAreaBoundary instance with loaded boundary

        Raises:
            ValueError: If file cannot be loaded or CRS is invalid
            FileNotFoundError: If file does not exist

        Examples:
            Load from GeoJSON with 2km buffer:

            >>> boundary = StudyAreaBoundary.from_file(
            ...     "durham_boundary.geojson",
            ...     crs="EPSG:3857",
            ...     buffer_km=2.0
            ... )

            Load Shapefile in local UTM zone:

            >>> boundary = StudyAreaBoundary.from_file(
            ...     "study_area.shp",
            ...     crs="EPSG:32617"  # UTM Zone 17N
            ... )
        """
        try:
            boundary_gdf = gpd.read_file(file_path)

            logger.info("📁 Loaded boundary from %s", file_path)
            logger.info("   Original CRS: %s, Features: %d", boundary_gdf.crs, len(boundary_gdf))

            return cls(boundary_gdf, crs, buffer_km)

        except Exception as e:
            logger.error("❌ Error loading boundary from %s: %s", file_path, e)
            raise ValueError(f"Could not load boundary from {file_path}: {e}")

    @classmethod
    def from_bounds(
        cls,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        input_crs: str = "EPSG:4326",
        crs: str = "EPSG:3857",
        buffer_km: float = 0.0,
    ) -> "StudyAreaBoundary":
        """
        Create rectangular boundary from bounding box coordinates.

        Args:
            minx: Minimum X coordinate (longitude if EPSG:4326)
            miny: Minimum Y coordinate (latitude if EPSG:4326)
            maxx: Maximum X coordinate (longitude if EPSG:4326)
            maxy: Maximum Y coordinate (latitude if EPSG:4326)
            input_crs: CRS of the input coordinates (default: EPSG:4326)
            crs: Target metric CRS for the boundary
            buffer_km: Buffer distance in kilometers

        Returns:
            StudyAreaBoundary instance with rectangular boundary

        Examples:
            Create boundary around Durham, NC:

            >>> boundary = StudyAreaBoundary.from_bounds(
            ...     minx=-79.0, miny=35.9,
            ...     maxx=-78.8, maxy=36.1,
            ...     crs="EPSG:3857"
            ... )
        """
        from shapely.geometry import box

        # Create rectangular polygon
        rect = box(minx, miny, maxx, maxy)
        boundary_gdf = gpd.GeoDataFrame({"geometry": [rect]}, crs=input_crs)

        return cls(boundary_gdf, crs, buffer_km)

    @classmethod
    def from_center_radius(
        cls,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        crs: str = "EPSG:3857",
    ) -> "StudyAreaBoundary":
        """
        Create circular boundary around a center point.

        Args:
            center_lat: Latitude of center point (EPSG:4326)
            center_lon: Longitude of center point (EPSG:4326)
            radius_km: Radius in kilometers
            crs: Target metric CRS for the boundary

        Returns:
            StudyAreaBoundary instance with circular boundary

        Examples:
            Create 10km radius around downtown Durham:

            >>> boundary = StudyAreaBoundary.from_center_radius(
            ...     center_lat=35.994,
            ...     center_lon=-78.899,
            ...     radius_km=10.0,
            ...     crs="EPSG:3857"
            ... )
        """
        from shapely.geometry import Point

        # Create point in geographic coordinates
        center_point = Point(center_lon, center_lat)
        point_gdf = gpd.GeoDataFrame({"geometry": [center_point]}, crs="EPSG:4326")

        # Convert to metric CRS and buffer
        point_metric = point_gdf.to_crs(crs)
        buffered = point_metric.buffer(radius_km * 1000)  # Convert km to meters

        boundary_gdf = gpd.GeoDataFrame({"geometry": buffered}, crs=crs)

        return cls(boundary_gdf, crs, buffer_km=0.0)  # Buffer already applied

    @classmethod
    def auto_detect_from_stops(
        cls, stops_gdf: gpd.GeoDataFrame, buffer_km: float = 5.0, crs: str = "EPSG:3857"
    ) -> "StudyAreaBoundary":
        """
        Auto-detect study area from transit stop distribution.

        Creates a convex hull around transit stops with an additional buffer
        to define the study area boundary.

        Args:
            stops_gdf: GeoDataFrame of transit stops
            buffer_km: Buffer distance around stop convex hull
            crs: Target metric CRS for the boundary

        Returns:
            StudyAreaBoundary instance covering transit network extent

        Examples:
            Auto-detect boundary from GTFS stops:

            >>> stops_gdf = gpd.read_file("gtfs_stops.geojson")
            >>> boundary = StudyAreaBoundary.auto_detect_from_stops(
            ...     stops_gdf,
            ...     buffer_km=3.0
            ... )
        """
        # Convert stops to target CRS if needed
        if stops_gdf.crs != crs:
            stops_metric = stops_gdf.to_crs(crs)
        else:
            stops_metric = stops_gdf

        # Create convex hull around all stops
        from shapely.ops import unary_union

        all_points = unary_union(stops_metric.geometry)
        convex_hull = all_points.convex_hull

        boundary_gdf = gpd.GeoDataFrame({"geometry": [convex_hull]}, crs=crs)

        return cls(boundary_gdf, crs, buffer_km)

    def _apply_buffer(
        self, boundary_gdf: gpd.GeoDataFrame, buffer_km: float | None = None
    ) -> gpd.GeoDataFrame:
        """
        Apply buffer to boundary geometry.

        Args:
            boundary_gdf: Boundary in metric CRS
            buffer_km: Buffer radius in kilometers (uses instance value if None)

        Returns:
            Buffered boundary GeoDataFrame

        Note:
            Assumes boundary is already in metric CRS for accurate buffering.
        """
        buffer_radius = buffer_km if buffer_km is not None else self.buffer_km

        if buffer_radius <= 0:
            return boundary_gdf

        # Convert km to meters for buffering
        buffer_m = buffer_radius * 1000
        buffered = boundary_gdf.copy()
        buffered.geometry = boundary_gdf.geometry.buffer(buffer_m)

        logger.info("📏 Applied %dkm buffer to boundary layer", buffer_radius)
        return buffered

    def filter_points(
        self, points_gdf: gpd.GeoDataFrame, output_crs: str | None = None
    ) -> gpd.GeoDataFrame:
        """
        Filter points to those within the study area boundary.

        Performs spatial join to select only points that fall within the
        boundary polygon. Handles CRS conversion automatically.

        Args:
            points_gdf: GeoDataFrame of points to filter (e.g., transit stops)
            output_crs: CRS for the output (defaults to boundary CRS)

        Returns:
            Filtered GeoDataFrame containing only points within boundary,
            in the specified output CRS

        Raises:
            ValueError: If no boundary has been set

        Examples:
            Filter GTFS stops to study area:

            >>> gtfs_stops = gpd.read_file("stops.geojson")
            >>> filtered_stops = boundary.filter_points(
            ...     gtfs_stops,
            ...     output_crs="EPSG:4326"
            ... )
            >>> print(f"Kept {len(filtered_stops)}/{len(gtfs_stops)} stops")
        """
        if self.boundary_gdf is None:
            raise ValueError(
                "No boundary set. Use set_boundary() or from_file() first."
            )

        # Determine output CRS
        if output_crs is None:
            output_crs = self.target_crs

        # Convert points to boundary CRS for filtering
        if points_gdf.crs != self.boundary_gdf.crs:
            points_for_filtering = points_gdf.to_crs(self.boundary_gdf.crs)
        else:
            points_for_filtering = points_gdf

        # Spatial filter using 'within' predicate
        filtered_points = gpd.sjoin(
            points_for_filtering, self.boundary_gdf, how="inner", predicate="within"
        )

        # Keep only original columns (remove join artifacts)
        original_columns = points_gdf.columns
        filtered_points = filtered_points[original_columns]

        # Convert to desired output CRS if different
        if filtered_points.crs != output_crs:
            logger.info("🔄 Converting output to %s", output_crs)
            filtered_points = filtered_points.to_crs(output_crs)

        logger.info("🔍 Filtered %d → %d points", len(points_gdf), len(filtered_points))
        return filtered_points

    def filter_grid(
        self,
        grid_gdf: gpd.GeoDataFrame,
        predicate: Literal["intersects", "within", "contains"] = "intersects",
        output_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter grid cells based on spatial relationship with boundary.

        Args:
            grid_gdf: GeoDataFrame of grid cells (e.g., hexagonal zones)
            predicate: Spatial relationship to test:
                      - 'intersects': cells that touch or overlap boundary
                      - 'within': cells completely inside boundary
                      - 'contains': cells that completely contain boundary
            output_crs: CRS for the output (defaults to boundary CRS)

        Returns:
            Filtered grid GeoDataFrame in specified CRS

        Raises:
            ValueError: If no boundary has been set

        Examples:
            Filter hexagonal grid to boundary:

            >>> hex_grid = create_hexagonal_grid(resolution_km=2.0)
            >>> filtered_grid = boundary.filter_grid(
            ...     hex_grid,
            ...     predicate="intersects"
            ... )
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        # Determine output CRS
        if output_crs is None:
            output_crs = self.target_crs

        # Convert grid to boundary CRS for filtering
        if grid_gdf.crs != self.boundary_gdf.crs:
            grid_for_filtering = grid_gdf.to_crs(self.boundary_gdf.crs)
        else:
            grid_for_filtering = grid_gdf

        # Spatial filter
        filtered_grid = gpd.sjoin(
            grid_for_filtering, self.boundary_gdf, how="inner", predicate=predicate
        )

        # Keep only original columns and remove duplicates
        original_columns = grid_gdf.columns
        filtered_grid = filtered_grid[original_columns].drop_duplicates()

        # Convert to desired output CRS if different
        if filtered_grid.crs != output_crs:
            logger.info("🔄 Converting output to %s", output_crs)
            filtered_grid = filtered_grid.to_crs(output_crs)

        logger.info("🔍 Filtered %d → %d grid cells", len(grid_gdf), len(filtered_grid))
        return filtered_grid

    def get_boundary(self, output_crs: str | None = None) -> gpd.GeoDataFrame:
        """
        Get boundary geometry in specified CRS.

        Args:
            output_crs: Desired CRS (defaults to boundary's current CRS)

        Returns:
            Copy of boundary GeoDataFrame in specified CRS

        Raises:
            ValueError: If no boundary has been set

        Examples:
            Get boundary in geographic coordinates for mapping:

            >>> boundary_geo = boundary.get_boundary("EPSG:4326")
            >>> boundary_geo.plot()
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        if output_crs is None or output_crs == self.boundary_gdf.crs:
            return self.boundary_gdf.copy()
        else:
            logger.info("🔄 Converting boundary: %s → %s", self.boundary_gdf.crs, output_crs)
            return self.boundary_gdf.to_crs(output_crs)

    def add_buffer(
        self, buffer_km: float, update_boundary: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Add buffer to the current boundary.

        Args:
            buffer_km: Buffer radius in kilometers (must be > 0)
            update_boundary: Whether to update the instance boundary or return new one

        Returns:
            Buffered boundary GeoDataFrame

        Raises:
            ValueError: If no boundary set or buffer_km <= 0

        Examples:
            Add 3km buffer to existing boundary:

            >>> buffered_boundary = boundary.add_buffer(3.0)
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        if buffer_km <= 0:
            raise ValueError("Buffer distance must be positive")

        buffered = self._apply_buffer(self.boundary_gdf, buffer_km)

        if update_boundary:
            self.boundary_gdf = buffered
            self.buffer_km = buffer_km
            logger.info("✅ Updated boundary with %dkm buffer", buffer_km)

        return buffered

    def visualize(self, ax=None, viz_crs: str | None = None, **plot_kwargs):
        """
        Visualize the boundary on a map.

        Args:
            ax: Matplotlib axis (created if None)
            viz_crs: CRS for visualization (defaults to boundary CRS)
            **plot_kwargs: Additional plotting arguments passed to GeoPandas plot()

        Returns:
            Matplotlib axis with boundary plotted

        Raises:
            ValueError: If no boundary has been set

        Examples:
            Quick visualization:

            >>> boundary.visualize()

            Custom styling:

            >>> ax = boundary.visualize(
            ...     viz_crs="EPSG:4326",
            ...     facecolor="lightgreen",
            ...     edgecolor="darkgreen",
            ...     alpha=0.7
            ... )
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        if viz_crs is None:
            viz_crs = self.boundary_gdf.crs

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get boundary in visualization CRS
        viz_boundary = self.get_boundary(output_crs=viz_crs)

        # Default styling
        default_kwargs = {
            "facecolor": "lightblue",
            "edgecolor": "darkblue",
            "alpha": 0.3,
        }
        default_kwargs.update(plot_kwargs)

        viz_boundary.plot(ax=ax, **default_kwargs)
        ax.set_title(f"Study Area (CRS: {viz_crs})")

        if viz_crs == "EPSG:4326":
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        return ax

    def __repr__(self) -> str:
        """String representation of StudyAreaBoundary."""
        if self.boundary_gdf is None:
            return f"StudyAreaBoundary(crs={self.target_crs}, no boundary set)"
        else:
            n_polygons = len(self.boundary_gdf)
            area_km2 = self.boundary_gdf.geometry.area.sum() / 1e6  # m² to km²
            return (
                f"StudyAreaBoundary(crs={self.target_crs}, "
                f"{n_polygons} polygon(s), {area_km2:.1f} km²)"
            )
