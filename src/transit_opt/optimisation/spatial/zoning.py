from typing import Any, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import Queen, lag_spatial
from shapely.geometry import Point

from transit_opt.optimisation.spatial.boundaries import StudyAreaBoundary


class HexagonalZoneSystem:
    """
    Optimized hexagonal zoning system with spatial analysis capabilities.

    Creates a spatial grid system for analyzing transit service distribution using
    hexagonal (currently square) zones. Provides efficient vehicle counting,
    spatial lag calculations, and boundary filtering.

    Key Features:
    - Metric CRS validation: Ensures accurate distance calculations
    - Boundary filtering: Optional geographic boundary constraints
    - Spatial indexing: Optimized O(S+Z) stop-to-zone mapping
    - Spatial weights: Queen contiguity matrix for neighbor analysis
    - Route caching: Pre-computed route-stop mappings for performance

    Attributes:
        gtfs_feed: GTFS feed object with transit data.
        hex_size_km (float): Size of hexagonal zones in kilometers.
        crs (str): Coordinate reference system (validated as metric).
        boundary (Optional[StudyAreaBoundary]): Geographic boundary filter.
        stops_gdf (gpd.GeoDataFrame): Transit stops in metric CRS.
        hex_grid (gpd.GeoDataFrame): Hexagonal zone polygons.
        stop_zone_mapping (dict[str, str]): Maps stop_id → zone_id.
        route_stops_cache (dict[str, Set[str]]): Maps service_id → stop_ids.
        weights_matrix (libpysal.weights.W): Spatial contiguity matrix (lazy-loaded).

    Args:
        gtfs_feed: GTFS feed object from GTFSKit or similar.
        hex_size_km (float, optional): Hexagon diameter in kilometers.
                                     Defaults to 2.0.
        crs (str, optional): Coordinate reference system. Must be metric.
                           Defaults to "EPSG:3857" (Web Mercator).
        boundary (Optional[StudyAreaBoundary], optional): Geographic boundary
                                                        for filtering zones and stops.
                                                        Defaults to None.

    Raises:
        ImportError: If required spatial analysis libraries are missing.
        ValueError: If CRS validation fails or no stops remain after filtering.

    Performance Notes:
        - Spatial join: O(S + Z) complexity vs O(S × Z) for naive approach
        - Route caching: Avoids repeated GTFS filtering operations
        - Lazy weights matrix: Created only when spatial lag is needed
        - Boundary filtering: Applied to both stops and zones for efficiency

    Example:
        zone_system = HexagonalZoneSystem(
            gtfs_feed=gtfs_feed,
            hex_size_km=1.5,
            crs="EPSG:3857"
        )

        # With boundary filtering
        bounded_system = HexagonalZoneSystem(
            gtfs_feed=gtfs_feed,
            hex_size_km=2.0,
            crs="EPSG:32633",  # UTM Zone 33N
            boundary=study_area_boundary
        )

        # Calculate vehicles per zone
        vehicles_data = zone_system._vehicles_per_zone(
            solution_matrix,
            optimization_data
        )
        print(f"Average vehicles: {vehicles_data['average']}")
    """

    def __init__(
        self,
        gtfs_feed,
        hex_size_km: float = 2.0,
        crs: str = "EPSG:3857",
        boundary: Optional["StudyAreaBoundary"] = None,
    ):
        self.gtfs_feed = gtfs_feed
        self.hex_size_km = hex_size_km
        self.crs = crs
        self.boundary = boundary

        # Validate that CRS is metric
        self._validate_metric_crs()

        # Create stop locations GeoDataFrame
        self.stops_gdf = self._create_stops_geodataframe()

        # Apply boundary filtering if provided
        if self.boundary is not None:
            print(f"🎯 Applying boundary filter to {len(self.stops_gdf)} stops...")
            self.stops_gdf = self.boundary.filter_points(
                self.stops_gdf, output_crs=self.crs
            )
            print(f"✅ Filtered to {len(self.stops_gdf)} stops within boundary")

        # Generate hexagonal grid
        self.hex_grid = self._create_hexagonal_grid()

        # Optionally filter grid to boundary as well
        if self.boundary is not None:
            print(f"🎯 Applying boundary filter to {len(self.hex_grid)} grid cells...")
            self.hex_grid = self.boundary.filter_grid(
                self.hex_grid, predicate="intersects", output_crs=self.crs
            )
            print(f"✅ Filtered to {len(self.hex_grid)} grid cells within boundary")

        # OPTIMIZED: Use spatial join instead of nested loops
        self.stop_zone_mapping = self._fast_map_stops_to_zones()

        # OPTIMIZED: Pre-compute route-stop mappings
        self._precompute_route_stop_mappings()

        # Initialize spatial weights matrix as None (created on demand)
        self._weights_matrix = None

    def _validate_metric_crs(self):
        """Validate that the CRS uses metric units."""
        try:
            import pyproj

            crs_info = pyproj.CRS(self.crs)

            # Check if units are metric
            if hasattr(crs_info.axis_info[0], "unit_name"):
                unit = crs_info.axis_info[0].unit_name.lower()
                if "metre" not in unit and "meter" not in unit:
                    print(
                        f"⚠️  Warning: CRS {self.crs} may not be metric (units: {unit})"
                    )
                    print(
                        "   Consider using EPSG:3857 (Web Mercator) or a local UTM zone"
                    )

        except ImportError:
            print("⚠️  pyproj not available - cannot validate CRS units")
        except Exception as e:
            print(f"⚠️  Could not validate CRS {self.crs}: {e}")

    def _create_stops_geodataframe(self) -> gpd.GeoDataFrame:
        """Create stops GeoDataFrame and reproject to metric CRS."""
        stops = self.gtfs_feed.stops.copy()

        # Create geometry from lat/lon (EPSG:4326)
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(stops["stop_lon"], stops["stop_lat"], strict=False)
        ]
        stops_gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs="EPSG:4326")

        # Reproject to target metric CRS
        stops_gdf = stops_gdf.to_crs(self.crs)

        print(f"🗺️  Reprojected {len(stops_gdf)} stops to {self.crs}")
        return stops_gdf

    def _create_hexagonal_grid(self) -> gpd.GeoDataFrame:
        """Create hexagonal grid using metric coordinates."""
        bounds = self.stops_gdf.total_bounds

        # Convert km to meters (since we're now using metric CRS)
        hex_size_m = self.hex_size_km * 1000
        buffer_m = hex_size_m * 0.5  # 50% buffer

        minx, miny, maxx, maxy = bounds
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

        hex_polygons = []
        zone_ids = []

        # Calculate steps using metric coordinates
        x_steps = int((maxx - minx) / hex_size_m) + 1
        y_steps = int((maxy - miny) / hex_size_m) + 1

        print(f"🔧 Creating {x_steps} × {y_steps} = {x_steps * y_steps} grid cells")
        print(
            f"   Grid bounds: ({minx:.0f}, {miny:.0f}) to ({maxx:.0f}, {maxy:.0f}) meters"
        )
        print(f"   Cell size: {hex_size_m}m × {hex_size_m}m")

        zone_id = 0
        for i in range(x_steps):
            for j in range(y_steps):
                x = minx + i * hex_size_m
                y = miny + j * hex_size_m

                # Create square cells (TODO: implement proper hexagons)
                from shapely.geometry import Polygon

                cell = Polygon(
                    [
                        (x, y),
                        (x + hex_size_m, y),
                        (x + hex_size_m, y + hex_size_m),
                        (x, y + hex_size_m),
                    ]
                )

                hex_polygons.append(cell)
                zone_ids.append(f"zone_{zone_id}")
                zone_id += 1

        hex_gdf = gpd.GeoDataFrame(
            {"zone_id": zone_ids, "geometry": hex_polygons}, crs=self.crs
        )

        print(f"✅ Created {len(hex_gdf)} hexagonal zones in {self.crs}")
        return hex_gdf

    def _fast_map_stops_to_zones(self) -> dict[str, str]:
        """OPTIMIZED: Use spatial join - O(S + Z) instead of O(S × Z)."""
        print("🚀 Using spatial join for zone mapping...")

        # Spatial join: finds containing zone for each stop in one operation
        # stops_with_zones = gpd.sjoin(
        #     self.stops_gdf, self.hex_grid, how="left", predicate="within"
        # )
        hex_grid_for_join = self.hex_grid.reset_index()
        if "zone_id" not in hex_grid_for_join.columns:
            hex_grid_for_join["zone_id"] = hex_grid_for_join.index

        # Rename to avoid conflicts in spatial join
        hex_grid_for_join = hex_grid_for_join.rename(columns={"zone_id": "hex_zone_id"})

        stops_with_zones = gpd.sjoin(
            self.stops_gdf, hex_grid_for_join, how="left", predicate="within"
        )

        # Convert to dictionary
        stop_zone_map = {}
        for idx, row in stops_with_zones.iterrows():
            if pd.notna(row["hex_zone_id"]):
                stop_zone_map[row["stop_id"]] = row["hex_zone_id"]
            else:
                # Handle stops not in any zone (find nearest)
                stop_point = row.geometry
                distances = self.hex_grid.geometry.distance(stop_point)
                nearest_zone_idx = distances.idxmin()
                stop_zone_map[row["stop_id"]] = self.hex_grid.loc[
                    nearest_zone_idx, "zone_id"
                ]

        print(f"✅ Mapped {len(stop_zone_map)} stops to zones")
        return stop_zone_map

    def _precompute_route_stop_mappings(self):
        """OPTIMIZED: Pre-compute all route → stops mappings to avoid repeated filtering."""
        print("🚀 Pre-computing route-stop mappings...")

        self.route_stops_cache = {}

        # Group trips by service_id once
        trips_by_service = (
            self.gtfs_feed.trips.groupby("service_id")["trip_id"].apply(list).to_dict()
        )

        # Group stop_times by trip_id once
        stop_times_by_trip = (
            self.gtfs_feed.stop_times.groupby("trip_id")["stop_id"].apply(set).to_dict()
        )

        for service_id, trip_ids in trips_by_service.items():
            # Get all unique stops for this service
            service_stops = set()
            for trip_id in trip_ids:
                if trip_id in stop_times_by_trip:
                    service_stops.update(stop_times_by_trip[trip_id])

            self.route_stops_cache[service_id] = service_stops

        print(f"✅ Cached stops for {len(self.route_stops_cache)} routes/services")

    def _create_contiguity_matrix(self):
        """
        Create Queen contiguity matrix using libpysal.

        Queen contiguity: zones are neighbors if they share an edge or vertex.
        Returns row-standardized weights where each row sums to 1.0.

        Returns:
            libpysal.weights.W: Spatial weights matrix
        """

        print(f"🔧 Creating Queen contiguity matrix for {len(self.hex_grid)} zones...")

        # Create weights directly from GeoDataFrame
        w = Queen.from_dataframe(self.hex_grid, ids=self.hex_grid["zone_id"])

        # Row-standardize the weights matrix
        w.transform = "r"  # Row standardization (each row sums to 1)

        # Verify the matrix
        print("✅ Contiguity matrix created:")
        print(f"   Zones: {w.n}")
        print(f"   Links: {w.s0}")
        print(f"   Islands: {len(w.islands)} zones")

        return w

    @property
    def weights_matrix(self):
        """Get spatial weights matrix (create if not exists)."""
        if self._weights_matrix is None:
            self._weights_matrix = self._create_contiguity_matrix()
        return self._weights_matrix

    # In cell 8e3cf72a, replace the _calculate_spatial_lag method with this corrected version:

    def _calculate_spatial_lag(self, vehicles_per_zone: np.ndarray) -> np.ndarray:
        """
        Calculate spatial lag using libpysal weights.

        Args:
            vehicles_per_zone: Array of vehicles per zone (n_zones,)

        Returns:
            Array of spatial lag values (n_zones,)
        """
        w = self.weights_matrix

        # Direct calculation - libpysal maintains GeoDataFrame order
        spatial_lag = lag_spatial(w, vehicles_per_zone)

        print("📊 Spatial lag calculated:")
        print(f"   Input mean: {np.mean(vehicles_per_zone):.2f}")
        print(f"   Spatial lag mean: {np.mean(spatial_lag):.2f}")
        print(f"   Non-zero lags: {np.sum(spatial_lag > 0)}")

        return spatial_lag

    def _calculate_accessibility_scores(
        self, vehicles_per_zone: np.ndarray, alpha: float = 0.1
    ) -> np.ndarray:
        """
        Calculate accessibility scores incorporating neighbor service levels.

        Creates "accessibility score" that blends direct service with neighbor service:
        accessibility_i = vehicles_i + α * spatial_lag_i

        Args:
            vehicles_per_zone: Array of vehicles per zone (n_zones,)
            alpha: Decay factor for neighbor influence (0-1).
                   0 = no neighbor effect, 1 = equal weight to neighbors

        Returns:
            Array of accessibility scores (n_zones,)
        """
        # Calculate spatial lag
        spatial_lag = self._calculate_spatial_lag(vehicles_per_zone)

        # Blend direct service with neighbor accessibility
        accessibility_scores = vehicles_per_zone + alpha * spatial_lag

        print(f"📊 Spatial lag calculated (α={alpha}):")
        print(f"   Mean direct service: {np.mean(vehicles_per_zone):.2f}")
        print(f"   Mean neighbor service: {np.mean(spatial_lag):.2f}")
        print(f"   Mean accessibility: {np.mean(accessibility_scores):.2f}")
        print(
            f"   Zones with improved access: {np.sum(accessibility_scores > vehicles_per_zone)}"
        )

        return accessibility_scores

    def _assign_population_to_zones(self, population_data: gpd.GeoDataFrame) -> None:
        """
        Assign population values to hexagonal zones.

        PLACEHOLDER: Population-zone assignment coming soon.

        Args:
            population_data: GeoDataFrame with population values

        Planned Features:
            - Spatial join with area-weighted assignment
            - Handle overlapping geometries (proportional allocation)
            - Support for different population data resolutions
            - Zero-population zone handling (industrial areas, water bodies)

        Side Effects:
            Will add 'population' column to self.hex_grid
        """
        raise NotImplementedError(
            "Population assignment coming in next update. "
            "Will handle spatial join with area weighting for accurate population allocation."
        )

    def get_zone_population_stats(self) -> dict:
        """
        Get population statistics by zone.

        PLACEHOLDER: Population analysis by zone coming soon.

        Returns:
            Dictionary with zone-level population statistics
        """
        raise NotImplementedError("Zone population analysis coming in next update.")

    def _vehicles_per_zone(
        self, solution_matrix: np.ndarray, optimization_data: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """
        Calculate vehicles per zone with all aggregation types.

        Args:
            solution_matrix: Decision matrix (n_routes × n_intervals)
            optimization_data: Optimization data structure from GTFSDataPreparator

        Returns:
            Dictionary with all aggregation types:
            - 'average': Array of mean vehicles per zone across time intervals
            - 'peak': Array of max vehicles per zone across time intervals
            - 'intervals': 2D array (n_zones × n_intervals) with vehicles per zone per interval
            - 'interval_labels': List of time interval labels from preparator
        """
        n_zones = len(self.hex_grid)
        n_intervals = optimization_data["n_intervals"]

        # Extract data from optimization structure
        route_ids = optimization_data["routes"]["ids"]
        allowed_headways = optimization_data["allowed_headways"]
        round_trip_times = optimization_data["routes"]["round_trip_times"]
        no_service_index = optimization_data["no_service_index"]
        interval_labels = optimization_data["intervals"]["labels"]

        # Calculate vehicles per zone for each interval
        vehicles_by_intervals = np.zeros((n_zones, n_intervals))

        for interval_idx in range(n_intervals):
            zone_counts = {zone_id: 0 for zone_id in self.hex_grid["zone_id"]}

            for route_idx, service_id in enumerate(route_ids):
                if service_id not in self.route_stops_cache:
                    continue

                service_stops = self.route_stops_cache[service_id]

                # Get choice for this specific interval
                choice_idx = solution_matrix[route_idx, interval_idx]

                # Skip no-service choices
                if choice_idx == no_service_index:
                    continue

                headway = allowed_headways[choice_idx]

                if headway < 9000:  # Valid service headway
                    round_trip = round_trip_times[route_idx]
                    vehicles_in_interval = max(1, int(np.ceil(round_trip / headway)))

                    # Add vehicles to all zones served by this route
                    zones_served = {
                        self.stop_zone_mapping[stop_id]
                        for stop_id in service_stops
                        if stop_id in self.stop_zone_mapping
                    }

                    for zone_id in zones_served:
                        zone_counts[zone_id] += vehicles_in_interval

            # Store results for this interval
            vehicles_by_intervals[:, interval_idx] = list(zone_counts.values())

        # Return all aggregation types
        return {
            "average": np.mean(vehicles_by_intervals, axis=1),
            "peak": np.max(vehicles_by_intervals, axis=1),
            "intervals": vehicles_by_intervals,
            "interval_labels": interval_labels,
        }

    def get_zone_statistics(
        self,
        solution_matrix: np.ndarray,
        optimization_data: dict[str, Any],
        aggregation: str = "average",
    ) -> dict[str, Any]:
        """
        Get detailed statistics about zone service distribution.

        Args:
            solution_matrix: Decision matrix
            optimization_data: Optimization data
            aggregation: Type of aggregation ('average', 'peak', 'intervals')
        """
        vehicles_data = self._vehicles_per_zone(solution_matrix, optimization_data)

        if aggregation == "intervals":
            # Return statistics for interval-based analysis
            intervals_data = vehicles_data["intervals"]
            return {
                "analysis_type": "by_intervals",
                "total_zones": len(self.hex_grid),
                "intervals": vehicles_data["interval_labels"],
                "vehicles_by_intervals": intervals_data,
                "zones_with_service_by_interval": np.sum(intervals_data > 0, axis=0),
                "total_vehicles_by_interval": np.sum(intervals_data, axis=0),
                "mean_vehicles_by_interval": np.mean(intervals_data, axis=0),
                "variance_by_interval": np.var(intervals_data, axis=0),
            }
        else:
            # Return statistics for aggregated analysis
            vehicles_per_zone = vehicles_data[aggregation]
            return {
                "analysis_type": aggregation,
                "total_zones": len(self.hex_grid),
                "zones_with_service": np.sum(vehicles_per_zone > 0),
                "zones_without_service": np.sum(vehicles_per_zone == 0),
                "total_vehicles": np.sum(vehicles_per_zone),
                "mean_vehicles_per_zone": np.mean(vehicles_per_zone),
                "std_vehicles_per_zone": np.std(vehicles_per_zone),
                "variance_vehicles_per_zone": np.var(vehicles_per_zone),
                "min_vehicles": np.min(vehicles_per_zone),
                "max_vehicles": np.max(vehicles_per_zone),
                "vehicles_distribution": vehicles_per_zone,
            }

    def visualize_zones_and_stops(self, figsize=(15, 10)):
        """Create a visualization of zones and stops."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot zones
        self.hex_grid.plot(ax=ax, alpha=0.3, edgecolor="blue", linewidth=0.5)

        # Reproject stops back to geographic for visualization
        stops_geo = self.stops_gdf.to_crs("EPSG:4326")
        stops_geo.plot(ax=ax, color="red", markersize=1, alpha=0.7)

        ax.set_title(
            f"Transit System Zones: {len(self.hex_grid)} zones, {len(self.stops_gdf)} stops"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                alpha=0.3,
                label=f"{self.hex_size_km}km Zones",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=5,
                label="Bus Stops",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        return fig, ax

    def visualize_spatial_coverage(
        self,
        solution_matrix: np.ndarray,
        optimization_data: dict[str, Any],
        aggregation: str = "average",
        interval_idx: int | None = None,
        figsize=(15, 12),
        show_stops=True,
        ax=None,
        vmin=None,  # Add this parameter
        vmax=None,  # Add this parameter
    ):
        """
        Create spatial visualization showing zones colored by vehicle count.

        Args:
            solution_matrix: Decision matrix
            optimization_data: Optimization data
            aggregation: 'average', 'peak', or 'intervals'
            interval_idx: Specific interval index (only used when aggregation='intervals')
            figsize: Figure size (only used if ax is None)
            show_stops: Whether to show transit stops
            ax: Optional matplotlib axis to plot on
            vmin: Minimum value for color scale (auto-calculated if None)
            vmax: Maximum value for color scale (auto-calculated if None)
        """
        vehicles_data = self._vehicles_per_zone(solution_matrix, optimization_data)

        if aggregation == "intervals" and interval_idx is not None:
            # Show specific interval
            vehicles_per_zone = vehicles_data["intervals"][:, interval_idx]
            interval_labels = vehicles_data["interval_labels"]
            title_suffix = f"(Interval {interval_idx}: {interval_labels[interval_idx]})"
        else:
            # Show aggregated data
            vehicles_per_zone = vehicles_data[aggregation]
            title_suffix = f"({aggregation.capitalize()} Service)"

        # Create a copy of hex_grid with vehicle counts
        zones_with_vehicles = self.hex_grid.copy()
        zones_with_vehicles["vehicles"] = vehicles_per_zone

        # Convert to geographic CRS for plotting
        zones_geo = zones_with_vehicles.to_crs("EPSG:4326")
        stops_geo = self.stops_gdf.to_crs("EPSG:4326")

        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_figure = True
        else:
            fig = ax.figure
            created_figure = False

        # Plot zones with vehicle-based coloring
        if vehicles_per_zone.max() > 0:
            zones_geo.plot(
                ax=ax,
                column="vehicles",
                cmap="YlOrRd",  # Yellow to red colormap
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                legend=True,
                vmin=vmin,  # Set consistent minimum
                vmax=vmax,  # Set consistent maximum
                legend_kwds={
                    "label": "Vehicles per Zone",
                    "orientation": "vertical",
                    "shrink": 0.6,
                    "pad": 0.1,
                },
            )
        else:
            # If no vehicles, plot in gray
            zones_geo.plot(
                ax=ax, color="lightgray", alpha=0.5, edgecolor="black", linewidth=0.5
            )

        # Add transit stops
        if show_stops:
            stops_geo.plot(
                ax=ax, color="blue", markersize=0.8, alpha=0.6, label="Transit Stops"
            )

        # Customize the plot (don't set title here - let caller do it)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)

        # Add statistics text box
        stats_text = self._create_stats_text(vehicles_per_zone)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Add legend for stops if shown
        if show_stops:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="blue",
                    markersize=5,
                    label="Transit Stops",
                    alpha=0.6,
                )
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Only show plot if we created the figure
        if created_figure:
            plt.tight_layout()
            plt.show()

        return fig, ax

    def _create_stats_text(self, vehicles_per_zone):
        """Create statistics text for the map."""
        total_zones = len(vehicles_per_zone)
        zones_with_service = np.sum(vehicles_per_zone > 0)
        zones_without_service = total_zones - zones_with_service

        stats = [
            "📊 ZONE STATISTICS:",
            f"Total Zones: {total_zones}",
            f"Zones with Service: {zones_with_service}",
            f"Zones without Service: {zones_without_service}",
            "",
            "🚌 VEHICLE DISTRIBUTION:",
            f"Total Vehicles: {vehicles_per_zone.sum():.0f}",
            f"Mean per Zone: {np.mean(vehicles_per_zone):.1f}",
            f"Max in Zone: {np.max(vehicles_per_zone):.0f}",
            f"Std Dev: {np.std(vehicles_per_zone):.1f}",
        ]

        return "\n".join(stats)
