import logging
from typing import Any, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import Queen, lag_spatial

logger = logging.getLogger(__name__)

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
        drt_config: dict | None = None,
    ):
        self.gtfs_feed = gtfs_feed
        self.hex_size_km = hex_size_km
        self.crs = crs
        self.boundary = boundary

        # Add evaluation counter for debug print management
        self._evaluation_count = 0
        self._print_frequency = 50  # Print every N evaluations (TODO: make configurable)

        # reusable buffers for per-evaluation arrays to avoid repeated allocations
        self._pt_vehicles_buffer = None   # shape: (n_intervals, n_zones) allocated on first use
        self._drt_vehicles_buffer = None  # shape: (n_intervals, n_hex_zones) allocated on first use


        # Validate that CRS is metric
        self._validate_metric_crs()

        # Create stop locations GeoDataFrame
        self.stops_gdf = self._create_stops_geodataframe()

        # Apply boundary filtering if provided
        if self.boundary is not None:
            logger.info("🎯 Applying boundary filter to %d stops...", len(self.stops_gdf))
            self.stops_gdf = self.boundary.filter_points(
                self.stops_gdf, output_crs=self.crs
            )
            logger.info("✅ Filtered to %d stops within boundary", len(self.stops_gdf))

        # Generate hexagonal grid
        self.hex_grid = self._create_hexagonal_grid()

        # Optionally filter grid to boundary as well
        if self.boundary is not None:
            logger.info("🎯 Applying boundary filter to %d grid cells...", len(self.hex_grid))
            self.hex_grid = self.boundary.filter_grid(
                self.hex_grid, predicate="intersects", output_crs=self.crs
            )
            logger.info("✅ Filtered to %d grid cells within boundary", len(self.hex_grid))

        # OPTIMIZED: Use spatial join instead of nested loops
        self.stop_zone_mapping = self._fast_map_stops_to_zones()

        # DRT zone mappings (if provided) - AFTER boundary filtering
        if drt_config is not None and drt_config.get('enabled', False):
            self._set_drt_zone_mappings(drt_config)

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
                    logger.warning(
                        f"⚠️  Warning: CRS {self.crs} may not be metric (units: {unit})"
                    )
                    logger.info(
                        "   Consider using EPSG:3857 (Web Mercator) or a local UTM zone"
                    )

        except ImportError:
            logger.warning("⚠️  pyproj not available - cannot validate CRS units")
        except Exception as e:
            logger.warning(f"⚠️  Could not validate CRS {self.crs}: {e}")

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

        logger.info("🗺️  Reprojected %d stops to %s", len(stops_gdf), self.crs)
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

        logger.info(f"🔧 Creating {x_steps} × {y_steps} = {x_steps * y_steps} grid cells")
        logger.info(
            f"   Grid bounds: ({minx:.0f}, {miny:.0f}) to ({maxx:.0f}, {maxy:.0f}) meters"
        )
        logger.info(f"   Cell size: {hex_size_m}m × {hex_size_m}m")

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

        logger.info("✅ Created %d hexagonal zones in %s", len(hex_gdf), self.crs)
        return hex_gdf

    def _fast_map_stops_to_zones(self) -> dict[str, str]:
        """OPTIMIZED: Use spatial join - O(S + Z) instead of O(S × Z)."""
        logger.info("Using spatial join for zone mapping...")

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

        logger.info("✅ Mapped %d stops to zones", len(stop_zone_map))
        return stop_zone_map

    def _precompute_route_stop_mappings(self):
        """OPTIMIZED: Pre-compute all route → stops mappings to avoid repeated filtering."""

        logger.info("Pre-computing route-stop mappings...")

        self.route_stops_cache = {}

        # Group trips by route_id instead of service_id
        trips_by_route = (
            self.gtfs_feed.trips.groupby("route_id")["trip_id"].apply(list).to_dict()
        )

        # Group stop_times by trip_id once
        stop_times_by_trip = (
            self.gtfs_feed.stop_times.groupby("trip_id")["stop_id"].apply(set).to_dict()
        )

        for route_id, trip_ids in trips_by_route.items():
            # Get all unique stops for this route
            route_stops = set()
            for trip_id in trip_ids:
                if trip_id in stop_times_by_trip:
                    route_stops.update(stop_times_by_trip[trip_id])

            self.route_stops_cache[route_id] = route_stops

        logger.info("✅ Cached stops for %d routes", len(self.route_stops_cache))

    def _create_contiguity_matrix(self):
        """
        Create Queen contiguity matrix using libpysal.

        Queen contiguity: zones are neighbors if they share an edge or vertex.
        Returns row-standardized weights where each row sums to 1.0.

        Returns:
            libpysal.weights.W: Spatial weights matrix
        """

        logger.info(f"🔧 Creating Queen contiguity matrix for {len(self.hex_grid)} zones...")

        # Create weights directly from GeoDataFrame
        w = Queen.from_dataframe(self.hex_grid, ids=self.hex_grid["zone_id"])

        # Row-standardize the weights matrix
        w.transform = "r"  # Row standardization (each row sums to 1)

        # Verify the matrix
        logger.info("✅ Contiguity matrix created:")
        logger.info("   Zones: %d", w.n)
        logger.info("   Links: %d", w.s0)
        logger.info("   Islands: %d zones", len(w.islands))

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

        logger.info(
            f"""📊 Spatial lag calculated:
            • Input mean: {np.mean(vehicles_per_zone):.2f}
            • Spatial lag mean: {np.mean(spatial_lag):.2f}
            • Non-zero lags: {np.sum(spatial_lag > 0)}
            """
            )

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

        logger.info(f"""
        📊 Spatial lag calculated (α={alpha:.2f}):
            • Mean direct service: {np.mean(vehicles_per_zone):.2f}
            • Mean neighbor service: {np.mean(spatial_lag):.2f}
            • Mean accessibility: {np.mean(accessibility_scores):.2f}
            • Zones with improved access: {np.sum(accessibility_scores > vehicles_per_zone)}
        """)

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
        self, solution_matrix: np.ndarray | dict, optimization_data: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """
         Calculate vehicles per zone with all aggregation types (PT + DRT).

        Args:
            solution_matrix:
                - PT-only: Decision matrix (n_routes × n_intervals)
                - PT+DRT: Dict with 'pt' and 'drt' keys
            optimization_data: Optimization data structure from GTFSDataPreparator

        Returns:
            Dictionary with all aggregation types:
            - 'average': Array of mean vehicles per zone across time intervals
            - 'peak': Array of max vehicles per zone across time intervals
            - 'intervals': 2D array (n_zones × n_intervals) with vehicles per zone per interval
            - 'interval_labels': List of time interval labels from preparator
        """
        # Increment counter for this evaluation
        self._evaluation_count += 1

         # Determine if DRT is enabled
        drt_enabled = optimization_data.get('drt_enabled', False)

        # Only print debug info every N evaluations
        should_print = (self._evaluation_count % self._print_frequency == 0 or
                    self._evaluation_count == 1)  # Always print first evaluation

        # PT + DRT case
        if drt_enabled and isinstance(solution_matrix, dict):
            if should_print:
                logger.debug("📊 Calculating PT+DRT vehicles per zone... (eval #%d)", self._evaluation_count)
            # Calculate PT vehicles
            pt_vehicles_data = self._calculate_pt_vehicles_by_interval(
                solution_matrix['pt'], optimization_data
            )

            # Calculate DRT vehicles
            drt_vehicles_data = self._calculate_drt_vehicles_by_interval(
                solution_matrix['drt'], optimization_data
            )

            # Combine PT and DRT data
            return self._combine_vehicle_data(pt_vehicles_data, drt_vehicles_data, should_print)
        # PT only case
        else:
            if should_print:
                logger.debug("📊 Calculating PT-only vehicles per zone... (eval #%d)", self._evaluation_count)
            if isinstance(solution_matrix, dict):
                # Extract PT part if dict format used
                solution_matrix = solution_matrix['pt']

            return self._calculate_pt_vehicles_by_interval(solution_matrix, optimization_data)


    def _calculate_pt_vehicles_by_interval(
        self, pt_solution_matrix: np.ndarray, optimization_data: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Calculate PT vehicles per zone for each interval."""
        n_zones = len(self.hex_grid)
        n_intervals = optimization_data["n_intervals"]

        # STEP 1: Allocate or reuse buffer
        if self._pt_vehicles_buffer is None or self._pt_vehicles_buffer.shape != (n_intervals, n_zones):
            logger.debug("Allocating PT vehicles buffer: shape (%d, %d)", n_intervals, n_zones)
            self._pt_vehicles_buffer = np.zeros((n_intervals, n_zones), dtype=float)
        else:
            # Reset buffer to zeros (much faster than new allocation)
            self._pt_vehicles_buffer.fill(0.0)

        # STEP 2: Compute into the reusable buffer
        vehicles_by_intervals = self._pt_vehicles_buffer

        # STEP 3: Extract data from optimization structure
        route_ids = optimization_data["routes"]["ids"]
        allowed_headways = optimization_data["allowed_headways"]
        round_trip_times = optimization_data["routes"]["round_trip_times"]
        no_service_index = optimization_data["no_service_index"]
        interval_labels = optimization_data["intervals"]["labels"]

        # STEP 4: Main computation loop (unchanged logic, different output location)
        for interval_idx in range(n_intervals):
            zone_counts = {zone_id: 0 for zone_id in self.hex_grid["zone_id"]}

            for route_idx, service_id in enumerate(route_ids):
                if service_id not in self.route_stops_cache:
                    continue

                service_stops = self.route_stops_cache[service_id]
                choice_idx = pt_solution_matrix[route_idx, interval_idx]

                if choice_idx == no_service_index:
                    continue

                headway = allowed_headways[choice_idx]

                if headway < 9000:  # Valid service headway
                    round_trip = round_trip_times[route_idx]
                    vehicles_in_interval = max(1, int(np.ceil(round_trip / headway)))

                    zones_served = {
                        self.stop_zone_mapping[stop_id]
                        for stop_id in service_stops
                        if stop_id in self.stop_zone_mapping
                    }

                    for zone_id in zones_served:
                        zone_counts[zone_id] += vehicles_in_interval

            # STEP 5: Write zone counts into buffer row
            vehicles_by_intervals[interval_idx, :] = list(zone_counts.values())

        # STEP 6: Identify peak interval
        total_vehicles_by_interval = np.sum(vehicles_by_intervals, axis=1)
        peak_interval_idx = int(np.argmax(total_vehicles_by_interval))

        # STEP 7: Return copies (not the buffer itself!)
        # This ensures callers can't accidentally keep the buffer alive
        return {
            "intervals": vehicles_by_intervals.copy(),
            "average": np.mean(vehicles_by_intervals, axis=0).copy(),
            "peak": vehicles_by_intervals[peak_interval_idx, :].copy(),
            "sum": np.sum(vehicles_by_intervals, axis=0).copy(),
            "interval_labels": interval_labels,
        }

    def _calculate_drt_vehicles_by_interval(
        self, drt_solution_matrix: np.ndarray, optimization_data: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """
        Calculate DRT vehicles per zone for each interval

        Step 1: Calculate Time to Cross a zone (zone: STUDY AREA zone, not DRT zone):
        - time = diameter / speed
        - This provides a basic estimate of how long a vehicle might spend traversing
        a zone. Using the diameter is a simple way to capture the zone's spatial
        scale. Assuming a constant speed is a necessary simplification without network details.

        Step 2: Calculate Coverage (Average Vehicles per Zone):
        - Coverage = Total Fleet Size / Number of Zones in Service Area
        - This distributes the total fleet across the service area,
        giving an average vehicle density per zone at any moment in time.
        It assumes uniform distribution, which is a simplification but a
        sensible starting point without demand data.

        Step 3: Calculate Vehicle Activity per Zone per Interval:
        - Vehicles per Zone = (Interval Length / Time to Cross Zone) * Coverage
        - This step combines the spatial density (Coverage) with the temporal aspect
        (how many times a zone could be crossed in the interval).
        - Interpretation: The result isn't strictly the number of unique vehicles passing
        through, nor the number simultaneously present. It's better interpreted as a measure of
        total vehicle activity or vehicle-presence-time within that zone during the interval.
        For example, a result of '12' could mean 1 vehicle spending 12 times the crossing duration in the zone,
        or 12 different vehicles each crossing once, or some combination. It represents the equivalent number
        of full zone crossings occurring during the interval, scaled by the average vehicle density.

        Args:
            drt_solution_matrix: DRT fleet decisions (n_drt_zones × n_intervals) with
                            fleet size choice indices
            optimization_data: Complete optimization data

        Returns:
            Dictionary with same structure as PT vehicles data:
            - 'intervals': Array of shape (n_intervals, n_hex_zones)
            - 'average': Array of shape (n_hex_zones,)
            - 'peak': Array of shape (n_hex_zones,)
            - 'sum': Array of shape (n_hex_zones,)
            - 'interval_labels': List of interval labels
        """
        if not optimization_data.get('drt_enabled', False):
            n_intervals = optimization_data['n_intervals']
            n_hex_zones = len(self.hex_grid)
            return {
                'intervals': np.zeros((n_intervals, n_hex_zones)),
                'average': np.zeros(n_hex_zones),
                'peak': np.zeros(n_hex_zones),
                'sum': np.zeros(n_hex_zones),
                'interval_labels': optimization_data['intervals']['labels']
            }

        # STEP 1: Allocate or reuse DRT buffer
        drt_zones = optimization_data['drt_config']['zones']
        n_intervals = optimization_data['n_intervals']
        n_hex_zones = len(self.hex_grid)

        if self._drt_vehicles_buffer is None or self._drt_vehicles_buffer.shape != (n_intervals, n_hex_zones):
            logger.debug("Allocating DRT vehicles buffer: shape (%d, %d)", n_intervals, n_hex_zones)
            self._drt_vehicles_buffer = np.zeros((n_intervals, n_hex_zones), dtype=float)
        else:
            # Reset buffer to zeros (much faster than new allocation)
            self._drt_vehicles_buffer.fill(0.0)

        # STEP 2: Compute into the reusable buffer
        drt_vehicles_by_interval = self._drt_vehicles_buffer

        # Use study area grid zone diameter (from spatial resolution)
        study_area_zone_diameter_km = self.hex_size_km  # This matches the hex grid size

        # STEP 3: Process each DRT zone and interval
        for drt_zone_idx, drt_zone in enumerate(drt_zones):
            # Get DRT speed for this zone
            drt_speed_kmh = drt_zone.get('drt_speed_kmh')
            if drt_speed_kmh is None:
                drt_speed_kmh = optimization_data['drt_config'].get('default_drt_speed_kmh', 25.0)
                logger.debug("Using default DRT speed %s km/h for zone %s", drt_speed_kmh, drt_zone.get('id'))

            for interval_idx in range(n_intervals):
                # Get fleet size for this DRT zone and interval
                fleet_choice_idx = drt_solution_matrix[drt_zone_idx, interval_idx]

                fleet_size = drt_zone['allowed_fleet_sizes'][int(fleet_choice_idx)]  # IndexError for out-of-range, negative

                # Calculate vehicle activity
                interval_length_minutes = optimization_data['intervals']['duration_minutes']

                # Time to cross a STUDY AREA zone (not DRT service area)
                time_to_cross_hours = study_area_zone_diameter_km / drt_speed_kmh
                time_to_cross_minutes = time_to_cross_hours * 60

                n_zones_in_drt_area = len(drt_zone.get('affected_hex_zones', []))
                if n_zones_in_drt_area == 0:
                    continue

                coverage = fleet_size / n_zones_in_drt_area
                vehicle_activity = (interval_length_minutes / time_to_cross_minutes) * coverage

                # Add to affected hexagonal zones for this interval
                affected_hex_zones = drt_zone.get('affected_hex_zones', [])
                for hex_zone_idx in affected_hex_zones:
                    if 0 <= hex_zone_idx < n_hex_zones:  # Safety check for bounds
                        drt_vehicles_by_interval[interval_idx, hex_zone_idx] += vehicle_activity
                    else:
                        logger.warning("Affected hex zone index %s out of bounds (0..%d); skipping", hex_zone_idx, n_hex_zones-1)

        # STEP 4: Identify time interval with peak total vehicles
        total_vehicles_by_interval = np.sum(drt_vehicles_by_interval, axis=1)
        peak_interval_idx = int(np.argmax(total_vehicles_by_interval)) if np.any(total_vehicles_by_interval > 0) else 0

        # STEP 5: Return copies (not the buffer itself!)
        # This ensures callers can't accidentally keep the buffer alive
        return {
            'intervals': drt_vehicles_by_interval.copy(),
            'average': np.mean(drt_vehicles_by_interval, axis=0).copy(),
            'peak': drt_vehicles_by_interval[peak_interval_idx, :].copy(),
            'sum': np.sum(drt_vehicles_by_interval, axis=0).copy(),
            'interval_labels': optimization_data['intervals']['labels']
        }

    def _combine_vehicle_data(
        self, pt_data: dict[str, np.ndarray], drt_data: dict[str, np.ndarray],
        should_print: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Combine PT and DRT vehicle data with temporally consistent peak calculation.

        Peak Calculation Logic:
        - Identifies the interval with highest total system demand (PT + DRT combined)
        - Returns vehicle counts from that specific interval for temporal consistency
        - Ensures PT and DRT peak values represent the SAME interval/time period
        """
        # STEP 1: Combine interval data
        combined_intervals = pt_data['intervals'] + drt_data['intervals']

        # STEP 2: Find system-wide peak interval (when total vehicles needed is highest)
        total_vehicles_by_interval = np.sum(combined_intervals, axis=1)
        peak_interval_idx = int(np.argmax(total_vehicles_by_interval))

        if should_print:
            logger.debug("🔄 Combined peak interval: %d (total vehicles: %.0f)",
                        peak_interval_idx, total_vehicles_by_interval[peak_interval_idx])

        # STEP 3: Get peak vehicles from the system peak interval
        peak_combined = combined_intervals[peak_interval_idx, :].copy()

        # STEP 4: Return combined results
        return {
            'intervals': combined_intervals,
            'average': np.mean(combined_intervals, axis=0).copy(),
            'peak': peak_combined,
            'sum': np.sum(combined_intervals, axis=0).copy(),
            'interval_labels': pt_data['interval_labels']
        }



    def set_drt_zone_mappings(self, opt_data: dict):
        """Set DRT zone mappings from optimization data using efficient spatial operations."""
        if not opt_data.get('drt_enabled', False):
            return

        # Access DRT zones from existing location
        drt_zones = opt_data['drt_config']['zones']

        logger.info("🗺️ Computing spatial intersections for %d DRT zones...", len(drt_zones))

        for drt_zone in drt_zones:
            # Use vectorized spatial operations instead of loops
            drt_geometry = drt_zone['geometry']  # Already in correct CRS

            # Efficient spatial intersection using GeoPandas
            mask = self.hex_grid.geometry.intersects(drt_geometry)
            affected_hex_indices = self.hex_grid.index[mask].tolist()

            drt_zone['affected_hex_zones'] = affected_hex_indices
            logger.debug("   DRT zone %d affects %d hexagonal zones", drt_zone['zone_id'], len(affected_hex_indices))

    def _set_drt_zone_mappings(self, drt_config: dict):
        """Set DRT zone mappings from config during initialization (after boundary filtering)."""
        if not drt_config.get('enabled', False):
            return

        drt_zones = drt_config['zones']
        logger.info("🗺️ Computing DRT spatial intersections for %d zones...", len(drt_zones))
        logger.info("   Hexagonal grid size: %d zones", len(self.hex_grid))

        for drt_zone in drt_zones:
            drt_geometry = drt_zone['geometry']  # Already in correct CRS

            # Find intersections with the CURRENT (filtered) hexagonal grid
            mask = self.hex_grid.geometry.intersects(drt_geometry)

            # Use positional indices (0-based sequential)
            affected_positions = np.where(mask)[0].tolist()

            drt_zone['affected_hex_zones'] = affected_positions

            # Validation check
            if affected_positions:
                max_pos = max(affected_positions)
                if max_pos >= len(self.hex_grid):
                    logger.error("   ❌ ERROR: Position %d exceeds grid size %d", max_pos, len(self.hex_grid))
                    # Filter out invalid positions as safety net
                    drt_zone['affected_hex_zones'] = [
                        pos for pos in affected_positions
                        if 0 <= pos < len(self.hex_grid)
                    ]

            logger.debug("   Zone %s: affects %d hexagonal zones", drt_zone['zone_id'], len(drt_zone['affected_hex_zones']))


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


    def _visualize_with_data(
        self,
        data_per_zone: np.ndarray,
        data_column_name: str,
        data_label: str,
        colormap: str,
        optimization_data: dict[str, Any],
        title_suffix: str = "",
        figsize=(15, 12),
        show_stops=True,
        show_drt_zones=None,
        ax=None,
        vmin=None,
        vmax=None,
    ):
        """
        Generic method to visualize any per-zone data as a choropleth map.

        This is the core visualization infrastructure used by both vehicle coverage
        and waiting time visualizations. It provides consistent styling, legends,
        and geographic handling across all objective types.

        **Technical Implementation**:
        1. **Data Preparation**: Attaches data to hexagonal grid GeoDataFrame
        2. **CRS Conversion**: Converts from metric CRS to EPSG:4326 for plotting
        3. **Choropleth Rendering**: Uses GeoPandas plot() with specified colormap
        4. **Overlay Addition**: Adds transit stops and DRT zones if requested
        5. **Annotation**: Adds statistics text box and legends

        **Geographic Workflow**:
        - Analysis CRS (metric): Used for distance calculations and spatial operations
        - Display CRS (EPSG:4326): Used for final map visualization
        - Automatic conversion ensures accuracy without user intervention

        **Color Scale Handling**:
        - If vmin/vmax provided: Uses fixed scale (good for comparisons)
        - If None: Auto-scales to data range (good for single maps)
        - Consistent color bars with proper labeling

        **DRT Zone Display Logic**:
        - show_drt_zones=None: Auto-detect from optimization_data['drt_enabled']
        - show_drt_zones=True: Force display DRT zones
        - show_drt_zones=False: Hide DRT zones even if available
        - DRT zones shown as dashed colored boundaries with legend

        **Legend Management**:
        - Color bar: Shows data scale and units
        - Point legend: Transit stops (if enabled)
        - Line legend: DRT zones (if available and enabled)
        - Combined legend positioned to avoid overlap

        Args:
            data_per_zone: Array of data values for each hexagonal zone
            data_column_name: Column name for the data in GeoDataFrame
            data_label: Human-readable label for color bar legend
            colormap: Matplotlib colormap name (e.g., 'YlOrRd', 'RdYlBu_r')
            optimization_data: Complete optimization data structure
            title_suffix: Additional text for plot title
            figsize: Figure size (only used if ax is None)
            show_stops: Whether to overlay transit stops
            show_drt_zones: DRT zone display setting (None=auto, True/False=force)
            ax: Optional matplotlib axis to plot on
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale

        Returns:
            Tuple of (figure, axis) objects

        **Used By**:
        - visualize_spatial_coverage(): Vehicle density maps
        - visualize_waiting_times(): Waiting time choropleth
        - Future objectives: Can reuse this infrastructure

        """
        # Auto-detect DRT visualization if not specified
        if show_drt_zones is None:
            show_drt_zones = optimization_data.get('drt_enabled', False)

        # Create zones with data
        zones_with_data = self.hex_grid.copy()
        zones_with_data[data_column_name] = data_per_zone

        # Convert to geographic CRS for plotting
        zones_geo = zones_with_data.to_crs("EPSG:4326")
        stops_geo = self.stops_gdf.to_crs("EPSG:4326")

        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_figure = True
        else:
            fig = ax.figure
            created_figure = False

        # Plot zones with data coloring
        if data_per_zone.max() > 0:
            zones_geo.plot(
                ax=ax,
                column=data_column_name,
                cmap=colormap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                legend=True,
                vmin=vmin,
                vmax=vmax,
                legend_kwds={
                    "label": data_label,
                    "orientation": "vertical",
                    "shrink": 0.6,
                    "pad": 0.1,
                },
            )
        else:
            zones_geo.plot(
                ax=ax, color="lightgray", alpha=0.5, edgecolor="black", linewidth=0.5
            )

        # Add DRT zones (reuse existing logic from visualize_spatial_coverage)
        self._add_drt_zones_to_plot(ax, optimization_data, show_drt_zones)

        # Add transit stops
        if show_stops:
            stops_geo.plot(ax=ax, color="blue", markersize=0.8, alpha=0.6)

        # Add statistics text
        stats_text = self._create_data_stats_text(data_per_zone, data_label, optimization_data)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_aspect("equal")

        if created_figure:
            plt.tight_layout()
            plt.show()

        return fig, ax

    def _add_drt_zones_to_plot(self, ax, optimization_data, show_drt_zones):
        """Add DRT zones to existing plot."""
        drt_legend_elements = []

        if show_drt_zones and optimization_data.get('drt_enabled', False):
            drt_zones = optimization_data.get('drt_config', {}).get('zones', [])

            if drt_zones:
                # Define colors for DRT zones
                drt_colors = ['purple', 'cyan', 'green', 'orange', 'magenta', 'yellow']

                for i, drt_zone in enumerate(drt_zones):
                    if 'geometry' in drt_zone:
                        drt_gdf = gpd.GeoDataFrame(
                            [{'zone_id': drt_zone['zone_id']}],
                            geometry=[drt_zone['geometry']],
                            crs=optimization_data['drt_config']['target_crs']
                        )
                        drt_geo = drt_gdf.to_crs("EPSG:4326")

                        color = drt_colors[i % len(drt_colors)]
                        drt_geo.plot(
                            ax=ax,
                            facecolor='none',
                            edgecolor=color,
                            linewidth=2.5,
                            linestyle='--',
                            alpha=0.8
                        )

                        # Add to legend
                        from matplotlib.lines import Line2D
                        drt_legend_elements.append(
                            Line2D([0], [0], color=color, linewidth=2.5, linestyle='--',
                                label=f"DRT: {drt_zone.get('zone_name', drt_zone['zone_id'])}")
                        )

        # Add combined legend if we have DRT zones
        if drt_legend_elements:
            ax.legend(handles=drt_legend_elements, loc="upper right")

    def _create_data_stats_text(self, data_per_zone, data_label, optimization_data):
        """Create statistics text for generic data visualization."""
        total_zones = len(data_per_zone)

        # Basic statistics
        zones_with_data = np.sum(data_per_zone > 0)
        zones_without_data = total_zones - zones_with_data

        if zones_with_data > 0:
            mean_value = np.mean(data_per_zone[data_per_zone > 0])
            min_value = np.min(data_per_zone[data_per_zone > 0])
            max_value = np.max(data_per_zone[data_per_zone > 0])
        else:
            mean_value = 0.0
            min_value = 0.0
            max_value = 0.0

        overall_mean = np.mean(data_per_zone)
        std_value = np.std(data_per_zone)

        # Build statistics text
        stats = [
            f"📊 {data_label.upper()} STATISTICS:",
            f"Total Zones: {total_zones}",
            f"Zones with Data: {zones_with_data}",
            f"Zones without Data: {zones_without_data}",
            "",
            "📈 VALUES:",
            f"Mean (All Zones): {overall_mean:.2f}",
            f"Mean (With Data): {mean_value:.2f}",
            f"Min Value: {min_value:.2f}",
            f"Max Value: {max_value:.2f}",
            f"Standard Deviation: {std_value:.2f}",
        ]

        # Add service type indicator
        is_drt_enabled = optimization_data.get('drt_enabled', False)
        if is_drt_enabled:
            stats.append("")
            stats.append("🚁 SERVICE TYPE: PT + DRT")
        else:
            stats.append("")
            stats.append("🚌 SERVICE TYPE: PT Only")

        return "\n".join(stats)
