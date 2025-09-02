import logging
import time
from datetime import datetime
from typing import Any

import gtfs_kit as gk
import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class GTFSDataPreparator:
    """
    Streamlined GTFS data extraction and preparation for discrete headway optimization.

    This class processes GTFS feeds to extract essential data for transit optimization
    algorithms, focusing on headway-based service frequency optimization.
    It creates a matrix-based representation suitable for metaheuristic algorithms
    like PSO, NSGA-II, and genetic algorithms.

    Key Features:
    - Extracts current headways by time interval from GTFS schedules
    - Calculates round-trip times for fleet sizing constraints
    - Creates discrete choice mappings for allowed headway values
    - Generates initial solutions from existing service patterns
    - Maintains GTFS feed for solution reconstruction

    Data Structure:
    - Matrix representation: (n_routes × n_intervals) decision variables
    - Discrete choices: Each cell contains index to allowed headway values
    - Time intervals: Configurable periods (1h, 2h, 3h, 4h, 6h, 8h, 12h, 24h)

    Example:
        >>> preparator = GTFSDataPreparator(
        ...     gtfs_path='transit_data.zip',
        ...     interval_hours=3,  # 8 periods per day
        ...     max_round_trip_minutes=180 # filter out long routes (possibly regional)
        ... )
        >>> allowed_headways = [10, 15, 30, 60, 120]  # minutes
        >>> opt_data = preparator.extract_optimization_data(allowed_headways)

    Args:
        gtfs_path: Path to GTFS ZIP file or directory containing GTFS files
        interval_hours: Time interval duration in hours (must divide 24 evenly).
                        Headways are calculated for each interval.
        date: Optional service date filter in YYYYMMDD format (None = use all dates)
        turnaround_buffer: Multiplier for round-trip times (default 1.15 = 15% buffer)
                           to account for layover
        default_round_trip_time: Fallback round-trip time in minutes for missing data
        max_round_trip_minutes: Maximum allowed round-trip time (filters out long routes)
        fleet_bounds_factor: Tuple of (min_factor, max_factor) for route fleet bounds.
                    E.g., (0.8, 1.2) means 80%-120% of current fleet size.
                    Set to (0.0, float('inf')) for no fleet constraints.
        fleet_bounds_level: How to apply fleet bounds constraints. Options:
                          - 'route': Bounds apply per route (each route's fleet can vary ±X%)
                                   Example: Route A can use 8-12 vehicles, Route B can use 3-5 vehicles
                                   Use when: Routes have dedicated fleets, different vehicle types
                          - 'interval': Bounds apply per time interval (total system fleet varies by time)
                                      Example: 7-9AM can use 100-120 vehicles, 10PM-6AM can use 20-40 vehicles
                                      Use when: Vehicles can be shared across routes, realistic deployment
                          - 'global': Single bound for total system fleet across all times
                                    Example: Total fleet must stay within 800-1200 vehicles
                                    Use when: Simple fleet constraint
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

    Raises:
        ValueError: If interval_hours doesn't divide 24 evenly
        FileNotFoundError: If GTFS path doesn't exist

    Attributes:
        feed: Original GTFS feed (gtfs_kit.feed.Feed)
        trips_df: Cached trips DataFrame with time conversions
        stop_times_df: Cached stop_times DataFrame with seconds columns
        routes_df: Cached routes DataFrame
        n_intervals: Number of time intervals per day (24 / interval_hours)
    """

    def __init__(
        self,
        gtfs_path: str,
        interval_hours: int,
        date: str | None = None,
        turnaround_buffer: float = 1.15,
        default_round_trip_time: float = 60.0,
        max_round_trip_minutes: float = 240.0,
        no_service_threshold_minutes: float = 480,
        log_level: str = "INFO",
    ):
        """
        Initialize GTFS data preparator with validation and caching.

        Loads GTFS feed, applies optional date filtering, and pre-processes
        time data for efficient headway and round-trip calculations.

        Args:
            gtfs_path: Path to GTFS ZIP file or directory
            interval_hours: Time interval duration in HOURS (1, 2, 3, 4, 6, 8, 12, 24)
            date: Optional service date filter (YYYYMMDD format, e.g., '20231201')
            turnaround_buffer: Round-trip time multiplier for vehicle scheduling
            default_round_trip_time: Fallback round-trip time in MINUTES
            max_round_trip_minutes: Maximum allowed round-trip time in MINUTES
                                  (filters out regional/express routes)
            no_service_threshold_minutes: Headways above this threshold in MINUTES
                                          are treated as no-service
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

        Raises:
            ValueError: If interval_hours doesn't divide 24 evenly OR is < 3
        """
        # Configure logging
        self._setup_logging(log_level)

        # Hardcoded constraint
        MIN_INTERVAL_HOURS = 3

        # Input validation
        if interval_hours < MIN_INTERVAL_HOURS:
            logger.error(
                f"interval_hours ({interval_hours}) must be ≥ {MIN_INTERVAL_HOURS}"
            )
            raise ValueError(
                f"interval_hours ({interval_hours}) must be ≥ {MIN_INTERVAL_HOURS}. "
                f"Smaller intervals may misclassify regular service as no-service."
            )

        # Input validation
        if 24 % interval_hours != 0:
            logger.error(
                f"Invalid interval_hours ({interval_hours}). Must divide 24 evenly."
            )
            raise ValueError(
                f"interval_hours ({interval_hours}) must divide 24 evenly. "
                f"Valid values: 1, 2, 3, 4, 6, 8, 12, 24"
            )

        logger.info(f"Initializing GTFSDataPreparator with {interval_hours}h intervals")
        logger.debug(
            f"Configuration: turnaround_buffer={turnaround_buffer}, "
            f"max_round_trip={max_round_trip_minutes}min, "
            f"no_service_threshold={no_service_threshold_minutes}min"
        )

        # Store configuration
        self.gtfs_path = gtfs_path
        self.date = date
        self.interval_hours = interval_hours
        self.n_intervals = 24 // interval_hours
        self.turnaround_buffer = turnaround_buffer
        self.default_round_trip_time = default_round_trip_time
        self.max_round_trip_minutes = max_round_trip_minutes
        self.no_service_threshold_minutes = no_service_threshold_minutes

        # Load and cache GTFS data
        self._load_gtfs()

    def _setup_logging(self, log_level: str) -> None:
        """Configure logging for this instance."""
        # Set level for module logger
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Only add handler if none exists (avoid duplicates)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def _load_gtfs(self) -> None:
        """
        Load GTFS feed and pre-process for optimization.

        Loads the GTFS feed using gtfs-kit, applies optional date filtering,
        and converts time strings to seconds for efficient calculations.
        Creates cached DataFrames for trips, stop_times, and routes.

        Side Effects:
            - Sets self.feed (original GTFS feed)
            - Sets self.trips_df, self.stop_times_df, self.routes_df (cached tables)
            - Adds 'departure_seconds' and 'arrival_seconds' columns to stop_times_df

        Note:
            Time conversion uses GTFS convention where times can exceed 24:00:00
            for services that continue past midnight.
        """
        logger.info(f"Loading GTFS feed from {self.gtfs_path}")
        start_time = time.time()

        try:
            # Load original feed (keep for reconstruction)
            self.feed = gk.read_feed(self.gtfs_path, dist_units="km")
            logger.debug(
                f"GTFS feed loaded successfully: {len(self.feed.routes)} routes, {len(self.feed.trips)} trips"
            )

            # Apply date filtering if specified
            if self.date:
                logger.info(f"Filtering GTFS for date: {self.date}")
                try:
                    original_trips = len(self.feed.trips)
                    self.feed = gk.filter_feed_by_dates(self.feed, [self.date])
                    filtered_trips = len(self.feed.trips)
                    logger.info(
                        f"Date filtering: {original_trips} → {filtered_trips} trips"
                    )
                except Exception as e:
                    logger.warning(f"Date filtering failed: {e}, using full feed")
            else:
                logger.info("Using full GTFS feed (all service periods)")

            # Cache tables for processing
            self.trips_df = self.feed.trips.copy()
            self.stop_times_df = self.feed.stop_times.copy()
            self.routes_df = self.feed.routes.copy()

            # Convert times to seconds for calculations
            logger.debug("Converting GTFS time strings to seconds")
            self.stop_times_df["departure_seconds"] = self.stop_times_df[
                "departure_time"
            ].apply(self._safe_timestr_to_seconds)
            self.stop_times_df["arrival_seconds"] = self.stop_times_df[
                "arrival_time"
            ].apply(self._safe_timestr_to_seconds)

            load_time = time.time() - start_time
            logger.info(f"GTFS loaded and cached in {load_time:.2f} seconds")
            logger.info(
                f"Dataset: {len(self.trips_df):,} trips, {len(self.stop_times_df):,} stop times"
            )

        except Exception as e:
            logger.error(f"Failed to load GTFS feed: {e}")
            raise

    def extract_optimization_data(
        self, allowed_headways: list[float]
    ) -> dict[str, Any]:
        """
        Extract and structure data for optimization algorithms.

        Creates a comprehensive data structure containing all information needed
        for discrete headway optimization, including decision variables, constraints,
        and reconstruction metadata.

        Args:
            allowed_headways: List of allowed headway values in MINUTES (e.g., [10, 15, 30, 60])

        Returns:
            Dictionary containing:
            - problem_type: Type identifier for optimization problem.
            - n_routes: Number of routes in the GTFS feed.
            - n_intervals: Number of time intervals per day (e.g., 8 for 3-hour intervals).
            - n_choices: Number of discrete headway choices, including the "no-service" option.
            - decision_matrix_shape: Shape of the decision variable matrix (n_routes × n_intervals).
            - initial_solution:
                A 2D matrix (n_routes × n_intervals) where each cell contains the index of the
                closest allowed headway value. The indices correspond to the `allowed_headways`
                array, with the last index representing the "no-service" option (`9999`).

                **Purpose**:
                    Provides a feasible starting point for optimization algorithms by mapping
                    current GTFS headways to the nearest allowed discrete values.

                **Structure**:
                    - Dimensions: (n_routes × n_intervals), where `n_routes` is the number of routes
                    and `n_intervals` is the number of time intervals per day.
                    - Each value is an **index** corresponding to an entry in the `allowed_headways` array.
                    - The last index corresponds to the "no-service" option (`9999`).

                **How It Works**:
                    - For each route and time interval:
                        - If the current headway is `NaN` (indicating no service), the cell is assigned
                        the index of the "no-service" option (`9999`).
                        - Otherwise, the closest allowed headway value is determined by finding the
                        minimum absolute difference between the current headway and the `allowed_headways`.
                    - The index of the closest allowed headway is stored in the matrix.

                **Example**:
                    - Inputs:
                        allowed_headways = [10, 15, 30, 60, 120, 9999]
                        current_headways = [
                            [12, 20, 35],
                            [NaN, 50, 100],
                            [15, NaN, 9999]
                        ]
                    - Conversion Process:
                        - Route 1, Interval 1: `12` is closest to `15` → Index `1`.
                        - Route 1, Interval 2: `20` is closest to `30` → Index `2`.
                        - Route 1, Interval 3: `35` is closest to `30` → Index `2`.
                        - Route 2, Interval 1: `NaN` → No-service → Index `5`.
                        - Route 2, Interval 2: `50` is closest to `60` → Index `3`.
                        - Route 2, Interval 3: `100` is closest to `120` → Index `4`.
                        - Route 3, Interval 1: `15` is closest to `15` → Index `1`.
                        - Route 3, Interval 2: `NaN` → No-service → Index `5`.
                        - Route 3, Interval 3: `9999` → No-service → Index `5`.
                    - Output:
                        initial_solution = [
                            [1, 2, 2],  # Route 1
                            [5, 3, 4],  # Route 2
                            [1, 5, 5]   # Route 3
                        ]
            - routes:
                Dictionary containing:
                - ids: List of route IDs.
                - round_trip_times: Array of round-trip times for each route.
                - current_headways: 2D array of current headways for each route and interval.
            - constraints:
                Dictionary containing:
                - fleet_data: Round-trip times and fleet constraints.
                - service_coverage: Minimum service coverage constraints.
            - intervals:
                Dictionary containing:
                - labels: Human-readable labels for each interval (e.g., `00-03h`).
                - hours: Start and end times for each interval.
                - duration_minutes: Duration of each interval in minutes.
            - reconstruction:
                Dictionary containing:
                - gtfs_feed: The original GTFS feed.
                - route_mapping: Mapping of route IDs to indices.
            - metadata:
                Dictionary containing:
                - gtfs_source: Path to the GTFS feed.
                - date_filter: Service date filter applied (if any).
                - creation_timestamp: Timestamp of when the data was prepared.
                - filter_stats: Statistics about the filtering process (e.g., number of routes retained).

        Example:
            >>> allowed_headways = [10, 15, 30, 60, 120]
            >>> opt_data = preparator.extract_optimization_data(allowed_headways)
            >>> print(opt_data['decision_matrix_shape'])  # (n_routes, n_intervals)
            >>> print(opt_data['n_choices'])  # 6 (5 headways + no-service)

        Note:
            Automatically adds 9999.0 as "no service" option to allowed_headways.
            Initial solution maps current GTFS headways to nearest allowed values.
        """
        logger.info(
            f"Extracting optimization data with {len(allowed_headways)} allowed headways"
        )
        logger.debug(f"Allowed headways: {allowed_headways}")

        # Extract route data first
        route_data = self._extract_route_essentials()
        n_routes = len(route_data)

        logger.info(f"Successfully extracted {n_routes} routes for optimization")

        # Analyze current fleet (baseline only)
        fleet_analysis = self._analyze_current_fleet(route_data)

        # Create headway mappings
        allowed_values = np.array(allowed_headways + [9999.0], dtype=np.float64)
        headway_to_index = {float(h): i for i, h in enumerate(allowed_values)}
        no_service_index = len(allowed_values) - 1

        logger.debug(
            f"Created discrete choice mapping: {len(allowed_values)} choices (including no-service)"
        )

        # Create aligned arrays
        route_ids = [r["service_id"] for r in route_data]
        round_trip_times = np.array(
            [r["round_trip_time"] for r in route_data], dtype=np.float64
        )
        current_headways = np.array(
            [r["headways_by_interval"] for r in route_data], dtype=np.float64
        )

        # Create initial solution matrix
        logger.debug("Creating initial solution matrix from current GTFS headways")
        initial_solution = self._create_initial_solution(
            current_headways, headway_to_index
        )

        # Log solution statistics
        total_cells = initial_solution.size
        no_service_cells = np.sum(initial_solution == no_service_index)
        service_cells = total_cells - no_service_cells
        logger.info(
            f"Initial solution: {service_cells}/{total_cells} cells have service "
            f"({100*service_cells/total_cells:.1f}%)"
        )

        # Build optimized structure
        optimization_data = {
            "problem_type": "discrete_headway_optimization",
            "n_routes": n_routes,
            "n_intervals": self.n_intervals,
            "n_choices": len(allowed_values),
            "decision_matrix_shape": (n_routes, self.n_intervals),
            "variable_bounds": (0, len(allowed_values) - 1),
            "initial_solution": initial_solution,
            "allowed_headways": allowed_values,
            "headway_to_index": headway_to_index,
            "no_service_index": no_service_index,
            "routes": {
                "ids": route_ids,
                "round_trip_times": round_trip_times,
                "current_headways": current_headways,
            },
            "constraints": {
                "fleet_analysis": fleet_analysis,
            },
            "intervals": {
                "labels": [
                    f"{i*self.interval_hours:02d}-{(i+1)*self.interval_hours:02d}h"
                    for i in range(self.n_intervals)
                ],
                "hours": [
                    (i * self.interval_hours, (i + 1) * self.interval_hours)
                    for i in range(self.n_intervals)
                ],
                "duration_minutes": self.interval_hours * 60,
            },
            "metadata": {
                "gtfs_source": self.gtfs_path,
                "date_filter": self.date,
                "creation_timestamp": datetime.now().isoformat(),
                "filter_stats": {
                    "final_routes": n_routes,
                },
            },
            "reconstruction": {
                "gtfs_feed": self.feed,
                "route_mapping": {route_id: i for i, route_id in enumerate(route_ids)},
            },
        }

        logger.info("Optimization data structure created successfully")
        logger.debug(
            f"Decision matrix shape: {optimization_data['decision_matrix_shape']}"
        )

        return optimization_data

    def _extract_route_essentials(self) -> list[dict[str, Any]]:
        """
        Extract only essential data: headways and round-trip times.

        Processes all services in the GTFS feed to extract headway patterns
        and round-trip times needed for optimization. Filters out routes
        with excessive round-trip times or invalid data.

        Returns:
            List of dictionaries, each containing:
            - service_id: GTFS service identifier
            - headways_by_interval: Array of headway values per time interval
            - round_trip_time: Calculated round-trip time in minutes
        """
        logger.info(
            f"Extracting route essentials with {self.interval_hours}-hour intervals"
        )

        all_services = self.trips_df["service_id"].unique()
        route_data = []
        filtered_count = 0
        failed_count = 0
        used_default_count = 0

        logger.debug(f"Processing {len(all_services)} services")

        for i, service_id in enumerate(all_services):
            # Progress logging for large datasets
            if i % 100 == 0 and i > 0:
                logger.debug(
                    f"Processed {i}/{len(all_services)} services "
                    f"({len(route_data)} retained, {failed_count} failed, {filtered_count} filtered)"
                )

            service_trips = self.trips_df[self.trips_df["service_id"] == service_id]

            if len(service_trips) == 0:
                logger.debug(f"Service {service_id}: No trips found, skipping")
                failed_count += 1
                continue

            # Calculate headways by interval
            headways_by_interval = self._calculate_service_headways(
                service_id, service_trips
            )

            # Skip if no service found
            if np.all(np.isnan(headways_by_interval)):
                logger.debug(f"Service {service_id}: No valid headways, skipping")
                failed_count += 1
                continue

            # Calculate round-trip time
            round_trip_time = self._calculate_round_trip_time(service_id, service_trips)

            # Track default usage
            if round_trip_time == self.default_round_trip_time:
                used_default_count += 1

            # Filter out services with excessive round-trip times
            if round_trip_time > self.max_round_trip_minutes:
                logger.warning(
                    f"Service {service_id}: Round-trip {round_trip_time:.1f}min "
                    f"exceeds limit ({self.max_round_trip_minutes}min), filtered out"
                )
                filtered_count += 1
                continue

            # Count active intervals
            active_intervals = np.nansum(~np.isnan(headways_by_interval))
            logger.debug(
                f"Service {service_id}: Round-trip {round_trip_time:.1f}min, "
                f"{active_intervals}/{len(headways_by_interval)} intervals active"
            )

            route_data.append(
                {
                    "service_id": service_id,
                    "headways_by_interval": headways_by_interval,
                    "round_trip_time": round_trip_time,
                }
            )

        # Final summary
        logger.info(
            f"Route extraction complete: {len(route_data)} routes retained from {len(all_services)} total"
        )
        if filtered_count > 0:
            logger.warning(
                f"Filtered out {filtered_count} routes (excessive round-trip time)"
            )
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} routes (no valid data)")
        if used_default_count > 0:
            logger.info(f"Used default round-trip time for {used_default_count} routes")

        return route_data

    def _calculate_service_headways(
        self, service_id: str, service_trips: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate average headway values for each time interval.

        For each time interval, calculates the average time between consecutive
        trip departures. Handles edge cases like single trips per interval
        and missing data gracefully.

        Args:
            service_id: GTFS service_id identifier
            service_trips: DataFrame of trips for this service

        Returns:
            Array of headway values in MINUTES for each interval.
            NaN indicates no service in that interval.

        Algorithm:
            1. Extract first departure time for each trip
            2. Group departures by time interval
            3. For intervals with ≥2 departures: calculate mean interval
            4. For intervals with 1 departure: assign 24-hour headway
            5. For intervals with 0 departures: assign NaN (no service)

        Note:
            - Uses GTFS convention where hour 25 = 1 AM next day
            - Handles overnight services that cross midnight boundary
            - Filters out negative or zero intervals (data quality issues)
        """
        headways = np.full(self.n_intervals, np.nan)

        try:
            trip_ids = service_trips["trip_id"].tolist()
            service_stop_times = self.stop_times_df[
                self.stop_times_df["trip_id"].isin(trip_ids)
            ].copy()

            if len(service_stop_times) == 0:
                logger.debug(f"Service {service_id}: No stop times found")
                return headways

            # Get first departure for each trip
            first_departures = service_stop_times.loc[
                service_stop_times.groupby("trip_id")["stop_sequence"].idxmin()
            ][["trip_id", "departure_seconds"]].copy()

            first_departures["departure_hour"] = (
                first_departures["departure_seconds"] // 3600
            ) % 24
            first_departures = first_departures.dropna()

            if len(first_departures) == 0:
                logger.debug(f"Service {service_id}: No valid departure times")
                return headways

            logger.debug(
                f"Service {service_id}: Processing {len(first_departures)} departures"
            )

            # Calculate headways for each interval
            active_intervals = 0
            for interval in range(self.n_intervals):
                start_hour = interval * self.interval_hours
                end_hour = (interval + 1) * self.interval_hours

                interval_departures = first_departures[
                    (first_departures["departure_hour"] >= start_hour)
                    & (first_departures["departure_hour"] < end_hour)
                ]["departure_seconds"].values

                if len(interval_departures) >= 2:
                    # Calculate average interval between departures
                    interval_departures = np.sort(interval_departures)
                    intervals = np.diff(interval_departures) / 60  # Convert to minutes
                    valid_intervals = intervals[intervals > 0]
                    if len(valid_intervals) > 0:
                        headway_value = np.mean(valid_intervals)
                        headways[interval] = headway_value
                        active_intervals += 1
                        logger.debug(
                            f"Service {service_id}, interval {interval}: "
                            f"{len(interval_departures)} departures → {headway_value:.1f}min headway"
                        )
                elif len(interval_departures) == 1:
                    # Single trip - once per day service
                    headways[interval] = 24 * 60  # 1440 minutes
                    active_intervals += 1
                    logger.debug(
                        f"Service {service_id}, interval {interval}: "
                        f"1 departure → 1440min headway (once-daily)"
                    )

            if active_intervals == 0:
                logger.debug(f"Service {service_id}: No active intervals found")

            return headways

        except Exception as e:
            logger.debug(f"Service {service_id}: Exception in headway calculation: {e}")
            return headways

    def _calculate_round_trip_time(
        self, service_id: str, service_trips: pd.DataFrame
    ) -> float:
        """
        Calculate round-trip time with turnaround buffer for fleet sizing.

        Estimates the total time a vehicle needs to complete a round trip,
        including turnaround time at terminals. Used for fleet size calculations
        in optimization constraints.

        Args:
            service_id: GTFS service_id identifier
            service_trips: DataFrame of trips for this service

        Returns:
            Round-trip time in MINUTES including turnaround buffer.
            Returns default_round_trip_time if calculation fails.

        Algorithm:
            1. For each trip, calculate duration from first departure to last arrival
            2. Take median one-way duration (robust to outliers)
            3. Double for round-trip and apply turnaround buffer
            4. Buffer accounts for terminal time, driver breaks, schedule recovery

        Example:
            If median one-way trip is 30 minutes with 1.15 buffer:
            Round-trip = 30 × 2 × 1.15 = 69 minutes

        Note:
            - Uses median instead of mean (robust to outliers/express trips)
            - Filters out negative durations (data quality issues)
            - Turnaround buffer typically 1.10-1.25 (10-25% extra time)
        """
        try:
            trip_ids = service_trips["trip_id"].tolist()
            service_stop_times = self.stop_times_df[
                self.stop_times_df["trip_id"].isin(trip_ids)
            ].copy()

            if len(service_stop_times) == 0:
                logger.debug(
                    f"Service {service_id}: No stop times, using default {self.default_round_trip_time}min"
                )
                return self.default_round_trip_time

            trip_durations = []
            for trip_id, trip_stops in service_stop_times.groupby("trip_id"):
                if len(trip_stops) >= 2:
                    trip_stops = trip_stops.sort_values("stop_sequence")
                    first_departure = trip_stops.iloc[0]["departure_seconds"]
                    last_arrival = trip_stops.iloc[-1]["arrival_seconds"]

                    if pd.notna(first_departure) and pd.notna(last_arrival):
                        duration_minutes = (last_arrival - first_departure) / 60.0
                        if duration_minutes > 0:
                            trip_durations.append(duration_minutes)

            if trip_durations:
                median_one_way = np.median(trip_durations)
                round_trip = median_one_way * 2.0 * self.turnaround_buffer
                logger.debug(
                    f"Service {service_id}: Calculated round-trip {round_trip:.1f}min "
                    f"(median one-way: {median_one_way:.1f}min, {len(trip_durations)} trips, "
                    f"buffer: {self.turnaround_buffer})"
                )
                return round_trip
            else:
                logger.debug(
                    f"Service {service_id}: No valid durations, using default {self.default_round_trip_time}min"
                )
                return self.default_round_trip_time

        except Exception as e:
            logger.debug(
                f"Service {service_id}: Exception calculating round-trip time: {e}, "
                f"using default {self.default_round_trip_time}min"
            )
            return self.default_round_trip_time

    def _create_initial_solution(
        self, current_headways: np.ndarray, headway_to_index: dict[float, int]
    ) -> np.ndarray:
        """
        Create initial solution matrix by mapping current GTFS headways to discrete choices.

        Converts the continuous headway values from GTFS to discrete choice indices
        that can be used by optimization algorithms. Provides a warm start based
        on existing service patterns.

        Args:
            current_headways: Array of current headways (n_routes × n_intervals)
            headway_to_index: Mapping from headway values to choice indices

        Returns:
            Integer matrix (n_routes × n_intervals) where each cell contains
            the index of the closest allowed headway value.

        Algorithm:
            1. For each route-interval cell:
            2. If current headway is NaN OR ≥ no_service_threshold → assign no-service index
            3. Else find closest allowed headway value (minimum absolute difference)
            4. Assign corresponding choice index

        Example:
            Allowed headways: [10, 15, 30, 60]
            Closest match: 15 minutes (difference = 2)
            Current headway: 17 minutes → maps to 15 minutes (closest allowed).
                             Assigned index: 1 (index of 15 in allowed list)
            Current headway: 1440 minutes → maps to no-service (above threshold).
                             Assigned index: 5 (index of 9999 in allowed list)
            Current headway: NaN → maps to no-service.
                             Assigned index: 5 (index of 9999 in allowed list)

        Note:
            - Provides optimization algorithms with feasible starting point
            - Preserves existing service patterns where possible
            - No-service periods (NaN) and irregular service (>threshold) map to no-service choice  # ← UPDATE
            - Threshold prevents once-daily trips from being treated as regular service.
        """
        n_routes, n_intervals = current_headways.shape
        initial_solution = np.zeros((n_routes, n_intervals), dtype=int)

        allowed_headway_values = list(headway_to_index.keys())[:-1]  # Exclude 9999
        no_service_index = headway_to_index[9999.0]

        mapping_stats = {choice: 0 for choice in headway_to_index.values()}
        large_differences = []
        threshold_mappings = 0

        logger.debug(
            f"Creating initial solution for {n_routes} routes × {n_intervals} intervals"
        )
        logger.debug(
            f"Using no-service threshold: {self.no_service_threshold_minutes:.0f} minutes"
        )  # ← ADD

        for i in range(n_routes):
            for j in range(n_intervals):
                current_headway = current_headways[i, j]

                # Handle no-service cases: NaN or headways above threshold
                if (
                    np.isnan(current_headway)
                    or current_headway >= self.no_service_threshold_minutes
                ):
                    choice_idx = no_service_index
                    initial_solution[i, j] = choice_idx
                    mapping_stats[choice_idx] += 1

                else:
                    # Find nearest allowed headway from valid optiona
                    distances = [
                        abs(current_headway - h) for h in allowed_headway_values
                    ]
                    best_idx = np.argmin(distances)
                    best_headway = allowed_headway_values[best_idx]
                    difference = abs(current_headway - best_headway)

                    initial_solution[i, j] = best_idx
                    mapping_stats[best_idx] += 1

                    # Track large mapping differences
                    if difference > 10:  # More than 10 minutes difference
                        large_differences.append(
                            (i, j, current_headway, best_headway, difference)
                        )

        # Log mapping statistics
        logger.debug("Initial solution mapping statistics:")
        for headway_val, choice_idx in headway_to_index.items():
            count = mapping_stats[choice_idx]
            if count > 0:
                if headway_val == 9999.0:
                    logger.debug(
                        f"  No service: {count} cells ({100*count/(n_routes*n_intervals):.1f}%)"
                    )
                else:
                    logger.debug(
                        f"  {headway_val:.0f}min: {count} cells ({100*count/(n_routes*n_intervals):.1f}%)"
                    )

        # Log threshold impact
        if threshold_mappings > 0:
            logger.info(
                f"Applied threshold: {threshold_mappings} cells with headways ≥{self.no_service_threshold_minutes:.0f}min "
                f"mapped to no-service ({100*threshold_mappings/(n_routes*n_intervals):.1f}%)"
            )

        if large_differences:
            logger.warning(
                f"Found {len(large_differences)} cells with >10min mapping difference"
            )
            # Log a few examples
            for i, (route_idx, interval_idx, current, mapped, diff) in enumerate(
                large_differences[:5]
            ):
                logger.debug(
                    f"  Route {route_idx}, interval {interval_idx}: "
                    f"{current:.1f}min → {mapped:.0f}min (diff: {diff:.1f}min)"
                )
            if len(large_differences) > 5:
                logger.debug(f"  ... and {len(large_differences) - 5} more")

        return initial_solution

    def _analyze_current_fleet(
        self, route_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze current GTFS fleet requirements by time interval (baseline analysis only).

        This method analyzes the existing GTFS service to determine current fleet
        requirements by calculating how many vehicles each route needs at each time
        interval, then aggregating to find total system fleet needs. This provides
        realistic baseline data that optimization classes can use to set constraints.

        **Key Insight**: Instead of naively summing each route's peak fleet requirements
        (which assumes all routes peak simultaneously), this method calculates the actual
        fleet needed at each time interval and takes the maximum across intervals.
        This typically results in 20-40% lower fleet estimates than the naive approach.

        **Example Calculation**:
            Time Period    | Route A | Route B | Route C | Total Needed
            --------------|---------|---------|---------|-------------
            06:00-09:00   |    4    |    3    |    0    |      7
            09:00-12:00   |    2    |    1    |    2    |      5
            12:00-15:00   |    2    |    2    |    1    |      5
            15:00-18:00   |    3    |    4    |    2    |      9  ← Peak

            Realistic total: 9 vehicles (maximum across intervals)
            Naive total: 4+4+2 = 10 vehicles (sum of route peaks)
            Efficiency gain: 1 vehicle saved

        Args:
            route_data: Output from _extract_route_essentials() containing:
                    - service_id: Route identifier (string)
                    - round_trip_time: Round-trip time in minutes (float)
                    - headways_by_interval: Array of current headway values per interval

        Returns:
            Dictionary containing baseline fleet analysis with these keys:

            **Fleet Analysis (Baseline Data)**:
            - current_fleet_per_route: Array (n_routes,) of peak vehicles needed per route
                                    Each value = max vehicles needed across all intervals for that route
            - current_fleet_by_interval: Array (n_intervals,) of total system vehicles per interval
                                        Each value = sum of all route needs at that specific time
            - total_current_fleet_peak: Maximum fleet needed across all intervals (realistic total)
                                    This is the actual number of vehicles the system needs
            - route_round_trip_times: Array of round-trip times (for optimization problem use)

            **Configuration (For Optimization Classes)**:
            - operational_buffer: Buffer factor used in vehicle calculations (typically 1.15)

            **Analysis Statistics**:
            - fleet_stats: Dictionary containing:
                * routes_with_service: Number of routes that have active service
                * peak_interval: Time interval index with highest fleet requirement
                * peak_interval_fleet: Number of vehicles needed during peak interval
                * off_peak_fleet: Minimum vehicles needed during any active interval
                * interval_utilization: List showing % of peak fleet used in each interval
                * fleet_efficiency_gain: Vehicles saved vs naive approach (positive = savings)
                * mean_fleet_per_route: Average peak fleet requirement per active route
                * max_route_fleet: Highest fleet requirement among all routes
                * fleet_distribution: Count of routes by fleet size category

        **Fleet Calculation Formula**:
            vehicles_needed = ceil((round_trip_time * operational_buffer) / headway)

            Where:
            - round_trip_time: Total time for vehicle to complete route and return (minutes)
            - operational_buffer: Extra time factor for maintenance, delays, crew relief (1.15 = 15%)
            - headway: Time between consecutive departures (minutes)
            - ceil(): Round up to next integer (can't have fractional vehicles)

        **Calculation Examples**:
            Route with 60-minute round-trip, 15-minute headway:
            vehicles = ceil((60 * 1.15) / 15) = ceil(4.6) = 5 vehicles

            Route with 90-minute round-trip, 30-minute headway:
            vehicles = ceil((90 * 1.15) / 30) = ceil(3.45) = 4 vehicles

            Route with no service (NaN headway):
            vehicles = 0 vehicles

        **Usage Note**:
            This method only analyzes existing GTFS data. It does NOT calculate optimization
            constraints or bounds - those are handled by optimization problem classes that
            use this baseline data to set their own constraint levels.
        """
        from ..optimisation.utils.fleet_calculations import calculate_fleet_requirements

        logger.debug("Analyzing current GTFS fleet requirements by interval")
        logger.debug(
            f"Processing {len(route_data)} routes across {self.n_intervals} intervals"
        )

        # Extract dimensions and set parameters
        n_routes = len(route_data)
        n_intervals = self.n_intervals
        operational_buffer = (
            1.15  # 15% buffer: accounts for maintenance, delays, crew changes
        )

        logger.debug(
            f"Using operational buffer: {operational_buffer} ({(operational_buffer-1)*100:.0f}% extra time)"
        )

        # Extract data for shared calculation
        round_trip_times = np.array([r["round_trip_time"] for r in route_data])
        headways_matrix = np.array([r["headways_by_interval"] for r in route_data])

        # Use shared calculation logic (ensures consistency with optimization)
        fleet_results = calculate_fleet_requirements(
            headways_matrix=headways_matrix,
            round_trip_times=round_trip_times,
            operational_buffer=operational_buffer,
            no_service_threshold=self.no_service_threshold_minutes,  # Use GTFSDataPreparator's parameter
        )

        # Extract results from shared calculation
        current_fleet_per_route = fleet_results["fleet_per_route"]
        current_fleet_by_interval = fleet_results["fleet_per_interval"]
        total_current_fleet_peak = fleet_results["total_peak_fleet"]
        route_fleet_matrix = fleet_results["route_fleet_matrix"]  # For detailed logging
        route_round_trip_times = round_trip_times

        # Calculate efficiency gain vs naive approach
        total_naive_sum = int(
            np.sum(current_fleet_per_route)
        )  # Sum of route peaks (naive)
        fleet_efficiency_gain = (
            total_naive_sum - total_current_fleet_peak
        )  # Positive = savings

        # Count routes that have any service
        routes_with_service = np.sum(current_fleet_per_route > 0)

        # Find peak interval (when most vehicles are needed)
        peak_interval_idx = int(np.argmax(current_fleet_by_interval))

        # Find minimum fleet during active periods (off-peak service level)
        active_intervals = current_fleet_by_interval > 0
        if np.any(active_intervals):
            off_peak_fleet = int(np.min(current_fleet_by_interval[active_intervals]))
        else:
            off_peak_fleet = 0

        # Calculate interval utilization (what % of peak fleet is used in each interval)
        if total_current_fleet_peak > 0:
            interval_utilization = (
                current_fleet_by_interval / total_current_fleet_peak
            ).tolist()
        else:
            interval_utilization = [0.0] * n_intervals

        # ===== KEEP ALL THE DETAILED LOGGING (unchanged) ===

        # Per-route logging with debug info
        for route_idx, route in enumerate(route_data):
            route_id = route["service_id"]
            if current_fleet_per_route[route_idx] > 0:
                active_intervals_count = np.sum(route_fleet_matrix[route_idx, :] > 0)
                logger.debug(
                    f"Route {route_id}: Peak {current_fleet_per_route[route_idx]:.0f} vehicles "
                    f"({active_intervals_count}/{n_intervals} intervals active)"
                )
            else:
                logger.debug(f"Route {route_id}: No service (0 vehicles)")

        # System summary logging
        logger.info("Fleet analysis by interval completed:")
        logger.info(
            f"  Fleet by interval: {current_fleet_by_interval.astype(int).tolist()}"
        )
        logger.info(
            f"  Peak interval {peak_interval_idx}: {total_current_fleet_peak} vehicles needed"
        )
        logger.info(f"  Off-peak minimum: {off_peak_fleet} vehicles")
        logger.info(f"  Active routes: {routes_with_service}/{n_routes}")

        # Highlight efficiency gain from interval-based calculation
        if fleet_efficiency_gain > 0:
            logger.info(
                f"  Efficiency gain: {fleet_efficiency_gain} vehicles saved vs naive sum "
                f"({100 * fleet_efficiency_gain / total_naive_sum:.1f}% reduction)"
            )
        elif fleet_efficiency_gain == 0:
            logger.info("  No efficiency gain (all routes peak simultaneously)")
        else:
            logger.warning(
                f"  Negative efficiency: interval approach needs {-fleet_efficiency_gain} more vehicles"
            )

        # ===== RETURN THE EXACT SAME STRUCTURE (unchanged) ===
        return {
            # ===== BASELINE FLEET ANALYSIS =====
            # Core data that optimization classes need for constraint setting
            "current_fleet_per_route": current_fleet_per_route.astype(
                int
            ),  # Peak per route
            "current_fleet_by_interval": current_fleet_by_interval.astype(
                int
            ),  # Total per interval
            "total_current_fleet_peak": total_current_fleet_peak,  # Realistic system total
            "route_round_trip_times": route_round_trip_times,  # For optimization use
            # ===== CONFIGURATION PARAMETERS =====
            # Parameters that optimization classes might need to replicate calculations
            "operational_buffer": operational_buffer,  # Buffer factor used
            "no_service_threshold_minutes": self.no_service_threshold_minutes,
            # ===== ANALYSIS STATISTICS =====
            # Summary information for logging, reporting, and validation
            "fleet_stats": {
                # Service coverage metrics
                "routes_with_service": int(
                    routes_with_service
                ),  # How many routes active
                # Peak period analysis
                "peak_interval": peak_interval_idx,  # When peak occurs
                "peak_interval_fleet": total_current_fleet_peak,  # Peak fleet size
                "off_peak_fleet": off_peak_fleet,  # Minimum active fleet
                "interval_utilization": interval_utilization,  # % utilization by interval
                # Efficiency metrics
                "fleet_efficiency_gain": fleet_efficiency_gain,  # Vehicles saved vs naive
                # Route-level statistics
                "mean_fleet_per_route": (
                    float(np.mean(current_fleet_per_route[current_fleet_per_route > 0]))
                    if routes_with_service > 0
                    else 0.0
                ),
                "max_route_fleet": int(
                    np.max(current_fleet_per_route)
                ),  # Largest single route
                # Fleet size distribution (helps understand system complexity)
                "fleet_distribution": {
                    "small_routes": int(
                        np.sum(  # Routes needing 1-5 vehicles
                            (current_fleet_per_route > 0)
                            & (current_fleet_per_route <= 5)
                        )
                    ),
                    "medium_routes": int(
                        np.sum(  # Routes needing 6-15 vehicles
                            (current_fleet_per_route > 5)
                            & (current_fleet_per_route <= 15)
                        )
                    ),
                    "large_routes": int(
                        np.sum(current_fleet_per_route > 15)
                    ),  # Routes needing >15 vehicles
                },
            },
        }

    def _safe_timestr_to_seconds(self, time_value: Any) -> float:
        """
        Safely convert GTFS time values to seconds from midnight.

        Handles various GTFS time formats and edge cases including:
        - Standard HH:MM:SS format
        - Times beyond 24:00:00 (e.g., 25:30:00 for 1:30 AM next day)
        - Missing/null values
        - Already-converted numeric values

        Args:
            time_value: GTFS time string, numeric value, or NaN/None

        Returns:
            Time in seconds since midnight as float.
            Returns NaN for invalid/missing values.

        Example:
            '06:30:00' → 23400.0 seconds
            '25:15:00' → 90900.0 seconds (1:15 AM next day)
            NaN → NaN
        """
        try:
            if pd.isna(time_value):
                return np.nan
            if isinstance(time_value, str):
                return gk.helpers.timestr_to_seconds(time_value)
            else:
                return float(time_value)
        except Exception:
            return np.nan
