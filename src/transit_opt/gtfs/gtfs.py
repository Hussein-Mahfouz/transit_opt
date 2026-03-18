import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SolutionConverter:
    """Convert optimization solutions back to GTFS-compatible formats."""

    def __init__(self, optimization_data: dict):
        """
        Initialize converter with optimization data structure.

        Args:
            optimization_data: Complete optimization data from GTFSDataPreparator
        """
        self.opt_data = optimization_data
        self.allowed_headways = optimization_data["allowed_headways"]
        self.no_service_index = optimization_data.get("no_service_index")
        self.route_ids = optimization_data["routes"]["ids"]
        self.interval_labels = optimization_data["intervals"]["labels"]
        self.interval_hours = optimization_data["intervals"]["hours"]

        # Extract original GTFS data for reconstruction
        self.gtfs_feed = optimization_data["reconstruction"]["gtfs_feed"]
        self.route_mapping = optimization_data["reconstruction"]["route_mapping"]

        logger.info(
            "SolutionConverter initialized for %d routes, %d intervals", len(self.route_ids), len(self.interval_labels)
        )

    # ========== CORE CONVERSION METHODS ========== #

    def solution_to_headways(self, solution_matrix: np.ndarray) -> dict[str, dict[str, float | None]]:
        """
        Convert solution matrix to actual headway values per route/interval.

        Args:
            solution_matrix: Array of shape (n_routes, n_intervals) containing headway indices

        Returns:
            Dict mapping route_id -> interval_label -> headway_minutes (or None for no service)

        Example:
            {
                'route_123': {
                    'Early Morning': 30.0,
                    'Morning Peak': 15.0,
                    'Midday': None,  # No service
                    'Evening Peak': 20.0
                },
                'route_456': {...}
            }
        """
        if solution_matrix.shape != (len(self.route_ids), len(self.interval_labels)):
            raise ValueError(
                f"Solution matrix shape {solution_matrix.shape} doesn't match "
                f"expected ({len(self.route_ids)}, {len(self.interval_labels)})"
            )

        headways_dict = {}

        for route_idx, route_id in enumerate(self.route_ids):
            route_headways = {}

            for interval_idx, interval_label in enumerate(self.interval_labels):
                headway_index = int(solution_matrix[route_idx, interval_idx])

                # Validate headway index
                if headway_index < 0 or headway_index >= len(self.allowed_headways):
                    logger.warning(
                        "Invalid headway index %d for route %s, interval %s. Using no service.",
                        headway_index,
                        route_id,
                        interval_label,
                    )
                    headway_minutes = None
                elif headway_index == self.no_service_index:
                    headway_minutes = None  # No service
                else:
                    headway_minutes = float(self.allowed_headways[headway_index])

                route_headways[interval_label] = headway_minutes

            headways_dict[route_id] = route_headways

        logger.debug("Converted solution matrix to headways for %d routes", len(headways_dict))
        return headways_dict

    def extract_route_templates(self) -> dict[str, dict[str, Any]]:
        """
        Extract template trips for each route with time-of-day variation.

        Crucially, if a route direction has multiple headsigns (branches/variations),
        this method enforces a PROPORTIONAL ROUND-ROBIN structure. It analyzes the
        historical trip frequency of each variation, finds the minimum common ratio,
        and populates the returned `dir_templates` list with identically duplicated
        templates matching those ratios. The `splitting_factor` is hardcoded to 1.0
        because branch assignment is handled purely via sequentially iterating through this
        weighted list during generation, inherently preserving the exact optimized headway
        and preventing ANY fleet inflation or 'excess' trips.

        Returns:
            Dict mapping route_id -> interval_label -> template_data
            Format: {
                'route_123': {
                    'Early Morning': {'trip_id': '...', 'stop_times': df, 'duration_minutes': 45.2, ...},
                    'Morning Peak': {'trip_id': '...', 'stop_times': df, 'duration_minutes': 52.1, ...},
                    'Midday': {'trip_id': '...', 'stop_times': df, 'duration_minutes': 43.8, ...},
                    'Evening Peak': {'trip_id': '...', 'stop_times': df, 'duration_minutes': 51.5, ...}
                }
            }
        """
        template_trips = {}

        for route_id in self.route_ids:
            logger.debug(f"🔍 Processing route {route_id}...")

            # Get all trips for this route
            route_trips = self.gtfs_feed.trips[self.gtfs_feed.trips.route_id == route_id]

            if route_trips.empty:
                logger.warning("No trips found for route %s", route_id)
                continue

            # Get all stop times for this route
            route_stop_times = self.gtfs_feed.stop_times[
                self.gtfs_feed.stop_times.trip_id.isin(route_trips.trip_id)
            ].copy()

            if route_stop_times.empty:
                logger.warning("No stop times found for route %s", route_id)
                continue

            # Extract templates for each interval
            route_templates = {}

            # Get available direction_ids for this route
            directions = [0]
            direction_source = "default"

            if "direction_id" in route_trips.columns:
                unique_dirs = route_trips["direction_id"].unique()
                # Handle case where direction_id might be NaN (common in some feeds)
                if len(unique_dirs) > 0 and not pd.isna(unique_dirs).all():
                    directions = [d for d in unique_dirs if not pd.isna(d)]
                    direction_source = "direction_id"

            # Fallback: Try using trip_headsign if direction_id is missing/empty
            if direction_source == "default" and "trip_headsign" in route_trips.columns:
                unique_headsigns = route_trips["trip_headsign"].value_counts()

                # Check if we have multiple headsigns
                if len(unique_headsigns) > 1:
                    # Spatial grouping into exactly TWO directions
                    # 1. Determine first and last stops for each trip to infer direction
                    route_trips = route_trips.copy()

                    # Sort stop times to safely get first/last
                    st_sorted = route_stop_times.sort_values(["trip_id", "stop_sequence"])

                    first_stops = st_sorted.groupby("trip_id")["stop_id"].first()
                    last_stops = st_sorted.groupby("trip_id")["stop_id"].last()

                    route_trips["inferred_first_stop"] = route_trips["trip_id"].map(first_stops)
                    route_trips["inferred_last_stop"] = route_trips["trip_id"].map(last_stops)

                    # Determine the two most common origins across all trips
                    first_stop_counts = route_trips["inferred_first_stop"].value_counts()
                    anchor_0_orig = first_stop_counts.index[0] if len(first_stop_counts) > 0 else None
                    anchor_1_orig = first_stop_counts.index[1] if len(first_stop_counts) > 1 else None

                    headsign_map = {}
                    for headsign in unique_headsigns.index:
                        hs_trips = route_trips[route_trips["trip_headsign"] == headsign]

                        f_stop = None
                        l_stop = None
                        if not hs_trips.empty and not hs_trips["inferred_first_stop"].isna().all():
                            f_stop = hs_trips["inferred_first_stop"].mode().iloc[0]
                        if not hs_trips.empty and not hs_trips["inferred_last_stop"].isna().all():
                            l_stop = hs_trips["inferred_last_stop"].mode().iloc[0]

                        # Use the user's origin-first grouping logic
                        if f_stop == anchor_0_orig:
                            headsign_map[headsign] = 0
                        elif f_stop == anchor_1_orig:
                            headsign_map[headsign] = 1
                        elif l_stop == anchor_1_orig:
                            # If it ends at Dir 1's start, it must be Dir 0
                            headsign_map[headsign] = 0
                        elif l_stop == anchor_0_orig:
                            # If it ends at Dir 0's start, it must be Dir 1
                            headsign_map[headsign] = 1
                        else:
                            # Fallback if logic fails to group
                            headsign_map[headsign] = 0

                    logger.info(
                        f"Route {route_id}: Spatially grouping {len(unique_headsigns)} headsigns into 2 directions using origin anchors."
                    )

                    # Apply mapping
                    if "direction_id" not in route_trips.columns:
                        route_trips = route_trips.copy()

                    route_trips["direction_id"] = route_trips["trip_headsign"].map(headsign_map)
                    directions = list(set(headsign_map.values()))
                    direction_source = "trip_headsign"
                else:
                    # Only 1 headsign, treat as direction 0
                    pass

            # Final check to ensure we have at least one direction
            if not directions:
                directions = [0]

            for interval_idx, (interval_label, (start_hour, end_hour)) in enumerate(
                zip(self.interval_labels, self.interval_hours, strict=False)
            ):
                interval_templates = {}

                for direction_id in directions:
                    if pd.isna(direction_id):
                        direction_id = 0  # Default to 0

                    # Find trips that start within this time interval AND match direction
                    try:
                        dir_trips = route_trips[route_trips["direction_id"] == direction_id]
                    except KeyError:
                        dir_trips = route_trips

                    if dir_trips.empty:
                        continue

                    interval_trips = self._get_trips_in_interval(route_stop_times, dir_trips, start_hour, end_hour)

                    dir_templates = []

                    if not interval_trips.empty:
                        # Find unique variations (headsigns) in this direction/interval
                        if "trip_headsign" in interval_trips.columns:
                            # PROPORTIONAL ROUND-ROBIN LOGIC:
                            # Instead of assigning a blanket template for all vehicles, we measure the
                            # historical frequency of each branch (identified by headsign) in the base GTFS.
                            # We then populate the `dir_templates` array with identical duplicates of branch
                            # templates matching their historical proportion.
                            # Example: If Branch A ran 10 times, and Branch B ran 2 times, the min_count is 2.
                            # We append Template A 5 times (10/2) and Template B 1 time (2/2).
                            # When trips are generated later, they cycle sequentially through this array,
                            # guaranteeing Branch A gets 5 vehicles for every 1 vehicle on Branch B, WITHOUT
                            # generating any extra / parallel trips or altering the aggregate headway.
                            hs_groups = interval_trips.groupby("trip_headsign")
                            total_dir_trips_count = len(interval_trips["trip_id"].unique())

                            branch_counts = {}
                            branch_templates = {}

                            for hs, hs_trips in hs_groups:
                                trip_count = len(hs_trips["trip_id"].unique())
                                try:
                                    template = self._extract_interval_template(hs_trips, interval_label)
                                    template["direction_id"] = int(direction_id)
                                    # Splitting factor is strictly 1.0! The total volume of vehicles is solely
                                    # dictated by the base headway. We only divide which path they take.
                                    template["splitting_factor"] = 1.0

                                    branch_templates[hs] = template
                                    branch_counts[hs] = trip_count
                                except Exception as e:
                                    # If a template cannot be extracted, it is dropped from branch_counts/templates.
                                    # Because vehicle volume is strictly governed by the aggregate headway, the
                                    # modulo selection array will just be smaller. The scheduled trips/vehicles for
                                    # this dropped branch are thus automatically redistributed among the other branches!
                                    logger.warning(
                                        f"Failed to extract pattern for headsign '{hs}' on route {route_id}: {e}. "
                                        f"Variation ignored. Its trips will be absorbed by other healthy variations."
                                    )
                                    pass

                            if branch_counts:
                                # Determine base denominator to scale down counts into integer ratios
                                min_count = max(1, min(branch_counts.values()))

                                for hs, count in branch_counts.items():
                                    # Calculate proportional weight (e.g. 10 base trips / 2 minimum = ratio of 5)
                                    ratio = max(1, round(count / min_count))

                                    # Populate the variation pool. A ratio of 5 adds 5 adjacent identical templates.
                                    for _ in range(ratio):
                                        dir_templates.append(branch_templates[hs])
                        else:
                            template = self._extract_interval_template(interval_trips, interval_label)
                            template["direction_id"] = int(direction_id)
                            template["splitting_factor"] = 1.0
                            dir_templates.append(template)

                    if not dir_templates:
                        # Fallback
                        try:
                            fallback_template = self._get_fallback_template(dir_trips, route_stop_times)
                            fallback_template["splitting_factor"] = 1.0
                            fallback_template["direction_id"] = int(direction_id)
                            dir_templates.append(fallback_template)
                        except Exception:
                            pass

                    if dir_templates:
                        interval_templates[int(direction_id)] = dir_templates

                if interval_templates:
                    route_templates[interval_label] = interval_templates

            # Validate extraction
            if not route_templates:
                logger.warning("Failed to extract any templates for route %s", route_id)
                continue

            template_trips[route_id] = route_templates

        # Clean stop times data for all templates
        logger.info("Cleaning stop times data for all templates...")
        for route_id, route_templates in template_trips.items():
            for interval_label, dir_templates in route_templates.items():
                for direction_id, templates_list in dir_templates.items():
                    for template in templates_list:
                        original_count = len(template["stop_times"])
                        template["stop_times"] = self._clean_stop_times(template["stop_times"])
                        cleaned_count = len(template["stop_times"])

                        if cleaned_count != original_count:
                            logger.debug(
                                "Cleaned stop times for %s %s (Dir %s): %d -> %d stops",
                                route_id,
                                interval_label,
                                direction_id,
                                original_count,
                                cleaned_count,
                            )

        logger.info("Extracted time-varying templates for %d routes", len(template_trips))
        return template_trips

    def generate_trips_and_stop_times(
        self, headways_dict: dict, templates: dict, service_id: str = "optimized_service"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate new trips and stop_times based on optimized headways with time-of-day templates.

        Crucially for branched routes, this function ensures that the AGGREGATE service frequency
        exactly matches the `headway` generated by the solver. It simply divides those scheduled
        dispatches proportionally among the various branches/headsigns using the scaled
        round-robin template list generated in `extract_route_templates()`. No parallel or
        additional subset trips are created.

        Args:
            headways_dict: Route headways by interval
            templates: Template data for each route (now with interval-specific templates)
            service_id: Service ID to use for all generated trips

        Returns:
            tuple: (new_trips_df, new_stop_times_df)
        """
        new_trips = []
        new_stop_times = []
        trip_counter = 0

        # Get interval information
        intervals = self.opt_data["intervals"]

        for route_id, route_headways in headways_dict.items():
            if route_id not in templates:
                logger.warning("No template found for route %s, skipping", route_id)
                continue

            route_templates = templates.get(route_id, {})

            for interval_idx, (interval_label, headway) in enumerate(route_headways.items()):
                if headway is None:
                    continue

                # Get interval time bounds
                start_hour, end_hour = intervals["hours"][interval_idx]

                # Get templates for this interval (by direction)
                interval_templates = route_templates.get(interval_label, {})

                # If no templates specifically for this interval, try to find ANY templates for this route
                # This acts as a fallback if specific interval data is missing
                if not interval_templates:
                    # Collect all available templates across all intervals
                    all_avail_templates_dict = {}  # direction_id -> template

                    # Scan all intervals to find at least one template per direction
                    for t_dict in route_templates.values():
                        for dir_id, t in t_dict.items():
                            if dir_id not in all_avail_templates_dict:
                                all_avail_templates_dict[dir_id] = t

                    if not all_avail_templates_dict:
                        logger.warning("No templates found for route %s in any interval", route_id)
                        continue

                    # Use these fallback templates
                    interval_templates = all_avail_templates_dict

                # Iterate through DIRECTION TEMPLATES
                for dir_index, (direction_id, templates_list) in enumerate(interval_templates.items()):
                    if not templates_list:
                        continue

                    # We just use the first template for some base metadata (assuming they share a lot)
                    base_template = templates_list[0]
                    splitting_factor = base_template.get("splitting_factor", 1.0)
                    effective_headway = headway * splitting_factor

                    # Calculate start offset to interleave departures
                    # If headway = 15, and 2 directions, Dir 0 starts at 0, Dir 1 starts at 15
                    start_offset = dir_index * headway

                    if splitting_factor > 1.0:
                        logger.debug(
                            f"Applying splitting factor {splitting_factor} to Route {route_id} Interval {interval_label} (Headway {headway} -> {effective_headway})"
                        )

                    # Generate trips for this interval/direction using ALL templates (round-robin)
                    interval_trips, interval_stop_times = self._generate_interval_trips(
                        route_id=route_id,
                        templates_list=templates_list,
                        start_hour=start_hour,
                        end_hour=end_hour,
                        headway_minutes=effective_headway,
                        trip_counter_start=trip_counter,
                        service_id=service_id,
                        direction_id=int(direction_id),  # EXPLICITLY pass direction_id
                        start_offset_minutes=start_offset,  # Interleaved departures
                    )

                    # Update trip IDs to include direction info to avoid collisions
                    # (Though _generate_interval_trips uses sequential IDs so it might be fine,
                    # but adding direction to GTFS is good practice)
                    for trip in interval_trips:
                        trip["direction_id"] = int(direction_id)
                        # Ensure trip_id is unique if counter was shared?
                        # Actually trip_counter increments after loop, so we should increment it here
                        # OR just let the list append.

                    new_trips.extend(interval_trips)
                    new_stop_times.extend(interval_stop_times)

                    # Increment counter by number of generated trips
                    trip_counter += len(interval_trips)

        # Convert to DataFrames
        trips_df = pd.DataFrame(new_trips) if new_trips else pd.DataFrame()
        stop_times_df = pd.DataFrame(new_stop_times) if new_stop_times else pd.DataFrame()

        logger.info(
            "Generated %d trips with %d stop times using time-varying templates", len(trips_df), len(stop_times_df)
        )
        return trips_df, stop_times_df

    def build_complete_gtfs(
        self,
        headways_dict: dict,
        templates: dict,
        output_dir: str = "output/optimized_gtfs",
        service_id: str = "optimized_service",
        start_date: str = None,
        end_date: str = None,
        copy_calendar_dates: bool = False,
        zip_output: bool = False,
    ) -> str:  # ✅ NEW parameter
        """
        Build complete GTFS feed from optimization solution.

        Args:
            headways_dict: Route headways by interval
            templates: Template data for each route
            output_dir: Output directory for GTFS files
            service_id: Service ID to use for generated trips
            start_date: Optional start date (YYYYMMDD format)
            end_date: Optional end date (YYYYMMDD format)
            copy_calendar_dates: Whether to copy calendar_dates.txt (default: False)
            zip_output: If True, create ZIP file instead of directory (default: False)

        Returns:
            Path to created GTFS directory or ZIP file
        """

        import os
        import tempfile
        import zipfile

        # ✅ Handle ZIP output vs directory output
        if zip_output:
            # Create temporary directory for files
            temp_dir = tempfile.mkdtemp(prefix="gtfs_temp_")
            working_dir = temp_dir
            logger.info("📦 Creating ZIP output: %s.zip", output_dir)
        else:
            # Create regular output directory
            os.makedirs(output_dir, exist_ok=True)
            working_dir = output_dir
            logger.info("📁 Creating directory output: %s", output_dir)

        # 1. Generate new trips and stop times with configurable service_id
        new_trips_df, new_stop_times_df = self.generate_trips_and_stop_times(
            headways_dict, templates, service_id=service_id
        )

        # 2. Copy unchanged files from original GTFS
        original_gtfs = self.opt_data["reconstruction"]["gtfs_feed"]

        # Write stops.txt (unchanged)
        original_stops = original_gtfs.stops.copy()
        # Ensure parent_station references are valid. Some parent stops may have been removed in preprocessing of original GTFS
        fixed_stops = self._fix_parent_station_references(original_stops)
        fixed_stops.to_csv(os.path.join(working_dir, "stops.txt"), index=False)
        logger.info("✅ Fixed and copied stops.txt: %d stops", len(fixed_stops))

        # Write routes.txt (unchanged)
        original_gtfs.routes.to_csv(os.path.join(working_dir, "routes.txt"), index=False)
        logger.info("✅ Copied routes.txt: %d routes", len(original_gtfs.routes))

        # Write agency.txt (unchanged)
        original_gtfs.agency.to_csv(os.path.join(working_dir, "agency.txt"), index=False)
        logger.info("✅ Copied agency.txt: %d agencies", len(original_gtfs.agency))

        # 3. Write new trip files
        new_trips_df.to_csv(os.path.join(working_dir, "trips.txt"), index=False)
        logger.info("✅ Generated trips.txt: %d trips", len(new_trips_df))

        new_stop_times_df.to_csv(os.path.join(working_dir, "stop_times.txt"), index=False)
        logger.info("✅ Generated stop_times.txt: %d stop times", len(new_stop_times_df))

        # 4. Smart calendar.txt creation
        calendar_start_date, calendar_end_date = self._determine_calendar_dates(start_date, end_date)

        calendar_df = pd.DataFrame(
            [
                {
                    "service_id": service_id,
                    "monday": 1,
                    "tuesday": 1,
                    "wednesday": 1,
                    "thursday": 1,
                    "friday": 1,
                    "saturday": 1,
                    "sunday": 1,
                    "start_date": calendar_start_date,
                    "end_date": calendar_end_date,
                }
            ]
        )
        calendar_df.to_csv(os.path.join(working_dir, "calendar.txt"), index=False)
        logger.info("✅ Generated calendar.txt: %s (%s to %s)", service_id, calendar_start_date, calendar_end_date)

        # 5. Copy shapes.txt if it exists
        if hasattr(original_gtfs, "shapes") and not original_gtfs.shapes.empty:
            original_gtfs.shapes.to_csv(os.path.join(working_dir, "shapes.txt"), index=False)
            logger.info("✅ Copied shapes.txt: %d shape points", len(original_gtfs.shapes))

            # Validate shape references
            shapes_used = set(new_trips_df["shape_id"].dropna()) if "shape_id" in new_trips_df.columns else set()
            shapes_available = set(original_gtfs.shapes["shape_id"].unique())
            missing_shapes = shapes_used - shapes_available
            if missing_shapes:
                logger.warning("⚠️  Missing shapes: %s", missing_shapes)
            else:
                logger.info("✅ All %d shape references validated", len(shapes_used))

        # 6. Handle calendar_dates.txt
        if copy_calendar_dates and hasattr(original_gtfs, "calendar_dates") and not original_gtfs.calendar_dates.empty:
            logger.info("⚠️  WARNING: Copying calendar_dates.txt with original service IDs")
            original_gtfs.calendar_dates.to_csv(os.path.join(working_dir, "calendar_dates.txt"), index=False)
            logger.info("✅ Copied calendar_dates.txt: %d exceptions", len(original_gtfs.calendar_dates))
        else:
            logger.info("✅ Skipped calendar_dates.txt: Using simplified calendar only")

        # ✅ 7. Handle ZIP creation
        if zip_output:
            # Create ZIP file
            zip_path = f"{output_dir}.zip"

            # Ensure output directory exists for ZIP file
            os.makedirs(os.path.dirname(zip_path) if os.path.dirname(zip_path) else ".", exist_ok=True)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add all .txt files to ZIP
                for filename in os.listdir(working_dir):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(working_dir, filename)
                        zipf.write(file_path, filename)  # Store with just filename (no path)

            # Clean up temporary directory
            import shutil

            shutil.rmtree(temp_dir)

            # Show ZIP contents and size
            zip_size = os.path.getsize(zip_path)
            with zipfile.ZipFile(zip_path, "r") as zipf:
                file_list = zipf.namelist()

            logger.info("📦 ZIP FILE CREATED: %s", zip_path)
            logger.info("   File size: %d bytes", zip_size)
            logger.info("   Contains: %s", ", ".join(file_list))

            return zip_path
        else:
            logger.info("🎯 COMPLETE GTFS FEED CREATED: %s", working_dir)
            return working_dir

    # ========= VALIDATION & SUMMARY METHODS ========== #

    def validate_solution(self, solution_matrix: np.ndarray) -> dict[str, Any]:
        """
        Validate that solution matrix contains reasonable values.

        Args:
            solution_matrix: Array of headway indices

        Returns:
            Dict with validation results and statistics
        """
        validation_result = {"valid": True, "errors": [], "warnings": [], "statistics": {}}

        # Check for None or invalid input types first
        if solution_matrix is None:
            validation_result["valid"] = False
            validation_result["errors"].append("Solution matrix is None")
            return validation_result

        # Check if input is a numpy array or can be converted to one
        if not isinstance(solution_matrix, np.ndarray):
            try:
                # Try to convert to numpy array
                solution_matrix = np.array(solution_matrix)
            except (ValueError, TypeError) as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Cannot convert input to numpy array: {e}")
                return validation_result

        # Check if array is empty
        if solution_matrix.size == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Solution matrix is empty")
            return validation_result

        # Check number of dimensions
        if solution_matrix.ndim != 2:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Solution matrix must be 2-dimensional, got {solution_matrix.ndim} dimensions"
            )
            return validation_result

        # Check shape
        expected_shape = (len(self.route_ids), len(self.interval_labels))
        if solution_matrix.shape != expected_shape:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Solution shape {solution_matrix.shape} doesn't match expected {expected_shape}"
            )
            return validation_result

        # Check value ranges
        min_val = np.min(solution_matrix)
        max_val = np.max(solution_matrix)
        max_allowed = len(self.allowed_headways) - 1

        if min_val < 0:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Solution contains negative values (min: {min_val})")

        if max_val > max_allowed:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Solution contains values > {max_allowed} (max: {max_val})")

        # Check for reasonable service patterns
        total_cells = solution_matrix.size
        no_service_cells = np.sum(solution_matrix == self.no_service_index) if self.no_service_index is not None else 0
        service_cells = total_cells - no_service_cells
        service_percentage = (service_cells / total_cells) * 100

        if service_percentage < 10:
            validation_result["warnings"].append(
                f"Very low service coverage: only {service_percentage:.1f}% of route-intervals have service"
            )
        elif service_percentage > 95:
            validation_result["warnings"].append(
                f"Very high service coverage: {service_percentage:.1f}% of route-intervals have service"
            )

        # Count headway frequency distribution
        headway_counts = {}
        for i, headway in enumerate(self.allowed_headways):
            count = np.sum(solution_matrix == i)
            if count > 0:
                if headway >= 9000:  # No service threshold
                    headway_counts["No Service"] = count
                else:
                    headway_counts[f"{headway:.0f}min"] = count

        # Check for at least some reasonable service frequencies
        frequent_service_count = 0
        for i, headway in enumerate(self.allowed_headways):
            if headway <= 30:  # Consider ≤30min as frequent
                frequent_service_count += np.sum(solution_matrix == i)

        frequent_percentage = (frequent_service_count / service_cells) * 100 if service_cells > 0 else 0

        if frequent_percentage < 20 and service_cells > 0:
            validation_result["warnings"].append(
                f"Low frequent service: only {frequent_percentage:.1f}% of service cells have ≤30min headways"
            )

        # Store statistics
        validation_result["statistics"] = {
            "total_cells": total_cells,
            "service_cells": service_cells,
            "no_service_cells": no_service_cells,
            "service_percentage": service_percentage,
            "frequent_service_percentage": frequent_percentage,
            "headway_distribution": headway_counts,
            "value_range": (min_val, max_val),
        }

        logger.info(
            "Solution validation: %s (%d errors, %d warnings)",
            "PASSED" if validation_result["valid"] else "FAILED",
            len(validation_result["errors"]),
            len(validation_result["warnings"]),
        )

        return validation_result

    def get_solution_summary(self, solution_matrix: np.ndarray) -> dict[str, Any]:
        """
        Get a comprehensive summary of the solution.

        Args:
            solution_matrix: Solution to summarize

        Returns:
            Dict with solution summary statistics
        """
        headways_dict = self.solution_to_headways(solution_matrix)
        validation = self.validate_solution(solution_matrix)

        # Count active routes per interval
        active_routes_per_interval = {}
        total_service_minutes_per_interval = {}

        for interval_idx, interval_label in enumerate(self.interval_labels):
            active_count = 0
            total_service_minutes = 0

            for route_idx, route_id in enumerate(self.route_ids):
                headway_idx = solution_matrix[route_idx, interval_idx]
                if headway_idx != self.no_service_index:
                    active_count += 1
                    headway_minutes = self.allowed_headways[headway_idx]
                    if headway_minutes < 9000:  # Valid service
                        interval_duration_minutes = (
                            self.interval_hours[interval_idx][1] - self.interval_hours[interval_idx][0]
                        ) * 60
                        trips_in_interval = max(1, interval_duration_minutes // headway_minutes)
                        total_service_minutes += trips_in_interval * headway_minutes

            active_routes_per_interval[interval_label] = active_count
            total_service_minutes_per_interval[interval_label] = total_service_minutes

        return {
            "validation": validation,
            "headways_dict": headways_dict,
            "active_routes_per_interval": active_routes_per_interval,
            "total_service_minutes_per_interval": total_service_minutes_per_interval,
            "overall_stats": {
                "total_routes": len(self.route_ids),
                "total_intervals": len(self.interval_labels),
                "service_percentage": validation["statistics"]["service_percentage"],
            },
        }

    # ========== UTILITY METHODS ========== #

    def get_interval_time_bounds(self, interval_label: str) -> tuple[int, int]:
        """
        Get start and end times (in seconds) for a time interval.

        Args:
            interval_label: Label like 'Morning Peak', 'Midday', etc.

        Returns:
            Tuple of (start_seconds, end_seconds)
        """
        try:
            interval_idx = self.interval_labels.index(interval_label)
            start_hour, end_hour = self.interval_hours[interval_idx]
            return start_hour * 3600, end_hour * 3600
        except ValueError:
            logger.error("Unknown interval label: %s", interval_label)
            raise ValueError(f"Unknown interval label: {interval_label}")

    # ========== INTERNAL HELPER METHODS ========== #

    def _generate_interval_trips(
        self,
        route_id: str,
        templates_list: list[dict],
        start_hour: int,
        end_hour: int,
        headway_minutes: float,
        trip_counter_start: int,
        service_id: str = "optimized_service",
        direction_id: int = 0,
        start_offset_minutes: float = 0.0,
    ) -> tuple[list, list]:
        """
        Generate trips and stop times for a single route/direction combination using templates.

        This method executes the PROPORTIONAL ROUND-ROBIN selection. Every `headway_minutes`,
        it generates exactly ONE new trip. It selects the path (branch template) for that trip
        by cycling sequentially through `templates_list` using modulo (`trip_idx % len()`).
        Because `extract_route_templates` pre-populated `templates_list` with properly weighted
        duplicate branch templates (matching exactly their historical dispatch ratio), iterating
        through this array natively distributes vehicles across variations. This absolutely guarantees
        that no 'extra' trips are created, and the total aggregate frequency explicitly matches the solver's headway.
        """
        trips = []
        stop_times = []

        # Get original trip info for shape_id and other metadata (as fallback)
        original_gtfs = self.opt_data["reconstruction"]["gtfs_feed"]
        original_route_trips = original_gtfs.trips[original_gtfs.trips.route_id == route_id]

        # Calculate trip start times
        interval_start_seconds = start_hour * 3600
        interval_end_seconds = end_hour * 3600
        headway_seconds = headway_minutes * 60

        # Generate trips every headway minutes
        current_start = interval_start_seconds + (start_offset_minutes * 60)

        # Iterate generating trips
        trip_idx = 0

        while current_start < interval_end_seconds:
            # Round-robin selection of template for variations
            template = templates_list[trip_idx % len(templates_list)]
            template_stop_times = template["stop_times"]
            trip_duration_seconds = template["duration_minutes"] * 60

            # Check if this specific trip completes before the interval ends
            if (current_start + trip_duration_seconds) > interval_end_seconds:
                # Instead of completely breaking, you could adjust logic here.
                # However, usually we don't start a trip if it can't finish in the interval
                # OR we let it finish. Often in schedule generation, you DO let the trip finish.
                # Let's break as per previous logic.
                break

            # Extract specific metadata for THIS template
            route_info = template.get("route_info", {})
            template_headsign = route_info.get("trip_headsign", None)
            template_shape_id = route_info.get("shape_id", None)

            fallback_shape_id = ""
            fallback_headsign = f"Route {route_id}"
            if not original_route_trips.empty:
                t_trip = original_route_trips.iloc[0]
                fallback_shape_id = t_trip.get("shape_id", "")
                th = t_trip.get("trip_headsign", fallback_headsign)
                fallback_headsign = fallback_headsign if pd.isna(th) else th

            final_shape_id = template_shape_id if template_shape_id is not None else fallback_shape_id
            final_headsign = template_headsign if template_headsign is not None else fallback_headsign
            if pd.isna(final_headsign):
                final_headsign = fallback_headsign

            # Create trip ID - Include direction_id to ensure uniqueness
            trip_id = f"opt_trip_{route_id}_{start_hour:02d}_{direction_id}_{trip_idx:03d}"

            # Create trip record
            trip_record = {
                "trip_id": trip_id,
                "route_id": route_id,
                "service_id": service_id,
                "trip_headsign": final_headsign,
                "direction_id": direction_id,
                "shape_id": final_shape_id,
            }
            trips.append(trip_record)

            # Create stop times for this trip
            trip_stop_times = self._create_trip_stop_times(
                trip_id=trip_id, template_stop_times=template_stop_times, trip_start_seconds=current_start
            )
            stop_times.extend(trip_stop_times)

            # Move to next trip
            current_start += headway_seconds
            trip_idx += 1

        return trips, stop_times

    def _create_trip_stop_times(self, trip_id: str, template_stop_times: pd.DataFrame, trip_start_seconds: int) -> list:
        """Create stop_times records for a single trip based on template."""

        stop_times = []

        # Get the first stop's departure time from template (as offset from trip start)
        template_first_departure = template_stop_times.iloc[0]["departure_seconds"]

        for _, template_stop in template_stop_times.iterrows():
            # Calculate time offset from first stop
            departure_offset = template_stop["departure_seconds"] - template_first_departure
            arrival_offset = template_stop["arrival_seconds"] - template_first_departure

            # Apply offset to new trip start time
            new_departure = trip_start_seconds + departure_offset
            new_arrival = trip_start_seconds + arrival_offset

            # Create stop time record
            stop_time_record = {
                "trip_id": trip_id,
                "stop_id": template_stop["stop_id"],
                "stop_sequence": template_stop["stop_sequence"],
                "arrival_time": self._seconds_to_gtfs_time(new_arrival),
                "departure_time": self._seconds_to_gtfs_time(new_departure),
                "arrival_seconds": new_arrival,
                "departure_seconds": new_departure,
                "pickup_type": template_stop.get("pickup_type", 0),
                "drop_off_type": template_stop.get("drop_off_type", 0),
            }
            stop_times.append(stop_time_record)

        return stop_times

    def _seconds_to_gtfs_time(self, seconds: int) -> str:
        """Convert seconds since midnight to HH:MM:SS format."""
        # Ensure seconds is an integer
        seconds = int(round(seconds))

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _determine_calendar_dates(self, start_date: str = None, end_date: str = None) -> tuple[str, str]:
        """
        Determine calendar start and end dates with smart defaults.

        Args:
            start_date: Optional start date (YYYYMMDD). If None, uses min from original
            end_date: Optional end date (YYYYMMDD). If None, uses max from original

        Returns:
            Tuple of (start_date, end_date) in YYYYMMDD format
        """
        original_gtfs = self.opt_data["reconstruction"]["gtfs_feed"]

        # Handle start_date
        if start_date is not None:
            calendar_start_date = start_date
            logger.info("📅 Using provided start_date: %s", start_date)
        else:
            # Find minimum start_date from original calendar
            if hasattr(original_gtfs, "calendar") and not original_gtfs.calendar.empty:
                min_start_date = original_gtfs.calendar["start_date"].min()
                calendar_start_date = str(min_start_date)
                logger.info("📅 Using min original start_date: %s", calendar_start_date)
            else:
                calendar_start_date = "20240101"
                logger.info("📅 Using default start_date: %s", calendar_start_date)

        # Handle end_date
        if end_date is not None:
            calendar_end_date = end_date
            logger.info("📅 Using provided end_date: %s", end_date)
        else:
            # Find maximum end_date from original calendar
            if hasattr(original_gtfs, "calendar") and not original_gtfs.calendar.empty:
                max_end_date = original_gtfs.calendar["end_date"].max()
                calendar_end_date = str(max_end_date)
                logger.info("📅 Using max original end_date: %s", calendar_end_date)
            else:
                calendar_end_date = "20241231"
                logger.info("📅 Using default end_date: %s", calendar_end_date)

        return calendar_start_date, calendar_end_date

    def _fix_parent_station_references(self, stops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix parent_station foreign key violations by clearing invalid references.

        Args:
            stops_df: DataFrame with stops data

        Returns:
            DataFrame with fixed parent_station references
        """
        stops_df = stops_df.copy()

        if "parent_station" not in stops_df.columns:
            return stops_df

        # Get all valid stop_ids that exist in the file
        valid_stop_ids = set(stops_df["stop_id"].astype(str))

        # Find rows where parent_station points to non-existent stop_id
        invalid_mask = (
            (stops_df["parent_station"].notna())  # Has a parent_station value
            & (stops_df["parent_station"].astype(str) != "")  # Not empty string
            & (~stops_df["parent_station"].astype(str).isin(valid_stop_ids))  # But it doesn't exist
        )

        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            invalid_refs = stops_df.loc[invalid_mask, "parent_station"].unique()

            logger.warning("Found %d stops with invalid parent_station references: %s", invalid_count, invalid_refs)
            logger.info("   Missing parent stations: %s", list(invalid_refs))

            # Clear the invalid parent_station references
            stops_df.loc[invalid_mask, "parent_station"] = ""
            logger.info("✅ Cleared %d invalid parent_station references", invalid_count)
        else:
            logger.info("✅ All parent_station references are valid")

        return stops_df

    def _get_trips_in_interval(
        self, route_stop_times: pd.DataFrame, route_trips: pd.DataFrame, start_hour: int, end_hour: int
    ) -> pd.DataFrame:
        """
        Find trips that start within a specific time interval.

        Args:
            route_stop_times: Stop times for the route
            route_trips: Trip records for the route
            start_hour: Interval start hour (0-23)
            end_hour: Interval end hour (0-23)

        Returns:
            DataFrame with stop times for trips starting in the interval
        """
        start_seconds = start_hour * 3600
        end_seconds = end_hour * 3600

        # Filter stop times to only include trips from the provided route_trips (which may be filtered by direction)
        if not route_trips.empty:
            relevant_trip_ids = route_trips["trip_id"].unique()
            route_stop_times = route_stop_times[route_stop_times["trip_id"].isin(relevant_trip_ids)]

        # Get first stop time for each trip (trip start time)
        trip_start_times = route_stop_times.groupby("trip_id")["departure_seconds"].min()

        # Filter trips that start within the interval
        trips_in_interval = trip_start_times[
            (trip_start_times >= start_seconds) & (trip_start_times < end_seconds)
        ].index

        # Return stop times for these trips
        interval_stop_times = route_stop_times[route_stop_times["trip_id"].isin(trips_in_interval)].copy()

        logger.debug("Found %d trips starting between %02d:00-%02d:00", len(trips_in_interval), start_hour, end_hour)
        return interval_stop_times

    def _extract_interval_template(self, interval_stop_times: pd.DataFrame, interval_label: str) -> dict[str, Any]:
        """
        Extract a representative template from trips in a specific time interval.

        Args:
            interval_stop_times: Stop times for trips in this interval
            interval_label: Name of the time interval

        Returns:
            Template trip data dictionary
        """
        if interval_stop_times.empty:
            raise ValueError("Cannot extract template from empty stop times")

        # Calculate trip durations for all trips in this interval
        trip_durations = []
        trip_info = {}

        for trip_id in interval_stop_times["trip_id"].unique():
            trip_stops = interval_stop_times[interval_stop_times["trip_id"] == trip_id].sort_values("stop_sequence")

            if len(trip_stops) < 2:  # Skip trips with insufficient stops
                continue

            first_departure = trip_stops.iloc[0]["departure_seconds"]
            last_arrival = trip_stops.iloc[-1]["arrival_seconds"]
            duration_minutes = (last_arrival - first_departure) / 60

            trip_durations.append(duration_minutes)
            trip_info[trip_id] = {
                "duration_minutes": duration_minutes,
                "n_stops": len(trip_stops),
                "stop_times": trip_stops.copy(),
            }

        if not trip_durations:
            raise ValueError(f"No valid trips found in interval {interval_label}")

        # Select median duration trip as template (most representative)
        median_duration = np.median(trip_durations)
        best_trip_id = min(trip_info.keys(), key=lambda x: abs(trip_info[x]["duration_minutes"] - median_duration))

        template_data = trip_info[best_trip_id]

        # Get trip metadata from original trips table
        original_trip = self.gtfs_feed.trips[self.gtfs_feed.trips.trip_id == best_trip_id]
        if not original_trip.empty:
            template_data["route_info"] = original_trip.iloc[0].to_dict()
        else:
            template_data["route_info"] = {}

        template_data["trip_id"] = best_trip_id
        template_data["interval_label"] = interval_label

        logger.debug(
            "Selected template trip %s for %s: %.1fmin duration",
            best_trip_id,
            interval_label,
            template_data["duration_minutes"],
        )

        return template_data

    def _get_fallback_template(self, route_trips: pd.DataFrame, route_stop_times: pd.DataFrame) -> dict[str, Any]:
        """
        Get fallback template when no trips exist in a specific interval.
        Uses the first available trip for the route.

        Args:
            route_trips: All trips for the route
            route_stop_times: All stop times for the route

        Returns:
            Template trip data dictionary
        """
        # Use first trip as fallback
        template_trip = route_trips.iloc[0]
        trip_id = template_trip.trip_id

        # Get stop times for this trip
        trip_stop_times = route_stop_times[route_stop_times["trip_id"] == trip_id].sort_values("stop_sequence").copy()

        if trip_stop_times.empty:
            raise ValueError(f"No stop times found for fallback trip {trip_id}")

        # Calculate duration
        first_departure = trip_stop_times.iloc[0]["departure_seconds"]
        last_arrival = trip_stop_times.iloc[-1]["arrival_seconds"]
        duration_minutes = (last_arrival - first_departure) / 60

        return {
            "trip_id": trip_id,
            "stop_times": trip_stop_times,
            "duration_minutes": duration_minutes,
            "n_stops": len(trip_stop_times),
            "route_info": template_trip.to_dict(),
            "interval_label": "fallback",
        }

    def _clean_stop_times(self, stop_times_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stop times data by filling NaN values with reasonable estimates.

        For missing arrival/departure times, attempts to interpolate from neighboring stops.
        Falls back to zero if no neighboring data is available.

        Args:
            stop_times_df: DataFrame with stop times data

        Returns:
            DataFrame with cleaned stop times
        """
        if stop_times_df.empty:
            return stop_times_df

        # Make a copy to avoid modifying original
        cleaned_df = stop_times_df.copy()

        # Sort by stop_sequence to ensure proper order for interpolation
        if "stop_sequence" in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values("stop_sequence").reset_index(drop=True)

        # Clean arrival_seconds
        if "arrival_seconds" in cleaned_df.columns:
            # Forward fill then backward fill to use neighboring values
            cleaned_df["arrival_seconds"] = cleaned_df["arrival_seconds"].ffill().bfill()
            # Fall back to zero for any remaining NaN values
            cleaned_df["arrival_seconds"] = cleaned_df["arrival_seconds"].fillna(0)

        # Clean departure_seconds
        if "departure_seconds" in cleaned_df.columns:
            # Forward fill then backward fill to use neighboring values
            cleaned_df["departure_seconds"] = cleaned_df["departure_seconds"].ffill().bfill()
            # Fall back to zero for any remaining NaN values
            cleaned_df["departure_seconds"] = cleaned_df["departure_seconds"].fillna(0)

        # Ensure departure >= arrival (basic consistency check)
        if "arrival_seconds" in cleaned_df.columns and "departure_seconds" in cleaned_df.columns:
            mask = cleaned_df["departure_seconds"] < cleaned_df["arrival_seconds"]
            cleaned_df.loc[mask, "departure_seconds"] = cleaned_df.loc[mask, "arrival_seconds"]

        # Filter out any remaining non-finite values
        if "arrival_seconds" in cleaned_df.columns:
            cleaned_df = cleaned_df[np.isfinite(cleaned_df["arrival_seconds"])]
        if "departure_seconds" in cleaned_df.columns:
            cleaned_df = cleaned_df[np.isfinite(cleaned_df["departure_seconds"])]

        return cleaned_df
