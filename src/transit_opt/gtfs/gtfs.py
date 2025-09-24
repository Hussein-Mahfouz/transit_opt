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
        self.allowed_headways = optimization_data['allowed_headways']
        self.no_service_index = optimization_data.get('no_service_index')
        self.route_ids = optimization_data['routes']['ids']
        self.interval_labels = optimization_data['intervals']['labels']
        self.interval_hours = optimization_data['intervals']['hours']

        # Extract original GTFS data for reconstruction
        self.gtfs_feed = optimization_data['reconstruction']['gtfs_feed']
        self.route_mapping = optimization_data['reconstruction']['route_mapping']

        logger.info(f"SolutionConverter initialized for {len(self.route_ids)} routes, "
                   f"{len(self.interval_labels)} intervals")


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
            raise ValueError(f"Solution matrix shape {solution_matrix.shape} doesn't match "
                           f"expected ({len(self.route_ids)}, {len(self.interval_labels)})")

        headways_dict = {}

        for route_idx, route_id in enumerate(self.route_ids):
            route_headways = {}

            for interval_idx, interval_label in enumerate(self.interval_labels):
                headway_index = int(solution_matrix[route_idx, interval_idx])

                # Validate headway index
                if headway_index < 0 or headway_index >= len(self.allowed_headways):
                    logger.warning(f"Invalid headway index {headway_index} for route {route_id}, "
                                 f"interval {interval_label}. Using no service.")
                    headway_minutes = None
                elif headway_index == self.no_service_index:
                    headway_minutes = None  # No service
                else:
                    headway_minutes = float(self.allowed_headways[headway_index])

                route_headways[interval_label] = headway_minutes

            headways_dict[route_id] = route_headways

        logger.debug(f"Converted solution matrix to headways for {len(headways_dict)} routes")
        return headways_dict


    def extract_route_templates(self) -> dict[str, dict[str, Any]]:
        """
        Extract template trips for each route with time-of-day variation.
        
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
            print(f"🔍 Processing route {route_id}...")

            # Get all trips for this route
            route_trips = self.gtfs_feed.trips[self.gtfs_feed.trips.route_id == route_id]

            if route_trips.empty:
                logger.warning(f"No trips found for route {route_id}")
                continue

            # Get all stop times for this route
            route_stop_times = self.gtfs_feed.stop_times[
                self.gtfs_feed.stop_times.trip_id.isin(route_trips.trip_id)
            ].copy()

            if route_stop_times.empty:
                logger.warning(f"No stop times found for route {route_id}")
                continue

            # Extract templates for each interval
            route_templates = {}

            for interval_idx, (interval_label, (start_hour, end_hour)) in enumerate(
                zip(self.interval_labels, self.interval_hours, strict=False)
            ):
                # Find trips that start within this time interval
                interval_trips = self._get_trips_in_interval(
                    route_stop_times, route_trips, start_hour, end_hour
                )

                if not interval_trips.empty:
                    # Extract template for this specific interval
                    template = self._extract_interval_template(interval_trips, interval_label)
                    route_templates[interval_label] = template
                    print(f"   ✅ {interval_label}: {template['duration_minutes']:.1f}min template")
                else:
                    # No trips in this interval - use fallback
                    fallback_template = self._get_fallback_template(route_trips, route_stop_times)
                    route_templates[interval_label] = fallback_template
                    print(f"   ⚠️  {interval_label}: Using fallback template ({fallback_template['duration_minutes']:.1f}min)")

            template_trips[route_id] = route_templates


        # Clean stop times data for all templates by filling NaNs with forward and backward fill
        logger.info("Cleaning stop times data for all templates...")
        for route_id, route_templates in template_trips.items():
            for interval_label, template in route_templates.items():
                # Clean the stop times data using the new cleaning method
                original_count = len(template['stop_times'])
                template['stop_times'] = self._clean_stop_times(template['stop_times'])
                cleaned_count = len(template['stop_times'])

                if cleaned_count != original_count:
                    logger.debug(f"Cleaned stop times for {route_id} {interval_label}: "
                            f"{original_count} -> {cleaned_count} stops")

        logger.info(f"Extracted time-varying templates for {len(template_trips)} routes")
        return template_trips




    def generate_trips_and_stop_times(self, headways_dict: dict, templates: dict,
                                service_id: str = 'optimized_service') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate new trips and stop_times based on optimized headways with time-of-day templates.
        
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
        intervals = self.opt_data['intervals']

        for route_id, route_headways in headways_dict.items():
            if route_id not in templates:
                logger.warning(f"No template found for route {route_id}, skipping")
                continue

            route_templates = templates[route_id]  # Now contains templates per interval

            for interval_idx, (interval_label, headway) in enumerate(route_headways.items()):
                if headway is None:  # No service in this interval
                    continue

                # ✅ NEW: Get interval-specific template
                if interval_label in route_templates:
                    template = route_templates[interval_label]
                    print(f"🕐 Using {interval_label} template for route {route_id}: {template['duration_minutes']:.1f}min")
                else:
                    # Fallback to first available template
                    template = list(route_templates.values())[0]
                    logger.warning(f"No template for {interval_label}, using fallback for route {route_id}")

                template_stop_times = template['stop_times']
                trip_duration_minutes = template['duration_minutes']

                # Get interval time bounds
                start_hour, end_hour = intervals['hours'][interval_idx]

                # Generate trips for this interval with the specific template
                interval_trips, interval_stop_times = self._generate_interval_trips(
                    route_id=route_id,
                    template_stop_times=template_stop_times,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    headway_minutes=headway,
                    trip_duration_minutes=trip_duration_minutes,
                    trip_counter_start=trip_counter,
                    service_id=service_id
                )

                new_trips.extend(interval_trips)
                new_stop_times.extend(interval_stop_times)
                trip_counter += len(interval_trips)

        # Convert to DataFrames
        trips_df = pd.DataFrame(new_trips) if new_trips else pd.DataFrame()
        stop_times_df = pd.DataFrame(new_stop_times) if new_stop_times else pd.DataFrame()

        logger.info(f"Generated {len(trips_df)} trips with {len(stop_times_df)} stop times using time-varying templates")
        return trips_df, stop_times_df



    def build_complete_gtfs(self, headways_dict: dict, templates: dict,
                        output_dir: str = "output/optimized_gtfs",
                        service_id: str = 'optimized_service',
                        start_date: str = None,
                        end_date: str = None,
                        copy_calendar_dates: bool = False,
                        zip_output: bool = False) -> str:  # ✅ NEW parameter
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
            temp_dir = tempfile.mkdtemp(prefix='gtfs_temp_')
            working_dir = temp_dir
            print(f"📦 Creating ZIP output: {output_dir}.zip")
        else:
            # Create regular output directory
            os.makedirs(output_dir, exist_ok=True)
            working_dir = output_dir
            print(f"📁 Creating directory output: {output_dir}")

        # 1. Generate new trips and stop times with configurable service_id
        new_trips_df, new_stop_times_df = self.generate_trips_and_stop_times(
            headways_dict, templates, service_id=service_id
        )

        # 2. Copy unchanged files from original GTFS
        original_gtfs = self.opt_data['reconstruction']['gtfs_feed']

        # Write stops.txt (unchanged)
        original_stops = original_gtfs.stops.copy()
        # Ensure parent_station references are valid. Some parent stops may have been removed in preprocessing of original GTFS
        fixed_stops = self._fix_parent_station_references(original_stops)
        fixed_stops.to_csv(os.path.join(working_dir, 'stops.txt'), index=False)
        print(f"✅ Fixed and copied stops.txt: {len(fixed_stops)} stops")

        # Write routes.txt (unchanged)
        original_gtfs.routes.to_csv(os.path.join(working_dir, 'routes.txt'), index=False)
        print(f"✅ Copied routes.txt: {len(original_gtfs.routes)} routes")

        # Write agency.txt (unchanged)
        original_gtfs.agency.to_csv(os.path.join(working_dir, 'agency.txt'), index=False)
        print(f"✅ Copied agency.txt: {len(original_gtfs.agency)} agencies")

        # 3. Write new trip files
        new_trips_df.to_csv(os.path.join(working_dir, 'trips.txt'), index=False)
        print(f"✅ Generated trips.txt: {len(new_trips_df)} trips")

        new_stop_times_df.to_csv(os.path.join(working_dir, 'stop_times.txt'), index=False)
        print(f"✅ Generated stop_times.txt: {len(new_stop_times_df)} stop times")

        # 4. Smart calendar.txt creation
        calendar_start_date, calendar_end_date = self._determine_calendar_dates(start_date, end_date)

        calendar_df = pd.DataFrame([{
            'service_id': service_id,
            'monday': 1, 'tuesday': 1, 'wednesday': 1, 'thursday': 1,
            'friday': 1, 'saturday': 1, 'sunday': 1,
            'start_date': calendar_start_date,
            'end_date': calendar_end_date
        }])
        calendar_df.to_csv(os.path.join(working_dir, 'calendar.txt'), index=False)
        print(f"✅ Generated calendar.txt: {service_id} ({calendar_start_date} to {calendar_end_date})")

        # 5. Copy shapes.txt if it exists
        if hasattr(original_gtfs, 'shapes') and not original_gtfs.shapes.empty:
            original_gtfs.shapes.to_csv(os.path.join(working_dir, 'shapes.txt'), index=False)
            print(f"✅ Copied shapes.txt: {len(original_gtfs.shapes)} shape points")

            # Validate shape references
            shapes_used = set(new_trips_df['shape_id'].dropna()) if 'shape_id' in new_trips_df.columns else set()
            shapes_available = set(original_gtfs.shapes['shape_id'].unique())
            missing_shapes = shapes_used - shapes_available
            if missing_shapes:
                print(f"⚠️  Missing shapes: {missing_shapes}")
            else:
                print(f"✅ All {len(shapes_used)} shape references validated")

        # 6. Handle calendar_dates.txt
        if copy_calendar_dates and hasattr(original_gtfs, 'calendar_dates') and not original_gtfs.calendar_dates.empty:
            print("⚠️  WARNING: Copying calendar_dates.txt with original service IDs")
            original_gtfs.calendar_dates.to_csv(os.path.join(working_dir, 'calendar_dates.txt'), index=False)
            print(f"✅ Copied calendar_dates.txt: {len(original_gtfs.calendar_dates)} exceptions")
        else:
            print("✅ Skipped calendar_dates.txt: Using simplified calendar only")

        # ✅ 7. Handle ZIP creation
        if zip_output:
            # Create ZIP file
            zip_path = f"{output_dir}.zip"

            # Ensure output directory exists for ZIP file
            os.makedirs(os.path.dirname(zip_path) if os.path.dirname(zip_path) else '.', exist_ok=True)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all .txt files to ZIP
                for filename in os.listdir(working_dir):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(working_dir, filename)
                        zipf.write(file_path, filename)  # Store with just filename (no path)

            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

            # Show ZIP contents and size
            zip_size = os.path.getsize(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()

            print(f"\n📦 ZIP FILE CREATED: {zip_path}")
            print(f"   File size: {zip_size:,} bytes")
            print(f"   Contains: {', '.join(file_list)}")

            return zip_path
        else:
            print(f"\n🎯 COMPLETE GTFS FEED CREATED: {working_dir}")
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
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

            # Check for None or invalid input types first
        if solution_matrix is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Solution matrix is None")
            return validation_result

        # Check if input is a numpy array or can be converted to one
        if not isinstance(solution_matrix, np.ndarray):
            try:
                # Try to convert to numpy array
                solution_matrix = np.array(solution_matrix)
            except (ValueError, TypeError) as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot convert input to numpy array: {e}")
                return validation_result

        # Check if array is empty
        if solution_matrix.size == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Solution matrix is empty")
            return validation_result

        # Check number of dimensions
        if solution_matrix.ndim != 2:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Solution matrix must be 2-dimensional, got {solution_matrix.ndim} dimensions"
            )
            return validation_result

        # Check shape
        expected_shape = (len(self.route_ids), len(self.interval_labels))
        if solution_matrix.shape != expected_shape:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Solution shape {solution_matrix.shape} doesn't match expected {expected_shape}"
            )
            return validation_result

        # Check value ranges
        min_val = np.min(solution_matrix)
        max_val = np.max(solution_matrix)
        max_allowed = len(self.allowed_headways) - 1

        if min_val < 0:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Solution contains negative values (min: {min_val})")

        if max_val > max_allowed:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Solution contains values > {max_allowed} (max: {max_val})"
            )

        # Check for reasonable service patterns
        total_cells = solution_matrix.size
        no_service_cells = np.sum(solution_matrix == self.no_service_index) if self.no_service_index is not None else 0
        service_cells = total_cells - no_service_cells
        service_percentage = (service_cells / total_cells) * 100

        if service_percentage < 10:
            validation_result['warnings'].append(
                f"Very low service coverage: only {service_percentage:.1f}% of route-intervals have service"
            )
        elif service_percentage > 95:
            validation_result['warnings'].append(
                f"Very high service coverage: {service_percentage:.1f}% of route-intervals have service"
            )

        # Count headway frequency distribution
        headway_counts = {}
        for i, headway in enumerate(self.allowed_headways):
            count = np.sum(solution_matrix == i)
            if count > 0:
                if headway >= 9000:  # No service threshold
                    headway_counts['No Service'] = count
                else:
                    headway_counts[f'{headway:.0f}min'] = count

        # Check for at least some reasonable service frequencies
        frequent_service_count = 0
        for i, headway in enumerate(self.allowed_headways):
            if headway <= 30:  # Consider ≤30min as frequent
                frequent_service_count += np.sum(solution_matrix == i)

        frequent_percentage = (frequent_service_count / service_cells) * 100 if service_cells > 0 else 0

        if frequent_percentage < 20 and service_cells > 0:
            validation_result['warnings'].append(
                f"Low frequent service: only {frequent_percentage:.1f}% of service cells have ≤30min headways"
            )

        # Store statistics
        validation_result['statistics'] = {
            'total_cells': total_cells,
            'service_cells': service_cells,
            'no_service_cells': no_service_cells,
            'service_percentage': service_percentage,
            'frequent_service_percentage': frequent_percentage,
            'headway_distribution': headway_counts,
            'value_range': (min_val, max_val)
        }

        logger.info(f"Solution validation: {'PASSED' if validation_result['valid'] else 'FAILED'} "
                   f"({len(validation_result['errors'])} errors, {len(validation_result['warnings'])} warnings)")

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
                            interval_duration_minutes = (self.interval_hours[interval_idx][1] -
                                                    self.interval_hours[interval_idx][0]) * 60
                            trips_in_interval = max(1, interval_duration_minutes // headway_minutes)
                            total_service_minutes += trips_in_interval * headway_minutes

                active_routes_per_interval[interval_label] = active_count
                total_service_minutes_per_interval[interval_label] = total_service_minutes

            return {
                'validation': validation,
                'headways_dict': headways_dict,
                'active_routes_per_interval': active_routes_per_interval,
                'total_service_minutes_per_interval': total_service_minutes_per_interval,
                'overall_stats': {
                    'total_routes': len(self.route_ids),
                    'total_intervals': len(self.interval_labels),
                    'service_percentage': validation['statistics']['service_percentage']
                }
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
            raise ValueError(f"Unknown interval label: {interval_label}")


    # ========== INTERNAL HELPER METHODS ========== #

    def _generate_interval_trips(self, route_id: str, template_stop_times: pd.DataFrame,
                            start_hour: int, end_hour: int, headway_minutes: float,
                            trip_duration_minutes: float, trip_counter_start: int,
                            service_id: str = 'optimized_service') -> tuple[list, list]:
        """Generate trips and stop times for a single route-interval combination."""

        trips = []
        stop_times = []

        # Get original trip info for shape_id and other metadata
        original_gtfs = self.opt_data['reconstruction']['gtfs_feed']
        original_route_trips = original_gtfs.trips[original_gtfs.trips.route_id == route_id]

        # Use first trip as template for shape_id and other fields
        if not original_route_trips.empty:
            template_trip = original_route_trips.iloc[0]
            shape_id = template_trip.get('shape_id', '')
            direction_id = template_trip.get('direction_id', 0)
            trip_headsign = template_trip.get('trip_headsign', f"Route {route_id}")
        else:
            shape_id = ''
            direction_id = 0
            trip_headsign = f"Route {route_id}"

        # Calculate trip start times
        interval_start_seconds = start_hour * 3600
        interval_end_seconds = end_hour * 3600
        trip_duration_seconds = trip_duration_minutes * 60
        headway_seconds = headway_minutes * 60

        # Latest possible start time (ensure trip completes within interval)
        latest_start = interval_end_seconds - trip_duration_seconds

        # Generate trips every headway minutes
        current_start = interval_start_seconds
        trip_idx = 0

        while current_start <= latest_start:
            # Create trip ID
            trip_id = f"opt_trip_{route_id}_{start_hour:02d}_{trip_idx:03d}"

            # Create trip record
            trip_record = {
                'trip_id': trip_id,
                'route_id': route_id,
                'service_id': service_id,  # Simplified service ID
                'trip_headsign': trip_headsign,
                'direction_id': direction_id,
                'shape_id': shape_id
            }
            trips.append(trip_record)

            # Create stop times for this trip
            trip_stop_times = self._create_trip_stop_times(
                trip_id=trip_id,
                template_stop_times=template_stop_times,
                trip_start_seconds=current_start
            )
            stop_times.extend(trip_stop_times)

            # Move to next trip
            current_start += headway_seconds
            trip_idx += 1

        return trips, stop_times

    def _create_trip_stop_times(self, trip_id: str, template_stop_times: pd.DataFrame,
                            trip_start_seconds: int) -> list:
        """Create stop_times records for a single trip based on template."""

        stop_times = []

        # Get the first stop's departure time from template (as offset from trip start)
        template_first_departure = template_stop_times.iloc[0]['departure_seconds']

        for _, template_stop in template_stop_times.iterrows():
            # Calculate time offset from first stop
            departure_offset = template_stop['departure_seconds'] - template_first_departure
            arrival_offset = template_stop['arrival_seconds'] - template_first_departure

            # Apply offset to new trip start time
            new_departure = trip_start_seconds + departure_offset
            new_arrival = trip_start_seconds + arrival_offset

            # Create stop time record
            stop_time_record = {
                'trip_id': trip_id,
                'stop_id': template_stop['stop_id'],
                'stop_sequence': template_stop['stop_sequence'],
                'arrival_time': self._seconds_to_gtfs_time(new_arrival),
                'departure_time': self._seconds_to_gtfs_time(new_departure),
                'arrival_seconds': new_arrival,
                'departure_seconds': new_departure,
                'pickup_type': template_stop.get('pickup_type', 0),
                'drop_off_type': template_stop.get('drop_off_type', 0)
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
        original_gtfs = self.opt_data['reconstruction']['gtfs_feed']

        # Handle start_date
        if start_date is not None:
            calendar_start_date = start_date
            print(f"📅 Using provided start_date: {start_date}")
        else:
            # Find minimum start_date from original calendar
            if hasattr(original_gtfs, 'calendar') and not original_gtfs.calendar.empty:
                min_start_date = original_gtfs.calendar['start_date'].min()
                calendar_start_date = str(min_start_date)
                print(f"📅 Using min original start_date: {calendar_start_date}")
            else:
                calendar_start_date = '20240101'
                print(f"📅 Using default start_date: {calendar_start_date}")

        # Handle end_date
        if end_date is not None:
            calendar_end_date = end_date
            print(f"📅 Using provided end_date: {end_date}")
        else:
            # Find maximum end_date from original calendar
            if hasattr(original_gtfs, 'calendar') and not original_gtfs.calendar.empty:
                max_end_date = original_gtfs.calendar['end_date'].max()
                calendar_end_date = str(max_end_date)
                print(f"📅 Using max original end_date: {calendar_end_date}")
            else:
                calendar_end_date = '20241231'
                print(f"📅 Using default end_date: {calendar_end_date}")

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

        if 'parent_station' not in stops_df.columns:
            return stops_df

        # Get all valid stop_ids that exist in the file
        valid_stop_ids = set(stops_df['stop_id'].astype(str))

        # Find rows where parent_station points to non-existent stop_id
        invalid_mask = (
            (stops_df['parent_station'].notna()) &  # Has a parent_station value
            (stops_df['parent_station'].astype(str) != '') &  # Not empty string
            (~stops_df['parent_station'].astype(str).isin(valid_stop_ids))  # But it doesn't exist
        )

        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            invalid_refs = stops_df.loc[invalid_mask, 'parent_station'].unique()

            logger.warning(f"Found {invalid_count} stops with invalid parent_station references: {invalid_refs}")
            print(f"⚠️  Found {invalid_count} stops with invalid parent_station references")
            print(f"   Missing parent stations: {list(invalid_refs)}")

            # Clear the invalid parent_station references
            stops_df.loc[invalid_mask, 'parent_station'] = ''
            print(f"✅ Cleared {invalid_count} invalid parent_station references")
        else:
            print("✅ All parent_station references are valid")

        return stops_df


    def _get_trips_in_interval(self, route_stop_times: pd.DataFrame, route_trips: pd.DataFrame,
                               start_hour: int, end_hour: int) -> pd.DataFrame:
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

        # Get first stop time for each trip (trip start time)
        trip_start_times = route_stop_times.groupby('trip_id')['departure_seconds'].min()

        # Filter trips that start within the interval
        trips_in_interval = trip_start_times[
            (trip_start_times >= start_seconds) & (trip_start_times < end_seconds)
        ].index

        # Return stop times for these trips
        interval_stop_times = route_stop_times[
            route_stop_times['trip_id'].isin(trips_in_interval)
        ].copy()

        logger.debug(f"Found {len(trips_in_interval)} trips starting between {start_hour:02d}:00-{end_hour:02d}:00")
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

        for trip_id in interval_stop_times['trip_id'].unique():
            trip_stops = interval_stop_times[interval_stop_times['trip_id'] == trip_id].sort_values('stop_sequence')

            if len(trip_stops) < 2:  # Skip trips with insufficient stops
                continue

            first_departure = trip_stops.iloc[0]['departure_seconds']
            last_arrival = trip_stops.iloc[-1]['arrival_seconds']
            duration_minutes = (last_arrival - first_departure) / 60

            trip_durations.append(duration_minutes)
            trip_info[trip_id] = {
                'duration_minutes': duration_minutes,
                'n_stops': len(trip_stops),
                'stop_times': trip_stops.copy()
            }

        if not trip_durations:
            raise ValueError(f"No valid trips found in interval {interval_label}")

        # Select median duration trip as template (most representative)
        median_duration = np.median(trip_durations)
        best_trip_id = min(trip_info.keys(),
                        key=lambda x: abs(trip_info[x]['duration_minutes'] - median_duration))

        template_data = trip_info[best_trip_id]

        # Get trip metadata from original trips table
        original_trip = self.gtfs_feed.trips[self.gtfs_feed.trips.trip_id == best_trip_id]
        if not original_trip.empty:
            template_data['route_info'] = original_trip.iloc[0].to_dict()
        else:
            template_data['route_info'] = {}

        template_data['trip_id'] = best_trip_id
        template_data['interval_label'] = interval_label

        logger.debug(f"Selected template trip {best_trip_id} for {interval_label}: "
                    f"{template_data['duration_minutes']:.1f}min duration")

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
        trip_stop_times = route_stop_times[
            route_stop_times['trip_id'] == trip_id
        ].sort_values('stop_sequence').copy()

        if trip_stop_times.empty:
            raise ValueError(f"No stop times found for fallback trip {trip_id}")

        # Calculate duration
        first_departure = trip_stop_times.iloc[0]['departure_seconds']
        last_arrival = trip_stop_times.iloc[-1]['arrival_seconds']
        duration_minutes = (last_arrival - first_departure) / 60

        return {
            'trip_id': trip_id,
            'stop_times': trip_stop_times,
            'duration_minutes': duration_minutes,
            'n_stops': len(trip_stop_times),
            'route_info': template_trip.to_dict(),
            'interval_label': 'fallback'
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
        if 'stop_sequence' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('stop_sequence').reset_index(drop=True)

        # Clean arrival_seconds
        if 'arrival_seconds' in cleaned_df.columns:
            # Forward fill then backward fill to use neighboring values
            cleaned_df['arrival_seconds'] = cleaned_df['arrival_seconds'].ffill().bfill()
            # Fall back to zero for any remaining NaN values
            cleaned_df['arrival_seconds'] = cleaned_df['arrival_seconds'].fillna(0)

        # Clean departure_seconds
        if 'departure_seconds' in cleaned_df.columns:
            # Forward fill then backward fill to use neighboring values
            cleaned_df['departure_seconds'] = cleaned_df['departure_seconds'].ffill().bfill()
            # Fall back to zero for any remaining NaN values
            cleaned_df['departure_seconds'] = cleaned_df['departure_seconds'].fillna(0)

        # Ensure departure >= arrival (basic consistency check)
        if 'arrival_seconds' in cleaned_df.columns and 'departure_seconds' in cleaned_df.columns:
            mask = cleaned_df['departure_seconds'] < cleaned_df['arrival_seconds']
            cleaned_df.loc[mask, 'departure_seconds'] = cleaned_df.loc[mask, 'arrival_seconds']

        # Filter out any remaining non-finite values
        if 'arrival_seconds' in cleaned_df.columns:
            cleaned_df = cleaned_df[np.isfinite(cleaned_df['arrival_seconds'])]
        if 'departure_seconds' in cleaned_df.columns:
            cleaned_df = cleaned_df[np.isfinite(cleaned_df['departure_seconds'])]

        return cleaned_df


