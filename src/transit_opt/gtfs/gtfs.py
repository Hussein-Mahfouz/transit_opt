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

    def get_template_trips(self) -> dict[str, dict[str, Any]]:
        """
        Extract template trips for each route from original GTFS.
        
        Returns:
            Dict mapping route_id to template trip information
        """
        template_trips = {}

        for route_id in self.route_ids:
            # Get trips for this route
            route_trips = self.gtfs_feed.trips[self.gtfs_feed.trips.route_id == route_id]

            if route_trips.empty:
                logger.warning(f"No trips found for route {route_id}")
                continue

            # Use first trip as template
            template_trip = route_trips.iloc[0]
            trip_id = template_trip.trip_id

            # Get stop times for this trip
            trip_stop_times = self.gtfs_feed.stop_times[
                self.gtfs_feed.stop_times.trip_id == trip_id
            ].sort_values('stop_sequence').copy()

            if trip_stop_times.empty:
                logger.warning(f"No stop times found for template trip {trip_id}")
                continue

            # Calculate trip duration
            first_departure = trip_stop_times.iloc[0].departure_seconds
            last_arrival = trip_stop_times.iloc[-1].arrival_seconds
            trip_duration_minutes = (last_arrival - first_departure) / 60

            template_trips[route_id] = {
                'trip_id': trip_id,
                'stop_times': trip_stop_times,
                'duration_minutes': trip_duration_minutes,
                'n_stops': len(trip_stop_times),
                'route_info': template_trip.to_dict()
            }

            logger.debug(f"Template for route {route_id}: trip {trip_id}, "
                        f"{len(trip_stop_times)} stops, {trip_duration_minutes:.1f}min duration")

        logger.info(f"Extracted template trips for {len(template_trips)} routes")
        return template_trips

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


    def generate_trips_and_stop_times(self, headways_dict: dict, templates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate new trips and stop_times based on optimized headways.
        
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

            template = templates[route_id]
            template_stop_times = template['stop_times']
            trip_duration_minutes = template['duration_minutes']

            for interval_idx, (interval_label, headway) in enumerate(route_headways.items()):
                if headway is None:  # No service in this interval
                    continue

                # Get interval time bounds
                start_hour, end_hour = intervals['hours'][interval_idx]

                # Generate trips for this interval
                interval_trips, interval_stop_times = self._generate_interval_trips(
                    route_id=route_id,
                    template_stop_times=template_stop_times,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    headway_minutes=headway,
                    trip_duration_minutes=trip_duration_minutes,
                    trip_counter_start=trip_counter
                )

                new_trips.extend(interval_trips)
                new_stop_times.extend(interval_stop_times)
                trip_counter += len(interval_trips)

        # Convert to DataFrames
        trips_df = pd.DataFrame(new_trips) if new_trips else pd.DataFrame()
        stop_times_df = pd.DataFrame(new_stop_times) if new_stop_times else pd.DataFrame()

        logger.info(f"Generated {len(trips_df)} trips with {len(stop_times_df)} stop times")
        return trips_df, stop_times_df

    def _generate_interval_trips(self, route_id: str, template_stop_times: pd.DataFrame,
                            start_hour: int, end_hour: int, headway_minutes: float,
                            trip_duration_minutes: float, trip_counter_start: int) -> tuple[list, list]:
        """Generate trips and stop times for a single route-interval combination."""

        trips = []
        stop_times = []

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
                'service_id': 'optimized_service',  # Simplified service ID
                'trip_headsign': f"Route {route_id}",
                'direction_id': template_stop_times.iloc[0].get('direction_id', 0)
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
