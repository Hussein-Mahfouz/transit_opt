from typing import Any

import gtfs_kit as gk
import numpy as np
import pandas as pd

# =============================================================================
# FIXED GTFS RECONSTRUCTOR
# =============================================================================


class SimplifiedGTFSReconstructor:
    """
    UPDATED: Reconstructor compatible with improved optimization data structure.
    """

    def __init__(
        self, optimization_data: dict[str, Any], optimization_result: dict[str, Any]
    ):
        self.optimization_data = optimization_data
        self.optimization_result = optimization_result

        # UPDATED: Access GTFS feed from new structure
        self.feed = optimization_data["reconstruction"]["gtfs_feed"]

        # Decode solution
        self.optimized_headways = self._decode_headway_solution()

    def _decode_headway_solution(self) -> np.ndarray:
        """Convert optimization solution indices to actual headway values."""
        solution_indices = self.optimization_result["headway_solution"]
        allowed_headways = self.optimization_data["allowed_headways"]
        no_service_index = self.optimization_data["no_service_index"]

        n_routes, n_intervals = solution_indices.shape
        headways = np.full((n_routes, n_intervals), np.nan)

        for i in range(n_routes):
            for j in range(n_intervals):
                choice_idx = solution_indices[i, j]
                headway_value = allowed_headways[choice_idx]

                if choice_idx == no_service_index or headway_value >= 9000:
                    headways[i, j] = np.nan  # No service
                else:
                    headways[i, j] = headway_value

        return headways

    def reconstruct_gtfs(self, use_frequencies: bool = False) -> Any:
        """Reconstruct GTFS with proper stop_times.txt."""
        print("=== RECONSTRUCTING GTFS WITH OPTIMIZED HEADWAYS ===")

        # Start with copy of original feed
        new_feed = self.feed.copy()

        # Generate new stop_times and trips
        new_stop_times, new_trips = self._generate_stop_times_and_trips()

        # Update feed
        new_feed.stop_times = new_stop_times
        new_feed.trips = new_trips

        # Handle frequencies (optional)
        if use_frequencies and len(new_trips) > 0:
            frequencies_df = self._create_frequencies_table(new_trips)
            if len(frequencies_df) > 0:
                new_feed.frequencies = frequencies_df
                print(f"   📊 Added {len(frequencies_df):,} frequency entries")
            else:
                new_feed.frequencies = None
                print("   ⚠️  No frequencies generated - skipping frequencies.txt")
        else:
            new_feed.frequencies = None
            print("   📊 Frequencies.txt disabled - using stop_times.txt only")

        print("✅ Reconstructed GTFS with stop_times.txt:")
        print(f"   📊 {len(new_trips):,} trips")
        print(f"   📊 {len(new_stop_times):,} stop times")

        return new_feed

    def _generate_stop_times_and_trips(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate both stop_times and trips tables with proper relationships."""
        new_stop_times_list = []
        new_trips_list = []
        trip_id_counter = 1

        # UPDATED: Use new data structure
        route_ids = self.optimization_data["routes"]["ids"]
        n_intervals = self.optimization_data["n_intervals"]
        interval_hours = self.optimization_data["intervals"]["duration_minutes"] // 60

        print(f"   🔄 Generating trips and stop_times for {len(route_ids)} routes")

        for route_idx, service_id in enumerate(route_ids):
            # Get original trips for this service
            original_trips = self.feed.trips[
                self.feed.trips["service_id"] == service_id
            ]

            if len(original_trips) == 0:
                continue

            # Use first trip as template
            template_trip = original_trips.iloc[0]
            template_trip_id = template_trip["trip_id"]

            # Get template stop_times
            template_stops = (
                self.feed.stop_times[
                    self.feed.stop_times["trip_id"] == template_trip_id
                ]
                .sort_values("stop_sequence")
                .copy()
            )

            if len(template_stops) == 0:
                continue

            # Convert template times to seconds for calculations
            template_stops["departure_seconds"] = template_stops[
                "departure_time"
            ].apply(self._safe_timestr_to_seconds)
            template_stops["arrival_seconds"] = template_stops["arrival_time"].apply(
                self._safe_timestr_to_seconds
            )

            # Generate trips for each interval with service
            route_trips_generated = 0
            for interval_idx in range(n_intervals):
                headway = self.optimized_headways[route_idx, interval_idx]

                # Skip intervals with no service
                if np.isnan(headway):
                    continue

                # UPDATED: Use interval hours from data structure
                start_hour, end_hour = self.optimization_data["intervals"]["hours"][
                    interval_idx
                ]
                interval_duration_minutes = end_hour * 60 - start_hour * 60

                # Calculate number of trips needed in this interval
                n_trips = max(1, int(interval_duration_minutes / headway))

                # Generate trips spaced by optimized headway
                for trip_num in range(n_trips):
                    # Calculate start time for this trip
                    trip_start_minutes = start_hour * 60 + (trip_num * headway)

                    # Don't exceed interval boundary
                    if trip_start_minutes >= end_hour * 60:
                        break

                    # Create new trip with unique ID
                    new_trip_id = f"opt_{service_id}_{interval_idx}_{trip_num}"
                    new_trip = template_trip.copy()
                    new_trip["trip_id"] = new_trip_id

                    # Clear any block_id to avoid conflicts
                    if "block_id" in new_trip:
                        new_trip["block_id"] = f"block_{trip_id_counter}"

                    new_trips_list.append(new_trip)

                    # Generate stop_times for this trip
                    trip_stop_times = self._create_trip_stop_times(
                        template_stops, new_trip_id, trip_start_minutes
                    )

                    if trip_stop_times is not None:
                        new_stop_times_list.append(trip_stop_times)
                        route_trips_generated += 1

                    trip_id_counter += 1

            if route_trips_generated > 0 and route_idx < 5:  # Log first few routes
                print(
                    f"   📍 Route {route_idx} ({service_id}): Generated {route_trips_generated} trips"
                )

        # Combine all data
        if new_trips_list and new_stop_times_list:
            new_trips = pd.DataFrame(new_trips_list).reset_index(drop=True)
            new_stop_times = pd.concat(new_stop_times_list, ignore_index=True)

            print(
                f"   ✅ Generated {len(new_trips):,} trips with {len(new_stop_times):,} stop times"
            )
        else:
            # Create empty but valid DataFrames
            new_trips = self.feed.trips.iloc[0:0].copy()
            new_stop_times = self.feed.stop_times.iloc[0:0].copy()
            print("   ⚠️  No trips generated - all routes mapped to no service")

        return new_stop_times, new_trips

    def _create_trip_stop_times(
        self, template_stops: pd.DataFrame, new_trip_id: str, trip_start_minutes: float
    ) -> pd.DataFrame | None:
        """Create stop_times for a single trip based on template."""
        try:
            # Calculate time offset
            template_start_seconds = template_stops.iloc[0]["departure_seconds"]
            if pd.isna(template_start_seconds):
                return None

            trip_start_seconds = trip_start_minutes * 60
            time_offset = trip_start_seconds - template_start_seconds

            # Create new stop_times
            new_stop_times = template_stops.copy()
            new_stop_times["trip_id"] = new_trip_id

            # Adjust all times
            new_stop_times["departure_seconds"] = (
                template_stops["departure_seconds"] + time_offset
            )
            new_stop_times["arrival_seconds"] = (
                template_stops["arrival_seconds"] + time_offset
            )

            # Convert back to GTFS time strings
            new_stop_times["departure_time"] = new_stop_times[
                "departure_seconds"
            ].apply(self._seconds_to_timestr)
            new_stop_times["arrival_time"] = new_stop_times["arrival_seconds"].apply(
                self._seconds_to_timestr
            )

            # Remove helper columns
            new_stop_times = new_stop_times.drop(
                ["departure_seconds", "arrival_seconds"], axis=1, errors="ignore"
            )

            return new_stop_times

        except Exception as e:
            print(f"   ⚠️  Failed to create stop_times for trip {new_trip_id}: {e}")
            return None

    def _create_frequencies_table(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Create frequencies.txt that uses ACTUAL trip IDs from the new trips."""
        frequencies_list = []
        n_intervals = self.optimization_data["n_intervals"]
        route_ids = self.optimization_data["routes"]["ids"]

        for route_idx, service_id in enumerate(route_ids):
            # Get trips that were actually generated for this service
            service_trips = trips_df[trips_df["service_id"] == service_id]

            if len(service_trips) == 0:
                continue

            # Create frequency entries for each interval that has service
            for interval_idx in range(n_intervals):
                headway = self.optimized_headways[route_idx, interval_idx]

                if np.isnan(headway):
                    continue

                # Find a trip that was actually generated for this interval
                interval_trips = service_trips[
                    service_trips["trip_id"].str.contains(f"_{interval_idx}_", na=False)
                ]

                if len(interval_trips) == 0:
                    continue

                # Use the first trip from this interval as the frequency template
                template_trip_id = interval_trips.iloc[0]["trip_id"]

                # UPDATED: Get interval hours from data structure
                start_hour, end_hour = self.optimization_data["intervals"]["hours"][
                    interval_idx
                ]

                frequency_entry = {
                    "trip_id": template_trip_id,
                    "start_time": f"{start_hour:02d}:00:00",
                    "end_time": f"{end_hour:02d}:00:00",
                    "headway_secs": int(headway * 60),
                    "exact_times": 0,
                }

                frequencies_list.append(frequency_entry)

        return pd.DataFrame(frequencies_list)

    # Helper methods remain the same
    def _safe_timestr_to_seconds(self, time_value: Any) -> float:
        """Safely convert GTFS time strings to seconds."""
        try:
            if pd.isna(time_value):
                return np.nan
            if isinstance(time_value, str):
                return gk.helpers.timestr_to_seconds(time_value)
            else:
                return float(time_value)
        except Exception:
            return np.nan

    def _seconds_to_timestr(self, seconds: float) -> str:
        """Convert seconds to GTFS time string format."""
        if pd.isna(seconds):
            return "00:00:00"

        # Handle times > 24 hours (GTFS allows this)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
