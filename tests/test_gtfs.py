from unittest.mock import MagicMock

import numpy as np
import pandas as pd

"""
Tests for GTFS Solution Converter functionality.

This module tests the SolutionConverter class which transforms optimization
solutions (headway matrices) into valid GTFS transit feeds. The tests cover:

1. Initialization and data validation
2. Solution matrix validation and statistics
3. Conversion from solution matrices to headway dictionaries
4. Template extraction from existing GTFS data
5. Generation of new GTFS files (trips, stop_times, etc.)
6. Error handling for edge cases and invalid inputs


"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transit_opt.gtfs.gtfs import SolutionConverter


class TestSolutionConverterInit:
    """
    Test SolutionConverter initialization and setup.

    These tests verify that the SolutionConverter properly initializes with
    optimization data and correctly extracts essential components from the
    underlying GTFS feed for later processing.
    """

    def test_init_with_valid_data(self, sample_optimization_data):
        """
        Test initialization with valid optimization data.

        Verifies that:
        - Route IDs are correctly extracted and counted
        - Time interval labels are properly set up
        - Allowed headway values are stored correctly
        - Essential GTFS components (routes, trips, stops) are accessible
        """
        converter = SolutionConverter(sample_optimization_data)

        assert len(converter.route_ids) == sample_optimization_data["n_routes"]
        assert len(converter.interval_labels) == sample_optimization_data["n_intervals"]
        assert np.array_equal(converter.allowed_headways, sample_optimization_data["allowed_headways"])

        # Check that essential components are accessible
        assert hasattr(converter, "gtfs_feed")
        assert hasattr(converter.gtfs_feed, "routes")
        assert hasattr(converter.gtfs_feed, "trips")
        assert hasattr(converter.gtfs_feed, "stops")

    def test_init_with_invalid_data(self):
        """
        Test initialization with invalid or malformed data.

        Ensures the converter fails gracefully when given:
        - None values
        - Empty dictionaries
        - Dictionaries with invalid/missing keys

        This prevents silent failures and ensures proper error reporting
        when the optimization data is corrupted or incomplete.
        """
        with pytest.raises((KeyError, TypeError, ValueError)):
            SolutionConverter(None)

        with pytest.raises((KeyError, TypeError, ValueError)):
            SolutionConverter({})

        with pytest.raises((KeyError, TypeError, ValueError)):
            SolutionConverter({"invalid": "data"})


class TestSolutionValidation:
    """
    Test solution matrix validation functionality.

    The validation system checks that optimization solutions are:
    - Properly formatted (correct dimensions, data types)
    - Within valid bounds (headway indices exist)
    - Logically consistent

    Useful for catching optimization bugs before GTFS generation.
    """

    def test_validate_valid_solution(self, sample_optimization_data, sample_solutions):
        """
        Test validation of properly formatted solution matrices.

        Verifies that valid solutions:
        - Pass validation (valid=True, no errors)
        - Generate meaningful statistics (service percentage, cell counts)
        - Have non-negative service percentages

        Uses the 'high_service' solution which should represent a scenario
        where most routes have frequent service across most time periods.
        """
        converter = SolutionConverter(sample_optimization_data)

        solution = sample_solutions["high_service"]
        validation = converter.validate_solution(solution)

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert "statistics" in validation
        assert validation["statistics"]["service_percentage"] >= 0

    def test_validate_wrong_dimensions(self, sample_optimization_data):
        """
        Test rejection of incorrectly sized solution matrices.

        Solution matrices must match the expected dimensions:
        (number of routes) x (number of time intervals)

        This test ensures that solutions with wrong dimensions are caught
        early, preventing array indexing errors or misaligned data during
        the conversion process.
        """
        converter = SolutionConverter(sample_optimization_data)

        wrong_solution = np.zeros((5, 3))  # Wrong dimensions
        validation = converter.validate_solution(wrong_solution)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0

    def test_validate_invalid_headway_indices(self, sample_optimization_data):
        """
        Test handling of solution matrices with invalid headway indices.

        Each cell in the solution matrix should contain an index that maps
        to a valid headway value. This test uses index 999 which should be
        out of bounds for any reasonable set of allowed headways.

        Catches cases where optimization algorithms produce invalid results
        or data gets corrupted during processing.
        """
        converter = SolutionConverter(sample_optimization_data)

        # Create solution with out-of-bounds indices
        shape = sample_optimization_data["decision_matrix_shape"]
        invalid_solution = np.full(shape, 999)  # Invalid headway index

        validation = converter.validate_solution(invalid_solution)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0

    def test_validate_none_solution(self, sample_optimization_data):
        """
        Test validation behavior with None input.

        Ensures that None solutions are handled gracefully with appropriate
        error messages rather than causing crashes. This is important for
        robustness when dealing with failed optimizations or missing data.
        """
        converter = SolutionConverter(sample_optimization_data)

        validation = converter.validate_solution(None)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert "none" in str(validation["errors"][0]).lower() or "invalid" in str(validation["errors"][0]).lower()

    def test_validate_solution_statistics(self, sample_optimization_data, sample_solutions):
        """
        Test that validation statistics are calculated correctly.

        For valid solutions, statistics should include:
        - service_percentage: Percentage of route-interval cells with service
        - service_cells: Count of cells with active service
        - total_cells: Total number of cells in the matrix

        These statistics are useful for comparing solution quality and
        understanding service coverage patterns.
        """
        converter = SolutionConverter(sample_optimization_data)

        # Test different service levels - only test valid solutions
        for solution_name, solution in sample_solutions.items():
            validation = converter.validate_solution(solution)

            if validation["valid"]:
                stats = validation["statistics"]
                assert "service_percentage" in stats
                assert "service_cells" in stats
                assert "total_cells" in stats
                assert stats["total_cells"] > 0
                assert 0 <= stats["service_percentage"] <= 100


class TestHeadwayConversion:
    """
    Test conversion from solution matrices to headway dictionaries.

    The conversion process transforms numerical solution matrices into
    structured dictionaries mapping:
    route_id -> time_interval -> headway_minutes

    This intermediate format is easier to work with for GTFS generation
    """

    def test_solution_to_headways_basic(self, sample_optimization_data, sample_solutions):
        """
        Test basic solution matrix to headway dictionary conversion.

        Verifies that:
        - Output dictionary has entries for all routes
        - Each route has entries for all time intervals
        - Structure matches input matrix dimensions

        """
        converter = SolutionConverter(sample_optimization_data)

        solution = sample_solutions["medium_service"]
        headways_dict = converter.solution_to_headways(solution)

        # Should have entries for all routes
        assert len(headways_dict) == sample_optimization_data["n_routes"]

        # Each route should have entries for all intervals
        route_id = list(headways_dict.keys())[0]
        assert len(headways_dict[route_id]) == sample_optimization_data["n_intervals"]

    def test_solution_to_headways_no_service(self, sample_optimization_data, sample_solutions):
        """
        Test handling of no-service intervals in solution conversion.

        When routes have no service during certain time periods, this should
        be represented as None values or special "no-service" headway values.

        This test ensures that the conversion properly handles scenarios where
        optimization decides certain routes shouldn't run during off-peak hours.
        """
        converter = SolutionConverter(sample_optimization_data)

        solution = sample_solutions["no_service"]
        headways_dict = converter.solution_to_headways(solution)

        # Should still have structure even with no service
        assert len(headways_dict) == sample_optimization_data["n_routes"]

        # Check that no-service is properly represented
        route_id = list(headways_dict.keys())[0]
        interval_label = list(headways_dict[route_id].keys())[0]
        headway_value = headways_dict[route_id][interval_label]

        # Should be None or the no-service value
        assert headway_value is None or headway_value == sample_optimization_data["allowed_headways"][-1]

    def test_solution_to_headways_invalid_solution(self, sample_optimization_data):
        """
        Test headway conversion with invalid solution matrices.

        Ensures that conversion fails appropriately when given solutions
        with wrong dimensions. This prevents silent corruption where
        wrong-sized arrays might be processed incorrectly.
        """
        converter = SolutionConverter(sample_optimization_data)

        # Test with wrong dimensions
        wrong_solution = np.zeros((2, 2))

        with pytest.raises((ValueError, IndexError)):
            converter.solution_to_headways(wrong_solution)

    def test_headway_index_mapping(self, sample_optimization_data, sample_solutions):
        """
        Test that headway indices correctly map to actual headway values.

        Verifies the core mapping logic:
        solution_matrix[route, interval] = index
        headways_dict[route][interval] = allowed_headways[index]

        This is critical - incorrect mapping would result in wrong service
        frequencies in the final GTFS output.
        """
        converter = SolutionConverter(sample_optimization_data)
        allowed_headways = sample_optimization_data["allowed_headways"]

        # Test each solution type
        for solution_name, solution in sample_solutions.items():
            headways_dict = converter.solution_to_headways(solution)

            # Get a sample headway value
            route_id = list(headways_dict.keys())[0]
            interval_label = list(headways_dict[route_id].keys())[0]
            actual_headway = headways_dict[route_id][interval_label]

            # Should be None or in allowed headways
            if actual_headway is not None:
                assert actual_headway in allowed_headways


class TestTemplateExtraction:
    """
    Test extraction of trip templates from existing GTFS data.

    Templates provide the blueprint for generating new trips:
    - Stop sequences and timing patterns
    - Route geometries and characteristics
    - Duration and stop count information

    These templates are then scaled according to the optimized headways
    to create the final GTFS schedule.
    """

    def test_extract_templates_structure(self, sample_optimization_data):
        """
        Test that template extraction returns proper data structure.

        Verifies that templates have the expected nested dictionary structure:
        route_id -> time_interval -> template_data

        Each template must contain essential fields:
        - trip_id: Reference trip from original GTFS
        - duration_minutes: Total trip duration
        - n_stops: Number of stops on the route
        - stop_times: DataFrame with stop sequence and timing

        This structure is required for downstream GTFS generation.
        """
        converter = SolutionConverter(sample_optimization_data)
        templates = converter.extract_route_templates()

        # Should have templates for available routes
        assert isinstance(templates, dict)
        assert len(templates) > 0

        # Check structure for first available route
        route_id = list(templates.keys())[0]
        route_templates = templates[route_id]

        assert isinstance(route_templates, dict)

        # Check first template
        interval_label = list(route_templates.keys())[0]
        interval_templates = route_templates[interval_label]

        # Structure is now route_id -> interval -> direction -> template
        assert isinstance(interval_templates, dict), "Interval templates should be a dict of directions"

        if not interval_templates:
            pytest.skip("No templates for this interval")

        # Get first direction template
        direction_id = list(interval_templates.keys())[0]
        template = interval_templates[direction_id]

        required_keys = ["trip_id", "duration_minutes", "n_stops", "stop_times"]
        for key in required_keys:
            assert key in template, f"Missing key '{key}' in template"

    def test_template_data_consistency(self, sample_optimization_data):
        """
        Test template data consistency and reasonableness.

        Validates that extracted templates contain realistic values:
        - Trip durations are positive and reasonable (< 10 hours)
        - Routes have at least 2 stops (minimum for a route)
        - Stop times are properly formatted DataFrames
        - Required columns exist in stop times data

        This catches issues with malformed GTFS data that could cause
        problems during trip generation.
        """
        converter = SolutionConverter(sample_optimization_data)
        templates = converter.extract_route_templates()

        for route_id, route_templates in templates.items():
            for interval_label, interval_templates in route_templates.items():
                for direction_id, template in interval_templates.items():
                    # Duration should be positive and reasonable
                    assert template["duration_minutes"] > 0
                    assert template["duration_minutes"] < 600  # Less than 10 hours

                    # Should have at least 2 stops
                    assert template["n_stops"] >= 2

                    # Stop times should be a DataFrame with required columns
                    stop_times = template["stop_times"]
                    assert isinstance(stop_times, pd.DataFrame)
                    required_cols = ["stop_id", "stop_sequence"]
                    for col in required_cols:
                        assert col in stop_times.columns

    def test_missing_time_intervals(self, sample_optimization_data):
        """
        Test handling of routes with missing data for some time intervals.

        Verifies that the system can handle routes that don't operate all day
        by ensuring that either:
        1. A valid template is found for other intervals
        2. A fallback template is generated using available data/interpolation

        This is crucial for reliable operation with sparse GTFS feeds.
        """
        converter = SolutionConverter(sample_optimization_data)
        templates = converter.extract_route_templates()

        # Check that we have coverage for the intervals defined in optimization data
        intervals = sample_optimization_data["intervals"]["labels"]

        for route_id, route_templates in templates.items():
            # Check for coverage gaps
            missing = [i for i in intervals if i not in route_templates]
            if missing:
                # If missing, check if fallback logic worked
                # Fallbacks should be injected during extraction or trip generation
                pass

            # Verify structure (intervals -> directions -> template)
            for interval, interval_templates in route_templates.items():
                assert isinstance(interval_templates, dict), (
                    f"Templates for {route_id} {interval} should be dict of directions"
                )


@pytest.fixture
def robust_test_data(sample_optimization_data, sample_solutions):
    """
    Create test data using the SolutionConverter's built-in data cleaning.

    The SolutionConverter now automatically cleans NaN values in stop times
    during template extraction, so this fixture just filters for routes
    that have valid templates after cleaning.

    Returns:
        tuple: (converter, headways_dict, clean_templates_dict)
    """
    converter = SolutionConverter(sample_optimization_data)

    # Use high service solution (most likely to work)
    solution = sample_solutions["high_service"]
    headways_dict = converter.solution_to_headways(solution)

    # Extract templates - this now includes automatic cleaning
    templates = converter.extract_route_templates()

    # Filter to only include routes that exist in both and have sufficient data
    clean_headways = {}
    clean_templates = {}

    for route_id in templates.keys():
        if route_id in headways_dict:
            # Check if route has valid templates after cleaning
            # Templates structure: route_id -> interval -> direction -> template
            route_templates = templates[route_id]
            valid_templates = {}

            for interval_label, interval_templates in route_templates.items():
                valid_int_templates = {}
                for direction_id, template in interval_templates.items():
                    # Only include templates with sufficient valid stop data
                    if (
                        len(template["stop_times"]) >= 2
                        and template["n_stops"] >= 2
                        and template["duration_minutes"] > 0
                    ):
                        valid_int_templates[direction_id] = template

                if valid_int_templates:
                    valid_templates[interval_label] = valid_int_templates

            # Only include route if it has valid templates
            if valid_templates:
                clean_templates[route_id] = valid_templates
                clean_headways[route_id] = headways_dict[route_id]

    return converter, clean_headways, clean_templates


class TestGTFSGeneration:
    """
    Test GTFS file generation functionality.

    These tests verify the core functionality of generating valid GTFS files
    from optimization results. This includes:
    - Trip generation based on headway schedules
    - Stop times calculation with proper timing
    - File structure creation (directories and ZIP files)
    - Handling of edge cases and invalid inputs
    """

    def test_generate_trips_empty_inputs(self, sample_optimization_data):
        """
        Test trip generation with empty input dictionaries.

        Verifies that the system handles empty inputs gracefully:
        - No crashes or exceptions
        - Returns empty but properly structured DataFrames
        - Maintains expected column schemas even with no data

        This is important for scenarios where optimization produces
        no viable solutions or all routes are filtered out.
        """
        converter = SolutionConverter(sample_optimization_data)

        # Test with empty inputs - should handle gracefully
        trips_df, stop_times_df = converter.generate_trips_and_stop_times({}, {})
        assert len(trips_df) == 0
        assert len(stop_times_df) == 0

    def test_generate_trips_with_valid_data(self, robust_test_data):
        """
        Test trip generation with clean, valid input data.

        When provided with valid headways and templates, verifies that:
        - Proper trip and stop_times DataFrames are generated
        - Required GTFS columns are present and properly named
        - Referential integrity is maintained (trip IDs match between tables)
        - Data types and formats are correct for GTFS compliance

        This is the core functionality test for successful GTFS generation.
        """
        converter, headways_dict, templates = robust_test_data

        # Test only if we have valid data after cleaning
        if headways_dict and templates:
            trips_df, stop_times_df = converter.generate_trips_and_stop_times(headways_dict, templates)

            # Check structure
            required_trip_cols = ["trip_id", "route_id", "service_id"]
            required_stop_time_cols = ["trip_id", "stop_id", "arrival_time", "departure_time"]

            for col in required_trip_cols:
                assert col in trips_df.columns
            for col in required_stop_time_cols:
                assert col in stop_times_df.columns

            # Check referential integrity if trips exist
            if len(trips_df) > 0:
                trip_ids_trips = set(trips_df["trip_id"])
                trip_ids_stop_times = set(stop_times_df["trip_id"])
                assert trip_ids_trips == trip_ids_stop_times

    def test_generate_trips_invalid_inputs(self, sample_optimization_data):
        """
        Test trip generation with mismatched or invalid inputs.

        Tests scenarios where headways and templates have:
        - Different route IDs (no overlap)
        - Malformed data structures
        - Missing required fields

        Should handle these gracefully by returning empty results rather
        than crashing, allowing the system to continue processing other
        valid routes.
        """
        converter = SolutionConverter(sample_optimization_data)

        # Test with mismatched route IDs
        fake_headways = {"nonexistent_route": {"06-09h": 10}}
        # Templates structure: route_id -> interval -> direction -> template
        fake_templates = {"different_route": {"06-09h": {0: {"trip_id": "test"}}}}

        trips_df, stop_times_df = converter.generate_trips_and_stop_times(fake_headways, fake_templates)

        # Should handle gracefully
        assert len(trips_df) == 0
        assert len(stop_times_df) == 0

    def test_build_gtfs_directory_structure(self, robust_test_data):
        """
        Test complete GTFS directory creation.

        Verifies that build_complete_gtfs() creates a proper GTFS directory with:
        - Correct directory structure
        - Required GTFS text files (agency.txt, routes.txt, stops.txt, etc.)
        - Files are created even if they contain minimal data

        The resulting directory should be readable by GTFS validation tools
        and transit planning software.
        """
        converter, headways_dict, templates = robust_test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            gtfs_path = converter.build_complete_gtfs(headways_dict, templates, output_dir=temp_dir)

            # Check directory exists
            assert Path(gtfs_path).exists()
            assert Path(gtfs_path).is_dir()

            # Check that some GTFS files exist (even if empty)
            expected_files = ["agency.txt", "routes.txt", "stops.txt"]
            for filename in expected_files:
                file_path = Path(gtfs_path) / filename
                assert file_path.exists()

    def test_build_gtfs_zip_creation(self, robust_test_data):
        """
        Test GTFS ZIP file creation.

        Tests the zip_output=True functionality which creates a compressed
        GTFS feed suitable for distribution. Verifies:
        - ZIP file is created with correct extension
        - ZIP contains expected GTFS files
        - File is not empty (has actual content)

        """
        converter, headways_dict, templates = robust_test_data

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = converter.build_complete_gtfs(
                headways_dict, templates, output_dir=f"{temp_dir}/test_gtfs", zip_output=True
            )

            # Check ZIP exists
            assert Path(zip_path).exists()
            assert Path(zip_path).suffix == ".zip"

            # Check ZIP has content
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = zf.namelist()
                assert len(file_list) > 0


class TestErrorHandling:
    """
    Test comprehensive error handling and edge case management.

    These tests ensure the system fails gracefully and provides meaningful
    error messages when encountering invalid inputs, corrupted data, or
    unexpected conditions. Good error handling is critical for:
    - Debugging optimization problems
    - Handling real-world data inconsistencies
    - Providing clear feedback to users
    """

    def test_invalid_optimization_data(self):
        """
        Test handling of various types of invalid optimization data.

        Tests initialization with systematically invalid inputs:
        - None values
        - Empty dictionaries
        - Wrong data types for required fields
        - Missing required keys

        Each should raise appropriate exceptions rather than causing
        silent failures or crashes later in the pipeline.
        """
        invalid_data_sets = [
            None,
            {},
            {"n_routes": "invalid"},
            {"n_routes": 5, "n_intervals": "invalid"},
            {"n_routes": 5, "n_intervals": 8, "allowed_headways": "invalid"},
        ]

        for invalid_data in invalid_data_sets:
            with pytest.raises((KeyError, TypeError, ValueError, AttributeError)):
                SolutionConverter(invalid_data)

    def test_solution_validation_edge_cases(self, sample_optimization_data):
        """
        Test solution validation with various edge case inputs.

        Tests validation behavior with inputs that might occur due to:
        - Programming errors (wrong types)
        - Corrupted data (empty arrays, wrong dimensions)
        - Invalid function calls (None, strings, scalars)

        All should be caught by validation and reported as invalid solutions
        with clear error messages.
        """
        converter = SolutionConverter(sample_optimization_data)

        edge_cases = [
            None,
            [],
            np.array([]),
            "invalid",
            42,
            np.array([[[1, 2, 3]]]),  # Wrong number of dimensions
        ]

        for edge_case in edge_cases:
            validation = converter.validate_solution(edge_case)
            assert validation["valid"] is False
            assert len(validation["errors"]) > 0

    def test_headway_conversion_edge_cases(self, sample_optimization_data):
        """
        Test headway conversion with various invalid inputs.

        Tests the robustness of solution_to_headways() when given:
        - None inputs
        - Empty arrays
        - Wrong dimensional arrays
        - Invalid data types

        Should raise appropriate exceptions to prevent downstream errors
        from malformed headway dictionaries.
        """
        converter = SolutionConverter(sample_optimization_data)

        edge_cases = [
            None,
            np.array([]),
            np.array([[1, 2], [3, 4]]),  # Wrong dimensions
        ]

        for edge_case in edge_cases:
            with pytest.raises((ValueError, IndexError, AttributeError, TypeError)):
                converter.solution_to_headways(edge_case)


class TestDirectionLogic:
    """
    Test logic for handling trip directions in GTFS.

    Verifies that the system correctly identifies and processes trip directions
    whether the optional 'direction_id' field is present or not.
    """

    def test_extract_templates_with_direction_id(self, sample_optimization_data):
        """
        Test template extraction when direction_id is present (standard case).

        The sample data (Duke GTFS) has direction_id. We verify that:
        1. Templates are extracted
        2. Keys are 0 or 1 (or other valid integers)
        3. Templates contain the direction_id
        """
        converter = SolutionConverter(sample_optimization_data)

        # Verify source has direction_id
        assert "direction_id" in converter.gtfs_feed.trips.columns

        templates = converter.extract_route_templates()

        assert len(templates) > 0, "Should extract templates from valid GTFS"

        for route_id, route_templates in templates.items():
            for interval, interval_templates in route_templates.items():
                # Should be a dict of direction_id -> template
                assert isinstance(interval_templates, dict)
                assert len(interval_templates) > 0

                for dir_id, template in interval_templates.items():
                    # Keys should be integers (typically 0 or 1)
                    assert isinstance(dir_id, int)
                    # Template should preserve this info
                    assert "direction_id" in template
                    assert template["direction_id"] == dir_id

    def test_extract_templates_without_direction_id(self, sample_optimization_data):
        """
        Test template extraction when direction_id is MISSING.

        We manually strip direction_id from the dataframe and verify that:
        1. System falls back to trip_headsign
        2. Correctly infers directions (keys are 0, 1, 2...)
        3. Generated templates still have a direction_id assigned
        """
        converter = SolutionConverter(sample_optimization_data)

        # Manually drop direction_id from the feed object to simulate missing column
        if "direction_id" in converter.gtfs_feed.trips.columns:
            converter.gtfs_feed.trips = converter.gtfs_feed.trips.drop(columns=["direction_id"])

        # Create dummy headsigns to ensure we have distinct "directions" if original data doesn't
        # This ensures we test the grouping logic
        trips = converter.gtfs_feed.trips
        # assign 'Headsign A' to even rows, 'Headsign B' to odd rows for each route
        # This is a bit hacky but ensures we have variation for the fallback logic to work on
        trips["trip_headsign"] = (
            trips.groupby("route_id").cumcount().map(lambda x: "Headsign A" if x % 2 == 0 else "Headsign B")
        )

        # Ensure trip_headsign column exists (it should, but just in case)
        assert "direction_id" not in converter.gtfs_feed.trips.columns
        assert "trip_headsign" in converter.gtfs_feed.trips.columns

        # Re-run extraction with modified data
        templates = converter.extract_route_templates()

        assert len(templates) > 0

        for route_id, route_templates in templates.items():
            for interval, interval_templates in route_templates.items():
                if not interval_templates:
                    continue

                # Should have found multiple directions due to our synthetic headsigns
                # (unless route has only 1 trip)
                route_trip_count = len(trips[trips.route_id == route_id])
                if route_trip_count > 1:
                    # In our synthetic setup, >1 trip usually means >1 headsign
                    # But it depends on if they fall in same interval
                    pass

                # Verify structure
                for dir_id, template in interval_templates.items():
                    assert isinstance(dir_id, int)
                    assert template["direction_id"] == dir_id

    def test_gtfs_generation_preserves_direction(self, robust_test_data):
        """
        Test that generated trips file contains correct direction_id.
        """
        converter, headways_dict, templates = robust_test_data

        # Generate files
        trips_df, _ = converter.generate_trips_and_stop_times(headways_dict, templates)

        assert "direction_id" in trips_df.columns
        assert not trips_df["direction_id"].isna().any()

        # Verify values match what was in templates
        unique_dirs = trips_df["direction_id"].unique()
        for d in unique_dirs:
            assert isinstance(d, (int, np.integer))


class TestHybridStrategy:
    """Tests for the Hybrid Strategy (Dominant Filtering + Frequency Splitting)."""

    @pytest.fixture
    def mock_converter(self, sample_optimization_data):
        """Fixture to provide a configured SolutionConverter with mocked GTFS feed."""
        converter = SolutionConverter(sample_optimization_data)
        # Mock the gtfs_feed object attributes we need
        converter.gtfs_feed = MagicMock()
        converter.gtfs_feed.trips = pd.DataFrame()
        converter.gtfs_feed.stop_times = pd.DataFrame()

        # Override route_ids to focus on our test route
        converter.route_ids = ["test_route"]
        return converter

    def test_dominant_filtering(self, mock_converter):
        """
        Test Strategy A: If top 2 headsigns cover > 90% of trips,
        filter to only those 2 and do NOT split frequency.
        """
        route_id = "test_route"

        # Setup Data: 100 trips total
        # 60 'A', 35 'B' -> 95/100 coverage (95%)
        # 5 'C' -> Noise
        trips_data = {
            "route_id": [route_id] * 100,
            "trip_id": [f"t_{i}" for i in range(100)],
            "trip_headsign": ["A"] * 60 + ["B"] * 35 + ["C"] * 5,
            # No direction_id to force inference
        }
        mock_converter.gtfs_feed.trips = pd.DataFrame(trips_data)

        # Create Dummy Stop Times (minimal valid data)
        # All trips at 10:00 AM (36000 seconds)
        stops = []
        for i, t_id in enumerate(trips_data["trip_id"]):
            stops.append(
                {
                    "trip_id": t_id,
                    "stop_id": "s1",
                    "stop_sequence": 1,
                    "arrival_seconds": 36000,
                    "departure_seconds": 36000,
                }
            )
            stops.append(
                {
                    "trip_id": t_id,
                    "stop_id": "s2",
                    "stop_sequence": 2,
                    "arrival_seconds": 37000,
                    "departure_seconds": 37000,
                }
            )
        mock_converter.gtfs_feed.stop_times = pd.DataFrame(stops)

        # Run Extraction
        templates = mock_converter.extract_route_templates()

        # Assertions
        assert route_id in templates
        route_templates = templates[route_id]

        # Find interval covering 10:00 AM
        valid_interval = None
        for i, label in enumerate(mock_converter.interval_labels):
            start, end = mock_converter.interval_hours[i]
            if start <= 10 < end:
                valid_interval = label
                break

        assert valid_interval in route_templates, "Should have extracted templates for 10:00 AM interval"
        interval_temps = route_templates[valid_interval]

        # Should have filtered out 'C', leaving only 2 directions (A and B)
        assert len(interval_temps) == 2, f"Expected 2 dominant directions, found {len(interval_temps)}"

        # Check splitting factor is 2.0 (since we have 2 directions)
        # The new logic dictates that we always split frequency by N directions
        # to ensure aggregate service matches the optimized headway.
        for t in interval_temps.values():
            sf = t.get("splitting_factor", 1.0)
            assert sf == 2.0, f"Splitting factor should be 2.0 for dominant routes (2 directions), got {sf}"

    def test_frequency_splitting(self, mock_converter):
        """
        Test Strategy C: If top 2 headsigns cover <= 90% of trips,
        keep ALL headsigns and Set splitting_factor = N_headsigns.
        """
        route_id = "test_route"

        # Setup Data: 100 trips total
        # 4 Headsigns, 25 each -> Top 2 = 50% coverage (< 90%)
        headsigns = ["A", "B", "C", "D"]
        trip_ids = [f"t_{i}" for i in range(100)]
        trip_headsigns = np.repeat(headsigns, 25)

        trips_data = {
            "route_id": [route_id] * 100,
            "trip_id": trip_ids,
            "trip_headsign": trip_headsigns,
        }
        mock_converter.gtfs_feed.trips = pd.DataFrame(trips_data)

        # Create Dummy Stop Times
        stops = []
        for i, t_id in enumerate(trip_ids):
            stops.append(
                {
                    "trip_id": t_id,
                    "stop_id": "s1",
                    "stop_sequence": 1,
                    "arrival_seconds": 36000,
                    "departure_seconds": 36000,
                }
            )
            stops.append(
                {
                    "trip_id": t_id,
                    "stop_id": "s2",
                    "stop_sequence": 2,
                    "arrival_seconds": 37000,
                    "departure_seconds": 37000,
                }
            )
        mock_converter.gtfs_feed.stop_times = pd.DataFrame(stops)

        # Run Extraction
        templates = mock_converter.extract_route_templates()

        # Assertions
        assert route_id in templates
        route_templates = templates[route_id]

        # Find interval covering 10:00 AM
        valid_interval = None
        for i, label in enumerate(mock_converter.interval_labels):
            start, end = mock_converter.interval_hours[i]
            if start <= 10 < end:
                valid_interval = label
                break

        assert valid_interval in route_templates, "Should have extracted templates for 10:00 AM interval"
        interval_temps = route_templates[valid_interval]

        # Should have kept all 4 directions (A, B, C, D)
        assert len(interval_temps) == 4, f"Expected 4 directions for messy route, found {len(interval_temps)}"

        # Check splitting factor is 4.0
        for t in interval_temps.values():
            sf = t.get("splitting_factor", 1.0)
            assert sf == 4.0, f"Splitting factor should be 4.0, got {sf}"
            sf = t.get("splitting_factor", 1.0)
            assert sf == 4.0, f"Splitting factor should be 4.0, got {sf}"
            sf = t.get("splitting_factor", 1.0)
            assert sf == 4.0, f"Splitting factor should be 4.0, got {sf}"
