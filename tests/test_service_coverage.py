"""
Comprehensive tests for HexagonalCoverageObjective functionality.

This test suite validates the spatial equity objective function used in transit
network optimization. The HexagonalCoverageObjective evaluates how evenly
transit service (measured in vehicles) is distributed across spatial zones.

Test Coverage:
- Basic objective creation and properties
- Objective function evaluation (variance calculation)
- Edge cases (no service scenarios)
- Advanced features (spatial lag modeling)
- Data structure validation
- Consistency checks between different aggregation methods
"""

import numpy as np

from transit_opt.optimisation.objectives import HexagonalCoverageObjective


class TestHexagonalCoverageObjective:
    """Test the service coverage objective."""

    def test_basic_objective_creation(self, sample_optimization_data):
        """
        Test that objective can be created and has expected properties.

        This test validates:
        - Successful instantiation of HexagonalCoverageObjective
        - Correct storage of configuration parameters
        - Creation of underlying spatial system components

        The spatial system includes:
        - Hexagonal zone grid for dividing study area
        - Transit stops mapped to zones
        - Stop-to-zone mapping for vehicle calculations
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,  # 3km hexagonal zones
            crs="EPSG:3857",  # Web Mercator projection
        )

        # Verify configuration parameters are stored correctly
        assert objective.spatial_resolution == 3.0
        assert objective.crs == "EPSG:3857"
        assert objective.time_aggregation == "average"  # default aggregation method
        assert objective.spatial_lag is False  # default: no neighbor effects

        # Verify spatial system components were created
        assert hasattr(objective, "spatial_system")
        assert len(objective.spatial_system.hex_grid) > 0  # Has hexagonal zones
        assert len(objective.spatial_system.stops_gdf) > 0  # Has transit stops

    def test_basic_evaluation(self, sample_optimization_data):
        """
        Test that objective evaluation works and returns reasonable values.

        The evaluate() method:
        1. Takes a solution matrix (routes × intervals with headway choices)
        2. Calculates vehicles needed per zone based on service frequencies
        3. Returns spatial variance (lower = more equitable distribution)

        This test ensures the core optimization objective functions properly.
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        # Evaluate with the initial GTFS-derived solution
        variance = objective.evaluate(sample_optimization_data["initial_solution"])

        # Validate objective function output properties
        assert isinstance(variance, float)  # Returns single numeric value
        assert variance >= 0  # Variance is always non-negative
        assert not np.isnan(variance)  # Should be a valid number
        assert not np.isinf(variance)  # Should be finite

    def test_no_service_solution(self, sample_optimization_data):
        """
        Test with a solution that has no service anywhere.

        Edge case testing: When all routes have no service, all zones should
        have zero vehicles, resulting in zero variance (perfect equality of nothing).

        This validates:
        - Proper handling of the "no service" headway choice
        - Correct variance calculation when all values are identical (zero)
        - Robustness of the objective function in extreme scenarios
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        # Create solution where every route-interval has "no service"
        no_service_solution = np.full(
            sample_optimization_data["initial_solution"].shape,
            sample_optimization_data[
                "no_service_index"
            ],  # Index for "no service" option
            dtype=int,
        )

        # Zero vehicles everywhere = zero variance (perfect equity)
        variance = objective.evaluate(no_service_solution)
        assert variance == 0.0

    def test_spatial_lag_mode(self, sample_optimization_data):
        """
        Test spatial lag functionality.

        Spatial lag modeling accounts for accessibility spillover effects:
        - Standard mode: Only counts direct service in each zone
        - Spatial lag mode: Includes α × (neighbor accessibility) in calculations

        This creates more realistic accessibility measures where proximity
        to well-served areas improves your effective accessibility.

        Formula: accessibility[i] = vehicles[i] + α × Σ(W[i,j] × vehicles[j])
        Where W[i,j] are spatial weights between zones i and j.
        """
        # Standard objective: Direct vehicle counts only
        standard_obj = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
            spatial_lag=False,  # No neighbor effects
        )

        # Spatial lag objective: Include neighbor accessibility
        spatial_obj = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
            spatial_lag=True,  # Enable neighbor effects
            alpha=0.2,  # 20% weight on neighbor accessibility
        )

        # Verify configuration is stored correctly
        assert spatial_obj.spatial_lag is True
        assert spatial_obj.alpha == 0.2

        # Evaluate both approaches with same solution
        standard_variance = standard_obj.evaluate(
            sample_optimization_data["initial_solution"]
        )
        spatial_variance = spatial_obj.evaluate(
            sample_optimization_data["initial_solution"]
        )

        # Both should produce valid variance values
        assert isinstance(standard_variance, float)
        assert isinstance(spatial_variance, float)
        assert standard_variance >= 0
        assert spatial_variance >= 0

        # Results should be mathematically valid (not NaN)
        assert not (np.isnan(standard_variance) or np.isnan(spatial_variance))

        # Test that the values are different
        # Note: Values may be similar but will generally differ due to spatial lag effects
        assert standard_variance != spatial_variance

    def test_detailed_analysis_structure(self, sample_optimization_data):
        """
        Test that detailed analysis returns expected structure.

        The get_detailed_analysis() method provides comprehensive metrics:
        - Vehicle distribution statistics (mean, variance, etc.)
        - Zone-level data for visualization and interpretation
        - Service coverage metrics (zones with/without service)

        This data is essential for understanding optimization results and
        validating that the objective function is working correctly.
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        analysis = objective.get_detailed_analysis(
            sample_optimization_data["initial_solution"]
        )

        # Verify required analysis components exist
        required_keys = [
            "vehicles_per_zone_average",  # Vehicle count per zone (averaged over time)
            "total_vehicles_average",  # Total fleet size (average across time)
            "variance_average",  # Spatial variance (the objective value)
            "zones_with_service_average",  # Count of zones with any service
        ]

        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"

        # Validate data structure and types
        n_zones = len(objective.spatial_system.hex_grid)

        # vehicles_per_zone_average should have one value per zone
        assert len(analysis["vehicles_per_zone_average"]) == n_zones

        # Scalar metrics should be numeric
        assert isinstance(analysis["total_vehicles_average"], (int, float))
        assert isinstance(analysis["variance_average"], float)
        assert isinstance(analysis["zones_with_service_average"], (int, np.integer))

    def test_interval_data_structure(self, sample_optimization_data):
        """
        Test that interval-specific data exists and has correct structure.

        Transit systems vary by time of day (rush hour vs off-peak), so the
        analysis must provide interval-specific data:
        - vehicles_per_zone_intervals: (n_zones × n_intervals) matrix
        - interval_labels: Human-readable time period names

        This granular data enables:
        - Time-of-day specific analysis
        - Validation of aggregation methods
        - Detailed visualization of service patterns
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        analysis = objective.get_detailed_analysis(
            sample_optimization_data["initial_solution"]
        )

        # Check that time-interval data exists
        assert "vehicles_per_zone_intervals" in analysis
        assert "interval_labels" in analysis

        # Get problem dimensions for validation
        n_zones = len(objective.spatial_system.hex_grid)
        n_intervals = sample_optimization_data["n_intervals"]

        # Extract interval-specific data
        interval_data = analysis["vehicles_per_zone_intervals"]
        interval_labels = analysis["interval_labels"]

        # Validate matrix dimensions: each zone × each time interval
        assert interval_data.shape == (
            n_zones,
            n_intervals,
        ), f"Expected shape ({n_zones}, {n_intervals}), got {interval_data.shape}"

        # Validate label count matches interval count
        assert (
            len(interval_labels) == n_intervals
        ), f"Expected {n_intervals} interval labels, got {len(interval_labels)}"

        # Validate data types
        assert isinstance(interval_data, np.ndarray)
        assert interval_data.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert isinstance(interval_labels, list)

        # Vehicle counts must be non-negative (can't have negative vehicles)
        assert np.all(interval_data >= 0), "Vehicle counts should be non-negative"

        # Labels should be human-readable strings
        assert all(
            isinstance(label, str) for label in interval_labels
        ), "All interval labels should be strings"

        # Log structure for inspection during test runs
        print(f"✅ Interval data shape: {interval_data.shape}")
        print(f"✅ Number of intervals: {n_intervals}")
        print(f"✅ Interval labels: {interval_labels}")
        print(f"✅ Sample interval data (first zone): {interval_data[0, :]}")

    def test_intervals_consistency_with_aggregated_data(self, sample_optimization_data):
        """
        Test that interval data is consistent with average/peak aggregations.

        Mathematical consistency check:
        - average_data should equal mean(interval_data, axis=1)
        - peak_data should equal max(interval_data, axis=1)

        This validates that the aggregation methods are implemented correctly
        and that there are no bugs in the temporal aggregation logic.
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        analysis = objective.get_detailed_analysis(
            sample_optimization_data["initial_solution"]
        )

        # Extract data for consistency checking
        intervals_data = analysis[
            "vehicles_per_zone_intervals"
        ]  # (n_zones, n_intervals)
        average_data = analysis["vehicles_per_zone_average"]  # (n_zones,)
        peak_data = analysis["vehicles_per_zone_peak"]  # (n_zones,)

        # Test average aggregation: should be mean across time intervals
        calculated_average = np.mean(intervals_data, axis=1)
        np.testing.assert_array_almost_equal(
            average_data,
            calculated_average,
            decimal=10,
            err_msg="Average data should match mean of intervals",
        )

        # Test peak aggregation: should be max across time intervals
        calculated_peak = np.max(intervals_data, axis=1)
        np.testing.assert_array_equal(
            peak_data,
            calculated_peak,
            err_msg="Peak data should match max of intervals",
        )

        print("✅ Average aggregation consistent with intervals")
        print("✅ Peak aggregation consistent with intervals")

    def test_interval_labels_match_optimization_data(self, sample_optimization_data):
        """
        Test that interval labels match those in optimization data.

        Data consistency check: The time interval labels from the spatial
        analysis should exactly match those from the original optimization data.

        This ensures:
        - No data transformation errors
        - Consistent interpretation across analysis components
        - Proper alignment between GTFS data and spatial analysis
        """
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
        )

        analysis = objective.get_detailed_analysis(
            sample_optimization_data["initial_solution"]
        )

        # Extract labels from both sources
        analysis_labels = analysis["interval_labels"]
        expected_labels = sample_optimization_data["intervals"]["labels"]

        # Labels should match exactly (same strings, same order)
        assert (
            analysis_labels == expected_labels
        ), f"Interval labels mismatch: {analysis_labels} vs {expected_labels}"

        # Count should match the declared number of intervals
        assert (
            len(analysis_labels) == sample_optimization_data["n_intervals"]
        ), "Label count mismatch with n_intervals"

        print("✅ Interval labels match optimization data")
        print(f"✅ Labels: {analysis_labels}")

    def test_different_aggregation_methods(self, sample_optimization_data):
        """
        Test different time aggregation methods.

        The objective function supports multiple ways to aggregate across time:
        - 'average': Use mean vehicles per zone across all time intervals
        - 'peak': Use maximum vehicles per zone across all time intervals

        Different aggregation methods serve different policy goals:
        - Average: Promotes consistent service throughout the day
        - Peak: Ensures adequate capacity during highest demand periods

        This test validates both methods work and have correct configuration.
        """
        # Test average aggregation (promotes all-day equity)
        obj_avg = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            time_aggregation="average",  # Focus on average service levels
        )

        # Test peak aggregation (ensures peak-hour capacity)
        obj_peak = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            time_aggregation="peak",  # Focus on peak service levels
        )

        # Both should evaluate successfully
        variance_avg = obj_avg.evaluate(sample_optimization_data["initial_solution"])
        variance_peak = obj_peak.evaluate(sample_optimization_data["initial_solution"])

        # Validate both produce valid variance values
        assert isinstance(variance_avg, float)
        assert isinstance(variance_peak, float)
        assert variance_avg >= 0
        assert variance_peak >= 0

        # Verify configuration is stored correctly
        assert obj_avg.time_aggregation == "average"
        assert obj_peak.time_aggregation == "peak"

        # Note: variance_avg and variance_peak will typically be different
        # because they measure equity at different temporal aggregation levels
