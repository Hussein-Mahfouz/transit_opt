"""Simple tests for HexagonalCoverageObjective functionality."""

import numpy as np

from transit_opt.optimisation.objectives import HexagonalCoverageObjective


class TestHexagonalCoverageObjective:
    """Test the service coverage objective."""

    def test_basic_objective_creation(self, sample_optimization_data):
        """Test that objective can be created and has expected properties."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        # Basic properties
        assert objective.spatial_resolution == 3.0
        assert objective.crs == "EPSG:3857"
        assert objective.time_aggregation == "average"  # default
        assert objective.spatial_lag is False  # default

        # Should have created spatial system
        assert hasattr(objective, 'spatial_system')
        assert len(objective.spatial_system.hex_grid) > 0
        assert len(objective.spatial_system.stops_gdf) > 0

    def test_basic_evaluation(self, sample_optimization_data):
        """Test that objective evaluation works and returns reasonable values."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        # Evaluate with initial solution
        variance = objective.evaluate(sample_optimization_data['initial_solution'])

        # Basic checks
        assert isinstance(variance, float)
        assert variance >= 0  # Variance is always non-negative
        assert not np.isnan(variance)
        assert not np.isinf(variance)

    def test_no_service_solution(self, sample_optimization_data):
        """Test with a solution that has no service anywhere."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        # Create no-service solution
        no_service_solution = np.full(
            sample_optimization_data['initial_solution'].shape,
            sample_optimization_data['no_service_index'],
            dtype=int
        )

        # Should return 0 variance (no variation)
        variance = objective.evaluate(no_service_solution)
        assert variance == 0.0

    def test_spatial_lag_mode(self, sample_optimization_data):
        """Test spatial lag functionality."""
        # Standard objective
        standard_obj = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
            spatial_lag=False
        )

        # Spatial lag objective
        spatial_obj = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857",
            spatial_lag=True,
            alpha=0.2
        )

        # Check properties
        assert spatial_obj.spatial_lag is True
        assert spatial_obj.alpha == 0.2

        # Evaluate both
        standard_variance = standard_obj.evaluate(sample_optimization_data['initial_solution'])
        spatial_variance = spatial_obj.evaluate(sample_optimization_data['initial_solution'])

        # Both should be valid
        assert isinstance(standard_variance, float)
        assert isinstance(spatial_variance, float)
        assert standard_variance >= 0
        assert spatial_variance >= 0

        # They should generally be different (unless very special case)
        # We'll be lenient here since they might be similar
        assert not (np.isnan(standard_variance) or np.isnan(spatial_variance))

    def test_detailed_analysis_structure(self, sample_optimization_data):
        """Test that detailed analysis returns expected structure."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        analysis = objective.get_detailed_analysis(sample_optimization_data['initial_solution'])

        # Check required keys exist
        required_keys = [
            'vehicles_per_zone_average',
            'total_vehicles_average',
            'variance_average',
            'zones_with_service_average'
        ]

        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"

        # Check data types and shapes
        n_zones = len(objective.spatial_system.hex_grid)
        assert len(analysis['vehicles_per_zone_average']) == n_zones
        assert isinstance(analysis['total_vehicles_average'], (int, float))
        assert isinstance(analysis['variance_average'], float)
        assert isinstance(analysis['zones_with_service_average'], (int, np.integer))

    def test_interval_data_structure(self, sample_optimization_data):
        """Test that interval-specific data exists and has correct structure."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        analysis = objective.get_detailed_analysis(sample_optimization_data['initial_solution'])

        # Check that interval-specific data exists
        assert 'vehicles_per_zone_intervals' in analysis
        assert 'interval_labels' in analysis

        # Get expected dimensions
        n_zones = len(objective.spatial_system.hex_grid)
        n_intervals = sample_optimization_data['n_intervals']

        # Test interval data structure
        interval_data = analysis['vehicles_per_zone_intervals']
        interval_labels = analysis['interval_labels']

        # Check shapes match expected dimensions
        assert interval_data.shape == (n_zones, n_intervals), \
            f"Expected shape ({n_zones}, {n_intervals}), got {interval_data.shape}"

        assert len(interval_labels) == n_intervals, \
            f"Expected {n_intervals} interval labels, got {len(interval_labels)}"

        # Check data types
        assert isinstance(interval_data, np.ndarray)
        assert interval_data.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert isinstance(interval_labels, list)

        # Check that interval data is non-negative (vehicle counts)
        assert np.all(interval_data >= 0), "Vehicle counts should be non-negative"

        # Check that interval labels are strings
        assert all(isinstance(label, str) for label in interval_labels), \
            "All interval labels should be strings"

        # Print for inspection
        print(f"✅ Interval data shape: {interval_data.shape}")
        print(f"✅ Number of intervals: {n_intervals}")
        print(f"✅ Interval labels: {interval_labels}")
        print(f"✅ Sample interval data (first zone): {interval_data[0, :]}")

    def test_intervals_consistency_with_aggregated_data(self, sample_optimization_data):
        """Test that interval data is consistent with average/peak aggregations."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        analysis = objective.get_detailed_analysis(sample_optimization_data['initial_solution'])

        # Get the data
        intervals_data = analysis['vehicles_per_zone_intervals']  # Shape: (n_zones, n_intervals)
        average_data = analysis['vehicles_per_zone_average']      # Shape: (n_zones,)
        peak_data = analysis['vehicles_per_zone_peak']            # Shape: (n_zones,)

        # Check that average is actually the mean across intervals
        calculated_average = np.mean(intervals_data, axis=1)
        np.testing.assert_array_almost_equal(
            average_data, calculated_average, decimal=10,
            err_msg="Average data should match mean of intervals"
        )

        # Check that peak is actually the max across intervals
        calculated_peak = np.max(intervals_data, axis=1)
        np.testing.assert_array_equal(
            peak_data, calculated_peak,
            err_msg="Peak data should match max of intervals"
        )

        print("✅ Average aggregation consistent with intervals")
        print("✅ Peak aggregation consistent with intervals")

    def test_interval_labels_match_optimization_data(self, sample_optimization_data):
        """Test that interval labels match those in optimization data."""
        objective = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            crs="EPSG:3857"
        )

        analysis = objective.get_detailed_analysis(sample_optimization_data['initial_solution'])

        # Get labels from both sources
        analysis_labels = analysis['interval_labels']
        expected_labels = sample_optimization_data['intervals']['labels']

        # They should match exactly
        assert analysis_labels == expected_labels, \
            f"Interval labels mismatch: {analysis_labels} vs {expected_labels}"

        # Check that the number matches n_intervals
        assert len(analysis_labels) == sample_optimization_data['n_intervals'], \
            "Label count mismatch with n_intervals"

        print("✅ Interval labels match optimization data")
        print(f"✅ Labels: {analysis_labels}")


    def test_different_aggregation_methods(self, sample_optimization_data):
        """Test different time aggregation methods."""
        # Test average aggregation
        obj_avg = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            time_aggregation='average'
        )

        # Test peak aggregation
        obj_peak = HexagonalCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=3.0,
            time_aggregation='peak'
        )

        # Both should work
        variance_avg = obj_avg.evaluate(sample_optimization_data['initial_solution'])
        variance_peak = obj_peak.evaluate(sample_optimization_data['initial_solution'])

        assert isinstance(variance_avg, float)
        assert isinstance(variance_peak, float)
        assert variance_avg >= 0
        assert variance_peak >= 0

        # Check properties are set correctly
        assert obj_avg.time_aggregation == 'average'
        assert obj_peak.time_aggregation == 'peak'
