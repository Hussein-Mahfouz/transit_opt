"""
Comprehensive tests for WaitingTimeObjective functionality.

This test suite validates the waiting time objective function used in transit
network optimization. The WaitingTimeObjective evaluates user waiting times
based on service frequency and vehicle counts across spatial zones.

Test Coverage:
- Basic objective creation and properties
- Waiting time calculation from vehicle counts
- Different metrics (total vs variance)
- Time aggregation methods (average, peak, sum, intervals)
- Population weighting functionality
"""


import numpy as np
import pytest

from transit_opt.optimisation.objectives import WaitingTimeObjective


class TestWaitingTimeObjective:
    """Test the waiting time objective."""

    def test_basic_objective_creation(self, sample_optimization_data):
        """
        Test that objective can be created and has expected properties.

        This test validates:
        - Successful instantiation of WaitingTimeObjective
        - Correct storage of configuration parameters
        - Creation of underlying spatial system components

        The spatial system includes:
        - Hexagonal zone grid for dividing study area
        - Transit stops mapped to zones
        - Vehicle-to-waiting-time conversion logic
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.5,  # 2.5km hexagonal zones
            crs="EPSG:3857",  # Web Mercator projection
            metric="total",  # Minimize total waiting time
            time_aggregation="average"  # Average across time intervals
        )

        # Verify configuration parameters are stored correctly
        assert objective.spatial_resolution == 2.5
        assert objective.crs == "EPSG:3857"
        assert objective.metric == "total"
        assert objective.time_aggregation == "average"
        assert objective.population_weighted is False  # default

        # Verify spatial system components were created
        assert hasattr(objective, "spatial_system")
        assert len(objective.spatial_system.hex_grid) > 0  # Has hexagonal zones
        assert len(objective.spatial_system.stops_gdf) > 0  # Has transit stops

    def test_parameter_validation(self, sample_optimization_data):
        """
        Test parameter validation for metric and time_aggregation.

        Validates that the objective correctly rejects invalid parameter values
        and provides helpful error messages for debugging.
        """
        # Test invalid metric
        with pytest.raises(ValueError, match="metric must be 'total' or 'variance'"):
            WaitingTimeObjective(
                sample_optimization_data,
                metric="invalid_metric"
            )

        # Test invalid time aggregation
        with pytest.raises(ValueError, match="time_aggregation must be"):
            WaitingTimeObjective(
                sample_optimization_data,
                time_aggregation="invalid_aggregation"
            )

        # Test valid parameters should work
        objective = WaitingTimeObjective(
            sample_optimization_data,
            metric="variance",
            time_aggregation="peak"
        )
        assert objective.metric == "variance"
        assert objective.time_aggregation == "peak"

    def test_basic_evaluation_total_metric(self, sample_optimization_data):
        """
        Test that objective evaluation works with total metric.

        The evaluate() method with total metric:
        1. Takes a solution matrix (routes × intervals with headway choices)
        2. Calculates vehicles per zone based on service frequencies
        3. Converts vehicles to waiting times (more vehicles = lower waiting)
        4. Returns total waiting time (lower = better)

        This test ensures the core total optimization objective functions properly.
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="total",  # Minimize total waiting time
            time_aggregation="average"
        )

        # Evaluate with the initial GTFS-derived solution
        total_waiting_time = objective.evaluate(sample_optimization_data["initial_solution"])

        # Validate objective function output properties
        assert isinstance(total_waiting_time, (int | float))  # Returns single numeric value
        assert total_waiting_time >= 0  # Total waiting time is non-negative
        assert not np.isnan(total_waiting_time)  # Should be a valid number
        assert not np.isinf(total_waiting_time)  # Should be finite (some service exists)

        print(f"✅ Total waiting time: {total_waiting_time:.2f} minutes")

    def test_basic_evaluation_variance_metric(self, sample_optimization_data):
        """
        Test that objective evaluation works with variance metric.

        The variance metric measures equity in waiting times:
        - Lower variance = more equitable service (similar waiting times across zones)
        - Higher variance = inequitable service (some zones much better than others)
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="variance",  # Minimize waiting time inequality
            time_aggregation="average"
        )

        # Evaluate with the initial GTFS-derived solution
        waiting_time_variance = objective.evaluate(sample_optimization_data["initial_solution"])

        # Validate objective function output properties
        assert isinstance(waiting_time_variance, (int | float))  # Returns single numeric value
        assert waiting_time_variance >= 0  # Variance is always non-negative
        assert not np.isnan(waiting_time_variance)  # Should be a valid number

        print(f"✅ Waiting time variance: {waiting_time_variance:.2f}")


    def test_vehicle_count_to_waiting_time_conversion(self, sample_optimization_data):
        """
        Test the core vehicle count to waiting time conversion logic.

        - More vehicles per zone → Higher frequency → Lower waiting time
        - Zero vehicles → Infinite waiting time (no service)
        - Conversion should be mathematically consistent
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0
        )
        interval_length = objective._get_interval_length_minutes()

        # Test zero vehicles
        waiting_time_zero = objective._convert_vehicle_count_to_waiting_time(0.0, interval_length)
        assert waiting_time_zero == interval_length # large value for no service (to avoid Inf)

        # Test single vehicle
        waiting_time_one = objective._convert_vehicle_count_to_waiting_time(1.0, interval_length)
        assert waiting_time_one > 0
        assert np.isfinite(waiting_time_one)

        # Test multiple vehicles - should have lower waiting time
        waiting_time_two = objective._convert_vehicle_count_to_waiting_time(2.0, interval_length)
        assert waiting_time_two > 0
        assert np.isfinite(waiting_time_two)
        assert waiting_time_two < waiting_time_one  # More vehicles = less waiting

        print(f"✅ Waiting times: 0 vehicles → ∞, 1 vehicle → {waiting_time_one:.2f}min, 2 vehicles → {waiting_time_two:.2f}min")

    def test_interval_length_calculation_and_caching(self, sample_optimization_data):
        """
        Test interval length calculation and caching functionality.

        The interval length determines how to convert vehicle counts to frequencies:
        - Should be calculated from decision_matrix_shape
        - Should be cached for performance
        - Should handle missing data gracefully
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0
        )

        # First call should calculate and cache
        length1 = objective._get_interval_length_minutes()
        expected_intervals = sample_optimization_data["decision_matrix_shape"][1]
        expected_length = (24 * 60) / expected_intervals  # 24 hours in minutes / intervals

        assert length1 == expected_length
        assert objective._interval_length_minutes == expected_length

        # Second call should use cache (no recalculation)
        length2 = objective._get_interval_length_minutes()
        assert length2 == length1

        print(f"✅ Interval length: {length1:.0f} minutes ({expected_intervals} intervals)")

    def test_different_time_aggregation_methods(self, sample_optimization_data):
        """
        Test different time aggregation methods.

        The objective function supports multiple ways to aggregate across time:
        - 'average': Mean waiting time across all time intervals
        - 'sum': Sum of waiting times across all time intervals  
        - 'peak': Waiting time from interval with most vehicles (best service)
        - 'intervals': Calculate objective per interval, then average

        Different methods serve different policy goals.
        """
        aggregation_methods = ["average", "sum", "peak", "intervals"]

        for method in aggregation_methods:
            print(f"Testing {method} aggregation...")

            objective = WaitingTimeObjective(
                optimization_data=sample_optimization_data,
                spatial_resolution_km=2.0,
                metric="total",
                time_aggregation=method
            )

            # Each method should evaluate successfully
            result = objective.evaluate(sample_optimization_data["initial_solution"])

            # Validate result properties
            assert isinstance(result, (int | float))
            assert not np.isnan(result)
            assert result >= 0

            print(f"✅ {method} aggregation result: {result:.2f}")

    def test_peak_interval_selection(self, sample_optimization_data):
        """
        Test peak interval selection logic.

        Peak aggregation should:
        - Find the SINGLE interval with most vehicles system-wide
        - Use waiting times from that interval for ALL zones
        - Represent system performance during peak service period
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            time_aggregation="peak"
        )

        # Test vehicle counts (zones × intervals)
        vehicles_intervals = np.array([
            [2, 3, 4, 1],  # Zone 0
            [1, 0, 2, 0],  # Zone 1
            [0, 1, 2, 0]   # Zone 2
        ])
        # Total by interval: [3, 4, 8, 1] → Peak is interval 2 (index 2)

        # Test interval waiting times matrix (intervals × zones)
        interval_waiting_times = np.array([
            [10.0, 5.0, 15.0],   # Interval 0
            [8.0, 12.0, 20.0],   # Interval 1
            [6.0, 8.0, 10.0],    # Interval 2 (PEAK - 8 total vehicles)
            [12.0, 15.0, 25.0]   # Interval 3
        ])

        peak_waiting_times = objective._get_peak_interval_waiting_times(
            interval_waiting_times, vehicles_intervals
        )

        # ALL zones should use interval 2 (system-wide peak)
        expected = np.array([6.0, 8.0, 10.0])  # All from interval 2
        np.testing.assert_array_equal(peak_waiting_times, expected)

        print("✅ Peak interval selection uses system-wide peak correctly")

    def test_intervals_separate_evaluation(self, sample_optimization_data):
        """
        Test separate evaluation of each interval.

        The 'intervals' aggregation method:
        - Calculates objective value for each time interval separately
        - Averages the interval-specific objective values
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="total",
            time_aggregation="intervals"
        )

        # Evaluate using intervals method
        result = objective.evaluate(sample_optimization_data["initial_solution"])

        # Should return valid result
        assert isinstance(result, (int |float))
        assert result >= 0
        assert not np.isnan(result)
        assert not np.isinf(result)

        print(f"✅ Intervals aggregation result: {result:.2f}")


    def test_mathematical_consistency(self, sample_optimization_data):
        """
        Test mathematical consistency of waiting time calculations.

        Verify that:
        - Doubling vehicles roughly halves waiting time
        - Calculations are deterministic
        - Edge cases are handled consistently
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0
        )
        interval_length = objective._get_interval_length_minutes()
        # Test mathematical relationship
        waiting_time_1 = objective._convert_vehicle_count_to_waiting_time(1.0, interval_length)
        waiting_time_2 = objective._convert_vehicle_count_to_waiting_time(2.0, interval_length)
        waiting_time_4 = objective._convert_vehicle_count_to_waiting_time(4.0, interval_length)

        # More vehicles should mean less waiting (inverse relationship)
        assert waiting_time_1 > waiting_time_2 > waiting_time_4

        # Doubling vehicles should roughly halve waiting time
        ratio_1_to_2 = waiting_time_1 / waiting_time_2
        ratio_2_to_4 = waiting_time_2 / waiting_time_4

        # Both ratios should be approximately 2.0
        assert 1.8 <= ratio_1_to_2 <= 2.2
        assert 1.8 <= ratio_2_to_4 <= 2.2

        print(f"✅ Mathematical consistency: 1 vehicle → {waiting_time_1:.1f}min, 2 vehicles → {waiting_time_2:.1f}min, 4 vehicles → {waiting_time_4:.1f}min")

    def test_spatial_summary(self, sample_optimization_data):
        """
        Test spatial summary generation.

        The spatial summary should provide human-readable information about:
        - Number of hexagonal zones
        - Population weighting status
        - Metric type (total/variance)
        - Time aggregation method
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="variance",
            time_aggregation="peak",
            population_weighted=False
        )

        summary = objective._get_spatial_summary()

        # Should contain key information
        assert "hexagonal zones" in summary
        assert "metric: variance" in summary
        assert "time aggregation: peak" in summary

        # Should NOT contain population weighting (disabled)
        assert "population weighted" not in summary

        print(f"✅ Spatial summary: {summary}")


class TestWaitingTimePopulationWeighting:
    """Test population weighting functionality for waiting time objective."""

    def test_population_interpolation_real_data(self, sample_optimization_data, usa_population_path):
        """
        Test population interpolation with real WorldPop data.
        
        This test validates:
        - Real WorldPop TIF file can be read and processed
        - Population data can be interpolated to hexagonal zones
        - Interpolated population values are reasonable
        """
        objective = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=1.5,
            metric="total",
            population_weighted=True,
            population_layer=usa_population_path
        )

        # Check population was interpolated successfully
        assert objective.population_per_zone is not None
        assert len(objective.population_per_zone) > 0
        assert np.all(objective.population_per_zone >= 0), "Population values should be non-negative"

        # Check that some zones have population (not all empty)
        assert np.sum(objective.population_per_zone) > 0, "Should have some population in study area"

        print(f"✅ Population interpolated: {len(objective.population_per_zone)} zones")
        print(f"✅ Total population: {np.sum(objective.population_per_zone):,.0f}")
        print(f"✅ Population range: {np.min(objective.population_per_zone):.0f} - {np.max(objective.population_per_zone):.0f}")

    def test_population_power_effects_real_data(self, sample_optimization_data, usa_population_path):
        """
        Test different population_power values with real WorldPop data.
        
        Different power values change how population weighting affects the objective:
        - power=0.5: Square root of population (less extreme weighting)
        - power=1.0: Linear population weighting
        - power=2.0: Squared population (more extreme weighting for large zones)
        """
        power_values = [0.5, 1.0, 2.0]
        total_objectives = []
        variance_objectives = []

        for power in power_values:
            # Test with total metric
            obj_total = WaitingTimeObjective(
                optimization_data=sample_optimization_data,
                spatial_resolution_km=2.0,
                metric="total",
                population_weighted=True,
                population_layer=usa_population_path,
                population_power=power
            )

            # Test with variance metric
            obj_variance = WaitingTimeObjective(
                optimization_data=sample_optimization_data,
                spatial_resolution_km=2.0,
                metric="variance",
                population_weighted=True,
                population_layer=usa_population_path,
                population_power=power
            )

            # Evaluate both metrics
            total_result = obj_total.evaluate(sample_optimization_data['initial_solution'])
            variance_result = obj_variance.evaluate(sample_optimization_data['initial_solution'])

            total_objectives.append(total_result)
            variance_objectives.append(variance_result)

            # Each should be valid
            assert total_result >= 0
            assert variance_result >= 0
            assert not np.isnan(total_result)
            assert not np.isnan(variance_result)

            print(f"✅ Power {power}: Total={total_result:.2f}, Variance={variance_result:.2f}")

        # Results should be different for different power values
        assert len(set(total_objectives)) > 1, "Different powers should give different total results"
        assert len(set(variance_objectives)) > 1, "Different powers should give different variance results"

    def test_population_vs_unweighted_comparison_real_data(self, sample_optimization_data, usa_population_path):
        """
        Compare population-weighted vs unweighted objectives with real data.
        
        This validates that population weighting produces meaningfully different
        results from unweighted objectives, demonstrating its value for equity analysis.
        """
        # Unweighted objectives
        obj_total_unweighted = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="total",
            population_weighted=False
        )

        obj_variance_unweighted = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="variance",
            population_weighted=False
        )

        # Population-weighted objectives
        obj_total_weighted = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="total",
            population_weighted=True,
            population_layer=usa_population_path
        )

        obj_variance_weighted = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="variance",
            population_weighted=True,
            population_layer=usa_population_path
        )

        # Evaluate all objectives
        solution = sample_optimization_data['initial_solution']

        total_unweighted = obj_total_unweighted.evaluate(solution)
        total_weighted = obj_total_weighted.evaluate(solution)
        variance_unweighted = obj_variance_unweighted.evaluate(solution)
        variance_weighted = obj_variance_weighted.evaluate(solution)

        # All should be valid values
        results = [total_unweighted, total_weighted, variance_unweighted, variance_weighted]
        for result in results:
            assert result >= 0
            assert not np.isnan(result)
            assert not np.isinf(result)

        # Population weighting should produce different results
        assert total_unweighted != total_weighted, "Population weighting should change total objective"
        assert variance_unweighted != variance_weighted, "Population weighting should change variance objective"

        print(f"✅ Total: Unweighted={total_unweighted:.2f}, Weighted={total_weighted:.2f}")
        print(f"✅ Variance: Unweighted={variance_unweighted:.2f}, Weighted={variance_weighted:.2f}")

    def test_real_population_data_consistency(self, sample_optimization_data, usa_population_path):
        """
        Test that population data is processed consistently across different objectives.
        
        Both total and variance objectives with the same population settings should
        interpolate identical population data.
        """
        obj1 = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="total",
            population_weighted=True,
            population_layer=usa_population_path,
            population_power=1.0
        )

        obj2 = WaitingTimeObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            metric="variance",  # Different metric, same population setup
            population_weighted=True,
            population_layer=usa_population_path,
            population_power=1.0
        )

        # Population arrays should be identical
        np.testing.assert_array_equal(
            obj1.population_per_zone,
            obj2.population_per_zone,
            err_msg="Same population settings should produce identical population arrays"
        )

        # Both should have the same number of zones
        assert len(obj1.population_per_zone) == len(obj2.population_per_zone)

        print(f"✅ Population consistency: {len(obj1.population_per_zone)} zones with identical population data")

if __name__ == "__main__":
    pytest.main([__file__])
