# filepath:
"""
Unit tests for equity metrics (Atkinson Index).
Tests mathematical correctness isolated from GTFS data.
"""

import numpy as np
import pytest

from transit_opt.optimisation.utils.equity import (
    calculate_atkinson_index, calculate_atkinson_index_from_vehicles)


class TestEquityMath:
    def test_perfect_equality(self):
        """If everyone has the same value, inequality should be 0."""
        # Case 1: Waiting times (lower is better, handled internally)
        times = np.array([10.0, 10.0, 10.0])
        assert calculate_atkinson_index(times) == 0.0

        # Case 2: Vehicles (higher is better)
        vehicles = np.array([5, 5, 5])
        assert calculate_atkinson_index_from_vehicles(vehicles) == 0.0

    def test_weights_influence(self):
        """Verify that population weights actually change the result."""
        # Scenario: One zone is good (100 vehicles), one is bad (1 vehicle)
        values = np.array([100.0, 1.0])

        print("\n--- Testing Weight Influence ---")

        # Reference: Unweighted (effectively 1 person in each zone)
        score_unweighted = calculate_atkinson_index_from_vehicles(values, weights=None, epsilon=2.0)
        print(f"Unweighted (1 rich, 1 poor): {score_unweighted:.4f}")

        # Case A: IGNORE the poor zone (weight near 0 for poor)
        # Should be near 0 (perfect equality among the relevant population)
        score_ignore_poor = calculate_atkinson_index_from_vehicles(values, weights=np.array([1000, 0.001]), epsilon=2.0)
        print(f"Weighted to ignore poor zone:   {score_ignore_poor:.4f}")

        # Case B: IGNORE the rich zone (weight near 0 for rich)
        # Should be near 0 (perfect equality among the relevant population)
        score_ignore_rich = calculate_atkinson_index_from_vehicles(values, weights=np.array([0.001, 1000]), epsilon=2.0)
        print(f"Weighted to ignore rich zone:   {score_ignore_rich:.4f}")

        # The unweighted score should be high (inequality exists)
        assert score_unweighted > 0.1

        # The weighted scores should be much lower (approaching 0) because we are weighting
        # towards a single homogeneous group
        assert score_ignore_poor < score_unweighted
        assert score_ignore_rich < score_unweighted

        # Also ensure exact symmetry doesn't occur by using 3 values
        # 10, 50, 100
        vals_3 = np.array([10.0, 50.0, 100.0])
        w_skewed = np.array([100, 1, 1])  # Focus on poor
        w_uniform = np.array([1, 1, 1])

        s_skewed = calculate_atkinson_index_from_vehicles(vals_3, weights=w_skewed, epsilon=1.0)
        s_uniform = calculate_atkinson_index_from_vehicles(vals_3, weights=w_uniform, epsilon=1.0)

        assert s_skewed != s_uniform

    def test_epsilon_sensitivity(self):
        """
        Verify that higher epsilon values (more aversion to inequality)
        result in higher Atkison index values for the same unequal distribution.
        """
        print("\n--- Testing Epsilon Sensitivity ---")
        values = np.array([100.0, 10.0]) # Unequal distribution

        # Low aversion
        score_low = calculate_atkinson_index_from_vehicles(values, epsilon=0.5)

        # Medium aversion
        score_med = calculate_atkinson_index_from_vehicles(values, epsilon=1.0)

        # High aversion (Standard)
        score_high = calculate_atkinson_index_from_vehicles(values, epsilon=2.0)

        # Very high aversion (Approaching Maximin/Rawlsian)
        score_vhigh = calculate_atkinson_index_from_vehicles(values, epsilon=5.0)

        print(f"Epsilon 0.5 (Low):       {score_low:.4f}")
        print(f"Epsilon 1.0 (Med):       {score_med:.4f}")
        print(f"Epsilon 2.0 (High):      {score_high:.4f}")
        print(f"Epsilon 5.0 (Very High): {score_vhigh:.4f}")

        # Inequality score should strictly increase as aversion increases
        # because the 'cost' of the low value (10.0) weighs heavier
        assert score_low < score_med < score_high < score_vhigh

    def test_waiting_time_inversion(self):
        """Check that waiting time logic inverts values correctly."""
        # 10 min vs 100 min waiting time
        # This converts to accessibility 0.1 vs 0.01 internally
        times = np.array([10.0, 100.0])
        score = calculate_atkinson_index(times, epsilon=2.0)

        assert 0.0 < score < 1.0

    def test_edge_cases(self):
        """Handle zeros and empty arrays gracefully."""
        # Empty array
        assert calculate_atkinson_index(np.array([])) == 0.0

        # Zero vehicles (should be capped internally)
        vehicles = np.array([0, 0, 10])
        score = calculate_atkinson_index_from_vehicles(vehicles)
        assert 0.0 < score <= 1.0
        assert np.isfinite(score) 
