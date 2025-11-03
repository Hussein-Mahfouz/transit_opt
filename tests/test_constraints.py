"""
Tests for constraint handlers in transit optimization.

This test suite validates constraint handlers using real GTFS data from the Duke University
transit system. The tests use deterministic fleet calculations to verify that constraints
work correctly with actual transit data.

TEST DATA FLOW:
1. Real GTFS data (duke-nc-us.zip) → GTFSDataPreparator
2. GTFSDataPreparator → sample_optimization_data (baseline fleet analysis)
3. Test solutions + fleet calculations → precalculated_fleet_data
4. Constraint handlers evaluate solutions → compare with expected values

CONSTRAINT TYPES TESTED:
- FleetTotalConstraintHandler: System-wide fleet budget limits
- FleetPerIntervalConstraintHandler: Fleet limits per time interval
- MinimumFleetConstraintHandler: Minimum service level requirements

SOLUTION TYPES:
- high_service: 10-minute headways everywhere (most vehicles needed)
- medium_service: 15-minute headways everywhere (moderate vehicles)
- low_service: 30-minute headways everywhere (fewer vehicles)
- no_service: No service anywhere (zero vehicles)
"""

import numpy as np
import pytest

from transit_opt.optimisation.problems.base import (
    FleetPerIntervalConstraintHandler,
    FleetTotalConstraintHandler,
    MinimumFleetConstraintHandler,
)

# ================================================================================================
# DATA STRUCTURE VALIDATION TESTS
# ================================================================================================


class TestConstraintHandlers:
    """
    Test basic constraint handler functionality and data structure validation.

    These tests ensure that:
    1. GTFS data has the expected structure for constraint handlers
    2. All required fields are present and have correct types
    3. Data dimensions are consistent across different parts of the structure
    """

    def test_data_structure_validity(self, sample_optimization_data):
        """
        Validate that optimization data has all required fields and structure.

        WHAT THIS TEST DOES:
        - Checks for required top-level keys
        - Validates fleet analysis section exists and has required fields
        - Prints actual data values for debugging
        - Ensures data types and dimensions are reasonable

        WHY THIS TEST IS IMPORTANT:
        - Constraint handlers depend on specific data structure
        - Missing fields cause cryptic errors later
        - Helps debug issues with GTFS data extraction
        """
        print("\n🔍 VALIDATING OPTIMIZATION DATA STRUCTURE:")

        # Check required top-level keys
        required_keys = [
            "n_routes",
            "n_intervals",
            "allowed_headways",
            "no_service_index",
        ]
        print(f"   📋 Checking top-level keys: {required_keys}")

        for key in required_keys:
            assert key in sample_optimization_data, f"Missing required key: {key}"
            print(f"      ✓ {key}: {sample_optimization_data[key]}")

        # Check constraints section exists
        print("\n   🔧 Checking constraints section:")
        assert (
            "constraints" in sample_optimization_data
        ), "Missing 'constraints' section"
        assert (
            "fleet_analysis" in sample_optimization_data["constraints"]
        ), "Missing 'fleet_analysis' section"

        fleet_analysis = sample_optimization_data["constraints"]["fleet_analysis"]
        print("      ✓ constraints.fleet_analysis found")

        # Check required fleet analysis fields
        required_fleet_keys = [
            "total_current_fleet_peak",
            "current_fleet_by_interval",
            "operational_buffer",
            "no_service_threshold_minutes",
        ]

        print(f"   🚌 Checking fleet analysis keys: {required_fleet_keys}")
        for key in required_fleet_keys:
            assert key in fleet_analysis, f"Missing fleet analysis key: {key}"
            print(f"      ✓ {key}: {fleet_analysis[key]}")

        # Print summary for debugging
        print("\n📊 DATA SUMMARY:")
        print(f"   Routes: {sample_optimization_data['n_routes']}")
        print(
            f"   Time intervals: {sample_optimization_data['n_intervals']} (intervals of {24//sample_optimization_data['n_intervals']}h each)"
        )
        print(f"   Allowed headways: {sample_optimization_data['allowed_headways']}")
        print(
            f"   Current peak fleet: {fleet_analysis['total_current_fleet_peak']} vehicles"
        )
        print(f"   Fleet by interval: {fleet_analysis['current_fleet_by_interval']}")
        print(
            f"   Operational buffer: {fleet_analysis['operational_buffer']} (15% extra time)"
        )

        # Validate data makes sense
        assert sample_optimization_data["n_routes"] > 0, "Should have at least 1 route"
        assert (
            sample_optimization_data["n_intervals"] > 0
        ), "Should have at least 1 interval"
        assert (
            fleet_analysis["total_current_fleet_peak"] >= 0
        ), "Peak fleet should be non-negative"
        assert (
            len(fleet_analysis["current_fleet_by_interval"])
            == sample_optimization_data["n_intervals"]
        ), "Fleet by interval length should match n_intervals"

        print("✅ All data structure validations passed!")


# ================================================================================================
# FLEET TOTAL CONSTRAINT HANDLER TESTS
# ================================================================================================


class TestFleetTotalConstraintHandler:
    """
    Test FleetTotalConstraintHandler - manages system-wide fleet budget limits.

    PURPOSE:
    These constraints prevent solutions from using too many vehicles system-wide.
    Example: "Don't use more than 120% of current peak fleet"

    CONSTRAINT FORMULA:
    violation = actual_fleet_measure - (baseline_fleet * (1 + tolerance))
    - Positive violation = constraint violated (too many vehicles)
    - Negative violation = constraint satisfied (within limit)

    MEASURES TESTED:
    - peak: Maximum vehicles needed across all time intervals
    - average: Average vehicles needed across all time intervals
    - total: Sum of vehicles across all intervals (rarely used)
    """

    def test_basic_constraint_creation(self, sample_optimization_data):
        """
        Test basic constraint handler initialization and configuration.

        WHAT THIS TEST DOES:
        - Creates a FleetTotalConstraintHandler with standard config
        - Verifies handler initializes correctly
        - Checks that configuration is stored properly
        - Validates n_constraints = 1 (total constraint always generates 1 constraint)
        """
        print("\n🏗️  TESTING FLEET TOTAL CONSTRAINT CREATION:")

        config = {
            "baseline": "current_peak",  # Use current GTFS peak as baseline
            "tolerance": 0.15,  # Allow 15% increase
            "measure": "peak",  # Measure peak fleet usage
        }

        print(f"   📋 Config: {config}")

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)

        # Verify handler properties
        print("   ✓ Handler created successfully")
        print(f"   📊 Number of constraints: {handler.n_constraints}")
        print(f"   🔧 Stored config: {handler.config}")

        # Assertions
        assert (
            handler.n_constraints == 1
        ), "FleetTotalConstraintHandler should always generate 1 constraint"
        assert (
            handler.config["baseline"] == "current_peak"
        ), "Baseline should be stored correctly"
        assert (
            handler.config["tolerance"] == 0.15
        ), "Tolerance should be stored correctly"
        assert handler.config["measure"] == "peak", "Measure should be stored correctly"

        print("✅ Basic constraint creation test passed!")

    def test_config_validation(self, sample_optimization_data):
        """
        Test configuration validation catches invalid settings.

        WHAT THIS TEST DOES:
        - Tests various invalid configurations
        - Ensures appropriate error messages are raised
        - Validates that constraint handler won't accept bad configurations

        INVALID CONFIGURATIONS TESTED:
        - Missing baseline
        - Invalid baseline values
        - Manual baseline without baseline_value
        """
        print("\n🔍 TESTING CONFIGURATION VALIDATION:")

        # Test 1: Missing baseline
        print("   Testing missing baseline...")
        with pytest.raises(ValueError, match="requires 'baseline'"):
            FleetTotalConstraintHandler({}, sample_optimization_data)
        print("   ✓ Correctly rejects missing baseline")

        # Test 2: Invalid baseline
        print("   Testing invalid baseline...")
        config = {"baseline": "invalid_baseline"}
        with pytest.raises(ValueError, match="baseline must be one of"):
            FleetTotalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects invalid baseline")

        # Test 3: Manual baseline without value
        print("   Testing manual baseline without value...")
        config = {"baseline": "manual"}
        with pytest.raises(
            ValueError, match="Manual baseline requires 'baseline_value'"
        ):
            FleetTotalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects manual baseline without value")

        print("✅ All configuration validation tests passed!")

    def test_peak_measure_constraint_with_known_values(
        self, sample_optimization_data, precalculated_fleet_data, sample_solutions
    ):
        """
        Test peak fleet constraint against precalculated values - MAIN DETERMINISTIC TEST.

        WHAT THIS TEST DOES:
        1. Sets up a peak fleet constraint with 20% tolerance
        2. Calculates expected limit = baseline_peak × 1.20
        3. For each test solution, calculates expected violation = solution_peak - limit
        4. Calls handler.evaluate() and compares actual vs expected violations
        5. Asserts they match within floating-point tolerance

        WHY THIS TEST IS IMPORTANT:
        - Tests the core constraint evaluation logic
        - Uses real GTFS data with deterministic calculations
        - Validates that constraint math works correctly
        - Provides detailed debugging output

        EXPECTED BEHAVIOR:
        - High service: Large positive violation (uses too many vehicles)
        - Medium service: Moderate violation (might be positive or negative)
        - Low service: Negative violation (under the limit)
        - No service: Large negative violation (well under the limit)
        """
        print("\n🎯 TESTING PEAK FLEET CONSTRAINT WITH KNOWN VALUES:")

        baseline = precalculated_fleet_data["baseline"]
        solutions = precalculated_fleet_data["solutions"]

        # Configure constraint: 20% increase allowed from current peak
        config = {
            "baseline": "current_peak",
            "tolerance": 0.20,  # 20% increase allowed
            "measure": "peak",
        }

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)
        baseline_peak = baseline["current_peak_fleet"]
        expected_limit = baseline_peak * 1.20

        print("   📊 CONSTRAINT SETUP:")
        print(f"      Baseline peak fleet: {baseline_peak} vehicles")
        print("      Tolerance: 20% increase allowed")
        print(f"      Calculated limit: {expected_limit:.1f} vehicles")
        print(f"      Formula: violation = solution_peak - {expected_limit:.1f}")

        # Test each solution type
        print("\n   🚌 TESTING EACH SOLUTION:")

        for solution_name, fleet_info in solutions.items():
            print(f"\n      → {solution_name.upper()}:")

            solution_peak = fleet_info["peak_fleet"]
            expected_violation = solution_peak - expected_limit

            print(f"        Solution peak fleet: {solution_peak:.1f} vehicles")
            print(
                f"        Expected violation: {solution_peak:.1f} - {expected_limit:.1f} = {expected_violation:.1f}"
            )

            # Call constraint handler
            actual_violations = handler.evaluate(sample_solutions[solution_name])
            actual_violation = actual_violations[0]

            print(f"        Actual violation: {actual_violation:.1f}")
            print(f"        Match? {abs(actual_violation - expected_violation) < 0.1}")

            # Assert they match within tolerance
            assert (
                abs(actual_violation - expected_violation) < 0.1
            ), f"Expected violation {expected_violation:.1f}, got {actual_violation:.1f} for {solution_name}"

            # Interpret result
            if actual_violation > 0:
                print(
                    f"        ❌ CONSTRAINT VIOLATED (uses {actual_violation:.1f} vehicles over limit)"
                )
            else:
                print(
                    f"        ✅ CONSTRAINT SATISFIED ({abs(actual_violation):.1f} vehicles under limit)"
                )

        print("\n✅ Peak measure constraint test passed!")

    def test_average_measure_constraint_with_known_values(
        self, sample_optimization_data, precalculated_fleet_data, sample_solutions
    ):
        """
        Test average fleet constraint - validates constraint works with different measures.

        WHAT THIS TEST DOES:
        - Tests 'average' measure instead of 'peak'
        - Uses precalculated average fleet values
        - Validates constraint evaluation returns reasonable results
        - Less strict than peak test (average fleet calculations can be more complex)

        WHY WE TEST THIS:
        - Ensures constraint handler works with different measures
        - Average fleet is sometimes used for budget constraints
        - Validates measure parameter is handled correctly
        """
        print("\n📊 TESTING AVERAGE FLEET CONSTRAINT:")

        baseline = precalculated_fleet_data["baseline"]
        solutions = precalculated_fleet_data["solutions"]

        config = {
            "baseline": "current_peak",  # Using peak baseline for simplicity
            "tolerance": 0.15,  # 15% tolerance
            "measure": "average",  # Test average measure
        }

        print(f"   📋 Config: {config}")

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)

        # Test one solution to verify constraint works
        solution_name = "medium_service"
        fleet_info = solutions[solution_name]
        solution_average = fleet_info["average_fleet"]

        print(f"   🚌 Testing with {solution_name}:")
        print(f"      Solution average fleet: {solution_average:.1f} vehicles")

        actual_violations = handler.evaluate(sample_solutions[solution_name])
        actual_violation = actual_violations[0]

        print(f"      Constraint violation: {actual_violation:.1f}")

        # Basic validations (less strict for average measures)
        assert len(actual_violations) == 1, "Should return exactly 1 violation"
        assert isinstance(actual_violation, (int, float)), "Violation should be numeric"
        assert not np.isnan(actual_violation), "Violation should not be NaN"

        if actual_violation > 0:
            print(f"      ❌ CONSTRAINT VIOLATED ({actual_violation:.1f} over limit)")
        else:
            print(
                f"      ✅ CONSTRAINT SATISFIED ({abs(actual_violation):.1f} under limit)"
            )

        print("✅ Average measure constraint test passed!")


    def test_fleet_total_constraint_with_drt_integration(self):
        """Test FleetTotalConstraintHandler with PT+DRT combined fleet calculation."""
        from pathlib import Path

        from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator

        # Create DRT-enabled optimization data
        test_data_dir = Path(__file__).parent / "data"
        gtfs_path = str(test_data_dir / "duke-nc-us.zip")

        preparator = GTFSDataPreparator(gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60, 120]

        # DRT configuration
        drt_config = {
            'enabled': True,
            'target_crs': 'EPSG:3857',
            'default_drt_speed_kmh': 25.0,
            'zones': [
                {
                    'zone_id': 'drt_duke_1',
                    'service_area_path': str(test_data_dir / "drt" / "drt_duke_1.shp"),
                    'allowed_fleet_sizes': [0, 10, 20, 30],
                    'zone_name': 'Duke Area 1',
                    'drt_speed_kmh': 20.0
                },
                {
                    'zone_id': 'drt_duke_2',
                    'service_area_path': str(test_data_dir / "drt" / "drt_duke_2.shp"),
                    'allowed_fleet_sizes': [0, 5, 15, 25],
                    'zone_name': 'Duke Area 2'
                }
            ]
        }

        # Extract PT+DRT optimization data
        opt_data = preparator.extract_optimization_data_with_drt(allowed_headways, drt_config)

        # Verify DRT is enabled
        assert opt_data['drt_enabled'] is True
        assert opt_data['n_drt_zones'] == 2

        # Create constraint handler
        constraint_config = {
            'baseline': 'current_peak',
            'tolerance': 0.30,
            'measure': 'peak'
        }
        constraint = FleetTotalConstraintHandler(constraint_config, opt_data)

        # ===== FIX: Use initial solution from opt_data (matches dimensions) =====
        initial_solution = opt_data['initial_solution']

        # For PT+DRT, initial_solution is already a flat array that needs decoding
        # Create a proper PT+DRT solution dict using the problem encoder
        from transit_opt.optimisation.objectives import StopCoverageObjective
        from transit_opt.optimisation.problems.transit_problem import TransitOptimizationProblem

        # Create a minimal problem to access encoding/decoding
        dummy_objective = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=2.0
        )
        problem = TransitOptimizationProblem(
            optimization_data=opt_data,
            objective=dummy_objective,
            constraints=[]
        )

        # Decode the initial solution to get proper PT+DRT dict format
        solution_dict = problem.decode_solution(initial_solution)

        print("\n🚌 Testing PT+DRT solution structure...")
        assert isinstance(solution_dict, dict)
        assert 'pt' in solution_dict
        assert 'drt' in solution_dict

        # Test constraint evaluation with proper dict format
        violations = constraint.evaluate(solution_dict)
        assert len(violations) == 1

        print(f"✅ Combined PT+DRT constraint violation: {violations[0]:.3f}")

        # ===== Verify PT and DRT components are both considered =====
        pt_fleet = constraint._calculate_fleet_from_solution(solution_dict['pt'])
        drt_fleet = constraint._calculate_drt_fleet_from_solution(solution_dict['drt'])

        print(f"   PT fleet by interval: {pt_fleet}")
        print(f"   DRT fleet by interval: {drt_fleet}")

        # Combined fleet should be sum of both
        combined_fleet = pt_fleet + drt_fleet
        baseline_pt_peak = opt_data['constraints']['fleet_analysis']['total_current_fleet_peak']
        fleet_limit = baseline_pt_peak * (1 + constraint_config['tolerance'])

        print(f"   PT baseline peak: {baseline_pt_peak}")
        print(f"   Fleet limit (with tolerance): {fleet_limit}")
        print(f"   Combined peak fleet: {np.max(combined_fleet)}")

        # Verify the calculation is correct
        expected_violation = np.max(combined_fleet) - fleet_limit
        assert abs(violations[0] - expected_violation) < 1e-6

        print("✅ PT+DRT constraint integration test passed!")


    def test_fleet_total_constraint_drt_disabled_compatibility(self):
        """Test that FleetTotalConstraintHandler works correctly when DRT is disabled."""
        from pathlib import Path

        from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator

        # Create PT-only optimization data
        test_data_dir = Path(__file__).parent / "data"
        gtfs_path = str(test_data_dir / "duke-nc-us.zip")

        preparator = GTFSDataPreparator(gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60, 120]

        # Extract PT-only data (no DRT config)
        opt_data = preparator.extract_optimization_data(allowed_headways)

        # Verify DRT is disabled
        assert opt_data.get('drt_enabled', False) is False

        # Create constraint handler
        constraint_config = {
            'baseline': 'current_peak',
            'tolerance': 0.20,
            'measure': 'peak'
        }
        constraint = FleetTotalConstraintHandler(constraint_config, opt_data)

        # ===== FIX: Use initial solution from opt_data (matches dimensions) =====
        pt_solution = opt_data['initial_solution']

        print(f"\n🚌 Testing PT-only solution...")
        print(f"   Solution shape: {pt_solution.shape}")
        print(f"   Expected shape: ({opt_data['n_routes']}, {opt_data['n_intervals']})")

        # Test 1: Standard matrix format (PT-only)
        violations_matrix = constraint.evaluate(pt_solution)
        assert len(violations_matrix) == 1

        print(f"✅ Matrix format violation: {violations_matrix[0]:.3f}")

        # Test 2: Dict format with 'pt' key (should also work)
        solution_dict = {'pt': pt_solution}
        violations_dict = constraint.evaluate(solution_dict)
        assert len(violations_dict) == 1

        print(f"✅ Dict format violation: {violations_dict[0]:.3f}")

        # Both should give same result
        np.testing.assert_array_almost_equal(violations_matrix, violations_dict)

        print("✅ PT-only constraint compatibility verified!")


# ================================================================================================
# FLEET PER-INTERVAL CONSTRAINT HANDLER TESTS
# ================================================================================================


class TestFleetPerIntervalConstraintHandler:
    """
    Test FleetPerIntervalConstraintHandler - manages fleet limits per time interval.

    PURPOSE:
    These constraints prevent solutions from overloading specific time periods.
    Example: "Don't use more than 125% of current fleet in any time interval"

    CONSTRAINT FORMULA (for each interval i):
    violation[i] = solution_fleet[i] - (baseline_fleet[i] * (1 + tolerance))
    - Positive violation = constraint violated for that interval
    - Negative violation = constraint satisfied for that interval

    KEY DIFFERENCES FROM TOTAL CONSTRAINT:
    - Generates n_intervals constraints (not just 1)
    - Each constraint applies to one time interval
    - Prevents peak-hour overloading
    """

    def test_basic_constraint_creation(self, sample_optimization_data):
        """
        Test basic per-interval constraint handler initialization.

        WHAT THIS TEST DOES:
        - Creates FleetPerIntervalConstraintHandler with standard config
        - Verifies n_constraints = n_intervals (one constraint per time interval)
        - Checks configuration storage
        """
        print("\n🏗️  TESTING PER-INTERVAL CONSTRAINT CREATION:")

        config = {
            "baseline": "current_by_interval",  # Use current fleet per interval as baseline
            "tolerance": 0.15,  # 15% increase allowed per interval
        }

        print(f"   📋 Config: {config}")

        handler = FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        n_intervals = sample_optimization_data["n_intervals"]

        print("   ✓ Handler created successfully")
        print(f"   📊 Number of intervals: {n_intervals}")
        print(f"   📊 Number of constraints: {handler.n_constraints}")

        # Assertions
        assert (
            handler.n_constraints == n_intervals
        ), f"Should generate {n_intervals} constraints, got {handler.n_constraints}"
        assert (
            handler.config["tolerance"] == 0.15
        ), "Tolerance should be stored correctly"

        print("✅ Basic per-interval constraint creation test passed!")

    def test_config_validation(self, sample_optimization_data):
        """
        Test configuration validation for per-interval constraints.

        Similar to FleetTotalConstraintHandler but specific to per-interval logic.
        """
        print("\n🔍 TESTING PER-INTERVAL CONFIGURATION VALIDATION:")

        # Missing both tolerance and min_fraction
        print("   Testing missing both tolerance and min_fraction...")
        config = {"baseline": "current_by_interval"}  # Neither tolerance nor min_fraction
        with pytest.raises(ValueError,
                           match="FleetPerIntervalConstraintHandler must specify either 'tolerance' "
                "or 'min_fraction' or both"):
            FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects missing both parameters")

        # Test missing baseline
        print("   Testing missing baseline...")
        with pytest.raises(ValueError,
                           match="FleetPerIntervalConstraintHandler must specify either 'tolerance' "
                "or 'min_fraction' or both"):
            FleetPerIntervalConstraintHandler({}, sample_optimization_data)
        print("   ✓ Correctly rejects missing baseline")

        # Test invalid baseline
        print("   Testing invalid baseline...")
        config = {"baseline": "invalid", "tolerance": 0.1}
        with pytest.raises(ValueError, match="baseline must be one of"):
            FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects invalid baseline")

        # Validate parameter ranges
        print("   Testing invalid min_fraction...")
        config = {"baseline": "current_by_interval", "min_fraction": 1.5}  # > 1.0
        with pytest.raises(ValueError, match="min_fraction must be between 0.0 and 1.0"):
            FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects invalid min_fraction")

        print("   Testing negative tolerance...")
        config = {"baseline": "current_by_interval", "tolerance": -0.1}  # < 0
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects negative tolerance")

        print("✅ Per-interval configuration validation tests passed!")

    def test_per_interval_constraint_with_known_values(
        self, sample_optimization_data, precalculated_fleet_data, sample_solutions
    ):
        """
        Test per-interval constraints against precalculated values - MAIN DETERMINISTIC TEST.

        WHAT THIS TEST DOES:
        1. Sets up per-interval fleet constraint with 25% tolerance
        2. Calculates expected limits = baseline_by_interval × 1.25
        3. For one test solution, calculates expected violations = solution_by_interval - limits
        4. Calls handler.evaluate() and compares actual vs expected violations for each interval
        5. Asserts they match within floating-point tolerance

        WHY THIS TEST IS IMPORTANT:
        - Tests per-interval constraint logic
        - Validates that constraints work correctly for each time interval
        - Uses real GTFS interval data
        - Provides detailed per-interval debugging

        EXPECTED BEHAVIOR:
        - Some intervals may violate (positive) while others satisfy (negative)
        - Pattern depends on solution type and baseline interval loads
        """
        print("\n🎯 TESTING PER-INTERVAL CONSTRAINT WITH KNOWN VALUES:")

        baseline = precalculated_fleet_data["baseline"]
        solutions = precalculated_fleet_data["solutions"]

        # Configure constraint: 25% increase allowed per interval
        config = {
            "baseline": "current_by_interval",
            "tolerance": 0.25,  # 25% increase allowed per interval
        }

        handler = FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        baseline_by_interval = np.array(baseline["current_fleet_by_interval"])
        expected_limits = baseline_by_interval * 1.25

        print("   📊 CONSTRAINT SETUP:")
        print(f"      Baseline fleet by interval: {baseline_by_interval}")
        print("      Tolerance: 25% increase allowed per interval")
        print(f"      Calculated limits: {expected_limits}")
        print("      Formula: violation[i] = solution_fleet[i] - limits[i]")

        # Test one solution with detailed comparison
        solution_name = "medium_service"
        fleet_info = solutions[solution_name]
        solution_fleet_by_interval = np.array(fleet_info["fleet_by_interval"])
        expected_violations = solution_fleet_by_interval - expected_limits

        print(f"\n   🚌 TESTING {solution_name.upper()}:")
        print(f"      Solution fleet by interval: {solution_fleet_by_interval}")
        print(f"      Expected violations: {expected_violations}")

        # Call constraint handler
        actual_violations = handler.evaluate(sample_solutions[solution_name])

        print(f"      Actual violations: {actual_violations}")

        # Detailed per-interval comparison
        print("\n   📋 PER-INTERVAL ANALYSIS:")
        violations_match = 0

        for i in range(len(expected_violations)):
            expected = expected_violations[i]
            actual = actual_violations[i]
            match = abs(actual - expected) < 0.1

            print(
                f"      Interval {i}: fleet={solution_fleet_by_interval[i]:.1f}, "
                f"limit={expected_limits[i]:.1f}, "
                f"expected={expected:.1f}, actual={actual:.1f} {'✓' if match else '❌'}"
            )

            if match:
                violations_match += 1

            # Individual assertion
            assert (
                abs(actual - expected) < 0.1
            ), f"Interval {i} violation mismatch: expected {expected:.1f}, got {actual:.1f}"

        print(
            f"\n   📊 SUMMARY: {violations_match}/{len(expected_violations)} intervals matched exactly"
        )
        print("✅ Per-interval constraint test passed!")

    def test_floor_only_constraint(self, sample_optimization_data, sample_solutions):
        """Test floor-only constraint"""
        print("\n🔽 TESTING FLOOR-ONLY CONSTRAINT:")

        config = {
            "baseline": "current_by_interval",
            "min_fraction": 0.7  # Only floor constraint, no ceiling
        }

        handler = FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        n_intervals = sample_optimization_data["n_intervals"]

        print(f"   📊 Expected constraints: {n_intervals} (floor only)")
        print(f"   📊 Actual constraints: {handler.n_constraints}")

        assert handler.n_constraints == n_intervals, f"Should generate {n_intervals} floor constraints"

        # Test with low service (likely to violate floor constraint)
        violations = handler.evaluate(sample_solutions["low_service"])
        print(f"   🧪 Violations: {violations}")

        assert len(violations) == n_intervals, "Should return one violation per interval"

        # Basic sanity check: violations should be numeric
        assert all(isinstance(v, (int, float)) for v in violations), "All violations should be numeric"
        print("✅ Floor-only constraint test passed!")

    def test_combined_ceiling_floor_constraint(self, sample_optimization_data, sample_solutions):
        """Test combined ceiling + floor constraints (new functionality)."""
        print("\n🔗 TESTING COMBINED CEILING + FLOOR CONSTRAINTS:")

        config = {
            "baseline": "current_by_interval",
            "tolerance": 0.25,      # Ceiling constraint
            "min_fraction": 0.6     # Floor constraint
        }

        handler = FleetPerIntervalConstraintHandler(config, sample_optimization_data)
        n_intervals = sample_optimization_data["n_intervals"]
        expected_constraints = 2 * n_intervals  # ceiling + floor

        print(f"   📊 Expected constraints: {expected_constraints} ({n_intervals} ceiling + {n_intervals} floor)")
        print(f"   📊 Actual constraints: {handler.n_constraints}")

        assert handler.n_constraints == expected_constraints

        # Test evaluation returns correct number of violations
        violations = handler.evaluate(sample_solutions["medium_service"])
        print(f"Violations shape: {violations.shape}")
        print(f"First {n_intervals} (ceiling): {violations[:n_intervals]}")
        print(f"Last {n_intervals} (floor): {violations[n_intervals:]}")

        assert len(violations) == expected_constraints
        print("✅ Combined ceiling + floor constraint test passed!")


# ================================================================================================
# MINIMUM FLEET CONSTRAINT HANDLER TESTS
# ================================================================================================


class TestMinimumFleetConstraintHandler:
    """
    Test MinimumFleetConstraintHandler - ensures minimum service levels are maintained.

    PURPOSE:
    These constraints prevent solutions from cutting service too drastically.
    Example: "Must maintain at least 60% of current peak fleet"

    CONSTRAINT FORMULA:
    violation = minimum_required - actual_fleet
    - Positive violation = constraint violated (not enough fleet/service)
    - Negative violation = constraint satisfied (sufficient fleet/service)

    NOTE: Sign convention is OPPOSITE of upper-bound constraints!

    LEVELS TESTED:
    - system: Single constraint for system-wide minimum
    - interval: One constraint per interval for interval-wise minimums
    """

    def test_basic_system_constraint_creation(self, sample_optimization_data):
        """
        Test basic system-level minimum constraint creation.

        WHAT THIS TEST DOES:
        - Creates MinimumFleetConstraintHandler with system-level config
        - Verifies n_constraints = 1 for system level
        - Checks configuration storage
        """
        print("\n🏗️  TESTING MINIMUM SYSTEM CONSTRAINT CREATION:")

        config = {
            "min_fleet_fraction": 0.8,  # Require 80% of current
            "level": "system",  # System-wide constraint
            "measure": "peak",  # Use peak measure
            "baseline": "current_peak",  # Compare to current peak
        }

        print(f"   📋 Config: {config}")

        handler = MinimumFleetConstraintHandler(config, sample_optimization_data)

        print("   ✓ Handler created successfully")
        print(f"   📊 Number of constraints: {handler.n_constraints}")
        print(f"   🔧 Min fleet fraction: {handler.config['min_fleet_fraction']}")

        # Assertions
        assert (
            handler.n_constraints == 1
        ), "System-level minimum should generate 1 constraint"
        assert (
            handler.config["min_fleet_fraction"] == 0.8
        ), "Min fleet fraction should be stored"

        print("✅ Basic system minimum constraint creation test passed!")

    def test_basic_interval_constraint_creation(self, sample_optimization_data):
        """
        Test basic interval-level minimum constraint creation.

        WHAT THIS TEST DOES:
        - Creates MinimumFleetConstraintHandler with interval-level config
        - Verifies n_constraints = n_intervals for interval level
        - Checks configuration storage
        """
        print("\n🏗️  TESTING MINIMUM INTERVAL CONSTRAINT CREATION:")

        config = {
            "min_fleet_fraction": 0.7,  # Require 70% of current per interval
            "level": "interval",  # Interval-level constraints
            "baseline": "current_by_interval",  # Compare to current per interval
        }

        print(f"   📋 Config: {config}")

        handler = MinimumFleetConstraintHandler(config, sample_optimization_data)
        n_intervals = sample_optimization_data["n_intervals"]

        print("   ✓ Handler created successfully")
        print(f"   📊 Number of intervals: {n_intervals}")
        print(f"   📊 Number of constraints: {handler.n_constraints}")

        # Assertions
        assert (
            handler.n_constraints == n_intervals
        ), f"Interval-level should generate {n_intervals} constraints"

        print("✅ Basic interval minimum constraint creation test passed!")

    def test_config_validation(self, sample_optimization_data):
        """
        Test configuration validation for minimum constraints.

        INVALID CONFIGURATIONS TESTED:
        - Missing min_fleet_fraction
        - Invalid fraction values (outside 0.0-1.0 range)
        - Invalid level values
        """
        print("\n🔍 TESTING MINIMUM CONSTRAINT CONFIGURATION VALIDATION:")

        # Test missing min_fleet_fraction
        print("   Testing missing min_fleet_fraction...")
        with pytest.raises(ValueError, match="requires 'min_fleet_fraction'"):
            MinimumFleetConstraintHandler({}, sample_optimization_data)
        print("   ✓ Correctly rejects missing min_fleet_fraction")

        # Test invalid fraction (too high)
        print("   Testing invalid fraction...")
        config = {"min_fleet_fraction": 1.5}
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            MinimumFleetConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects fraction > 1.0")

        # Test invalid level
        print("   Testing invalid level...")
        config = {"min_fleet_fraction": 0.8, "level": "invalid"}
        with pytest.raises(ValueError, match="level must be 'system' or 'interval'"):
            MinimumFleetConstraintHandler(config, sample_optimization_data)
        print("   ✓ Correctly rejects invalid level")

        print("✅ Minimum constraint configuration validation tests passed!")

    def test_system_minimum_constraint_with_known_values(
        self, sample_optimization_data, precalculated_fleet_data, sample_solutions
    ):
        """
        Test system-level minimum constraint against precalculated values - MAIN DETERMINISTIC TEST.

        WHAT THIS TEST DOES:
        1. Sets up system minimum constraint requiring 40% of current peak
        2. Calculates minimum_required = baseline_peak × 0.4
        3. For each test solution, calculates expected violation = minimum_required - solution_peak
        4. Calls handler.evaluate() and compares actual vs expected violations
        5. Asserts they match within floating-point tolerance

        WHY THIS TEST IS IMPORTANT:
        - Tests minimum constraint logic (opposite sign convention)
        - Validates system-level minimum service requirements
        - Uses real GTFS baseline data
        - Provides detailed debugging for each solution type

        EXPECTED BEHAVIOR:
        - High service: Large negative violation (exceeds minimum easily)
        - Medium service: Moderate negative violation (exceeds minimum)
        - Low service: Small negative violation or positive (might violate)
        - No service: Large positive violation (violates minimum)
        """
        print("\n🎯 TESTING SYSTEM MINIMUM CONSTRAINT WITH KNOWN VALUES:")

        baseline = precalculated_fleet_data["baseline"]
        solutions = precalculated_fleet_data["solutions"]

        # Configure constraint: Require 40% of current peak (lenient for testing)
        config = {
            "min_fleet_fraction": 0.4,  # Require 40% of current
            "level": "system",  # System-wide minimum
            "measure": "peak",  # Use peak measure
            "baseline": "current_peak",  # Compare to current peak
        }

        handler = MinimumFleetConstraintHandler(config, sample_optimization_data)
        baseline_peak = baseline["current_peak_fleet"]
        minimum_required = baseline_peak * 0.4

        print("   📊 CONSTRAINT SETUP:")
        print(f"      Baseline peak fleet: {baseline_peak} vehicles")
        print("      Min fraction: 40% required")
        print(f"      Minimum required: {minimum_required:.1f} vehicles")
        print(f"      Formula: violation = {minimum_required:.1f} - solution_peak")
        print("      NOTE: Positive violation = not enough fleet (BAD)")
        print("            Negative violation = sufficient fleet (GOOD)")

        # Test each solution type
        print("\n   🚌 TESTING EACH SOLUTION:")

        for solution_name, fleet_info in solutions.items():
            print(f"\n      → {solution_name.upper()}:")

            solution_peak = fleet_info["peak_fleet"]
            expected_violation = minimum_required - solution_peak

            print(f"        Solution peak fleet: {solution_peak:.1f} vehicles")
            print(
                f"        Expected violation: {minimum_required:.1f} - {solution_peak:.1f} = {expected_violation:.1f}"
            )

            # Call constraint handler
            actual_violations = handler.evaluate(sample_solutions[solution_name])
            actual_violation = actual_violations[0]

            print(f"        Actual violation: {actual_violation:.1f}")
            print(f"        Match? {abs(actual_violation - expected_violation) < 0.1}")

            # Assert they match
            assert (
                abs(actual_violation - expected_violation) < 0.1
            ), f"Expected violation {expected_violation:.1f}, got {actual_violation:.1f} for {solution_name}"

            # Interpret result
            if actual_violation > 0:
                print(
                    f"        ❌ MINIMUM VIOLATED (need {actual_violation:.1f} more vehicles)"
                )
            else:
                print(
                    f"        ✅ MINIMUM SATISFIED ({abs(actual_violation):.1f} vehicles above minimum)"
                )

        print("\n✅ System minimum constraint test passed!")


# ================================================================================================
# CONSTRAINT INTEGRATION TESTS
# ================================================================================================


class TestConstraintIntegration:
    """
    Test combinations of multiple constraints working together.

    PURPOSE:
    - Validates that multiple constraint types can be used simultaneously
    - Tests realistic optimization scenarios with mixed constraints
    - Ensures constraint handlers don't interfere with each other
    """

    def test_multiple_constraint_combination(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test combining different constraint types in a realistic scenario.

        WHAT THIS TEST DOES:
        - Creates 3 different constraint handlers (total, per-interval, minimum)
        - Configures them with lenient settings to avoid failures
        - Evaluates all constraints on a test solution
        - Validates total constraint count and basic functionality

        WHY THIS TEST IS IMPORTANT:
        - Real optimization problems use multiple constraint types
        - Ensures constraints can coexist without conflicts
        - Validates total constraint count calculations
        """
        print("\n🔗 TESTING MULTIPLE CONSTRAINT COMBINATION:")

        n_intervals = sample_optimization_data["n_intervals"]

        # Create constraint combination (lenient settings to avoid failures)
        print("   🏗️  Creating constraint handlers:")

        fleet_total = FleetTotalConstraintHandler(
            {
                "baseline": "current_peak",
                "tolerance": 0.50,  # Very lenient - 50% increase allowed
                "measure": "peak",
            },
            sample_optimization_data,
        )
        print("      ✓ FleetTotal: 1 constraint (50% tolerance)")

        fleet_intervals = FleetPerIntervalConstraintHandler(
            {"baseline": "current_by_interval", "tolerance": 0.50},  # Very lenient
            sample_optimization_data,
        )
        print(
            f"      ✓ FleetPerInterval: {fleet_intervals.n_constraints} constraints (50% tolerance)"
        )

        minimum_fleet = MinimumFleetConstraintHandler(
            {
                "min_fleet_fraction": 0.1,  # Very lenient - only 10% minimum
                "level": "system",
                "measure": "peak",
                "baseline": "current_peak",
            },
            sample_optimization_data,
        )
        print("      ✓ MinimumFleet: 1 constraint (10% minimum)")

        constraints = [fleet_total, fleet_intervals, minimum_fleet]

        # Calculate total constraints
        total_constraints = sum(c.n_constraints for c in constraints)
        expected_constraints = 1 + n_intervals + 1

        print("   📊 CONSTRAINT COUNT:")
        print(f"      Total + PerInterval + Minimum = {total_constraints}")
        print(f"      Expected: 1 + {n_intervals} + 1 = {expected_constraints}")

        assert (
            total_constraints == expected_constraints
        ), f"Expected {expected_constraints} total constraints, got {total_constraints}"

        # Test constraint evaluation
        print("\n   🧪 TESTING CONSTRAINT EVALUATION:")

        def evaluate_all_constraints(solution):
            all_violations = []
            for i, constraint in enumerate(constraints):
                violations = constraint.evaluate(solution)
                print(
                    f"      Constraint {i+1}: {len(violations)} violations = {violations}"
                )
                all_violations.extend(violations)
            return np.array(all_violations)

        # Test with medium service solution
        solution_name = "medium_service"
        print(f"   Testing with {solution_name}...")

        all_violations = evaluate_all_constraints(sample_solutions[solution_name])

        print("\n   📊 EVALUATION RESULTS:")
        print(f"      Total violations returned: {len(all_violations)}")
        print(f"      Expected: {total_constraints}")
        print(f"      All violations: {all_violations}")

        assert (
            len(all_violations) == total_constraints
        ), f"Expected {total_constraints} violations, got {len(all_violations)}"

        # Basic sanity checks
        satisfied = np.sum(all_violations <= 0)
        violated = np.sum(all_violations > 0)

        print(f"      Satisfied: {satisfied}/{total_constraints}")
        print(f"      Violated: {violated}/{total_constraints}")

        # All violations should be numeric and finite
        assert all(
            isinstance(v, (int, float)) for v in all_violations
        ), "All violations should be numeric"
        assert all(
            not np.isnan(v) for v in all_violations
        ), "No violations should be NaN"

        print("✅ Multiple constraint combination test passed!")

    def test_constraint_info_methods(self, sample_optimization_data):
        """
        Test constraint info methods for debugging and introspection.

        WHAT THIS TEST DOES:
        - Creates a constraint handler
        - Calls get_constraint_info() method
        - Validates returned information structure
        - Ensures debugging info is available
        """
        print("\n🔍 TESTING CONSTRAINT INFO METHODS:")

        config = {"baseline": "current_peak", "tolerance": 0.15, "measure": "peak"}

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)
        info = handler.get_constraint_info()

        print(f"   📊 Constraint info returned: {info}")

        # Validate info structure
        assert "handler_type" in info, "Should include handler_type"
        assert "n_constraints" in info, "Should include n_constraints"
        assert "config" in info, "Should include config"

        assert (
            info["handler_type"] == "FleetTotalConstraintHandler"
        ), "Should identify handler type"
        assert info["n_constraints"] == 1, "Should report correct constraint count"
        assert (
            info["config"]["baseline"] == "current_peak"
        ), "Should include config details"

        print("✅ Constraint info methods test passed!")

    def test_constraint_edge_cases(self, sample_optimization_data):
        """
        Test edge cases and boundary conditions.

        WHAT THIS TEST DOES:
        - Tests with all-no-service solution (extreme case)
        - Validates upper-bound constraints are satisfied with zero fleet
        - Validates minimum constraints are violated with zero fleet
        - Ensures constraint handlers handle edge cases gracefully
        """
        print("\n⚠️  TESTING CONSTRAINT EDGE CASES:")

        n_routes = sample_optimization_data["n_routes"]
        n_intervals = sample_optimization_data["n_intervals"]
        no_service_index = sample_optimization_data["no_service_index"]

        # Create extreme test case: no service anywhere
        all_no_service = np.full((n_routes, n_intervals), no_service_index)

        print("   🚫 Testing with ALL-NO-SERVICE solution:")
        print(f"      Solution shape: {all_no_service.shape}")
        print(f"      All values: {no_service_index} (no service)")

        # Test 1: Upper-bound constraint should be satisfied (0 vehicles < any limit)
        print("\n   📈 Testing upper-bound constraint (should be satisfied):")

        config = {
            "baseline": "current_peak",
            "tolerance": 0.0,  # Very strict - no increase allowed
            "measure": "peak",
        }

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)
        violations = handler.evaluate(all_no_service)

        print(f"      Violation: {violations[0]:.1f}")
        print("      Expected: <= 0 (satisfied)")

        assert (
            violations[0] <= 0
        ), "No service should satisfy upper-bound constraint (0 vehicles < limit)"
        print("      ✅ Upper-bound constraint satisfied as expected")

        # Test 2: Minimum constraint should be violated (0 vehicles < minimum)
        print("\n   📉 Testing minimum constraint (should be violated):")

        min_config = {
            "min_fleet_fraction": 0.3,  # Require 30% of current (lenient but not zero)
            "level": "system",
            "measure": "peak",
            "baseline": "current_peak",
        }

        min_handler = MinimumFleetConstraintHandler(
            min_config, sample_optimization_data
        )
        min_violations = min_handler.evaluate(all_no_service)

        print(f"      Violation: {min_violations[0]:.1f}")
        print("      Expected: > 0 (violated)")

        assert (
            min_violations[0] > 0
        ), "No service should violate minimum constraint (0 vehicles < minimum)"
        print("      ✅ Minimum constraint violated as expected")

        print("✅ Constraint edge cases test passed!")

    def test_constraint_consistency(self, sample_optimization_data, sample_solutions):
        """
        Test that constraint evaluations are consistent and repeatable.

        WHAT THIS TEST DOES:
        - Evaluates same solution multiple times
        - Ensures results are identical (deterministic)
        - Validates constraint handlers are stateless
        """
        print("\n🔄 TESTING CONSTRAINT CONSISTENCY:")

        config = {"baseline": "current_peak", "tolerance": 0.15, "measure": "peak"}

        handler = FleetTotalConstraintHandler(config, sample_optimization_data)
        solution = sample_solutions["medium_service"]

        print("   🧪 Evaluating same solution 3 times:")

        # Evaluate same solution multiple times
        violations1 = handler.evaluate(solution)
        violations2 = handler.evaluate(solution)
        violations3 = handler.evaluate(solution)

        print(f"      Evaluation 1: {violations1}")
        print(f"      Evaluation 2: {violations2}")
        print(f"      Evaluation 3: {violations3}")

        # Results should be identical
        assert np.allclose(
            violations1, violations2
        ), "Evaluation 1 and 2 should be identical"
        assert np.allclose(
            violations2, violations3
        ), "Evaluation 2 and 3 should be identical"

        print("      ✅ All evaluations identical")
        print("✅ Constraint consistency test passed!")


# ================================================================================================
# MAIN TEST EXECUTION
# ================================================================================================

if __name__ == "__main__":
    """
    Run tests directly with detailed output.

    Usage: python test_constraints.py
    This will run all tests with verbose output and print statements.
    """
    print("🚀 Running constraint handler tests with detailed output...")
    pytest.main([__file__, "-v", "-s"])
    pytest.main([__file__, "-v", "-s"])
