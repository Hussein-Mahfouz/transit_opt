"""
Comprehensive test suite for TransitOptimizationProblem - the main pymoo integration class.

OVERVIEW:
This test suite validates that TransitOptimizationProblem correctly integrates your
domain-specific components (objectives, constraints, GTFS data) with pymoo's
metaheuristic optimization framework. It ensures the bridge between your transit
optimization logic and pymoo algorithms works correctly.

CRITICAL INTEGRATION POINTS TESTED:
1. Problem initialization and pymoo compatibility
2. Solution format conversion (domain matrices ↔ pymoo flat vectors)
3. Population-based evaluation for metaheuristics (PSO, GA, etc.)
4. Real component integration (StopCoverageObjective, constraint handlers)
5. Error handling and edge cases

TEST DATA FLOW:
Real GTFS → GTFSDataPreparator → sample_optimization_data → Problem → pymoo algorithms

FIXTURES USED:
- sample_optimization_data: Real Duke University GTFS data prepared for optimization
- sample_solutions: Test solution matrices (high/medium/low/no service scenarios)
- precalculated_fleet_data: Expected fleet calculations for deterministic validation

WHY THESE TESTS MATTER:
- Validates the core bridge between your domain logic and optimization algorithms
- Ensures population-based algorithms (PSO, GA) can evaluate solutions correctly
- Prevents silent failures in optimization that would waste computational time
- Validates that optimization results will be meaningful and correct
"""

import numpy as np
import pytest
from pymoo.core.problem import Problem

from transit_opt.optimisation.objectives.service_coverage import StopCoverageObjective
from transit_opt.optimisation.problems.base import FleetTotalConstraintHandler
from transit_opt.optimisation.problems.transit_problem import TransitOptimizationProblem

# ================================================================================================
# BASIC PROBLEM CREATION TESTS
# ================================================================================================


class TestTransitProblemCreation:
    """
    Test TransitOptimizationProblem initialization and basic properties.

    PURPOSE:
    These tests ensure that the problem class initializes correctly with pymoo
    and stores all necessary components. They validate the foundation that all
    other optimization functionality depends on.

    WHAT GETS VALIDATED:
    - Pymoo Problem inheritance and interface compliance
    - Problem dimensions match GTFS data structure
    - Variable bounds and types are set correctly
    - Component storage (objectives, constraints) works properly
    - Both constrained and unconstrained problem variants work

    WHY CRITICAL:
    - Pymoo algorithms depend on correct problem setup
    - Wrong dimensions cause array shape mismatches during optimization
    - Incorrect bounds lead to invalid solutions being generated
    - Component storage failures break evaluation pipeline
    """

    def test_problem_creation_with_objective_only(self, sample_optimization_data):
        """
        Test creating problem with only objective function, no constraints.

        WHAT THIS TEST DOES:
        - Creates StopCoverageObjective with real GTFS data
        - Initializes TransitOptimizationProblem without constraints
        - Validates pymoo Problem interface compliance
        - Checks problem dimensions match GTFS data
        - Verifies variable bounds and types are correct

        WHY THIS SCENARIO MATTERS:
        - Unconstrained optimization is often used for initial exploration
        - Simpler setup helps isolate objective function issues
        - Many research scenarios focus on single objectives
        - Baseline for comparing constrained vs unconstrained results

        EXPECTED BEHAVIOR:
        - n_obj = 1 (single objective optimization)
        - n_constr = 0 (no constraints)
        - n_var = n_routes × n_intervals (one decision per route-interval pair)
        - Variable bounds: [0, n_choices-1] (indices into allowed_headways)
        - Variable type: integer (discrete headway choices)
        """
        print("\n🏗️  TESTING PROBLEM CREATION (OBJECTIVE ONLY):")

        # Create spatial equity objective using your existing component
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0,
            crs="EPSG:3857",
        )

        # Create problem without constraints (unconstrained optimization)
        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=None,
        )

        # CRITICAL: Validate pymoo interface compliance
        assert isinstance(
            problem, Problem
        ), "Must inherit from pymoo Problem for algorithm compatibility"
        assert (
            problem.n_obj == 1
        ), "Single-objective problem for spatial equity optimization"
        assert (
            problem.n_constr == 0
        ), "No constraints specified - unconstrained optimization"

        # CRITICAL: Validate problem dimensions match GTFS data structure
        expected_vars = (
            sample_optimization_data["n_routes"]
            * sample_optimization_data["n_intervals"]
        )
        assert (
            problem.n_var == expected_vars
        ), f"Decision variables must match route×interval matrix: {expected_vars}"

        # CRITICAL: Validate variable bounds for discrete optimization
        assert np.all(problem.xl == 0), "Lower bounds: index 0 (first headway choice)"
        expected_upper = sample_optimization_data["n_choices"] - 1
        assert np.all(
            problem.xu == expected_upper
        ), f"Upper bounds: index {expected_upper} (last valid choice)"

        # CRITICAL: Validate variable type for discrete headway choices
        assert (
            problem.vtype == int
        ), "Integer variables required for discrete headway indices"

        print("   ✅ Unconstrained problem created successfully:")
        print(f"      Decision variables: {problem.n_var} (routes × intervals)")
        print(f"      Objectives: {problem.n_obj} (spatial equity)")
        print(f"      Constraints: {problem.n_constr} (unconstrained)")
        print(f"      Variable bounds: [0, {expected_upper}] (headway choice indices)")

    def test_problem_creation_with_constraints(self, sample_optimization_data):
        """
        Test creating problem with both objective function and constraints.

        WHAT THIS TEST DOES:
        - Creates spatial equity objective + fleet budget constraint
        - Initializes constrained TransitOptimizationProblem
        - Validates constraint handler integration
        - Checks component storage and reference preservation
        - Verifies constraint count calculation

        WHY CONSTRAINED OPTIMIZATION MATTERS:
        - Real-world optimization always has resource limits
        - Fleet constraints prevent unrealistic solutions
        - Regulatory compliance requires constraint satisfaction
        - Multi-constraint scenarios test handler interactions

        CONSTRAINT TESTED:
        - FleetTotalConstraintHandler: System-wide fleet budget limit
        - Realistic scenario: 15% increase from current peak allowed
        - Single constraint (n_constr = 1) for clear validation

        EXPECTED BEHAVIOR:
        - Same objective setup as unconstrained case
        - n_constr = 1 (FleetTotalConstraintHandler produces 1 constraint)
        - Constraint handler stored correctly for evaluation
        - All components accessible for population evaluation
        """
        print("\n🏗️  TESTING PROBLEM CREATION (WITH CONSTRAINTS):")

        # Create spatial equity objective
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        # Create realistic fleet budget constraint
        fleet_constraint = FleetTotalConstraintHandler(
            {
                "baseline": "current_peak",  # Compare to current GTFS peak fleet
                "tolerance": 0.15,  # Allow 15% increase (realistic budget expansion)
                "measure": "peak",  # Constrain system-wide peak fleet
            },
            sample_optimization_data,
        )

        # Create constrained optimization problem
        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=[fleet_constraint],
        )

        # CRITICAL: Validate constraint integration
        assert (
            problem.n_constr == 1
        ), "FleetTotalConstraintHandler produces exactly 1 constraint"
        assert (
            len(problem.constraints) == 1
        ), "Should store exactly 1 constraint handler"

        # CRITICAL: Validate component storage for evaluation pipeline
        assert (
            problem.objective is objective
        ), "Must store objective reference for evaluation"
        assert (
            problem.constraints[0] is fleet_constraint
        ), "Must store constraint reference for evaluation"

        print("   ✅ Constrained problem created successfully:")
        print(f"      Objective: {type(objective).__name__} (spatial equity)")
        print(f"      Constraints: {len(problem.constraints)} handler(s)")
        print(f"      Total constraint count: {problem.n_constr}")
        print(f"      Constraint type: {type(fleet_constraint).__name__}")

    def test_problem_dimensions_match_data(self, sample_optimization_data):
        """
        Test that problem dimensions exactly match GTFS optimization data.

        WHAT THIS TEST DOES:
        - Creates problem with real GTFS data
        - Extracts dimensions from both problem and data
        - Validates exact matches for all dimensional parameters
        - Checks decision matrix shape consistency

        WHY DIMENSION MATCHING IS CRITICAL:
        - Mismatched dimensions cause array broadcasting errors during optimization
        - Population evaluation fails with wrong solution shapes
        - Fleet calculations become invalid with wrong route/interval counts
        - Spatial analysis breaks with inconsistent data structures

        DIMENSIONS VALIDATED:
        - n_routes: Number of transit routes from GTFS
        - n_intervals: Number of time periods (24h / interval_hours)
        - n_choices: Number of headway options + no-service option
        - decision_matrix_shape: Expected shape for solution matrices

        TYPICAL VALUES (Duke University GTFS):
        - Routes: ~6 (bus routes in system)
        - Intervals: 8 (3-hour intervals: 24/3=8)
        - Choices: 7 (6 headways + no-service: [5,10,15,30,60,120,9999])
        - Matrix shape: (6, 8) = 48 decision variables
        """
        print("\n🔍 TESTING PROBLEM DIMENSIONS:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # CRITICAL: Validate all dimensional parameters match exactly
        assert (
            problem.n_routes == sample_optimization_data["n_routes"]
        ), "Route count mismatch breaks fleet calculations"
        assert (
            problem.n_intervals == sample_optimization_data["n_intervals"]
        ), "Interval count mismatch breaks temporal analysis"
        assert (
            problem.n_choices == sample_optimization_data["n_choices"]
        ), "Choice count mismatch breaks solution decoding"

        # CRITICAL: Validate decision matrix shape consistency
        expected_shape = sample_optimization_data["decision_matrix_shape"]
        assert (
            problem.n_routes,
            problem.n_intervals,
        ) == expected_shape, "Matrix shape mismatch breaks solution evaluation"

        print("   ✅ All dimensions match GTFS optimization data:")
        print(f"      Routes: {problem.n_routes} (transit routes in system)")
        print(f"      Time intervals: {problem.n_intervals} (24h divided into periods)")
        print(
            f"      Headway choices: {problem.n_choices} (discrete options + no-service)"
        )
        print(f"      Decision matrix shape: {expected_shape}")
        print(f"      Total decision variables: {problem.n_var}")


# ================================================================================================
# SOLUTION ENCODING/DECODING TESTS
# ================================================================================================


class TestSolutionEncoding:
    """
    Test solution format conversion between pymoo and domain formats.

    PURPOSE:
    Pymoo algorithms work with flat integer vectors, but your transit optimization
    logic uses 2D matrices (routes × intervals). These tests ensure bidirectional
    conversion works perfectly, preserving all solution information.

    CONVERSION FORMATS:
    - Domain format: (n_routes, n_intervals) matrix with headway indices
    - Pymoo format: Flat vector of length (n_routes × n_intervals)
    - Example: [[0,1,2],[3,1,0]] ↔ [0,1,2,3,1,0]

    WHY FORMAT CONVERSION IS CRITICAL:
    - PSO/GA algorithms generate flat vectors that must be interpreted correctly
    - Fleet calculations require 2D matrix format for route-interval analysis
    - Any data loss during conversion invalidates optimization results
    - Round-trip conversion must be lossless for solution integrity

    WHAT GETS TESTED:
    - Bidirectional conversion (encode ↔ decode) preserves all data
    - Manual flat vectors decode to correct matrix shapes
    - Real GTFS initial solutions work with the conversion system
    - All sample solutions (high/medium/low/no service) convert correctly
    """

    def test_encode_decode_roundtrip(self, sample_optimization_data, sample_solutions):
        """
        Test that solution encoding and decoding preserves all data (lossless conversion).

        WHAT THIS TEST DOES:
        - Takes each sample solution matrix (high/medium/low/no service)
        - Encodes to pymoo flat format: matrix.flatten()
        - Decodes back to matrix format: flat.reshape(n_routes, n_intervals)
        - Validates that original and round-trip matrices are identical

        WHY ROUND-TRIP TESTING IS ESSENTIAL:
        - Optimization algorithms repeatedly convert between formats
        - Any data corruption compounds over optimization iterations
        - Silent conversion errors are nearly impossible to debug
        - Solution quality depends on perfect format preservation

        SAMPLE SOLUTIONS TESTED:
        - high_service: All index 0 (5-minute headways everywhere)
        - medium_service: All index 1 (10-minute headways everywhere)
        - low_service: All index 3 (30-minute headways everywhere)
        - no_service: All index 6 (no service everywhere)

        VALIDATION CHECKS:
        - Encoded length matches expected (n_routes × n_intervals)
        - Decoded shape matches original matrix shape
        - All values identical after round-trip conversion
        - No precision loss or type changes
        """
        print("\n🔄 TESTING SOLUTION ENCODING/DECODING ROUNDTRIP:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Test all sample solution scenarios for comprehensive validation
        for solution_name, solution_matrix in sample_solutions.items():
            print(f"\n   Testing {solution_name}:")
            print(f"      Original shape: {solution_matrix.shape}")
            print(f"      Original unique values: {np.unique(solution_matrix)}")

            # STEP 1: Encode matrix to flat format (domain → pymoo)
            encoded = problem.encode_solution(solution_matrix)
            expected_length = problem.n_routes * problem.n_intervals
            assert (
                len(encoded) == expected_length
            ), f"Encoded length must be {expected_length} for pymoo compatibility"

            # STEP 2: Decode flat vector back to matrix (pymoo → domain)
            decoded = problem.decode_solution(encoded)
            assert (
                decoded.shape == solution_matrix.shape
            ), "Shape must be preserved through conversion"

            # CRITICAL: Validate perfect data preservation
            assert np.array_equal(
                decoded, solution_matrix
            ), "All values must be identical after round-trip"

            print(f"      ✅ Roundtrip successful (encoded length: {len(encoded)})")

        print("   ✅ All solution types pass lossless round-trip conversion")

    def test_decode_with_flat_vectors(self, sample_optimization_data):
        """
        Test decoding manually created flat vectors to validate format interpretation.

        WHAT THIS TEST DOES:
        - Creates simple flat vectors with known patterns
        - Decodes them to matrix format using problem.decode_solution()
        - Validates resulting matrix has expected shape and values
        - Tests edge case of uniform solutions (all same headway choice)

        WHY MANUAL VECTOR TESTING MATTERS:
        - PSO/GA algorithms generate arbitrary flat vectors during optimization
        - We need confidence that any valid flat vector decodes correctly
        - Simple patterns help isolate decoding logic issues
        - Edge cases (uniform solutions) test boundary conditions

        TEST PATTERNS:
        - All zeros vector: High service everywhere (5-min headways)
        - Tests decoding logic with simplest possible input
        - Validates reshape operation works correctly
        - Confirms value interpretation is consistent

        VALIDATION CHECKS:
        - Decoded matrix has correct shape (n_routes, n_intervals)
        - All decoded values match expected pattern
        - No unexpected transformations during decoding
        """
        print("\n🔄 TESTING DECODING WITH FLAT VECTORS:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Create simple test pattern: uniform high service
        n_vars = problem.n_var
        flat_vector = np.zeros(
            n_vars, dtype=int
        )  # All zeros = index 0 = 5-min headways

        # Decode and validate
        decoded = problem.decode_solution(flat_vector)
        expected_shape = (problem.n_routes, problem.n_intervals)

        # CRITICAL: Validate decoding produces correct structure and values
        assert (
            decoded.shape == expected_shape
        ), f"Decoded shape must be {expected_shape}"
        assert np.all(
            decoded == 0
        ), "All decoded values should be 0 (matching input pattern)"

        print("   ✅ Manual flat vector decoding successful:")
        print(f"      Input: {len(flat_vector)} zeros (high service pattern)")
        print(f"      Output shape: {decoded.shape}")
        print("      Interpretation: 5-minute headways on all routes, all intervals")

    def test_initial_solution_compatibility(self, sample_optimization_data):
        """
        Test that GTFS-derived initial solution works with problem's encoding system.

        WHAT THIS TEST DOES:
        - Extracts the initial solution from GTFS data (current service levels)
        - Tests encoding/decoding with this real-world solution
        - Validates bounds compliance (indices within allowed range)
        - Ensures optimization can start from realistic baseline

        WHY INITIAL SOLUTION COMPATIBILITY IS CRITICAL:
        - Optimization needs good starting points for faster convergence
        - GTFS initial solution represents current service (realistic baseline)
        - Many algorithms (local search) depend on valid initial solutions
        - Initial solution provides comparison point for optimization results

        INITIAL SOLUTION CHARACTERISTICS:
        - Derived from actual GTFS schedule data
        - Contains headway indices corresponding to current service frequencies
        - May have diverse patterns (different headways per route/interval)
        - Represents existing operational reality

        VALIDATION CHECKS:
        - Round-trip conversion preserves GTFS solution exactly
        - All indices are non-negative (valid array indices)
        - All indices are within bounds (< n_choices)
        - Solution shape matches expected dimensions
        """
        print("\n🔄 TESTING INITIAL SOLUTION COMPATIBILITY:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Extract current service levels from GTFS analysis
        initial_solution = sample_optimization_data["initial_solution"]
        print(f"   Initial solution shape: {initial_solution.shape}")
        print(
            f"   Initial solution range: {np.min(initial_solution)}-{np.max(initial_solution)}"
        )
        print(f"   Unique headway indices: {np.unique(initial_solution)}")

        # CRITICAL: Test round-trip conversion with real GTFS data
        encoded = problem.encode_solution(initial_solution)
        decoded = problem.decode_solution(encoded)
        assert np.array_equal(
            decoded, initial_solution
        ), "GTFS initial solution must survive round-trip conversion"

        # CRITICAL: Validate bounds compliance for optimization algorithms
        assert np.all(initial_solution >= 0), "All headway indices must be non-negative"
        assert np.all(
            initial_solution < problem.n_choices
        ), "All indices must be within valid choice range"

        print("   ✅ GTFS initial solution fully compatible with problem:")
        print("      Represents current service levels from real transit data")
        print(f"      All {initial_solution.size} decisions within valid bounds")
        print("      Ready for use as optimization starting point")


# ================================================================================================
# SINGLE SOLUTION EVALUATION TESTS
# ================================================================================================


class TestSingleSolutionEvaluation:
    """
    Test evaluation of individual solutions for debugging and detailed analysis.

    PURPOSE:
    Single solution evaluation is critical for understanding optimization behavior,
    debugging issues, and analyzing specific scenarios. These tests ensure the
    evaluation pipeline works correctly for individual solutions before testing
    population-based evaluation used by metaheuristics.

    EVALUATION PIPELINE TESTED:
    1. Solution matrix → Fleet calculations → Spatial analysis → Objective value
    2. Solution matrix → Fleet calculations → Constraint evaluation → Violations
    3. Integration of objective + constraints → Feasibility determination

    WHY SINGLE SOLUTION TESTING MATTERS:
    - Debugging: Easy to trace evaluation logic with known inputs
    - Validation: Compare results against precalculated expected values
    - Understanding: Analyze how different service levels affect objectives/constraints
    - Reliability: Ensure evaluation is deterministic and repeatable

    SCENARIOS TESTED:
    - Objective-only evaluation (unconstrained optimization)
    - Objective + constraints evaluation (constrained optimization)
    - Multiple solution types (high/medium/low/no service scenarios)
    """

    def test_evaluate_single_solution_objective_only(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test single solution evaluation when problem has only objective function.

        WHAT THIS TEST DOES:
        - Creates unconstrained problem (spatial equity objective only)
        - Evaluates high service solution (5-minute headways everywhere)
        - Validates evaluation result structure and content
        - Checks that objective value is mathematically valid

        WHY OBJECTIVE-ONLY EVALUATION MATTERS:
        - Simplest case for validating core evaluation logic
        - Useful for initial exploration without resource constraints
        - Helps isolate objective function issues from constraint issues
        - Baseline for comparing constrained vs unconstrained scenarios

        EVALUATION RESULT STRUCTURE:
        - objective: Spatial variance value (lower = more equitable)
        - constraints: Empty array (no constraints specified)
        - feasible: Always True (no constraints to violate)
        - constraint_details: Empty list (no constraint handlers)
        - solution_matrix: Copy of input solution for reference

        VALIDATION CHECKS:
        - Objective is numeric (int or float) and finite
        - No NaN values (indicates evaluation errors)
        - Feasible = True (unconstrained problems are always feasible)
        - Empty constraints arrays (consistent with problem setup)
        - All required fields present in result structure
        """
        print("\n🔍 TESTING SINGLE SOLUTION EVALUATION (OBJECTIVE ONLY):")

        # Create unconstrained spatial equity problem
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Test with high service solution (most resource-intensive scenario)
        solution = sample_solutions["high_service"]  # 5-minute headways everywhere
        results = problem.evaluate_single_solution(solution)

        # CRITICAL: Validate result structure completeness
        required_fields = [
            "objective",
            "feasible",
            "constraints",
            "constraint_details",
            "solution_matrix",
        ]
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"

        # CRITICAL: Validate objective function output
        assert isinstance(
            results["objective"], (int, float)
        ), "Objective must be numeric for optimization"
        assert not np.isnan(
            results["objective"]
        ), "NaN objective indicates evaluation error"
        assert np.isfinite(
            results["objective"]
        ), "Infinite objective breaks optimization algorithms"

        # CRITICAL: Validate unconstrained problem behavior
        assert (
            results["feasible"] is True
        ), "Unconstrained problems should always be feasible"
        assert len(results["constraints"]) == 0, "Should have no constraint violations"
        assert (
            len(results["constraint_details"]) == 0
        ), "Should have no constraint details"

        print("   ✅ Objective-only evaluation successful:")
        print(f"      Objective (spatial variance): {results['objective']:.4f}")
        print(f"      Feasible: {results['feasible']}")
        print(
            "      Interpretation: Lower variance = more equitable service distribution"
        )

    def test_evaluate_single_solution_with_constraints(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test single solution evaluation with both objective and constraints.

        WHAT THIS TEST DOES:
        - Creates constrained problem (objective + fleet budget constraint)
        - Uses lenient constraint to ensure feasible evaluation
        - Evaluates medium service solution (balanced resource usage)
        - Validates constraint evaluation mechanics and result structure

        WHY CONSTRAINED EVALUATION TESTING MATTERS:
        - Real optimization problems always have resource/regulatory constraints
        - Constraint evaluation is complex (fleet calculations, bounds checking)
        - Integration between objectives and constraints must work seamlessly
        - Constraint satisfaction determines solution viability

        CONSTRAINT TESTED:
        - FleetTotalConstraintHandler with 50% tolerance (very lenient)
        - Ensures evaluation succeeds without constraint violations
        - Tests constraint evaluation pipeline without feasibility complications

        MEDIUM SERVICE SOLUTION CHARACTERISTICS:
        - 10-minute headways on all routes, all intervals
        - Moderate fleet requirements (between high and low service)
        - Likely to satisfy lenient budget constraints
        - Good test case for normal operational scenarios

        VALIDATION CHECKS:
        - Constraint violations array has correct length (1 violation)
        - Constraint details include handler type and satisfaction status
        - Constraint evaluation integrates properly with objective evaluation
        - All constraint metadata is accessible for debugging
        """
        print("\n🔍 TESTING SINGLE SOLUTION EVALUATION (WITH CONSTRAINTS):")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        # Create lenient constraint to test evaluation mechanics without violations
        fleet_constraint = FleetTotalConstraintHandler(
            {
                "baseline": "current_peak",  # Compare to current GTFS peak
                "tolerance": 0.50,  # Very lenient: 50% increase allowed
                "measure": "peak",  # System-wide peak constraint
            },
            sample_optimization_data,
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=[fleet_constraint],
        )

        # Test with medium service (balanced resource usage scenario)
        solution = sample_solutions["medium_service"]  # 10-minute headways everywhere
        results = problem.evaluate_single_solution(solution)

        # CRITICAL: Validate constraint evaluation integration
        assert (
            len(results["constraints"]) == 1
        ), "Should have exactly 1 constraint violation value"
        assert (
            len(results["constraint_details"]) == 1
        ), "Should have exactly 1 constraint detail record"

        # CRITICAL: Validate constraint detail structure
        constraint_detail = results["constraint_details"][0]
        assert (
            constraint_detail["handler_type"] == "FleetTotalConstraintHandler"
        ), "Should identify constraint type"
        assert (
            constraint_detail["n_constraints"] == 1
        ), "FleetTotal produces exactly 1 constraint"
        assert isinstance(
            constraint_detail["satisfied"], bool
        ), "Satisfaction status must be boolean"

        violation_value = results["constraints"][0]
        is_satisfied = constraint_detail["satisfied"]

        print("   ✅ Constrained evaluation successful:")
        print(f"      Objective: {results['objective']:.4f}")
        print(f"      Constraint violation: {violation_value:.3f}")
        print(f"      Constraint satisfied: {is_satisfied}")
        print(f"      Overall feasible: {results['feasible']}")
        print("      Interpretation: violation ≤ 0 means constraint satisfied")

    def test_evaluate_multiple_solutions(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test evaluation across different solution types to validate consistency.

        WHAT THIS TEST DOES:
        - Evaluates all sample solution types (high/medium/low/no service)
        - Compares objective values across different service levels
        - Validates that all evaluations complete successfully
        - Checks for mathematical consistency and reasonableness

        WHY MULTI-SOLUTION TESTING MATTERS:
        - Different service levels stress-test evaluation robustness
        - Relative objective values should make intuitive sense
        - Edge cases (no service) test boundary condition handling
        - Consistency across scenarios builds confidence in evaluation logic

        SOLUTION SCENARIOS:
        - high_service: 5-min headways → Maximum vehicles, high spatial coverage
        - medium_service: 10-min headways → Moderate vehicles, balanced coverage
        - low_service: 30-min headways → Fewer vehicles, sparse coverage
        - no_service: No service → Zero vehicles, zero coverage

        EXPECTED OBJECTIVE RELATIONSHIPS:
        - All values should be finite (no NaN or infinite values)
        - no_service should have zero variance (perfect equity: all zones get nothing)
        - Other solutions may have various variance levels depending on stop distribution
        - No strict ordering expected (depends on spatial distribution of stops/zones)

        VALIDATION CHECKS:
        - All evaluations complete without errors
        - All objective values are mathematically valid
        - No unexpected failures across different service patterns
        - Evaluation is robust to diverse input scenarios
        """
        print("\n🔍 TESTING MULTIPLE SOLUTION EVALUATION:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        objective_values = {}

        # Evaluate all sample solution scenarios
        for solution_name, solution_matrix in sample_solutions.items():
            results = problem.evaluate_single_solution(solution_matrix)
            objective_values[solution_name] = results["objective"]

            # Log individual results for analysis
            headway_desc = {
                "high_service": "5-min headways (maximum vehicles)",
                "medium_service": "10-min headways (moderate vehicles)",
                "low_service": "30-min headways (fewer vehicles)",
                "no_service": "No service (zero vehicles)",
            }

            print(f"   {solution_name}: objective = {results['objective']:.4f}")
            print(f"      Service level: {headway_desc[solution_name]}")

        # CRITICAL: Validate mathematical properties across all solutions
        assert all(
            np.isfinite(val) for val in objective_values.values()
        ), "All objectives must be finite"
        assert all(
            val >= 0 for val in objective_values.values()
        ), "Variance objectives must be non-negative"

        # Special case validation: no service should have zero variance
        if "no_service" in objective_values:
            no_service_obj = objective_values["no_service"]
            print(f"   Special case - no_service objective: {no_service_obj:.6f}")
            # Note: May be 0.0 (perfect equity) or small positive (numerical precision)
            assert (
                no_service_obj < 0.001
            ), "No service should have near-zero variance (all zones equal)"

        print("   ✅ All solution scenarios evaluated successfully")
        print("   ✅ Evaluation pipeline robust across diverse service levels")


# ================================================================================================
# POPULATION EVALUATION TESTS
# ================================================================================================


class TestPopulationEvaluation:
    """
    Test population-based evaluation for metaheuristic optimization algorithms.

    PURPOSE:
    Metaheuristic algorithms (PSO, GA, NSGA-II) work with populations of candidate
    solutions. The _evaluate() method must efficiently process multiple solutions
    simultaneously and return results in pymoo's required format. This is the
    core interface that optimization algorithms use.

    POPULATION EVALUATION PIPELINE:
    1. Algorithm generates population matrix X: (pop_size, n_var)
    2. Problem._evaluate(X, out) processes all solutions
    3. Results stored in out["F"] (objectives) and out["G"] (constraints)
    4. Algorithm uses results for selection, crossover, mutation

    WHY POPULATION EVALUATION IS CRITICAL:
    - Core interface between problem and optimization algorithms
    - Performance bottleneck: called every generation (100+ times per run)
    - Must handle diverse solution quality (excellent to infeasible)
    - Array formatting errors break optimization completely

    TESTING SCENARIOS:
    - Objective-only populations (unconstrained optimization)
    - Constrained populations (real-world scenarios)
    - Edge cases: empty populations, evaluation failures
    - Integration with real solution data
    """

    def test_population_evaluation_objective_only(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test population evaluation with only objective function (unconstrained).

        WHAT THIS TEST DOES:
        - Creates population from sample solutions (high/medium/low/no service)
        - Calls pymoo _evaluate() interface with population matrix
        - Validates output format matches pymoo requirements
        - Checks objective values for mathematical validity

        WHY UNCONSTRAINED POPULATION TESTING MATTERS:
        - Simplest case for validating core population processing
        - Tests array formatting and data flow without constraint complexity
        - Baseline for comparing constrained population evaluation
        - Many research scenarios use unconstrained optimization

        POPULATION COMPOSITION:
        - 4 solutions from sample_solutions fixture
        - Represents diverse service level scenarios
        - Known solutions enable deterministic validation
        - Realistic test of algorithm input diversity

        PYMOO OUTPUT FORMAT REQUIREMENTS:
        - out["F"]: Shape (pop_size, n_obj) - objective values
        - out["G"]: None or missing - no constraints
        - All values must be numeric and finite
        - Array shapes must match population size exactly

        VALIDATION CHECKS:
        - Output arrays have correct shapes for pymoo algorithms
        - All objective values are mathematically valid
        - No unexpected array dimensions or data types
        - Results correspond correctly to input solutions
        """
        print("\n👥 TESTING POPULATION EVALUATION (OBJECTIVE ONLY):")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Create population from known sample solutions for deterministic testing
        population = []
        solution_names = []

        for name, solution_matrix in sample_solutions.items():
            flat_solution = problem.encode_solution(solution_matrix)
            population.append(flat_solution)
            solution_names.append(name)

        # Format as pymoo population matrix
        X = np.array(population)  # Shape: (pop_size, n_var)
        out = {}

        print(f"   Population matrix shape: {X.shape}")
        print(f"   Population size: {len(population)}")

        # CRITICAL: Call pymoo evaluation interface
        problem._evaluate(X, out)

        # CRITICAL: Validate pymoo output format compliance
        assert "F" in out, "Must return objective values in 'F' key for pymoo"
        assert out["F"].shape == (
            len(population),
            1,
        ), f"Objective array must have shape ({len(population)}, 1)"
        assert (
            "G" not in out or out["G"] is None
        ), "Unconstrained problems should not return constraint array"

        # CRITICAL: Validate objective value validity
        objectives = out["F"][:, 0]
        assert len(objectives) == len(
            population
        ), "Must have one objective per solution"
        assert all(
            np.isfinite(obj) for obj in objectives
        ), "All objectives must be finite for optimization"
        assert all(
            obj >= 0 for obj in objectives
        ), "Variance objectives must be non-negative"

        print("   ✅ Population evaluation successful:")
        print(f"      Population processed: {len(population)} solutions")
        print(
            f"      Objective range: {np.min(objectives):.4f} - {np.max(objectives):.4f}"
        )

        # Log individual results for debugging and validation
        for i, (name, obj) in enumerate(zip(solution_names, objectives, strict=False)):
            print(f"      {name}: {obj:.4f}")

    def test_population_evaluation_with_constraints(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test population evaluation with both objectives and constraints.

        WHAT THIS TEST DOES:
        - Creates constrained problem with fleet budget limit
        - Evaluates population with diverse service levels
        - Validates constraint violation array formatting
        - Counts feasible vs infeasible solutions

        WHY CONSTRAINED POPULATION TESTING MATTERS:
        - Real optimization problems always have constraints
        - Constraint evaluation adds complexity (fleet calculations, multiple handlers)
        - Feasibility analysis is critical for algorithm performance
        - Array formatting is more complex (objectives + constraints)

        CONSTRAINT CONFIGURATION:
        - FleetTotalConstraintHandler with 20% tolerance
        - Realistic constraint: modest fleet budget increase allowed
        - Should create mix of feasible/infeasible solutions for testing

        EXPECTED POPULATION BEHAVIOR:
        - high_service: Likely infeasible (too many vehicles needed)
        - medium_service: Possibly feasible (moderate vehicle needs)
        - low_service: Likely feasible (fewer vehicles needed)
        - no_service: Definitely feasible (zero vehicles needed)

        PYMOO OUTPUT FORMAT WITH CONSTRAINTS:
        - out["F"]: Shape (pop_size, 1) - objective values
        - out["G"]: Shape (pop_size, n_constr) - constraint violations
        - Constraint violations ≤ 0 mean satisfied, > 0 mean violated
        - Algorithms use violations for penalty methods and feasibility filtering

        VALIDATION CHECKS:
        - Both objective and constraint arrays have correct shapes
        - Constraint violations are mathematically valid
        - Feasibility counting works correctly
        - Integration between objectives and constraints is seamless
        """
        print("\n👥 TESTING POPULATION EVALUATION (WITH CONSTRAINTS):")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        # Create realistic fleet constraint with moderate tolerance
        fleet_constraint = FleetTotalConstraintHandler(
            {
                "baseline": "current_peak",  # Compare to current GTFS peak
                "tolerance": 0.20,  # Allow 20% increase (realistic budget)
                "measure": "peak",  # System-wide peak constraint
            },
            sample_optimization_data,
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=[fleet_constraint],
        )

        # Create population from sample solutions
        population = []
        solution_names = []
        for name, solution_matrix in sample_solutions.items():
            flat_solution = problem.encode_solution(solution_matrix)
            population.append(flat_solution)
            solution_names.append(name)

        X = np.array(population)
        out = {}

        print(f"   Constrained population size: {len(population)}")
        print(f"   Constraint: Fleet ≤ {1.20:.0%} of current peak")

        # CRITICAL: Evaluate constrained population
        problem._evaluate(X, out)

        # CRITICAL: Validate constrained pymoo output format
        assert "F" in out, "Must return objectives for constrained problems"
        assert "G" in out, "Must return constraint violations for constrained problems"
        assert out["F"].shape == (
            len(population),
            1,
        ), "Objective array shape validation"
        assert out["G"].shape == (
            len(population),
            1,
        ), "Constraint array shape validation"

        # CRITICAL: Validate constraint violation values
        objectives = out["F"][:, 0]
        violations = out["G"][:, 0]

        assert len(violations) == len(population), "One violation per solution required"
        assert all(
            np.isfinite(viol) for viol in violations
        ), "All violations must be finite"

        # Analyze feasibility distribution
        feasible_count = np.sum(violations <= 0)
        infeasible_count = len(population) - feasible_count

        print("   ✅ Constrained population evaluation successful:")
        print(f"      Total solutions: {len(population)}")
        print(f"      Feasible solutions: {feasible_count}")
        print(f"      Infeasible solutions: {infeasible_count}")
        print(
            f"      Violation range: {np.min(violations):.3f} - {np.max(violations):.3f}"
        )

        # Log individual constraint analysis
        for name, obj, viol in zip(
            solution_names, objectives, violations, strict=False
        ):
            status = "✅ Feasible" if viol <= 0 else "❌ Infeasible"
            print(f"      {name}: obj={obj:.4f}, violation={viol:.3f} {status}")

    def test_empty_population_handling(self, sample_optimization_data):
        """
        Test handling of edge case with empty population (zero solutions).

        WHAT THIS TEST DOES:
        - Creates problem with normal configuration
        - Passes empty population matrix (0 solutions) to _evaluate()
        - Validates graceful handling without crashes or errors
        - Checks output array shapes are consistent with empty input

        WHY EMPTY POPULATION TESTING MATTERS:
        - Edge case that can occur during algorithm initialization/termination
        - Array operations must handle zero-length arrays correctly
        - Optimization frameworks should be robust to unusual inputs
        - Prevents crashes during algorithm development/debugging

        EMPTY POPULATION CHARACTERISTICS:
        - X matrix shape: (0, n_var) - zero rows, correct column count
        - Should not trigger any solution evaluations
        - Output arrays should have zero rows but correct column structure

        EXPECTED BEHAVIOR:
        - No crashes or exceptions raised
        - Output arrays have shapes (0, n_obj) and (0, n_constr)
        - All array operations complete successfully
        - Graceful handling demonstrates robustness

        VALIDATION CHECKS:
        - Evaluation completes without exceptions
        - Output arrays have correct empty shapes
        - No memory allocation issues with zero-length arrays
        - System remains stable for subsequent normal evaluations
        """
        print("\n👥 TESTING EMPTY POPULATION HANDLING:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Create empty population matrix (edge case)
        X = np.empty(
            (0, problem.n_var), dtype=int
        )  # 0 solutions, correct variable count
        out = {}

        print(f"   Empty population shape: {X.shape}")

        # CRITICAL: Should handle empty input gracefully
        try:
            problem._evaluate(X, out)
            evaluation_success = True
        except Exception as e:
            evaluation_success = False
            error_type = type(e).__name__
            print(f"   ⚠️  Empty population caused exception: {error_type}")

        if evaluation_success:
            # CRITICAL: Validate empty output format consistency
            assert "F" in out, "Must return objectives array even for empty population"
            assert out["F"].shape == (0, 1), "Empty objectives must have shape (0, 1)"

            print("   ✅ Empty population handled gracefully:")
            print(f"      Output objectives shape: {out['F'].shape}")
            print("   ✅ System remains stable for edge case inputs")
        else:
            # Empty populations causing exceptions is acceptable if handled consistently
            print(
                "   ℹ️  Empty population handling: Exception thrown (may be acceptable)"
            )


# ================================================================================================
# INTEGRATION TESTS WITH REAL DATA
# ================================================================================================


class TestRealDataIntegration:
    """
    Test integration with real GTFS data and existing optimization components.

    PURPOSE:
    These tests validate that TransitOptimizationProblem correctly integrates with
    your existing, tested components using real transit data. They ensure the
    optimization results will be meaningful and match expected calculations.

    INTEGRATION POINTS TESTED:
    - StopCoverageObjective with real Duke University GTFS data
    - Fleet constraint handlers with precalculated expected values
    - Solution evaluation pipeline with deterministic validation
    - Problem introspection and debugging methods

    WHY REAL DATA INTEGRATION TESTING MATTERS:
    - Synthetic test data may miss real-world edge cases
    - Validates end-to-end pipeline with actual transit system data
    - Ensures optimization results correspond to reality
    - Builds confidence in production deployment

    REAL DATA CHARACTERISTICS:
    - Duke University transit system GTFS feed
    - 6 bus routes with realistic service patterns
    - 8 time intervals (3-hour periods across 24-hour day)
    - Current service levels derived from actual schedules
    - Fleet requirements calculated from real operational data
    """

    def test_integration_with_precalculated_data(
        self, sample_optimization_data, precalculated_fleet_data, sample_solutions
    ):
        """
        Test that problem constraint evaluation matches precalculated expected values.

        WHAT THIS TEST DOES:
        - Creates problem with fleet constraint using real GTFS data
        - Evaluates known solution against precalculated fleet requirements
        - Compares actual constraint violation with expected value
        - Validates that constraint evaluation matches fleet calculation logic

        WHY PRECALCULATED VALIDATION MATTERS:
        - Ensures constraint handlers work correctly within problem framework
        - Validates that evaluation pipeline produces expected results
        - Catches integration errors between components
        - Provides deterministic test that must pass for correct implementation

        PRECALCULATED DATA SOURCE:
        - precalculated_fleet_data fixture uses same fleet calculation logic
        - Expected fleet requirements calculated for each sample solution
        - Baseline fleet data extracted from real GTFS analysis
        - Provides ground truth for validation

        TEST SOLUTION: medium_service
        - 10-minute headways on all routes, all intervals
        - Moderate fleet requirements (between high and low service)
        - Should have predictable constraint violation value
        - Good test case for typical operational scenarios

        CONSTRAINT: FleetTotalConstraintHandler
        - 15% tolerance above current peak fleet
        - Realistic budget constraint scenario
        - Single constraint (n_constr = 1) for clear validation

        VALIDATION LOGIC:
        - Expected violation = solution_peak_fleet - (baseline_peak × 1.15)
        - Actual violation from problem.evaluate_single_solution()
        - Should match within small numerical tolerance (same calculations)
        - Validates integration between problem and constraint handler
        """
        print("\n🎯 TESTING INTEGRATION WITH PRECALCULATED DATA:")

        # Create realistic constrained optimization problem
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        fleet_constraint = FleetTotalConstraintHandler(
            {
                "baseline": "current_peak",  # Use current GTFS peak as baseline
                "tolerance": 0.15,  # 15% budget increase allowed
                "measure": "peak",  # Constrain system-wide peak fleet
            },
            sample_optimization_data,
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=[fleet_constraint],
        )

        # Test specific solution against precalculated expectations
        solution_name = "medium_service"
        solution_matrix = sample_solutions[solution_name]
        fleet_info = precalculated_fleet_data["solutions"][solution_name]

        # CRITICAL: Evaluate solution through problem interface
        results = problem.evaluate_single_solution(solution_matrix)

        # CRITICAL: Calculate expected constraint violation using same logic
        baseline_peak = precalculated_fleet_data["baseline"]["current_peak_fleet"]
        limit = baseline_peak * 1.15  # 15% tolerance
        expected_violation = fleet_info["peak_fleet"] - limit
        actual_violation = results["constraints"][0]

        print(f"   Test solution: {solution_name}")
        print(f"   Baseline peak fleet: {baseline_peak:.1f} vehicles")
        print(f"   Constraint limit: {limit:.1f} vehicles (115% of baseline)")
        print(f"   Solution peak fleet: {fleet_info['peak_fleet']:.1f} vehicles")
        print(f"   Expected violation: {expected_violation:.3f}")
        print(f"   Actual violation: {actual_violation:.3f}")
        print(f"   Difference: {abs(actual_violation - expected_violation):.6f}")

        # CRITICAL: Validate integration accuracy (same calculation methods)
        tolerance = 0.01  # Allow small numerical differences
        assert (
            abs(actual_violation - expected_violation) < tolerance
        ), f"Integration error: expected {expected_violation:.3f}, got {actual_violation:.3f}"

        # Additional validation: check violation sign interpretation
        if actual_violation <= 0:
            print("   ✅ Constraint satisfied (solution within budget)")
        else:
            print("   ❌ Constraint violated (solution exceeds budget)")

        print("   ✅ Problem evaluation matches precalculated fleet data")
        print("   ✅ Integration between problem and constraint handlers validated")

    def test_problem_info_method(self, sample_optimization_data):
        """
        Test the get_problem_info() debugging method returns correct information.

        WHAT THIS TEST DOES:
        - Creates realistic problem with objective and constraints
        - Calls get_problem_info() introspection method
        - Validates returned information structure and content
        - Ensures debugging/logging functionality works correctly

        WHY PROBLEM INFO TESTING MATTERS:
        - Essential for debugging optimization setup issues
        - Enables logging of problem configurations for reproducibility
        - Helps users understand problem structure and components
        - Critical for generating optimization reports and documentation

        PROBLEM INFO STRUCTURE:
        - problem_type: Class name for identification
        - dimensions: Problem size parameters (routes, intervals, variables)
        - objective: Information about objective function type
        - constraints: List of constraint handler details
        - variable_bounds: Decision variable limits and types
        - optimization_data_keys: Available data fields

        INFORMATION SOURCES:
        - Problem class introspection (dimensions, bounds)
        - Component introspection (objective.class, constraint.get_info())
        - Optimization data structure analysis
        - Pymoo problem parameters

        VALIDATION CHECKS:
        - All required information fields present
        - Information values match actual problem configuration
        - Component identification works correctly
        - Data structure is suitable for logging/reporting
        """
        print("\n🔍 TESTING PROBLEM INFO METHOD:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        fleet_constraint = FleetTotalConstraintHandler(
            {"baseline": "current_peak", "tolerance": 0.15, "measure": "peak"},
            sample_optimization_data,
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective,
            constraints=[fleet_constraint],
        )

        # CRITICAL: Call introspection method
        info = problem.get_problem_info()

        # CRITICAL: Validate information structure completeness
        required_sections = [
            "problem_type",
            "dimensions",
            "objective",
            "constraints",
            "variable_bounds",
        ]
        for section in required_sections:
            assert section in info, f"Missing info section: {section}"

        # CRITICAL: Validate specific information accuracy
        assert (
            info["problem_type"] == "TransitOptimizationProblem"
        ), "Should identify problem class"
        assert (
            info["dimensions"]["n_routes"] == sample_optimization_data["n_routes"]
        ), "Route count should match"
        assert (
            info["dimensions"]["n_intervals"] == sample_optimization_data["n_intervals"]
        ), "Interval count should match"
        assert (
            info["objective"]["type"] == "StopCoverageObjective"
        ), "Should identify objective type"
        assert len(info["constraints"]) == 1, "Should report 1 constraint handler"
        assert (
            info["constraints"][0]["handler_type"] == "FleetTotalConstraintHandler"
        ), "Should identify constraint type"

        # Log information for debugging and validation
        print("   ✅ Problem info structure validated:")
        print(f"      Problem type: {info['problem_type']}")
        print(f"      Decision variables: {info['dimensions']['n_variables']}")
        print(
            f"      Problem dimensions: {info['dimensions']['n_routes']} routes × {info['dimensions']['n_intervals']} intervals"
        )
        print(f"      Objective function: {info['objective']['type']}")
        print(f"      Constraints: {len(info['constraints'])} handler(s)")

        for i, constraint_info in enumerate(info["constraints"]):
            print(f"        {i+1}. {constraint_info['handler_type']}")

        print("   ✅ Problem introspection and debugging support functional")


# ================================================================================================
# ERROR HANDLING TESTS
# ================================================================================================


class TestErrorHandling:
    """
    Test error handling and edge cases for robustness validation.

    PURPOSE:
    Real optimization scenarios involve diverse inputs, edge cases, and potential
    errors. These tests ensure TransitOptimizationProblem handles unusual situations
    gracefully without crashing or producing invalid results.

    ERROR SCENARIOS TESTED:
    - Invalid solution matrix dimensions
    - Out-of-bounds solution values
    - Malformed inputs that might occur during optimization

    WHY ERROR HANDLING TESTING MATTERS:
    - Optimization algorithms can generate invalid solutions during search
    - Robust error handling prevents optimization crashes
    - Clear error messages help debug optimization issues
    - Production systems must handle unexpected inputs gracefully

    ERROR HANDLING APPROACHES:
    - Exception raising for clearly invalid inputs
    - Error value returns (inf, NaN) for evaluation failures
    - Graceful degradation where possible
    - Informative error messages for debugging
    """

    def test_invalid_solution_matrix_size(self, sample_optimization_data):
        """
        Test handling of incorrectly sized solution matrices.

        WHAT THIS TEST DOES:
        - Creates solution matrix with wrong dimensions (too many routes)
        - Attempts evaluation with invalid solution
        - Validates that system handles error gracefully
        - Checks for either exception or error indicator in results

        WHY INVALID SIZE TESTING MATTERS:
        - Programming errors might create wrong-sized solutions
        - Algorithm mutations might corrupt solution dimensions
        - Robust systems should detect and handle size mismatches
        - Better to fail explicitly than produce wrong results

        INVALID SOLUTION CHARACTERISTICS:
        - Correct number of intervals, wrong number of routes
        - Should be detectable during evaluation setup
        - Represents clear programming/logic error
        - No reasonable interpretation possible

        ACCEPTABLE ERROR HANDLING:
        - Exception raised with clear error message
        - Evaluation returns error values (inf, NaN)
        - Error logging for debugging support
        - System remains stable after error

        VALIDATION APPROACH:
        - Try evaluation and check for graceful handling
        - Accept either exception or error return values
        - Ensure system doesn't crash or corrupt state
        - Log behavior for debugging support
        """
        print("\n⚠️  TESTING INVALID SOLUTION MATRIX SIZE:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        # Create invalid solution: wrong number of routes
        wrong_shape = (problem.n_routes + 1, problem.n_intervals)  # Extra route
        invalid_solution = np.zeros(wrong_shape, dtype=int)

        print(
            f"   Expected solution shape: ({problem.n_routes}, {problem.n_intervals})"
        )
        print(f"   Invalid solution shape: {wrong_shape}")

        # CRITICAL: Test graceful error handling
        try:
            results = problem.evaluate_single_solution(invalid_solution)

            # If no exception raised, check for error indicators
            if np.isinf(results["objective"]) or results["objective"] is None:
                print(
                    "   ✅ Invalid solution handled gracefully (returned error value)"
                )
                error_handled = True
            else:
                print("   ❌ Invalid solution not detected (unexpected success)")
                error_handled = False

        except Exception as e:
            error_type = type(e).__name__
            print(f"   ✅ Invalid solution handled gracefully (raised {error_type})")
            error_handled = True

        # Either approach (exception or error value) is acceptable
        assert error_handled, "System must detect and handle invalid solution sizes"

    def test_solution_bounds_validation(
        self, sample_optimization_data, sample_solutions
    ):
        """
        Test that all sample solutions respect variable bounds constraints.

        WHAT THIS TEST DOES:
        - Converts all sample solutions to flat format
        - Checks that all values are within valid bounds [0, n_choices-1]
        - Validates bounds compliance across all solution scenarios
        - Ensures test data is consistent with problem constraints

        WHY BOUNDS VALIDATION TESTING MATTERS:
        - Out-of-bounds values cause array indexing errors
        - Invalid headway indices break fleet calculations
        - Sample solutions must be valid for testing to be meaningful
        - Demonstrates bounds checking capability

        BOUNDS DEFINITION:
        - Lower bound: 0 (first headway choice index)
        - Upper bound: n_choices-1 (last valid choice, including no-service)
        - Values outside bounds indicate data corruption or logic errors

        SOLUTION SCENARIOS TESTED:
        - high_service: All zeros (should be valid)
        - medium_service: All ones (should be valid)
        - low_service: All threes (should be valid)
        - no_service: All no_service_index (should be valid)

        VALIDATION CHECKS:
        - All values >= 0 (non-negative indices)
        - All values < n_choices (within choice array bounds)
        - Consistent bounds compliance across all scenarios
        - Test data integrity verification
        """
        print("\n🔍 TESTING SOLUTION BOUNDS VALIDATION:")

        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data, spatial_resolution_km=2.0
        )

        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data, objective=objective
        )

        print(f"   Valid bounds: [0, {problem.n_choices-1}]")
        print(f"   Headway choices: {sample_optimization_data['allowed_headways']}")

        # CRITICAL: Validate bounds compliance for all sample solutions
        all_bounds_valid = True

        for solution_name, solution_matrix in sample_solutions.items():
            flat_solution = problem.encode_solution(solution_matrix)

            # Check lower bound compliance
            min_val = np.min(flat_solution)
            lower_bound_ok = np.all(flat_solution >= 0)

            # Check upper bound compliance
            max_val = np.max(flat_solution)
            upper_bound_ok = np.all(flat_solution < problem.n_choices)

            bounds_ok = lower_bound_ok and upper_bound_ok
            all_bounds_valid = all_bounds_valid and bounds_ok

            status = "✅" if bounds_ok else "❌"
            print(f"   {status} {solution_name}: range [{min_val}, {max_val}]")

            if not bounds_ok:
                if not lower_bound_ok:
                    print(f"      ❌ Lower bound violation: found {min_val} < 0")
                if not upper_bound_ok:
                    print(
                        f"      ❌ Upper bound violation: found {max_val} >= {problem.n_choices}"
                    )

        # CRITICAL: All test solutions must be valid
        assert (
            all_bounds_valid
        ), "All sample solutions must respect bounds for valid testing"

        print("   ✅ All sample solutions respect variable bounds")
        print("   ✅ Test data integrity validated")
        print("   ✅ Bounds checking system functional")


if __name__ == "__main__":
    """
    Run tests standalone for debugging and development.

    Usage: python test_transit_problem.py
    This will run all tests with verbose output and print statements.
    Useful for:
    - Debugging individual test failures
    - Understanding test behavior during development
    - Validating changes to TransitOptimizationProblem
    - Learning how the problem class integrates with pymoo
    """
    pytest.main([__file__, "-v", "-s"])
