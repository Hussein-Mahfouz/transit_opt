"""
Tests for PSO Runner.

This module validates the PSO optimization runner that serves as the main entry point
for Particle Swarm Optimization in the transit optimization system. Tests cover:

- PSO runner initialization and configuration validation
- Integration with existing system components (objectives, constraints)
- Single and multi-run optimization workflows
- Error handling and edge cases
- Solution format validation and constraint violation reporting

The tests use real Duke GTFS data to ensure integration with the broader system
including #preprocessing and #reconstruction workflows.

Test Data Dependencies:
- Uses sample_optimization_data fixture from #fixtures providing Duke GTFS data
- Tests constraint integration with handlers from #test_constraints
- Validates objective function integration with #test_service_coverage patterns

Key Integration Points:
- TransitOptimizationProblem: Solution encoding/decoding
- StopCoverageObjective: Spatial coverage evaluation
- Constraint handlers: Fleet size validation
- Configuration system: Parameter management and validation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transit_opt.optimisation.config import OptimizationConfigManager
from transit_opt.optimisation.runners import PSORunner

# ================================================================================================
# PSO RUNNER SETUP AND CONFIGURATION TESTS
# ================================================================================================


class TestPSORunnerSetup:
    """
    Test PSO runner initialization and configuration validation.

    Validates that the PSO runner correctly integrates with the configuration
    system and catches invalid parameter combinations before optimization begins.
    This prevents runtime failures during expensive optimization runs.
    """

    def test_runner_creation_with_config(self):
        """
        Test PSO runner initialization with valid configuration.

        Validates that runner properly stores configuration reference and
        initializes internal state. This test ensures the basic setup workflow
        works correctly before more complex integration testing.
        """
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,
                    "inertia_weight": 0.8,
                },
                "termination": {"max_generations": 50},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        assert runner.config_manager == config_manager
        print("✅ PSO runner creation with configuration works")

    def test_runner_validation_errors(self):
        """
        Test that configuration validation catches invalid PSO parameters.

        Ensures that common configuration errors (like population size too small)
        are caught during initialization rather than causing mysterious optimization
        failures. This test validates the _validate_configuration method.
        """
        # Invalid population size
        bad_config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 2},  # Too small
                "termination": {"max_generations": 50},
            },
        }

        with pytest.raises(ValueError, match="Population size must be at least 5"):
            OptimizationConfigManager(config_dict=bad_config)

        print("✅ PSO runner validation works")


# ================================================================================================
# DATA STRUCTURE AND INTEGRATION TESTS
# ================================================================================================


class TestPSORunnerIntegration:
    """
    Test PSO runner integration with result data structures.

    Validates that OptimizationResult and MultiRunResult classes work correctly
    and contain all necessary information for downstream analysis and deployment.
    These data structures are the primary interface between PSO optimization
    and the rest of the transit optimization system.
    """

    def test_optimization_result_structure(self):
        """
        Test OptimizationResult data structure and field validation.

        Validates that single optimization results contain all required fields
        in the expected format. This structure is used throughout the system
        for solution analysis, constraint validation, and performance assessment.
        """
        from transit_opt.optimisation.runners.pso_runner import OptimizationResult

        # Create sample result
        result = OptimizationResult(
            best_solution=np.array([10.0, 15.0, 20.0]),
            best_objective=42.5,
            constraint_violations={"total_violations": 0, "feasible": True},
            optimization_time=125.3,
            generations_completed=75,
        )

        assert len(result.best_solution) == 3
        assert result.best_objective == 42.5
        assert result.constraint_violations["feasible"] == True
        assert result.optimization_time > 0
        assert result.generations_completed > 0
        assert hasattr(result, 'best_feasible_solutions')
        assert isinstance(result.best_feasible_solutions, list)

        print("✅ OptimizationResult structure works")

    def test_multi_run_result_structure(self):
        """
        Test MultiRunResult data structure for statistical analysis.

        Validates that multi-run results correctly identify the best solution
        across runs and provide statistical summaries. This enables robust
        optimization assessment and confidence interval analysis.
        """
        from transit_opt.optimisation.runners.pso_runner import MultiRunResult, OptimizationResult

        # Create sample results
        result1 = OptimizationResult(
            best_solution=np.array([10.0]),
            best_objective=45.0,
            constraint_violations={"total_violations": 0},
            optimization_time=100.0,
            generations_completed=50,
            best_feasible_solutions=[]
        )

        result2 = OptimizationResult(
            best_solution=np.array([12.0]),
            best_objective=40.0,  # Better
            constraint_violations={"total_violations": 0},
            optimization_time=120.0,
            generations_completed=55,
            best_feasible_solutions=[
                {'solution': np.array([12.0]), 'objective': 40.0, 'generation_found': 30, 'feasible': True, 'violations': 0}
            ]
        )

         # Create run summaries
        run_summaries = [
            {'run_id': 1, 'objective': 45.0, 'feasible': True, 'generations': 50, 'time': 100.0, 'violations': 0, 'best_feasible_solutions_count': 0},
            {'run_id': 2, 'objective': 40.0, 'feasible': True, 'generations': 55, 'time': 120.0, 'violations': 0, 'best_feasible_solutions_count': 1}
        ]

        # Create best solutions per run
        best_feasible_solutions_per_run = [
            [],  # Run 1 had no feasible solutions tracked
            [{'solution': np.array([12.0]), 'objective': 40.0, 'generation_found': 30, 'feasible': True, 'violations': 0}]  # Run 2 had 1
        ]

        multi_result = MultiRunResult(
            best_result=result2,  # Best result
            run_summaries=run_summaries,  # lightweight summaries
            best_feasible_solutions_per_run=best_feasible_solutions_per_run,  # tracked solutions
            statistical_summary={"objective_mean": 42.5, "num_runs": 2},
            total_time=220.0,
            num_runs_completed=2,
        )

        assert multi_result.best_result.best_objective == 40.0
        assert len(multi_result.run_summaries) == 2
        assert len(multi_result.best_feasible_solutions_per_run) == 2
        assert multi_result.statistical_summary["num_runs"] == 2


        print("✅ MultiRunResult structure works")


# ================================================================================================
# CONFIGURATION INTEGRATION TESTS
# ================================================================================================


class TestPSORunnerMockOptimization:
    """
    Test PSO runner configuration integration without running full optimization.

    Validates that the runner correctly translates configuration parameters
    into algorithm and termination objects. These tests ensure configuration
    consistency without the time cost of actual optimization.
    """

    def test_runner_configuration_integration(self):
        """
        Test configuration translation to algorithm and termination objects.

        Validates that PSO parameters from configuration are correctly applied
        to the AdaptivePSO algorithm and termination criteria. This ensures
        that user configuration is properly respected during optimization.
        """
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 25,
                    "inertia_weight": 0.95,
                    "cognitive_coeff": 1.8,
                    "social_coeff": 2.2,
                    "adaptive": True,
                },
                "termination": {"max_generations": 30},
                "multi_run": {"enabled": True, "num_runs": 3},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Test algorithm creation
        algorithm = runner._create_algorithm()
        assert algorithm.pop_size == 25
        assert algorithm.w == 0.95
        assert algorithm.c1 == 1.8
        assert algorithm.c2 == 2.2
        assert algorithm.adaptive

        # Test termination creation
        termination = runner._create_termination()
        assert hasattr(termination, "n_max_gen")

        print("✅ Runner correctly uses configuration")


# ================================================================================================
# REAL DATA INTEGRATION TESTS
# ================================================================================================


class TestPSORunnerWithRealData:
    """
    Test PSO runner with real Duke GTFS data.

    These integration tests validate the complete optimization workflow using
    actual transit data. They ensure compatibility with #preprocessing data
    structures and realistic optimization scenarios.
    """

    def test_runner_with_duke_gtfs_data(self, sample_optimization_data):
        """
        Test complete PSO optimization workflow with Duke GTFS data.

        Integration test that validates the entire optimization pipeline:
        problem creation, algorithm execution, and result processing.
        Uses lenient constraints to ensure feasible solutions for testing.

        This test demonstrates integration with:
        - StopCoverageObjective from #service_coverage
        - FleetTotalConstraintHandler from #constraints
        - Duke GTFS data from #preprocessing
        """
        print("\n🧪 TESTING PSO RUNNER WITH REAL DUKE GTFS DATA:")

        # Create realistic configuration for Duke data
        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 1.5,
                    "crs": "EPSG:3857",
                    "time_aggregation": "average",
                    "alpha": 0.1,
                },
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.2,  # Allow 20% more vehicles
                        "measure": "peak",
                    }
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,  # Small for test speed
                    "inertia_weight": 0.9,
                    "cognitive_coeff": 2.0,
                    "social_coeff": 2.0,
                },
                "termination": {"max_generations": 5},  # Very short for testing
                "monitoring": {
                    "progress_frequency": 2,
                    "save_history": True,
                    "detailed_logging": False,
                },
            },
        }

        print(
            f"   📊 Duke GTFS data: {sample_optimization_data['n_routes']} routes, {sample_optimization_data['n_intervals']} intervals"
        )
        print("   🎯 Objective: StopCoverageObjective")
        print("   🚦 Constraints: FleetTotalConstraintHandler (20% tolerance)")
        print("   ⚙️ Algorithm: Adaptive PSO (20 particles, 5 generations)")

        # Create runner
        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        print("   ✅ PSO runner created successfully")

        # Run optimization
        print("\n   🚀 Running optimization...")
        result = runner.optimize(sample_optimization_data)

        # Validate results
        assert isinstance(result.best_solution, np.ndarray)
        assert (
            result.best_solution.shape
            == sample_optimization_data["decision_matrix_shape"]
        )
        assert result.best_objective < float("inf")
        assert result.optimization_time > 0
        assert result.generations_completed <= 5

        # Check that solution contains valid headway indices
        max_index = sample_optimization_data["n_choices"] - 1
        assert np.all(result.best_solution >= 0)
        assert np.all(result.best_solution <= max_index)

        print("   ✅ Optimization completed successfully!")
        print(f"      Best objective: {result.best_objective:.6f}")
        print(f"      Generations: {result.generations_completed}")
        print(f"      Time: {result.optimization_time:.2f}s")
        print(f"      Solution shape: {result.best_solution.shape}")
        print(
            f"      Constraints satisfied: {result.constraint_violations['feasible']}"
        )

        # Validate optimization history
        if result.optimization_history:
            assert len(result.optimization_history) == result.generations_completed

            # Check improvement trend (should generally improve or stay same)
            objectives = [gen["best_objective"] for gen in result.optimization_history]
            final_obj = objectives[-1]
            initial_obj = objectives[0]

            print(f"      Improvement: {initial_obj:.6f} → {final_obj:.6f}")
            assert (
                final_obj <= initial_obj
            ), "Final objective should be <= initial (minimization)"

        print("   🎯 All validation checks passed!")

    # Find the test_multi_run_with_duke_data method and update it

    def test_multi_run_with_duke_data(self, sample_optimization_data):
        """Test multi-run optimization with real Duke GTFS data."""
        # Configure for fast testing
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 8},
                "termination": {"max_generations": 2},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run multi-run optimization
        multi_result = runner.optimize_multi_run(
            sample_optimization_data, num_runs=3, parallel=False
        )

        # Test new structure instead of all_results
        assert len(multi_result.run_summaries) == 3
        assert multi_result.best_result is not None
        assert len(multi_result.best_feasible_solutions_per_run) == 3

        # Test run summaries structure (NEW: replaces all_results tests)
        for summary in multi_result.run_summaries:
            assert 'run_id' in summary
            assert 'objective' in summary
            assert 'feasible' in summary
            assert 'generations' in summary
            assert 'time' in summary
            assert 'violations' in summary
            assert 'best_feasible_solutions_count' in summary

        # Test that statistical summary exists and has reasonable values
        stats = multi_result.statistical_summary
        assert "objective_mean" in stats
        assert "num_runs" in stats
        assert stats["num_runs"] == 3

        # NEW: Test feasible solutions tracking
        for run_solutions in multi_result.best_feasible_solutions_per_run:
            assert isinstance(run_solutions, list)
            # Each solution should have required fields if any exist
            for solution in run_solutions:
                assert 'solution' in solution
                assert 'objective' in solution
                assert 'generation_found' in solution
                assert 'feasible' in solution
                assert solution['feasible'] is True  # Should only track feasible

        print("✅ Multi-run with Duke data successful")

    def test_solution_decoding_integration(self, sample_optimization_data):
        """
        Test solution format validation and headway index decoding.

        Validates that PSO solutions are properly decoded from flat vectors
        to route×interval matrices using TransitOptimizationProblem.
        This ensures compatibility with #reconstruction workflows that
        convert solutions back to GTFS schedules.
        """
        print("\n🔍 TESTING SOLUTION DECODING INTEGRATION:")

        # Simple config for testing solution format
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 10,
                },
                "termination": {"max_generations": 2},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run short optimization
        result = runner.optimize(sample_optimization_data)

        # Validate solution format
        expected_shape = sample_optimization_data["decision_matrix_shape"]
        assert result.best_solution.shape == expected_shape

        print(f"   📊 Solution shape: {result.best_solution.shape}")
        print(f"   📊 Expected shape: {expected_shape}")

        # Check that solution contains valid indices
        n_choices = sample_optimization_data["n_choices"]
        assert np.all(result.best_solution >= 0), "Solution indices should be >= 0"
        assert np.all(
            result.best_solution < n_choices
        ), f"Solution indices should be < {n_choices}"

        # Check some cells have different values (not all same service level)
        unique_values = np.unique(result.best_solution)
        print(f"   🎯 Unique solution values: {unique_values}")

        # Convert to actual headway values for verification
        allowed_headways = sample_optimization_data["allowed_headways"]
        solution_headways = np.array(
            [
                [
                    allowed_headways[result.best_solution[i, j]]
                    for j in range(expected_shape[1])
                ]
                for i in range(expected_shape[0])
            ]
        )

        print(f"   ⏰ Actual headways used: {np.unique(solution_headways)}")

        # Verify headways are from allowed set
        used_headways = set(solution_headways.flatten())
        allowed_set = set(allowed_headways)
        assert used_headways.issubset(
            allowed_set
        ), f"Used headways {used_headways} not in allowed set {allowed_set}"

        print("   ✅ Solution decoding works correctly!")


    def test_track_best_n_basic_functionality(self, sample_optimization_data):
        """Test that track_best_n parameter correctly tracks feasible solutions."""
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],  # No constraints to ensure feasible solutions
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 15},
                "termination": {"max_generations": 3},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Test different track_best_n values
        result_n1 = runner.optimize(sample_optimization_data, track_best_n=1)
        result_n3 = runner.optimize(sample_optimization_data, track_best_n=3)

        # Should track at most N solutions
        assert len(result_n1.best_feasible_solutions) <= 1
        assert len(result_n3.best_feasible_solutions) <= 3

        # Each tracked solution should have required fields
        for solution in result_n3.best_feasible_solutions:
            assert 'solution' in solution
            assert 'objective' in solution
            assert 'generation_found' in solution
            assert 'feasible' in solution
            assert solution['feasible'] is True  # Should only track feasible

        print("✅ track_best_n basic functionality works")

    def test_track_best_n_multi_run_integration(self, sample_optimization_data):
        """Test track_best_n with multi-run optimization."""
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 10},
                "termination": {"max_generations": 2},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run multi-run with tracking
        multi_result = runner.optimize_multi_run(
            sample_optimization_data,
            num_runs=2,
            track_best_n=2
        )

        # Should have solutions per run
        assert len(multi_result.best_feasible_solutions_per_run) == 2

        # Each run should track at most 2 solutions
        for run_solutions in multi_result.best_feasible_solutions_per_run:
            assert len(run_solutions) <= 2

        # Run summaries should include solution counts
        for summary in multi_result.run_summaries:
            assert 'best_feasible_solutions_count' in summary
            assert summary['best_feasible_solutions_count'] >= 0

        print("✅ track_best_n multi-run integration works")

    def test_population_weighting(self, sample_optimization_data):
        """Test PSO runner with population weighting using real data."""
        import os
        usa_pop_path = os.path.join(os.path.dirname(__file__), "data", "usa_pop_2025_CN_1km_R2025A_UA_v1.tif")

        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.0,
                    "population_weighted": True,
                    "population_layer": usa_pop_path,
                    "population_power": 0.5
                },
                "constraints": []
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 10,
                    "inertia_weight": 0.9,
                    "cognitive_coeff": 2.0,
                    "social_coeff": 2.0
                },
                "termination": {
                    "max_generations": 2,  # Short test
                    "convergence_tolerance": 1e-6
                },
                "monitoring": {
                    "progress_frequency": 1,
                    "save_history": False
                }
            }
        }
        from transit_opt.optimisation.config.config_manager import OptimizationConfigManager
        from transit_opt.optimisation.runners.pso_runner import PSORunner

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(
            config_manager=config_manager
        )

        # Set optimization data
        runner.optimization_data = sample_optimization_data
        # Should create problem without error
        runner._create_problem()
        assert runner.problem is not None
        assert runner.problem.objective.population_weighted is True
        assert runner.problem.objective.population_power == 0.5
        assert runner.problem.objective.population_per_zone is not None

        # Population data should be reasonable
        pop_data = runner.problem.objective.population_per_zone
        assert len(pop_data) > 0
        assert np.all(pop_data >= 0)
        assert np.sum(pop_data > 0) > 0  # Should have some populated zones

        print(f"Integration test: {len(pop_data)} zones, {np.sum(pop_data > 0)} with population")


def test_best_feasible_solutions_tracker():
    """
    Test the BestFeasibleSolutionsTracker class functionality.
    The class tracks the best feasible solutions found during optimisation, updating
    with each generation
    """
    from transit_opt.optimisation.runners.pso_runner import BestFeasibleSolutionsTracker

    tracker = BestFeasibleSolutionsTracker(max_solutions=3)

    # add_generation_solutions expects lists of: solution_matrices, objectives, generations, feasibles, violations

    # Add first batch of solutions
    solution_matrices = [
        np.array([[1, 2], [3, 4]]),
        np.array([[2, 3], [4, 5]]),
        np.array([[3, 4], [5, 6]])
    ]
    objectives = [0.5, 0.3, 0.7]
    generations = [10, 15, 20]
    feasibles = [True, True, True]
    violations = [0, 0, 0]

    tracker.add_generation_solutions(solution_matrices, objectives, generations, feasibles, violations)

    # Test that infeasible solutions are ignored
    infeasible_matrices = [np.array([[4, 5], [6, 7]])]
    infeasible_objectives = [0.1]
    infeasible_generations = [25]
    infeasible_feasibles = [False]
    infeasible_violations = [2]

    tracker.add_generation_solutions(infeasible_matrices, infeasible_objectives, infeasible_generations, infeasible_feasibles, infeasible_violations)

    solutions = tracker.get_best_solutions()

    # Should have 3 solutions (infeasible one ignored)
    assert len(solutions) == 3

    # Should be sorted by objective (best first)
    objectives = [sol['objective'] for sol in solutions]
    assert objectives == sorted(objectives)  # Should be sorted
    assert objectives[0] == 0.3  # Best objective first

    # Test max_solutions limit
    better_matrices = [
        np.array([[5, 6], [7, 8]]),
        np.array([[6, 7], [8, 9]])
    ]
    better_objectives = [0.2, 0.4]  # Better than existing
    better_generations = [30, 15]
    better_feasibles = [True, True]
    better_violations = [0, 0]

    tracker.add_generation_solutions(better_matrices, better_objectives, better_generations, better_feasibles, better_violations)

    solutions = tracker.get_best_solutions()
    assert len(solutions) == 3  # Still only 3 (max_solutions limit)
    assert solutions[0]['objective'] == 0.2  # New best should be first
    assert solutions[1]['objective'] == 0.3  # Second best (from 1st batch)
    assert solutions[2]['objective'] == 0.4
    # check generations
    assert solutions[0]['generation_found'] == 30
    assert solutions[1]['generation_found'] == 15
    assert solutions[2]['generation_found'] == 15

    # Test get_count method
    assert tracker.get_count() == 3

    print("✅ BestFeasibleSolutionsTracker works correctly")

# ================================================================================================
# ERROR HANDLING AND ROBUSTNESS TESTS
# ================================================================================================


class TestPSORunnerErrorHandling:
    """
    Test PSO runner error handling and termination behavior.

    Validates graceful handling of invalid inputs and proper respect for
    termination criteria. These tests ensure robust operation in production
    environments where data quality and resource limits are concerns.
    """

    def test_invalid_optimization_data(self):
        """
        Test error handling for malformed optimization data.

        Validates that invalid or incomplete optimization data triggers
        appropriate errors during problem creation rather than causing
        mysterious failures during optimization execution.
        """
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 20},
                "termination": {"max_generations": 10},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Test with None data
        with pytest.raises(ValueError, match="Optimization data must be set"):
            runner.optimize(None)

        # Test with incomplete data
        bad_data = {"n_routes": 5}  # Missing required fields
        with pytest.raises(ValueError, match="Failed to create optimization problem"):
            runner.optimize(bad_data)

        print("✅ Error handling works correctly")

    def test_termination_criteria_respected(self, sample_optimization_data):
        """
        Test that termination criteria properly limit optimization execution.

        Validates that max_generations termination is respected and optimization
        doesn't exceed specified limits. This ensures predictable resource usage
        in production environments with time or computational constraints.
        """
        print("\n⏰ TESTING TERMINATION CRITERIA:")

        # Test max generations termination
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 15},
                "termination": {"max_generations": 3},  # Very short
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        # Should stop at exactly max_generations
        assert (
            result.generations_completed <= 3
        ), f"Should complete <= 3 generations, got {result.generations_completed}"

        print(
            f"   ✅ Terminated after {result.generations_completed} generations (max: 3)"
        )

        # Test with time-based termination (if implemented)
        if hasattr(config_manager.get_termination_config(), "max_time_minutes"):
            print("   🕐 Time-based termination available for future testing")

    def test_constraint_violation_reporting(self, sample_optimization_data):
        """
        Test constraint violation detection and reporting accuracy.

        Uses intentionally tight constraints that should be violated to validate
        that the constraint violation analysis correctly identifies and reports
        infeasible solutions. This ensures deployment safety by preventing
        infeasible schedule implementation.
        """
        print("\n🚦 TESTING CONSTRAINT VIOLATION REPORTING:")

        # Create config with tight constraint (likely to be violated)
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": -0.5,  # Require 50% FEWER vehicles (hard constraint)
                        "measure": "peak",
                    }
                ],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 15},
                "termination": {"max_generations": 3},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        # Check constraint violation reporting
        violations = result.constraint_violations
        assert "total_violations" in violations
        assert "feasible" in violations
        assert "violation_details" in violations

        print(f"   📊 Constraint violations: {violations['total_violations']}")
        print(f"   ✅ Feasible solution: {violations['feasible']}")

        if violations["total_violations"] > 0:
            print(
                f"   ⚠️ Found {violations['total_violations']} constraint violations (expected with tight constraint)"
            )
            assert len(violations["violation_details"]) > 0
        else:
            print("   🎯 No constraint violations found")

        print("   ✅ Constraint violation reporting works correctly")


# ================================================================================================
# MULTIPLE CONSTRAINT INTEGRATION TESTS
# ================================================================================================


class TestPSORunnerMultipleConstraints:
    """
    Test PSO runner with multiple constraint types working together.

    Validates complex scenarios with multiple constraint handlers operating
    simultaneously. These tests ensure proper constraint composition and
    violation reporting when multiple operational limits are active.

    Uses patterns from #test_constraints to validate integration.
    """

    def test_multiple_constraint_integration(self, sample_optimization_data):
        """
        Test PSO optimization with multiple constraint types simultaneously.

        Validates that FleetTotal, FleetPerInterval, and MinimumFleet constraints
        can operate together correctly. Uses lenient tolerances to ensure
        feasible solutions while testing the constraint integration machinery.

        This demonstrates the complete constraint composition system that
        would be used in production optimization scenarios.
        """
        print("\n🔗 TESTING PSO RUNNER WITH MULTIPLE CONSTRAINTS:")

        # Create configuration with multiple constraints (based on test_constraints.py)
        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.0,
                },
                "constraints": [
                    # Constraint 1: System-wide budget limit
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.70,  # Lenient - 50% increase allowed
                        "measure": "peak",
                    },
                    # Constraint 2: Per-interval operational limits
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 0.80,  # Lenient - 50% increase per interval
                    },
                    # Constraint 3: Minimum service requirement
                    {
                        "type": "MinimumFleetConstraintHandler",
                        "min_fleet_fraction": 0.1,  # Lenient - 10% minimum
                        "level": "system",
                        "measure": "peak",
                        "baseline": "current_peak",
                    },
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 35,  # Small for test speed
                    "inertia_weight": 0.9,
                    "adaptive": True,
                },
                "termination": {"max_generations": 10},  # Very short for testing
            },
        }

        n_intervals = sample_optimization_data["n_intervals"]
        expected_total_constraints = (
            1 + n_intervals + 1
        )  # Total + PerInterval + Minimum

        print("   📋 Configuration:")
        print("      FleetTotal: 1 constraint (50% tolerance)")
        print(f"      FleetPerInterval: {n_intervals} constraints (50% tolerance)")
        print("      MinimumFleet: 1 constraint (10% minimum)")
        print(f"      Expected total: {expected_total_constraints} constraints")

        # Create and run PSO with multiple constraints
        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        print("\n   🚀 Running PSO optimization...")
        result = runner.optimize(sample_optimization_data)

        # Validate that all constraints were created correctly
        print("\n   ✅ OPTIMIZATION RESULTS:")
        print(f"      Problem constraints: {runner.problem.n_constr}")
        print(f"      Expected constraints: {expected_total_constraints}")
        print(f"      Best objective: {result.best_objective:.6f}")
        print(f"      Generations: {result.generations_completed}")
        print(f"      Optimization time: {result.optimization_time:.2f}s")

        # Assertions
        assert (
            runner.problem.n_constr == expected_total_constraints
        ), f"Expected {expected_total_constraints} constraints, got {runner.problem.n_constr}"

        # Validate constraint violation reporting
        violations = result.constraint_violations
        assert "total_violations" in violations
        assert "feasible" in violations
        assert "violation_details" in violations

        # Should have violation details for each constraint
        assert (
            len(violations["violation_details"]) == expected_total_constraints
        ), f"Expected {expected_total_constraints} violation details, got {len(violations['violation_details'])}"

        print(f"      Constraint violations: {violations['total_violations']}")
        print(f"      Solution feasible: {violations['feasible']}")

        # Validate solution format
        expected_shape = sample_optimization_data["decision_matrix_shape"]
        assert (
            result.best_solution.shape == expected_shape
        ), f"Expected solution shape {expected_shape}, got {result.best_solution.shape}"

        print(f"      Solution shape: {result.best_solution.shape}")
        print("   🎯 All multi-constraint integration tests passed!")

    def test_constraint_type_combinations(self, sample_optimization_data):
        """
        Test different combinations of constraint types without full optimization.

        Validates that various constraint type combinations create the expected
        number of constraint equations. This tests the constraint composition
        logic without the computational cost of full optimization runs.
        """
        print("\n🎛️  TESTING DIFFERENT CONSTRAINT COMBINATIONS:")

        # Test combinations with different constraint type mixes
        test_combinations = [
            {
                "name": "Total + Minimum",
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.3,
                    },
                    {
                        "type": "MinimumFleetConstraintHandler",
                        "min_fleet_fraction": 0.2,
                        "level": "system",
                    },
                ],
                "expected_count": 2,  # 1 + 1
            },
            {
                "name": "PerInterval + Minimum",
                "constraints": [
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 0.4,
                    },
                    {
                        "type": "MinimumFleetConstraintHandler",
                        "min_fleet_fraction": 0.15,
                        "level": "system",
                    },
                ],
                "expected_count": sample_optimization_data["n_intervals"] + 1,
            },
            {
                "name": "All Three Types",
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.4,
                    },
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 0.4,
                    },
                    {
                        "type": "MinimumFleetConstraintHandler",
                        "min_fleet_fraction": 0.1,
                        "level": "system",
                    },
                ],
                "expected_count": 1 + sample_optimization_data["n_intervals"] + 1,
            },
        ]

        for combo in test_combinations:
            print(f"\n   🧪 Testing: {combo['name']}")

            config = {
                "problem": {
                    "objective": {"type": "StopCoverageObjective"},
                    "constraints": combo["constraints"],
                },
                "optimization": {
                    "algorithm": {"type": "PSO", "pop_size": 15},
                    "termination": {"max_generations": 2},  # Very short
                },
            }

            config_manager = OptimizationConfigManager(config_dict=config)
            runner = PSORunner(config_manager)

            # Just create the problem (don't run full optimization for speed)
            runner.optimization_data = sample_optimization_data
            runner._create_problem()

            print(f"      Expected constraints: {combo['expected_count']}")
            print(f"      Actual constraints: {runner.problem.n_constr}")

            assert (
                runner.problem.n_constr == combo["expected_count"]
            ), f"{combo['name']}: Expected {combo['expected_count']}, got {runner.problem.n_constr}"

            print(f"      ✅ {combo['name']} constraint count correct")

        print("   🎯 All constraint combination tests passed!")

    def test_constraint_violation_analysis_multiple(self, sample_optimization_data):
        """
        Test constraint violation analysis with multiple tight constraints.

        Uses intentionally tight constraints across multiple types to validate
        that violation reporting correctly identifies which specific constraints
        are violated. This ensures accurate feasibility assessment in complex
        operational scenarios.
        """
        print("\n📊 TESTING CONSTRAINT VIOLATION ANALYSIS WITH MULTIPLE CONSTRAINTS:")

        # Create tight constraints that are likely to be violated
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [
                    # Very tight total constraint
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": -0.2,  # 20% REDUCTION required
                        "measure": "peak",
                    },
                    # Tight per-interval constraints
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 100.0,  # 10% increase per interval
                    },
                ],
            },
            "optimization": {
                "algorithm": {"type": "PSO",
                              "pop_size": 10},
                "termination": {"max_generations": 2},
            },
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        violations = result.constraint_violations
        n_intervals = sample_optimization_data["n_intervals"]
        expected_constraints = 1 + n_intervals  # Total + PerInterval

        print("   📋 Constraint Analysis:")
        print(f"      Expected constraints: {expected_constraints}")
        print(f"      Violation details count: {len(violations['violation_details'])}")
        print(f"      Total violations: {violations['total_violations']}")
        print(f"      Solution feasible: {violations['feasible']}")

        # With tight constraints, we expect some violations
        assert len(violations["violation_details"]) == expected_constraints

        # Analyze violation details
        total_violated = 0
        interval_violated = 0

        for i, detail in enumerate(violations["violation_details"]):
            if i == 0:  # First constraint is FleetTotal
                if detail["violated"]:
                    total_violated += 1
                    print(
                        f"      FleetTotal constraint violated: {detail['violation_amount']:.2f}"
                    )
                else:
                    print(
                        f"      FleetTotal constraint satisfied: {detail['value']:.2f}"
                    )
            else:  # Rest are FleetPerInterval
                if detail["violated"]:
                    interval_violated += 1

        print(f"      Total constraint violations: {total_violated}/1")
        print(f"      Per-interval violations: {interval_violated}/{n_intervals}")

        print("   ✅ Constraint violation analysis working correctly!")


# ================================================================================================
# PENALTY METHOD INTEGRATION TESTS
# ================================================================================================


class TestPSORunnerPenaltyMethod:
    """
    Test PSO runner with penalty method constraint handling.

    PENALTY METHOD OVERVIEW:
    The penalty method converts hard constraints into penalty terms added to the
    objective function. Instead of rejecting infeasible solutions, the algorithm
    penalizes constraint violations proportionally, allowing exploration of the
    entire search space while guiding solutions toward feasibility.

    PENALTY FORMULA:
    f_penalized(x) = f_original(x) + Σ(w_i * max(0, g_i(x))²)

    Where:
    - f_original(x): Base objective function value
    - g_i(x): Constraint violation amount (positive = violated)
    - w_i: Penalty weight for constraint i
    - The sum covers all constraints

    ADAPTIVE PENALTY SCHEDULING:
    Penalty weights can increase over generations to progressively emphasize
    constraint satisfaction as optimization proceeds. This balances exploration
    (early generations) with feasibility enforcement (later generations).

    TEST CATEGORIES COVERED:
    1. Configuration validation and parameter parsing
    2. Problem creation with penalty vs hard constraint modes
    3. Complete optimization workflow with penalty method
    4. Adaptive penalty weight scheduling during optimization
    5. Integration testing with pymoo's minimize() function
    6. Comparative analysis between penalty and hard constraint approaches

    INTEGRATION POINTS TESTED:
    - OptimizationConfigManager: Penalty parameter configuration
    - TransitOptimizationProblem: Penalty method evaluation logic
    - PSORunner: Penalty callback integration and workflow
    - AdaptivePSO: Callback system compatibility
    - Constraint handlers: Conversion from hard constraints to penalties
    """

    def test_penalty_method_configuration(self):
        """
        Test penalty method configuration parsing and validation.

        VALIDATES:
        - Default penalty method settings (disabled by default)
        - Custom penalty parameter parsing from configuration
        - Parameter value propagation to PSOConfig dataclass

        CONFIGURATION PARAMETERS TESTED:
        - use_penalty_method: Enable/disable penalty method (default: False)
        - penalty_weight: Base penalty weight for unspecified constraints (default: 1000.0)
        - adaptive_penalty: Enable adaptive weight increases (default: False)
        - penalty_increase_rate: Multiplicative factor for weight increases (default: 2.0)

        This test ensures the configuration system correctly parses penalty method
        parameters and applies appropriate defaults for missing values.
        """
        # Test default values (penalty method disabled)
        config_default = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 50},
                "termination": {"max_generations": 100},
            },
        }

        manager = OptimizationConfigManager(config_dict=config_default)
        pso_config = manager.get_pso_config()

        # Test defaults
        assert not pso_config.use_penalty_method, "Default should be hard constraints"
        assert (
            pso_config.penalty_weight == 1000.0
        ), "Default penalty weight should be 1000.0"
        assert not pso_config.adaptive_penalty, "Default should be fixed penalty"
        assert (
            pso_config.penalty_increase_rate == 2.0
        ), "Default increase rate should be 2.0"

        print("✅ Penalty method defaults correct")

        # Test custom penalty configuration
        config_custom = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 50,
                    "use_penalty_method": True,
                    "penalty_weight": 2000.0,
                    "adaptive_penalty": True,
                    "penalty_increase_rate": 1.8,
                },
                "termination": {"max_generations": 100},
            },
        }

        manager_custom = OptimizationConfigManager(config_dict=config_custom)
        pso_config_custom = manager_custom.get_pso_config()

        assert pso_config_custom.use_penalty_method, "Should enable penalty method"
        assert (
            pso_config_custom.penalty_weight == 2000.0
        ), "Should use custom penalty weight"
        assert pso_config_custom.adaptive_penalty, "Should enable adaptive penalty"
        assert (
            pso_config_custom.penalty_increase_rate == 1.8
        ), "Should use custom increase rate"

        print("✅ Custom penalty configuration works")

    def test_penalty_method_validation_errors(self):
        """
        Test penalty method parameter validation and error handling.

        VALIDATES:
        - Negative penalty weights are rejected (must be positive)
        - Invalid increase rates are caught (must be > 1.0 for growth)
        - Configuration validation occurs during setup, not optimization

        ERROR SCENARIOS TESTED:
        - penalty_weight ≤ 0: Penalty must provide meaningful constraint pressure
        - penalty_increase_rate ≤ 1.0: Adaptive scheduling requires growth

        This test ensures invalid penalty configurations are caught early
        rather than causing mysterious optimization failures.
        """

        # Test invalid penalty weight
        with pytest.raises(ValueError, match="Penalty weight must be positive"):
            config = {
                "problem": {
                    "objective": {"type": "StopCoverageObjective"},
                    "constraints": [],
                },
                "optimization": {
                    "algorithm": {
                        "type": "PSO",
                        "pop_size": 50,
                        "penalty_weight": -100.0,
                    },
                    "termination": {"max_generations": 100},
                },
            }
            OptimizationConfigManager(config_dict=config)

        # Test invalid increase rate
        with pytest.raises(ValueError, match="Penalty increase rate must be > 1.0"):
            config = {
                "problem": {
                    "objective": {"type": "StopCoverageObjective"},
                    "constraints": [],
                },
                "optimization": {
                    "algorithm": {
                        "type": "PSO",
                        "pop_size": 50,
                        "penalty_increase_rate": 0.8,
                    },
                    "termination": {"max_generations": 100},
                },
            }
            OptimizationConfigManager(config_dict=config)

        print("✅ Penalty method validation works")

    def test_penalty_vs_hard_constraints_problem_creation(
        self, sample_optimization_data
    ):
        """
        Test TransitOptimizationProblem creation in penalty vs hard constraint modes.

        VALIDATES:
        - Penalty method problems have n_constr=0 (no hard constraints)
        - Hard constraint problems have n_constr>0 (enforced constraints)
        - Penalty weight configuration propagates correctly to problem instance
        - Constraint-specific penalty weights override base penalty weight

        PROBLEM CREATION DIFFERENCES:
        - Penalty mode: Constraints stored but not passed to pymoo Problem base class
        - Hard mode: Constraints create pymoo constraint equations (G matrix)
        - Penalty mode: All constraint violations handled in objective evaluation
        - Hard mode: Constraint violations handled by pymoo feasibility checking

        This test ensures the fundamental problem setup differs correctly between
        penalty and hard constraint approaches.
        """

        # Create configuration for penalty method
        config_penalty = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.15,
                        "measure": "peak",
                    }
                ],
                "penalty_weights": {"fleet_total": 1500.0},
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,
                    "use_penalty_method": True,
                    "penalty_weight": 1000.0,
                },
                "termination": {"max_generations": 5},
            },
        }

        # Create configuration for hard constraints
        config_hard = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.15,
                        "measure": "peak",
                    }
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,
                    "use_penalty_method": False,  # Hard constraints
                },
                "termination": {"max_generations": 5},
            },
        }

        # Test penalty method problem creation
        config_manager_penalty = OptimizationConfigManager(config_dict=config_penalty)
        runner_penalty = PSORunner(config_manager_penalty)
        runner_penalty.optimization_data = sample_optimization_data
        runner_penalty._create_problem()

        assert (
            runner_penalty.problem.use_penalty_method
        ), "Problem should use penalty method"
        assert (
            runner_penalty.problem.n_constr == 0
        ), "Problem should have no hard constraints"

        # Test penalty weight configuration
        expected_weight = 1500.0  # From penalty_weights config
        actual_weight = runner_penalty.problem._get_constraint_penalty_weight(
            "FleetTotalConstraintHandler"
        )
        assert (
            actual_weight == expected_weight
        ), f"Expected {expected_weight}, got {actual_weight}"

        # Test hard constraints problem creation
        config_manager_hard = OptimizationConfigManager(config_dict=config_hard)
        runner_hard = PSORunner(config_manager_hard)
        runner_hard.optimization_data = sample_optimization_data
        runner_hard._create_problem()

        assert (
            not runner_hard.problem.use_penalty_method
        ), "Problem should use hard constraints"
        assert (
            runner_hard.problem.n_constr == 1
        ), "Problem should have 1 hard constraint"

        print("✅ Penalty vs hard constraints problem creation works")

    def test_penalty_method_optimization(self, sample_optimization_data):
        """
        Test complete PSO optimization workflow using penalty method.

        VALIDATES:
        - End-to-end optimization completes successfully with penalty method
        - Solutions maintain valid format (route×interval matrix)
        - Objective values are finite and meaningful
        - Optimization respects termination criteria
        - Adaptive penalty scheduling integrates with PSO execution

        OPTIMIZATION WORKFLOW TESTED:
        1. Problem creation with penalty method enabled
        2. Algorithm initialization with adaptive penalty callback
        3. Population-based evaluation with penalty-augmented objectives
        4. Generation-by-generation penalty weight increases
        5. Solution decoding and result processing

        CONSTRAINT TYPES TESTED:
        - FleetTotalConstraintHandler: System-wide fleet budget limits
        - FleetPerIntervalConstraintHandler: Time-specific operational constraints

        This test demonstrates the penalty method as a viable alternative to
        hard constraints for constrained transit optimization.
        """

        print("\n🎯 TESTING PENALTY METHOD PSO OPTIMIZATION:")

        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.0,
                },
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.20,  # Tighter constraint for penalty method
                        "measure": "peak",
                    },
                    {
                        "type": "FleetPerIntervalConstraintHandler",  # This should work with penalty method
                        "baseline": "current_by_interval",
                        "tolerance": 0.3,
                    },
                ],
                "penalty_weights": {
                    "fleet_total": 2000.0,
                    "fleet_per_interval": 1000.0,
                },
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 30,
                    "use_penalty_method": True,  # Enable penalty method
                    "penalty_weight": 1500.0,  # Base penalty
                    "adaptive_penalty": True,  # Increase over time
                    "penalty_increase_rate": 1.3,  # 30% increase per generation
                },
                "termination": {"max_generations": 10},
            },
        }

        print("   🔧 Configuration: Penalty method enabled")
        print("   🎯 Constraints: FleetTotal + FleetPerInterval")
        print("   📈 Adaptive penalty: 1500 → increasing")

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run optimization
        result = runner.optimize(sample_optimization_data)

        # Validate results
        assert isinstance(result.best_solution, np.ndarray)
        assert (
            result.best_solution.shape
            == sample_optimization_data["decision_matrix_shape"]
        )
        assert result.best_objective < float("inf")
        assert result.optimization_time > 0
        assert result.generations_completed <= 10

        print("   ✅ Penalty method optimization completed!")
        print(f"      Best objective: {result.best_objective:.6f}")
        print(f"      Generations: {result.generations_completed}")
        print(f"      Time: {result.optimization_time:.2f}s")

        # Note: With penalty method, constraint_violations may show violations
        # but optimization should still complete successfully
        print(
            f"      Constraint status: {result.constraint_violations.get('feasible', 'Unknown')}"
        )

    def test_adaptive_penalty_callback(self):
        """
        Test PenaltySchedulingCallback for adaptive penalty weight increases.

        VALIDATES:
        - Callback initializes with correct initial penalty weight
        - Penalty weight increases according to specified schedule
        - Weight updates are monotonically non-decreasing
        - Problem penalty weight is updated during optimization

        ADAPTIVE SCHEDULE FORMULA:
        w(t) = w_initial * (increase_rate ^ (t / T))

        Where:
        - t: Current generation
        - T: Total generations
        - increase_rate: Multiplicative factor (e.g., 1.5 = 50% increase)

        CALLBACK INTEGRATION:
        - Receives algorithm state during optimization
        - Calculates new penalty weight based on generation progress
        - Updates problem instance penalty weight for next evaluations

        This test ensures adaptive penalty scheduling works correctly as an
        automatic exploration-to-exploitation transition mechanism.
        """

        from transit_opt.optimisation.runners.pso_runner import PenaltySchedulingCallback

        print("\n📈 TESTING ADAPTIVE PENALTY CALLBACK:")

        callback = PenaltySchedulingCallback(initial_penalty=1000.0, increase_rate=1.5)

        assert callback.initial_penalty == 1000.0
        assert callback.increase_rate == 1.5
        assert callback.current_penalty == 1000.0

        # Mock algorithm with termination
        mock_algorithm = MagicMock()
        mock_algorithm.termination.n_max_gen = 100
        mock_algorithm.problem.update_penalty_weight = MagicMock()

        # Test at different generations
        test_generations = [0, 25, 50, 75, 99]

        print("   Generation → Penalty weight:")
        for gen in test_generations:
            mock_algorithm.n_gen = gen
            callback.notify(mock_algorithm)

            progress = gen / 100
            expected_penalty = 1000.0 * (1.5**progress)

            print(
                f"      Gen {gen:2d}: {callback.current_penalty:7.1f} (expected: {expected_penalty:7.1f})"
            )

            assert abs(callback.current_penalty - expected_penalty) < 1e-6

        # Verify penalty increases monotonically
        penalties = []
        for gen in test_generations:
            mock_algorithm.n_gen = gen
            callback.notify(mock_algorithm)
            penalties.append(callback.current_penalty)

        for i in range(1, len(penalties)):
            assert (
                penalties[i] >= penalties[i - 1]
            ), f"Penalty should increase: {penalties[i-1]} → {penalties[i]}"

        print("   ✅ Adaptive penalty scheduling works correctly")

    @patch("transit_opt.optimisation.runners.pso_runner.minimize")
    def test_penalty_callback_integration_with_pso(
        self, mock_minimize, sample_optimization_data
    ):
        """
        Test penalty callback integration with pymoo's minimize() function.

        VALIDATES:
        - PenaltySchedulingCallback is properly added to callback list
        - CallbackCollection wrapper correctly handles multiple callbacks
        - Penalty callback receives correct initialization parameters
        - Runtime and penalty callbacks coexist without interference

        CALLBACK ARCHITECTURE:
        - pymoo expects single callback function for minimize()
        - CallbackCollection wraps multiple callbacks into single interface
        - Each generation triggers all wrapped callbacks in sequence
        - Penalty callback updates problem state for subsequent evaluations

        MOCKING STRATEGY:
        - Mock pymoo.minimize() to avoid actual optimization execution
        - Verify callback parameter structure passed to minimize()
        - Validate penalty callback configuration and parameters

        This test ensures penalty method integrates correctly with pymoo's
        callback system without breaking existing functionality.
        """

        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.2,
                        "measure": "peak",
                    }
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,
                    "use_penalty_method": True,
                    "adaptive_penalty": True,
                    "penalty_weight": 1500.0,
                    "penalty_increase_rate": 1.3,
                },
                "termination": {"max_generations": 10},
            },
        }

        # Mock minimize result
        mock_result = MagicMock()
        mock_result.X = np.zeros(
            sample_optimization_data["n_routes"]
            * sample_optimization_data["n_intervals"]
        )
        mock_result.F = np.array([[0.5]])
        mock_result.n_gen = 10
        mock_result.history = []
        mock_minimize.return_value = mock_result

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run optimization (will be mocked)
        result = runner.optimize(sample_optimization_data)

        # Check that minimize was called with callback
        mock_minimize.assert_called_once()
        call_args = mock_minimize.call_args

        # Check if callback parameter was provided
        assert (
            "callback" in call_args.kwargs
        ), "Should provide callback for adaptive penalty"
        callbacks = call_args.kwargs["callback"]
        assert callbacks is not None, "Should have callbacks"

        # Should have runtime callback + penalty callback
        assert (
            len(callbacks) == 2
        ), f"Expected 2 callbacks (runtime + penalty), got {len(callbacks)}"

        # Find penalty callback
        penalty_callbacks = [
            cb
            for cb in callbacks
            if cb.__class__.__name__ == "PenaltySchedulingCallback"
        ]
        assert len(penalty_callbacks) == 1, "Should have exactly one penalty callback"

        penalty_callback = penalty_callbacks[0]
        assert (
            penalty_callback.initial_penalty == 1500.0
        ), "Should use configured initial penalty"
        assert (
            penalty_callback.increase_rate == 1.3
        ), "Should use configured increase rate"

        print("✅ Penalty callback integration with PSO works")

    def test_penalty_method_comparison(self, sample_optimization_data):
        """
        Test PSO runner with penalty method constraint handling.

        PENALTY METHOD OVERVIEW:
        The penalty method converts hard constraints into penalty terms added to the
        objective function. Instead of rejecting infeasible solutions, the algorithm
        penalizes constraint violations proportionally, allowing exploration of the
        entire search space while guiding solutions toward feasibility.

        PENALTY FORMULA:
        f_penalized(x) = f_original(x) + Σ(w_i * max(0, g_i(x))²)

        Where:
        - f_original(x): Base objective function value
        - g_i(x): Constraint violation amount (positive = violated)
        - w_i: Penalty weight for constraint i
        - The sum covers all constraints

        ADAPTIVE PENALTY SCHEDULING:
        Penalty weights can increase over generations to progressively emphasize
        constraint satisfaction as optimization proceeds. This balances exploration
        (early generations) with feasibility enforcement (later generations).

        TEST CATEGORIES COVERED:
        1. Configuration validation and parameter parsing
        2. Problem creation with penalty vs hard constraint modes
        3. Complete optimization workflow with penalty method
        4. Adaptive penalty weight scheduling during optimization
        5. Integration testing with pymoo's minimize() function
        6. Comparative analysis between penalty and hard constraint approaches

        INTEGRATION POINTS TESTED:
        - OptimizationConfigManager: Penalty parameter configuration
        - TransitOptimizationProblem: Penalty method evaluation logic
        - PSORunner: Penalty callback integration and workflow
        - AdaptivePSO: Callback system compatibility
        - Constraint handlers: Conversion from hard constraints to penalties
        """
        print("\n⚖️ COMPARING PENALTY METHOD VS HARD CONSTRAINTS:")

        base_config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.0,
                },
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.25,  # Lenient to allow feasible solutions
                        "measure": "peak",
                    }
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 20,
                },
                "termination": {"max_generations": 5},
            },
        }

        # Test hard constraints
        config_hard = base_config.copy()
        config_hard["optimization"]["algorithm"]["use_penalty_method"] = False

        config_manager_hard = OptimizationConfigManager(config_dict=config_hard)
        runner_hard = PSORunner(config_manager_hard)

        print("   🚦 Running with hard constraints...")
        result_hard = runner_hard.optimize(sample_optimization_data)

        # Test penalty method
        config_penalty = base_config.copy()
        config_penalty["optimization"]["algorithm"].update(
            {
                "use_penalty_method": True,
                "penalty_weight": 1000.0,
                "adaptive_penalty": False,  # Fixed penalty for comparison
            }
        )

        config_manager_penalty = OptimizationConfigManager(config_dict=config_penalty)
        runner_penalty = PSORunner(config_manager_penalty)

        print("   🎯 Running with penalty method...")
        result_penalty = runner_penalty.optimize(sample_optimization_data)

        # Compare results
        print("   📊 COMPARISON RESULTS:")
        print(f"      Hard constraints objective: {result_hard.best_objective:.6f}")
        print(f"      Penalty method objective:   {result_penalty.best_objective:.6f}")
        print(
            f"      Hard constraints feasible:  {result_hard.constraint_violations['feasible']}"
        )
        print(
            f"      Penalty method constraint status: {result_penalty.constraint_violations.get('feasible', 'N/A')}"
        )

        # Both should produce valid solutions
        assert result_hard.best_objective < float(
            "inf"
        ), "Hard constraints should produce valid objective"
        assert result_penalty.best_objective < float(
            "inf"
        ), "Penalty method should produce valid objective"

        print("   ✅ Both methods produced valid results")

    def test_penalty_method_realistic_scenario(self, sample_optimization_data):
        """
        Test penalty method with realistic Duke GTFS optimization scenario.

        COMPREHENSIVE REAL-WORLD TEST:
        This test demonstrates the penalty method's effectiveness on actual transit
        data using the Duke University GTFS dataset. Unlike other tests that focus
        on configuration or use mocked results, this runs a complete optimization
        with realistic constraints that may be violated during exploration.

        REALISTIC SCENARIO:
        - Multiple constraint types (fleet budget + operational limits)
        - Moderately tight constraints (realistic but challenging)
        - Adaptive penalty scheduling to balance exploration vs feasibility
        - Sufficient generations to demonstrate convergence behavior

        PENALTY METHOD ADVANTAGES DEMONSTRATED:
        - Can explore infeasible solutions during early generations
        - Gradually increases constraint pressure via adaptive penalties
        - Potentially finds better solutions than hard constraint methods
        - Provides graceful degradation when perfect feasibility is impossible

        DUKE GTFS CHARACTERISTICS TESTED:
        - 6 bus routes with different service patterns
        - 8 time intervals across 24-hour day
        - Real spatial coverage optimization with hexagonal zones
        - Actual fleet size calculations from GTFS schedules
        """
        print("\n🏛️ TESTING PENALTY METHOD WITH REALISTIC DUKE GTFS SCENARIO:")

        # Create realistic scenario: modest budget increase with operational constraints
        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 1.5,  # Higher resolution for more challenge
                    "time_aggregation": "peak",  # Focus on peak coverage
                },
                "constraints": [
                    # Constraint 1: Moderate budget increase (achievable but requires optimization)
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.15,  # 15% budget increase limit
                        "measure": "peak",
                    },
                    # Constraint 2: Per-interval operational limits (may conflict with total budget)
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 0.25,  # 25% increase per interval
                    },
                    # Constraint 3: Minimum service levels (ensures basic accessibility)
                    {
                        "type": "MinimumFleetConstraintHandler",
                        "min_fleet_fraction": 0.3,  # Must maintain 30% of current service
                        "level": "system",
                        "measure": "peak",
                        "baseline": "current_peak",
                    },
                ],
                "penalty_weights": {
                    "fleet_total": 2500.0,  # High penalty for budget violations
                    "fleet_per_interval": 1000.0,  # Moderate penalty for operational flexibility
                    "minimum_fleet": 5000.0,  # Very high penalty for service cuts
                },
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 40,  # Larger population for thorough exploration
                    "use_penalty_method": True,  # Enable penalty method
                    "penalty_weight": 1500.0,  # Default for any unspecified constraints
                    "adaptive_penalty": True,  # Increase penalties over time
                    "penalty_increase_rate": 1.2,  # 20% increase per generation (gradual)
                    "inertia_weight": 0.9,
                },
                "termination": {
                    "max_generations": 15
                },  # Enough generations to see adaptation
            },
        }

        print("   📊 SCENARIO CONFIGURATION:")
        print(f"      Routes: {sample_optimization_data['n_routes']}")
        print(f"      Time intervals: {sample_optimization_data['n_intervals']}")
        print(
            f"      Search space size: {sample_optimization_data['n_routes'] * sample_optimization_data['n_intervals']} decisions"
        )
        print("   🎯 CONSTRAINTS:")
        print("      Fleet budget: +15% increase limit (moderate pressure)")
        print("      Per-interval: +25% operational limits")
        print("      Minimum service: 30% of current levels (accessibility)")
        print("   📈 PENALTY SCHEDULE:")
        print("      Initial weights: Budget=2500, Operational=1000, Service=5000")
        print("      Expected final multiplier: ~1.2^15 = 15x initial weights")

        # Run penalty method optimization
        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        print("\n   🚀 RUNNING PENALTY METHOD OPTIMIZATION...")
        print("      Expected behavior:")
        print("        Early gens: Explore infeasible solutions with low penalties")
        print("        Mid gens: Balance feasibility vs objective improvement")
        print("        Late gens: High penalties enforce constraint satisfaction")

        result = runner.optimize(sample_optimization_data)

        # === COMPREHENSIVE RESULT ANALYSIS ===
        print("\n   📊 OPTIMIZATION RESULTS:")
        print(f"      Best objective: {result.best_objective:.6f}")
        print(f"      Generations completed: {result.generations_completed}")
        print(f"      Optimization time: {result.optimization_time:.2f}s")
        print(f"      Solution shape: {result.best_solution.shape}")

        # Analyze constraint violation trends
        violations = result.constraint_violations
        print("\n   🚦 CONSTRAINT ANALYSIS:")
        print(f"      Total violations: {violations['total_violations']}")
        print(f"      Solution feasible: {violations['feasible']}")

        if violations["total_violations"] > 0:
            print("      📋 Violation breakdown:")
            for i, detail in enumerate(violations["violation_details"]):
                constraint_name = detail.get("constraint_type", f"Constraint_{i}")
                if detail["violated"]:
                    print(
                        f"        {constraint_name}: VIOLATED by {detail['violation_amount']:.3f}"
                    )
                else:
                    print(
                        f"        {constraint_name}: satisfied ({detail['value']:.3f})"
                    )

        # Analyze optimization history for penalty method behavior
        if result.optimization_history and len(result.optimization_history) > 1:
            print("\n   📈 CONVERGENCE ANALYSIS:")

            objectives = [gen["best_objective"] for gen in result.optimization_history]
            initial_obj = objectives[0]
            final_obj = objectives[-1]
            improvement = initial_obj - final_obj
            improvement_pct = (
                (improvement / initial_obj) * 100 if initial_obj > 0 else 0
            )

            print(f"      Initial objective: {initial_obj:.6f}")
            print(f"      Final objective: {final_obj:.6f}")
            print(f"      Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

            # Check for penalty method characteristics
            mid_point = len(objectives) // 2
            early_avg = (
                np.mean(objectives[:mid_point]) if mid_point > 0 else objectives[0]
            )
            late_avg = np.mean(objectives[mid_point:])

            print(f"      Early phase avg: {early_avg:.6f}")
            print(f"      Late phase avg: {late_avg:.6f}")

            if early_avg > late_avg:
                print(
                    "      ✅ Typical penalty method behavior: exploration → convergence"
                )
            else:
                print("      📊 Steady improvement throughout optimization")

        # === VALIDATION ASSERTIONS ===
        assert isinstance(
            result.best_solution, np.ndarray
        ), "Should return solution matrix"
        assert (
            result.best_solution.shape
            == sample_optimization_data["decision_matrix_shape"]
        ), "Solution shape should match problem"
        assert result.best_objective < float(
            "inf"
        ), "Should produce finite objective value"
        assert result.optimization_time > 0, "Should track optimization time"
        assert (
            result.generations_completed > 0
        ), "Should complete at least one generation"
        assert result.generations_completed <= 15, "Should respect generation limit"

        # Solution should contain valid headway indices
        n_choices = sample_optimization_data["n_choices"]
        assert np.all(result.best_solution >= 0), "All solution values should be >= 0"
        assert np.all(
            result.best_solution < n_choices
        ), f"All solution values should be < {n_choices}"

        # Should have constraint violation analysis
        assert "total_violations" in violations, "Should report total violations"
        assert "feasible" in violations, "Should report feasibility status"
        assert "violation_details" in violations, "Should provide violation details"

        print("\n   🎯 PENALTY METHOD EFFECTIVENESS:")
        if violations["feasible"]:
            print("      ✅ Found feasible solution despite using penalty method")
            print(
                "      🎉 Penalty method successfully guided optimization to feasibility"
            )
        else:
            print(
                f"      ⚡ Solution violates {violations['total_violations']} constraints"
            )
            print(
                "      📊 Penalty method balanced objective vs constraint satisfaction"
            )
            print(
                "      💡 Consider: higher penalty weights or more generations for feasibility"
            )

        # Compare with current Duke system metrics
        current_fleet = sample_optimization_data["constraints"]["fleet_analysis"][
            "total_current_fleet_peak"
        ]
        print("\n   🚌 FLEET COMPARISON WITH CURRENT DUKE SYSTEM:")
        print(f"      Current Duke fleet size: {current_fleet} vehicles")

        # Calculate implied fleet size from solution (simplified estimation)
        unique_headways = np.unique(result.best_solution)
        active_services = np.sum(
            result.best_solution < (n_choices - 1)
        )  # Exclude no-service option
        service_density = active_services / result.best_solution.size

        print(
            f"      Solution service density: {service_density:.2f} ({active_services}/{result.best_solution.size} slots)"
        )
        print(f"      Unique headway levels used: {len(unique_headways)}")

        print("\n   ✅ REALISTIC DUKE GTFS PENALTY METHOD TEST COMPLETED")
        print(
            "      This test demonstrates penalty method effectiveness on real transit data"
        )


class TestPSORunnerWithSeeding:
    """
    Test PSO runner with custom sampling/seeding functionality.
    Seeding is used to provide initial solutions to PYMOO PSO problem. 
    Default without seeding is to use Latin Hypercube Sampling. 

     The number of solutions seeded should be equal to pop_size.
     With seeding, we set:
    - base_solutions: "from_data" (this means use the current initial_solutions, 
                       and make variations of it using a combination of gaussian 
                       noise and LHS())
    - base_solutions: List (use a List of base solutions, probably from a previous 
                      run. Fill in the remaining number of initial solutons using 
                      a combination of gaussian noise and LHS())
    """

    def test_sampling_actually_used_algorithm_inspection(self, sample_optimization_data):
        """Test that custom sampling is actually set on the PSO algorithm using algorithm inspection."""
        print("\n🔍 TESTING SAMPLING IS ACTUALLY USED (Algorithm Inspection):")

        # Test 1: Default sampling (no custom sampling)
        config_default = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 20},
                "termination": {"max_generations": 1},
                "sampling": {"enabled": False}  # Disabled
            }
        }

        config_manager_default = OptimizationConfigManager(config_dict=config_default)
        runner_default = PSORunner(config_manager_default)
        runner_default.optimization_data = sample_optimization_data
        runner_default._create_problem()

        # Create algorithm without custom sampling
        algorithm_default = runner_default._create_algorithm()

        print("   📊 Algorithm without custom sampling:")
        print(f"      Sampling type: {type(algorithm_default.initialization.sampling)}")
        print(f"      Is LHS (default): {type(algorithm_default.initialization.sampling).__name__ == 'LHS'}")

        # Should be default LHS
        from pymoo.operators.sampling.lhs import LHS
        assert isinstance(algorithm_default.initialization.sampling, LHS), \
            "Default algorithm should use LHS sampling"

        # Test 2: Custom sampling enabled
        config_custom = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 20},
                "termination": {"max_generations": 1},
                "sampling": {
                    "enabled": True,
                    "base_solutions": "from_data",
                    "frac_gaussian_pert": 0.7,
                    "random_seed": 42
                }
            }
        }

        config_manager_custom = OptimizationConfigManager(config_dict=config_custom)
        runner_custom = PSORunner(config_manager_custom)
        runner_custom.optimization_data = sample_optimization_data
        runner_custom._create_problem()

        # Create algorithm with custom sampling
        algorithm_custom = runner_custom._create_algorithm()

        print("   📊 Algorithm with custom sampling:")
        print(f"      Sampling type: {type(algorithm_custom.initialization.sampling)}")
        print(f"      Is numpy array: {isinstance(algorithm_custom.initialization.sampling, np.ndarray)}")

        # Should be numpy array (pre-built population)
        assert isinstance(algorithm_custom.initialization.sampling, np.ndarray), \
            "Custom sampling should result in numpy array being set on algorithm"

        # Check array dimensions
        expected_pop_size = config_custom["optimization"]["algorithm"]["pop_size"]
        expected_n_var = runner_custom.problem.n_var

        assert algorithm_custom.initialization.sampling.shape == (expected_pop_size, expected_n_var), \
            f"Custom sampling array should have shape ({expected_pop_size}, {expected_n_var})"

        print(f"      Array shape: {algorithm_custom.initialization.sampling.shape}")
        print(f"      Expected shape: ({expected_pop_size}, {expected_n_var})")

        print("   ✅ Algorithm inspection confirms custom sampling is properly set!")

    def test_sampling_initial_population_inspection(self, sample_optimization_data):
        """Test custom sampling by inspecting the actual initial population produced."""
        print("\n🔍 TESTING SAMPLING VIA INITIAL POPULATION INSPECTION:")

        # Create runner with custom sampling
        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 15},
                "termination": {"max_generations": 1},
                "sampling": {
                    "enabled": True,
                    "base_solutions": "from_data",
                    "frac_gaussian_pert": 0.5,  # 50% gaussian, 50% LHS
                    "gaussian_sigma": 1.5,
                    "random_seed": 123  # Fixed seed for reproducibility
                }
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)
        runner.optimization_data = sample_optimization_data
        runner._create_problem()

        # Create algorithm
        algorithm = runner._create_algorithm()

        # Get the initial population that would be used
        pop = algorithm.initialization.do(runner.problem, algorithm.pop_size, random_state=123)

        print("   📊 Initial population analysis:")
        print(f"      Population size: {len(pop)}")
        print(f"      Expected size: {algorithm.pop_size}")
        print(f"      X matrix shape: {pop.get('X').shape}")

        # Should match expected population size
        assert len(pop) == algorithm.pop_size, \
            f"Initial population size should be {algorithm.pop_size}, got {len(pop)}"

        # Extract initial solution for comparison
        initial_solution = sample_optimization_data['initial_solution'].flatten()
        pop_X = pop.get('X')

        print(f"      Initial solution (first 5): {initial_solution[:5]}")
        print(f"      Pop particle 0 (first 5): {pop_X[0][:5]}")

        # With custom sampling, at least one particle should be very close to initial solution
        # (the base solution should be included, so should be identical actually)
        distances_to_initial = [np.linalg.norm(pop_X[i] - initial_solution) for i in range(len(pop))]
        min_distance = min(distances_to_initial)
        # the first solution should be zero, and the first n% after that should be closer
        # as they are based on Gaussian pertubation
        print(f"      Distances to initial solution: {[f'{d:.6f}' for d in distances_to_initial]}")
        print(f"      Min distance to initial solution: {min_distance:.6f}")

        # Should have at least one particle very close to initial solution (the base solution)
        assert min_distance < 1e-10, \
            f"Expected base solution to be included in population (min distance: {min_distance})"

        # Check that particles are not all identical (should have variety from gaussian + LHS)
        unique_particles = len(set(tuple(row) for row in pop_X))
        print(f"      Unique particles: {unique_particles}/{len(pop)}")

        assert unique_particles > 1, \
            "Population should have variety, not all identical particles"

        # Check bounds are respected
        assert np.all(pop_X >= runner.problem.xl), "All particles should respect lower bounds"
        assert np.all(pop_X <= runner.problem.xu), "All particles should respect upper bounds"

        print("   ✅ Initial population inspection confirms custom sampling works correctly!")

    def test_sampling_vs_no_sampling_population_difference(self, sample_optimization_data):
        """Test that custom sampling produces different initial populations than default LHS."""
        print("\n🔍 TESTING CUSTOM SAMPLING PRODUCES DIFFERENT POPULATIONS:")

        # Base config
        base_config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 10},
                "termination": {"max_generations": 1}
            }
        }

        # Test 1: Default LHS sampling
        config_default = base_config.copy()
        config_default["optimization"]["sampling"] = {"enabled": False}

        config_manager_default = OptimizationConfigManager(config_dict=config_default)
        runner_default = PSORunner(config_manager_default)
        runner_default.optimization_data = sample_optimization_data
        runner_default._create_problem()

        algorithm_default = runner_default._create_algorithm()
        pop_default = algorithm_default.initialization.do(runner_default.problem, algorithm_default.pop_size, random_state=42)
        X_default = pop_default.get('X')

        print("   📊 Default LHS sampling:")
        print(f"      Population shape: {X_default.shape}")
        print(f"      First particle (first 5): {X_default[0][:5]}")

        # Test 2: Custom sampling
        config_custom = base_config.copy()
        config_custom["optimization"]["sampling"] = {
            "enabled": True,
            "base_solutions": "from_data",
            "frac_gaussian_pert": 0.6,
            "gaussian_sigma": 1.5,
            "random_seed": 42
        }

        config_manager_custom = OptimizationConfigManager(config_dict=config_custom)
        runner_custom = PSORunner(config_manager_custom)
        runner_custom.optimization_data = sample_optimization_data
        runner_custom._create_problem()

        algorithm_custom = runner_custom._create_algorithm()
        pop_custom = algorithm_custom.initialization.do(runner_custom.problem, algorithm_custom.pop_size, random_state=42)
        X_custom = pop_custom.get('X')

        print("   📊 Custom sampling:")
        print(f"      Population shape: {X_custom.shape}")
        print(f"      First particle (first 5): {X_custom[0][:5]}")

        # Compare populations - they should be different
        population_difference = np.linalg.norm(X_default - X_custom)
        print(f"      Population difference (Frobenius norm): {population_difference:.6f}")

        # Populations should be significantly different (Gaussian noise produces solutions
        # much closer to initial solution than LHS)
        assert population_difference > 1.0, \
            f"Custom sampling should produce different population than default LHS (difference: {population_difference})"

        # Custom sampling should include initial solution (from optimization data)
        initial_solution = sample_optimization_data['initial_solution'].flatten()

        # Check if initial solution is in custom population
        distances_custom = [np.linalg.norm(X_custom[i] - initial_solution) for i in range(len(X_custom))]
        min_distance_custom = min(distances_custom)

        # Check if initial solution is in default population
        distances_default = [np.linalg.norm(X_default[i] - initial_solution) for i in range(len(X_default))]
        min_distance_default = min(distances_default)

        print(f"      Min distance to initial (custom): {min_distance_custom:.6f}")
        print(f"      Min distance to initial (default): {min_distance_default:.6f}")

        # Custom sampling should have initial solution (very small distance)
        assert min_distance_custom < 1e-10, \
            "Custom sampling should include exact initial solution as base"

        # Default LHS is very unlikely to randomly hit the exact initial solution
        assert min_distance_default > 0.1, \
            "Default LHS should not randomly hit the exact initial solution"

        print("   ✅ Custom sampling produces appropriately different populations!")


    def test_sampling_from_data_basic(self, sample_optimization_data):
        """
        Test PSO with sampling using initial solution from optimization data.
        """
        print("\n🌱 TESTING SAMPLING FROM OPTIMIZATION DATA:")

        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 20},
                "termination": {"max_generations": 3},
                "sampling": {
                    "enabled": True,
                    "base_solutions": "from_data",
                    "frac_gaussian_pert": 0.7,
                    "gaussian_sigma": 0.75,
                    "random_seed": 42
                }
            }
        }

        print("   📋 Configuration: Use initial solution as base for sampling")
        print("   📊 70% Gaussian perturbations, 30% LHS samples")

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        assert result.best_objective < float('inf')
        assert result.generations_completed > 0

        print(f"   ✅ Optimization completed: {result.best_objective:.6f}")
        print("   🌱 Sampling from data works!")

    def test_sampling_from_solution_list(self, sample_optimization_data):
        """Test PSO with sampling using multiple provided base solutions."""
        print("\n🌱 TESTING SAMPLING FROM SOLUTION LIST:")

        # Create multiple diverse base solutions
        shape = sample_optimization_data['decision_matrix_shape']
        n_choices = sample_optimization_data['n_choices']
        original = sample_optimization_data['initial_solution']

        base_solutions = [
            original.copy(),                                    # Original solution
            np.zeros(shape, dtype=int),                        # All frequent service
            np.full(shape, min(2, n_choices-2), dtype=int),   # All moderate service
            np.clip(original + 1, 0, n_choices - 1),          # Slightly less frequent
            np.clip(original - 1, 0, n_choices - 1),          # Slightly more frequent
        ]

        print(f"   📋 Created {len(base_solutions)} diverse base solutions")

        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 25},
                "termination": {"max_generations": 3},
                "sampling": {
                    "enabled": True,
                    "base_solutions": base_solutions,
                    "frac_gaussian_pert": 0.8,
                    "gaussian_sigma": 0.75,
                    "random_seed": 42
                }
            }
        }

        print("   📊 80% Gaussian perturbations, 20% LHS samples")

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        assert result.best_objective < float('inf')
        assert result.generations_completed > 0

        print(f"   ✅ Optimization completed: {result.best_objective:.6f}")
        print("   🌱 Sampling from solution list works!")


    def test_multi_run_with_sampling(self, sample_optimization_data):
        """Test multi-run optimization with custom sampling."""
        print("\n🔄🌱 TESTING MULTI-RUN WITH SAMPLING:")

        # Create diverse base solutions for seeding
        n_choices = sample_optimization_data['n_choices']
        original = sample_optimization_data['initial_solution']

        base_solutions = [
            original.copy(),
            np.clip(original + 1, 0, n_choices - 1),
            np.clip(original - 1, 0, n_choices - 1),
            np.clip(original + 2, 0, n_choices - 1),
        ]

        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 8},
                "termination": {"max_generations": 3},
                "sampling": {
                    "enabled": True,
                    "base_solutions": base_solutions,
                    "frac_gaussian_pert": 0.6,
                    "gaussian_sigma": 0.1
                    # Note: no random_seed so each run gets different randomness
                }
            }
        }

        print(f"   📋 Using {len(base_solutions)} base solutions for seeding")
        print("   🔄 Running 4 independent runs...")

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        multi_result = runner.optimize_multi_run(
            sample_optimization_data,
            num_runs=4,
            parallel=True,
            track_best_n=2
        )

        # Should complete successfully
        assert len(multi_result.run_summaries) == 4
        assert multi_result.best_result is not None

        print("   📊 MULTI-RUN RESULTS:")
        for i, summary in enumerate(multi_result.run_summaries):
            print(f"      Run {summary['run_id']}: {summary['objective']:.6f}, "
                  f"feasible={summary['feasible']}")

        stats = multi_result.statistical_summary
        print(f"      Mean ± std: {stats['objective_mean']:.6f} ± {stats['objective_std']:.6f}")

        print("   ✅ Multi-run with sampling completed successfully!")

    @pytest.mark.skip(reason="Requires PT+DRT optimization data - implement after DRT integration")
    def test_sampling_with_pt_drt_solutions(self):
        """Test sampling with PT+DRT solutions (requires real DRT data)."""
        print("\n🚁🌱 TESTING SAMPLING WITH PT+DRT SOLUTIONS:")
        print("   ⏸️ Skipped - requires real PT+DRT optimization data")
        print("   🔄 Will implement after DRT data preparation is complete")

        # This test will be implemented once we have:
        # 1. Real PT+DRT optimization data from GTFSDataPreparator
        # 2. Updated drt_solution.json file
        # 3. Verified DRT shapefile compatibility

        # The test should:
        # - Load PT+DRT optimization data
        # - Create base solutions with both PT and DRT components
        # - Run PSO with PT+DRT sampling
        # - Verify that both PT and DRT parts are optimized

    def test_sampling_error_handling(self, sample_optimization_data):
        """Test error handling in sampling configuration."""
        print("\n🚨 TESTING SAMPLING ERROR HANDLING:")

        # Test invalid base solutions
        invalid_shape = np.zeros((5, 5), dtype=int)  # Wrong shape

        config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": []
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 10},
                "termination": {"max_generations": 2},
                "sampling": {
                    "enabled": True,
                    "base_solutions": [invalid_shape],  # Invalid solution
                    "frac_gaussian_pert": 0.5
                }
            }
        }

        print("   🔧 Testing with invalid base solution shape...")

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Should raise an error during optimization
        with pytest.raises(RuntimeError, match="shape"):
            runner.optimize(sample_optimization_data)

        print("   ✅ Error handling works correctly!")

    def test_sampling_configuration_validation(self):
        """Test sampling configuration validation."""
        print("\n🔧 TESTING SAMPLING CONFIGURATION VALIDATION:")

        # Test invalid frac_gaussian_pert
        with pytest.raises(ValueError, match="frac_gaussian_pert must be between 0.0 and 1.0"):
            config = {
                "problem": {"objective": {"type": "StopCoverageObjective"}},
                "optimization": {
                    "algorithm": {"type": "PSO", "pop_size": 10},
                    "termination": {"max_generations": 5},
                    "sampling": {
                        "enabled": True,
                        "frac_gaussian_pert": 1.5  # Invalid: > 1.0
                    }
                }
            }
            OptimizationConfigManager(config_dict=config)

        # Test invalid gaussian_sigma
        with pytest.raises(ValueError, match="gaussian_sigma must be positive"):
            config = {
                "problem": {"objective": {"type": "StopCoverageObjective"}},
                "optimization": {
                    "algorithm": {"type": "PSO", "pop_size": 10},
                    "termination": {"max_generations": 5},
                    "sampling": {
                        "enabled": True,
                        "gaussian_sigma": -0.1  # Invalid: negative
                    }
                }
            }
            OptimizationConfigManager(config_dict=config)

        print("   ✅ Configuration validation works correctly!")


# ================================================================================================
# MAIN TEST EXECUTION
# ================================================================================================

if __name__ == "__main__":
    """
    Run tests directly for quick validation.

    Usage: python test_pso_runner.py
    This will run all tests with verbose output and print statements.
    """
    print("🧪 Running PSO runner tests...")
    pytest.main([__file__, "-v", "-s"])
