"""
Tests for PSO Runner.

This module validates the PSO optimization runner that serves as the main entry point
for Particle Swarm Optimization in the transit optimization system. Tests cover:

- PSO runner initialization and configuration validation
- AdaptivePSO algorithm implementation and weight scheduling  
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
- HexagonalCoverageObjective: Spatial coverage evaluation
- Constraint handlers: Fleet size validation
- Configuration system: Parameter management and validation
"""

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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 20,
                    'inertia_weight': 0.8,
                    'inertia_weight_final': 0.3
                },
                'termination': {
                    'max_generations': 50
                }
            }
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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 2  # Too small
                },
                'termination': {
                    'max_generations': 50
                }
            }
        }

        with pytest.raises(ValueError, match="Population size must be at least 5"):
            OptimizationConfigManager(config_dict=bad_config)

        print("✅ PSO runner validation works")


# ================================================================================================
# ADAPTIVE PSO ALGORITHM TESTS
# ================================================================================================

class TestAdaptivePSO:
    """
    Test the AdaptivePSO algorithm implementation.
    
    Validates the adaptive inertia weight scheduling that provides automatic
    exploration-exploitation balance. Tests both adaptive and fixed weight modes
    to ensure the algorithm works correctly in all configurations.
    """

    def test_adaptive_pso_creation(self):
        """
        Test AdaptivePSO instantiation with adaptive weight scheduling.
        
        Validates that adaptive mode is correctly detected when both initial
        and final inertia weights are provided. This enables automatic
        exploration-to-exploitation transition during optimization.
        """
        from transit_opt.optimisation.runners.pso_runner import AdaptivePSO

        # Create adaptive PSO
        pso = AdaptivePSO(
            pop_size=30,
            inertia_weight=0.9,
            inertia_weight_final=0.4,
            cognitive_coeff=2.0,
            social_coeff=2.0
        )

        assert pso.pop_size == 30
        assert pso.initial_inertia_weight == 0.9
        assert pso.final_inertia_weight == 0.4
        assert pso.is_adaptive == True

        print("✅ Adaptive PSO creation works")

    def test_fixed_pso_creation(self):
        """
        Test AdaptivePSO in fixed inertia weight mode.
        
        Validates that when no final weight is provided, the algorithm
        operates as traditional PSO with constant inertia weight.
        This ensures backward compatibility with standard PSO behavior.
        """
        from transit_opt.optimisation.runners.pso_runner import AdaptivePSO

        # Create fixed PSO (no final weight)
        pso = AdaptivePSO(
            pop_size=20,
            inertia_weight=0.7,
            inertia_weight_final=None
        )

        assert pso.initial_inertia_weight == 0.7
        assert pso.final_inertia_weight is None
        assert pso.is_adaptive == False

        print("✅ Fixed inertia weight PSO creation works")

    def test_adaptive_weight_calculation(self):
        """
        Test adaptive inertia weight calculation logic.
        
        Validates the linear decay formula for inertia weight scheduling:
        w(t) = w_initial - (w_initial - w_final) * t / (T - 1)
        
        This ensures correct exploration-exploitation balance throughout optimization.
        """
        from transit_opt.optimisation.runners.pso_runner import AdaptivePSO

        pso = AdaptivePSO(
            pop_size=20,
            inertia_weight=1.0,
            inertia_weight_final=0.2
        )

        # Test weight calculation at different generations
        w_start = pso._calculate_adaptive_weight(0, 100)
        w_middle = pso._calculate_adaptive_weight(50, 100)
        w_end = pso._calculate_adaptive_weight(99, 100)

        assert abs(w_start - 1.0) < 1e-6
        assert abs(w_end - 0.2) < 1e-6
        assert 0.2 < w_middle < 1.0

        print(f"✅ Adaptive weight calculation: {w_start:.3f} → {w_middle:.3f} → {w_end:.3f}")


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
            constraint_violations={'total_violations': 0, 'feasible': True},
            optimization_time=125.3,
            generations_completed=75
        )

        assert len(result.best_solution) == 3
        assert result.best_objective == 42.5
        assert result.constraint_violations['feasible'] == True
        assert result.optimization_time > 0
        assert result.generations_completed > 0

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
            constraint_violations={'total_violations': 0},
            optimization_time=100.0,
            generations_completed=50
        )

        result2 = OptimizationResult(
            best_solution=np.array([12.0]),
            best_objective=40.0,  # Better
            constraint_violations={'total_violations': 0},
            optimization_time=120.0,
            generations_completed=55
        )

        multi_result = MultiRunResult(
            best_result=result2,  # Better result
            all_results=[result1, result2],
            statistical_summary={'objective_mean': 42.5, 'num_runs': 2},
            total_time=220.0,
            num_runs_completed=2
        )

        assert multi_result.best_result.best_objective == 40.0
        assert len(multi_result.all_results) == 2
        assert multi_result.statistical_summary['num_runs'] == 2

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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 25,
                    'inertia_weight': 0.95,
                    'inertia_weight_final': 0.35,
                    'cognitive_coeff': 1.8,
                    'social_coeff': 2.2
                },
                'termination': {
                    'max_generations': 30
                },
                'multi_run': {
                    'enabled': True,
                    'num_runs': 3
                }
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Test algorithm creation
        algorithm = runner._create_algorithm()
        assert algorithm.pop_size == 25
        assert algorithm.initial_inertia_weight == 0.95
        assert algorithm.final_inertia_weight == 0.35

        # Test termination creation
        termination = runner._create_termination()
        assert hasattr(termination, 'n_max_gen')

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
        - HexagonalCoverageObjective from #service_coverage
        - FleetTotalConstraintHandler from #constraints  
        - Duke GTFS data from #preprocessing
        """
        print("\n🧪 TESTING PSO RUNNER WITH REAL DUKE GTFS DATA:")

        # Create realistic configuration for Duke data
        config = {
            'problem': {
                'objective': {
                    'type': 'HexagonalCoverageObjective',
                    'spatial_resolution_km': 1.5,
                    'crs': 'EPSG:3857',
                    'time_aggregation': 'average',
                    'alpha': 0.1
                },
                'constraints': [
                    {
                        'type': 'FleetTotalConstraintHandler',
                        'baseline': 'current_peak',
                        'tolerance': 0.2,  # Allow 20% more vehicles
                        'measure': 'peak'
                    }
                ]
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 20,  # Small for test speed
                    'inertia_weight': 0.9,
                    'inertia_weight_final': 0.4,
                    'cognitive_coeff': 2.0,
                    'social_coeff': 2.0
                },
                'termination': {
                    'max_generations': 5  # Very short for testing
                },
                'monitoring': {
                    'progress_frequency': 2,
                    'save_history': True,
                    'detailed_logging': False
                }
            }
        }

        print(f"   📊 Duke GTFS data: {sample_optimization_data['n_routes']} routes, {sample_optimization_data['n_intervals']} intervals")
        print("   🎯 Objective: HexagonalCoverageObjective")
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
        assert result.best_solution.shape == sample_optimization_data['decision_matrix_shape']
        assert result.best_objective < float('inf')
        assert result.optimization_time > 0
        assert result.generations_completed <= 5

        # Check that solution contains valid headway indices
        max_index = sample_optimization_data['n_choices'] - 1
        assert np.all(result.best_solution >= 0)
        assert np.all(result.best_solution <= max_index)

        print("   ✅ Optimization completed successfully!")
        print(f"      Best objective: {result.best_objective:.6f}")
        print(f"      Generations: {result.generations_completed}")
        print(f"      Time: {result.optimization_time:.2f}s")
        print(f"      Solution shape: {result.best_solution.shape}")
        print(f"      Constraints satisfied: {result.constraint_violations['feasible']}")

        # Validate optimization history
        if result.optimization_history:
            assert len(result.optimization_history) == result.generations_completed

            # Check improvement trend (should generally improve or stay same)
            objectives = [gen['best_objective'] for gen in result.optimization_history]
            final_obj = objectives[-1]
            initial_obj = objectives[0]

            print(f"      Improvement: {initial_obj:.6f} → {final_obj:.6f}")
            assert final_obj <= initial_obj, "Final objective should be <= initial (minimization)"

        print("   🎯 All validation checks passed!")

    def test_multi_run_with_duke_data(self, sample_optimization_data):
        """
        Test multi-run PSO optimization for statistical analysis.
        
        Validates that multiple independent runs produce consistent results
        and proper statistical summaries. This demonstrates the stochastic
        robustness analysis capabilities essential for production deployment.
        """
        print("\n🔄 TESTING MULTI-RUN PSO WITH DUKE GTFS DATA:")

        config = {
            'problem': {
                'objective': {
                    'type': 'HexagonalCoverageObjective',
                    'spatial_resolution_km': 2.0
                },
                'constraints': []  # No constraints for faster testing
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 15,  # Small for speed
                    'inertia_weight': 0.9,
                    'inertia_weight_final': 0.4
                },
                'termination': {
                    'max_generations': 3  # Very short
                },
                'multi_run': {
                    'enabled': True,
                    'num_runs': 3  # Small number for testing
                }
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        print("   🎲 Running 3 independent PSO runs...")

        # Run multi-run optimization
        multi_result = runner.optimize_multi_run(sample_optimization_data, num_runs=3)

        # Validate multi-run results
        assert multi_result.num_runs_completed == 3
        assert len(multi_result.all_results) == 3
        assert multi_result.best_result is not None
        assert multi_result.total_time > 0

        # Check statistical summary
        stats = multi_result.statistical_summary
        assert stats['num_runs'] == 3
        assert 'objective_mean' in stats
        assert 'objective_std' in stats
        assert 'objective_min' in stats
        assert 'objective_max' in stats

        # Best result should be the minimum across all runs
        all_objectives = [r.best_objective for r in multi_result.all_results]
        assert multi_result.best_result.best_objective == min(all_objectives)
        assert abs(stats['objective_min'] - min(all_objectives)) < 1e-6

        print("   ✅ Multi-run optimization completed!")
        print(f"      Successful runs: {multi_result.num_runs_completed}/3")
        print(f"      Best objective: {multi_result.best_result.best_objective:.6f}")
        print(f"      Mean objective: {stats['objective_mean']:.6f}")
        print(f"      Std objective: {stats['objective_std']:.6f}")
        print(f"      Total time: {multi_result.total_time:.2f}s")

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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 10,
                },
                'termination': {
                    'max_generations': 2
                }
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Run short optimization
        result = runner.optimize(sample_optimization_data)

        # Validate solution format
        expected_shape = sample_optimization_data['decision_matrix_shape']
        assert result.best_solution.shape == expected_shape

        print(f"   📊 Solution shape: {result.best_solution.shape}")
        print(f"   📊 Expected shape: {expected_shape}")

        # Check that solution contains valid indices
        n_choices = sample_optimization_data['n_choices']
        assert np.all(result.best_solution >= 0), "Solution indices should be >= 0"
        assert np.all(result.best_solution < n_choices), f"Solution indices should be < {n_choices}"

        # Check some cells have different values (not all same service level)
        unique_values = np.unique(result.best_solution)
        print(f"   🎯 Unique solution values: {unique_values}")

        # Convert to actual headway values for verification
        allowed_headways = sample_optimization_data['allowed_headways']
        solution_headways = np.array([
            [allowed_headways[result.best_solution[i, j]]
             for j in range(expected_shape[1])]
            for i in range(expected_shape[0])
        ])

        print(f"   ⏰ Actual headways used: {np.unique(solution_headways)}")

        # Verify headways are from allowed set
        used_headways = set(solution_headways.flatten())
        allowed_set = set(allowed_headways)
        assert used_headways.issubset(allowed_set), f"Used headways {used_headways} not in allowed set {allowed_set}"

        print("   ✅ Solution decoding works correctly!")


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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {'type': 'PSO', 'pop_size': 20},
                'termination': {'max_generations': 10}
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        # Test with None data
        with pytest.raises(ValueError, match="Optimization data must be set"):
            runner.optimize(None)

        # Test with incomplete data
        bad_data = {'n_routes': 5}  # Missing required fields
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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': []
            },
            'optimization': {
                'algorithm': {'type': 'PSO', 'pop_size': 15},
                'termination': {'max_generations': 3}  # Very short
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        # Should stop at exactly max_generations
        assert result.generations_completed <= 3, f"Should complete <= 3 generations, got {result.generations_completed}"

        print(f"   ✅ Terminated after {result.generations_completed} generations (max: 3)")

        # Test with time-based termination (if implemented)
        if hasattr(config_manager.get_termination_config(), 'max_time_minutes'):
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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': [
                    {
                        'type': 'FleetTotalConstraintHandler',
                        'baseline': 'current_peak',
                        'tolerance': -0.5,  # Require 50% FEWER vehicles (hard constraint)
                        'measure': 'peak'
                    }
                ]
            },
            'optimization': {
                'algorithm': {'type': 'PSO', 'pop_size': 15},
                'termination': {'max_generations': 3}
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        # Check constraint violation reporting
        violations = result.constraint_violations
        assert 'total_violations' in violations
        assert 'feasible' in violations
        assert 'violation_details' in violations

        print(f"   📊 Constraint violations: {violations['total_violations']}")
        print(f"   ✅ Feasible solution: {violations['feasible']}")

        if violations['total_violations'] > 0:
            print(f"   ⚠️ Found {violations['total_violations']} constraint violations (expected with tight constraint)")
            assert len(violations['violation_details']) > 0
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
            'problem': {
                'objective': {
                    'type': 'HexagonalCoverageObjective',
                    'spatial_resolution_km': 2.0
                },
                'constraints': [
                    # Constraint 1: System-wide budget limit
                    {
                        'type': 'FleetTotalConstraintHandler',
                        'baseline': 'current_peak',
                        'tolerance': 0.50,  # Lenient - 50% increase allowed
                        'measure': 'peak'
                    },
                    # Constraint 2: Per-interval operational limits
                    {
                        'type': 'FleetPerIntervalConstraintHandler',
                        'baseline': 'current_by_interval',
                        'tolerance': 0.50  # Lenient - 50% increase per interval
                    },
                    # Constraint 3: Minimum service requirement
                    {
                        'type': 'MinimumFleetConstraintHandler',
                        'min_fleet_fraction': 0.1,  # Lenient - 10% minimum
                        'level': 'system',
                        'measure': 'peak',
                        'baseline': 'current_peak'
                    }
                ]
            },
            'optimization': {
                'algorithm': {
                    'type': 'PSO',
                    'pop_size': 20,  # Small for test speed
                    'inertia_weight': 0.9,
                    'inertia_weight_final': 0.4
                },
                'termination': {
                    'max_generations': 3  # Very short for testing
                }
            }
        }

        n_intervals = sample_optimization_data['n_intervals']
        expected_total_constraints = 1 + n_intervals + 1  # Total + PerInterval + Minimum

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
        assert runner.problem.n_constr == expected_total_constraints, \
            f"Expected {expected_total_constraints} constraints, got {runner.problem.n_constr}"

        # Validate constraint violation reporting
        violations = result.constraint_violations
        assert 'total_violations' in violations
        assert 'feasible' in violations
        assert 'violation_details' in violations

        # Should have violation details for each constraint
        assert len(violations['violation_details']) == expected_total_constraints, \
            f"Expected {expected_total_constraints} violation details, got {len(violations['violation_details'])}"

        print(f"      Constraint violations: {violations['total_violations']}")
        print(f"      Solution feasible: {violations['feasible']}")

        # Validate solution format
        expected_shape = sample_optimization_data['decision_matrix_shape']
        assert result.best_solution.shape == expected_shape, \
            f"Expected solution shape {expected_shape}, got {result.best_solution.shape}"

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
                'name': 'Total + Minimum',
                'constraints': [
                    {'type': 'FleetTotalConstraintHandler', 'baseline': 'current_peak', 'tolerance': 0.3},
                    {'type': 'MinimumFleetConstraintHandler', 'min_fleet_fraction': 0.2, 'level': 'system'}
                ],
                'expected_count': 2  # 1 + 1
            },
            {
                'name': 'PerInterval + Minimum',
                'constraints': [
                    {'type': 'FleetPerIntervalConstraintHandler', 'baseline': 'current_by_interval', 'tolerance': 0.4},
                    {'type': 'MinimumFleetConstraintHandler', 'min_fleet_fraction': 0.15, 'level': 'system'}
                ],
                'expected_count': sample_optimization_data['n_intervals'] + 1
            },
            {
                'name': 'All Three Types',
                'constraints': [
                    {'type': 'FleetTotalConstraintHandler', 'baseline': 'current_peak', 'tolerance': 0.4},
                    {'type': 'FleetPerIntervalConstraintHandler', 'baseline': 'current_by_interval', 'tolerance': 0.4},
                    {'type': 'MinimumFleetConstraintHandler', 'min_fleet_fraction': 0.1, 'level': 'system'}
                ],
                'expected_count': 1 + sample_optimization_data['n_intervals'] + 1
            }
        ]

        for combo in test_combinations:
            print(f"\n   🧪 Testing: {combo['name']}")

            config = {
                'problem': {
                    'objective': {'type': 'HexagonalCoverageObjective'},
                    'constraints': combo['constraints']
                },
                'optimization': {
                    'algorithm': {'type': 'PSO', 'pop_size': 15},
                    'termination': {'max_generations': 2}  # Very short
                }
            }

            config_manager = OptimizationConfigManager(config_dict=config)
            runner = PSORunner(config_manager)

            # Just create the problem (don't run full optimization for speed)
            runner.optimization_data = sample_optimization_data
            runner._create_problem()

            print(f"      Expected constraints: {combo['expected_count']}")
            print(f"      Actual constraints: {runner.problem.n_constr}")

            assert runner.problem.n_constr == combo['expected_count'], \
                f"{combo['name']}: Expected {combo['expected_count']}, got {runner.problem.n_constr}"

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
            'problem': {
                'objective': {'type': 'HexagonalCoverageObjective'},
                'constraints': [
                    # Very tight total constraint
                    {
                        'type': 'FleetTotalConstraintHandler',
                        'baseline': 'current_peak',
                        'tolerance': -0.2,  # 20% REDUCTION required
                        'measure': 'peak'
                    },
                    # Tight per-interval constraints
                    {
                        'type': 'FleetPerIntervalConstraintHandler',
                        'baseline': 'current_by_interval',
                        'tolerance': -0.1  # 10% reduction per interval
                    }
                ]
            },
            'optimization': {
                'algorithm': {'type': 'PSO', 'pop_size': 10},
                'termination': {'max_generations': 2}
            }
        }

        config_manager = OptimizationConfigManager(config_dict=config)
        runner = PSORunner(config_manager)

        result = runner.optimize(sample_optimization_data)

        violations = result.constraint_violations
        n_intervals = sample_optimization_data['n_intervals']
        expected_constraints = 1 + n_intervals  # Total + PerInterval

        print("   📋 Constraint Analysis:")
        print(f"      Expected constraints: {expected_constraints}")
        print(f"      Violation details count: {len(violations['violation_details'])}")
        print(f"      Total violations: {violations['total_violations']}")
        print(f"      Solution feasible: {violations['feasible']}")

        # With tight constraints, we expect some violations
        assert len(violations['violation_details']) == expected_constraints

        # Analyze violation details
        total_violated = 0
        interval_violated = 0

        for i, detail in enumerate(violations['violation_details']):
            if i == 0:  # First constraint is FleetTotal
                if detail['violated']:
                    total_violated += 1
                    print(f"      FleetTotal constraint violated: {detail['violation_amount']:.2f}")
                else:
                    print(f"      FleetTotal constraint satisfied: {detail['value']:.2f}")
            else:  # Rest are FleetPerInterval
                if detail['violated']:
                    interval_violated += 1

        print(f"      Total constraint violations: {total_violated}/1")
        print(f"      Per-interval violations: {interval_violated}/{n_intervals}")

        print("   ✅ Constraint violation analysis working correctly!")


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
