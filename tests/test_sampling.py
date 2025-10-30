"""
Tests for custom sampling functionality.

Tests the solution loading, population building, and integration components
that enable PSO to start with custom initial populations.
"""

import numpy as np
import pytest

from transit_opt.optimisation.utils.population_builder import PopulationBuilder
from transit_opt.optimisation.utils.solution_loader import SolutionLoader


class TestSolutionLoaderBasic:
    """Test basic solution loading functionality."""

    def test_load_from_optimization_data_pt_only(self, sample_optimization_data):
        """Test loading initial solution from optimization data (PT-only)."""
        loader = SolutionLoader()

        solutions = loader.load_solutions('from_data', sample_optimization_data)

        assert len(solutions) == 1
        assert isinstance(solutions[0], np.ndarray)
        assert solutions[0].shape == sample_optimization_data['decision_matrix_shape']

        # Should match the initial solution from optimization data
        np.testing.assert_array_equal(solutions[0], sample_optimization_data['initial_solution'])

    def test_load_from_list_pt_only(self, sample_optimization_data):
        """Test loading solutions from provided list (PT-only problem)."""
        loader = SolutionLoader()

        # Create test solutions
        shape = sample_optimization_data['decision_matrix_shape']
        n_choices = sample_optimization_data['n_choices']

        test_solutions = [
            np.zeros(shape, dtype=int),  # All minimum headways
            np.ones(shape, dtype=int),   # All second headways
            np.full(shape, n_choices - 2, dtype=int),  # All maximum service headways
        ]

        solutions = loader.load_solutions(test_solutions, sample_optimization_data)

        assert len(solutions) == 3
        for i, solution in enumerate(solutions):
            np.testing.assert_array_equal(solution, test_solutions[i])

    def test_validation_errors_pt_only(self, sample_optimization_data):
        """Test solution validation catches errors for PT-only problems."""
        loader = SolutionLoader()

        # Test wrong shape
        bad_shape = np.zeros((5, 5), dtype=int)
        with pytest.raises(ValueError, match="shape"):
            loader.load_solutions([bad_shape], sample_optimization_data)

        # Test invalid values (too high)
        bad_values = np.full(sample_optimization_data['decision_matrix_shape'], 999, dtype=int)
        with pytest.raises(ValueError, match="invalid indices"):
            loader.load_solutions([bad_values], sample_optimization_data)

        # Test negative values
        negative_values = np.full(sample_optimization_data['decision_matrix_shape'], -1, dtype=int)
        with pytest.raises(ValueError, match="negative indices"):
            loader.load_solutions([negative_values], sample_optimization_data)

    def test_empty_solution_list(self, sample_optimization_data):
        """Test handling of empty solution list."""
        loader = SolutionLoader()

        solutions = loader.load_solutions([], sample_optimization_data)
        assert len(solutions) == 0

    def test_invalid_config_spec(self, sample_optimization_data):
        """Test handling of invalid configuration specifications."""
        loader = SolutionLoader()

        with pytest.raises(ValueError, match="Invalid base_solutions specification"):
            loader.load_solutions("invalid_spec", sample_optimization_data)

        with pytest.raises(ValueError, match="Invalid base_solutions specification"):
            loader.load_solutions(123, sample_optimization_data)


class TestSolutionLoaderDRT:
    """Test solution loading with DRT (PT+DRT) functionality."""

    def test_load_from_optimization_data_pt_drt(self):
        """Test loading initial solution from PT+DRT optimization data."""
        # Create mock PT+DRT optimization data
        pt_drt_data = {
            'n_routes': 3,
            'n_intervals': 4,
            'n_choices': 6,
            'decision_matrix_shape': (3, 4),
            'drt_enabled': True,
            'n_drt_zones': 2,
            # initial solution should be a dictionary with ["pt"] and ["drt"] keys
            'initial_solution': {
                'pt': np.array([[0, 1, 2, 3],
                                [1, 2, 3, 4],
                                [2, 3, 4, 5]], dtype=int),
                'drt': np.array([[0, 1, 2, 3],
                                 [1, 2, 3, 2]], dtype=int)
            },
            'variable_structure': {
                'pt_size': 12,  # 3 routes × 4 intervals
                'drt_size': 8,  # 2 zones × 4 intervals
                'drt_shape': (2, 4)
            },
            'drt_config': {
                'zones': [
                    {'allowed_fleet_sizes': [0, 5, 10, 15]},  # 4 choices
                    {'allowed_fleet_sizes': [0, 8, 16, 24]}   # 4 choices
                ]
            }
        }

        loader = SolutionLoader()
        solutions = loader.load_solutions('from_data', pt_drt_data)

        assert len(solutions) == 1
        solution = solutions[0]

        # Should be dict format for PT+DRT
        assert isinstance(solution, dict)
        assert 'pt' in solution
        assert 'drt' in solution

        # Check PT part
        assert solution['pt'].shape == (3, 4)

        # Check DRT part
        assert solution['drt'].shape == (2, 4)

    def test_load_from_list_pt_drt(self):
        """Test loading PT+DRT solutions from provided list."""
        # Create mock PT+DRT optimization data
        pt_drt_data = {
            'n_routes': 2,
            'n_intervals': 2,
            'n_choices': 4,
            'decision_matrix_shape': (2, 2),
            'drt_enabled': True,
            'n_drt_zones': 1,
            'drt_config': {
                'zones': [
                    {'allowed_fleet_sizes': [0, 5, 10]}  # 3 choices
                ]
            }
        }

        # Create test PT+DRT solutions
        test_solutions = [
            {
                'pt': np.array([[0, 1], [2, 0]], dtype=int),
                'drt': np.array([[0, 1]], dtype=int)
            },
            {
                'pt': np.array([[1, 2], [0, 1]], dtype=int),
                'drt': np.array([[1, 2]], dtype=int)
            }
        ]

        loader = SolutionLoader()
        solutions = loader.load_solutions(test_solutions, pt_drt_data)

        assert len(solutions) == 2
        for i, solution in enumerate(solutions):
            assert isinstance(solution, dict)
            np.testing.assert_array_equal(solution['pt'], test_solutions[i]['pt'])
            np.testing.assert_array_equal(solution['drt'], test_solutions[i]['drt'])

    def test_validation_errors_pt_drt(self):
        """Test validation errors for PT+DRT solutions."""
        pt_drt_data = {
            'n_routes': 2,
            'n_intervals': 2,
            'n_choices': 4,
            'decision_matrix_shape': (2, 2),
            'drt_enabled': True,
            'n_drt_zones': 1,
            'drt_config': {
                'zones': [
                    {'allowed_fleet_sizes': [0, 5, 10]}  # 3 choices
                ]
            }
        }

        loader = SolutionLoader()

        # Test missing DRT key
        bad_solution = {'pt': np.array([[0, 1], [2, 0]], dtype=int)}  # Missing 'drt'
        with pytest.raises(ValueError, match="must have 'pt' and 'drt' keys"):
            loader.load_solutions([bad_solution], pt_drt_data)

        # Test wrong DRT shape
        bad_drt_shape = {
            'pt': np.array([[0, 1], [2, 0]], dtype=int),
            'drt': np.array([[0, 1, 2]], dtype=int)  # Wrong shape
        }
        with pytest.raises(ValueError, match="DRT solution shape"):
            loader.load_solutions([bad_drt_shape], pt_drt_data)

        # Test invalid DRT values
        bad_drt_values = {
            'pt': np.array([[0, 1], [2, 0]], dtype=int),
            'drt': np.array([[5, 1]], dtype=int)  # Invalid: only 3 choices (0,1,2)
        }
        with pytest.raises(ValueError, match="invalid indices"):
            loader.load_solutions([bad_drt_values], pt_drt_data)


class TestPopulationBuilder:
    """Test population building for PSO initialization."""

    def test_build_population_basic(self, sample_optimization_data):
        """Test basic population building workflow."""
        from transit_opt.optimisation.objectives.service_coverage import StopCoverageObjective
        from transit_opt.optimisation.problems.transit_problem import TransitOptimizationProblem

        # Create mock problem
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0
        )
        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective
        )

        loader = SolutionLoader()
        builder = PopulationBuilder(loader)

        pop_size = 20
        population = builder.build_initial_population(
            problem=problem,
            pop_size=pop_size,
            optimization_data=sample_optimization_data,
            base_solutions='from_data',
            frac_gaussian_pert=0.6,
            gaussian_sigma=0.1
        )

        # Validate population
        assert population.shape == (pop_size, problem.n_var)
        assert np.all(population >= 0)
        assert np.all(population <= problem.xu)


    def test_build_population_with_solution_list(self, sample_optimization_data):
        """Test population building with provided list of base solutions."""
        from transit_opt.optimisation.objectives.service_coverage import StopCoverageObjective
        from transit_opt.optimisation.problems.transit_problem import TransitOptimizationProblem

        # Create problem
        objective = StopCoverageObjective(
            optimization_data=sample_optimization_data,
            spatial_resolution_km=2.0
        )
        problem = TransitOptimizationProblem(
            optimization_data=sample_optimization_data,
            objective=objective
        )

        # Create multiple synthetic solutions
        shape = sample_optimization_data['decision_matrix_shape']
        n_choices = sample_optimization_data['n_choices']

        base_solutions = [
            # Solution 1: All minimum headways (frequent service)
            np.zeros(shape, dtype=int),

            # Solution 2: All maximum non-service headways
            np.full(shape, n_choices - 2, dtype=int),  # n_choices-1 is no-service, so n_choices-2 is max service

            # Solution 3: Mixed solution (some routes high frequency, others low)
            np.random.randint(0, n_choices - 1, size=shape),

            # Solution 4: Conservative solution (moderate headways)
            np.full(shape, min(2, n_choices - 2), dtype=int),
        ]

        print(f"   📋 Created {len(base_solutions)} synthetic base solutions")
        print(f"   📊 Solution shapes: {[sol.shape for sol in base_solutions]}")

        loader = SolutionLoader()
        builder = PopulationBuilder(loader)

        pop_size = 25
        population = builder.build_initial_population(
            problem=problem,
            pop_size=pop_size,
            optimization_data=sample_optimization_data,
            base_solutions=base_solutions,  # Pass list instead of 'from_data'
            frac_gaussian_pert=0.6,
            gaussian_sigma=0.1,
            random_seed=42
        )

        # Validate population
        assert population.shape == (pop_size, problem.n_var)
        assert np.all(population >= 0)
        assert np.all(population <= problem.xu)

        # Expected distribution: 4 base + 15 gaussian (60% of 25) + 6 LHS
        expected_distribution = {
            'base': len(base_solutions),
            'gaussian': int(0.6 * pop_size),
            'lhs': pop_size - len(base_solutions) - int(0.6 * pop_size)
        }

        print("   ✅ Population created successfully:")
        print(f"      Total size: {population.shape[0]}")
        print(f"      Expected distribution: {expected_distribution}")
        print(f"      All values in valid range: {np.all(population >= 0) and np.all(population <= problem.xu)}")

        print("   🎯 Multiple base solutions test passed!")

    def test_build_population_with_pt_drt_solution_list(self):
        """Test population building with provided PT+DRT base solutions."""
        # Create mock PT+DRT optimization data
        pt_drt_data = {
            'n_routes': 2,
            'n_intervals': 2,
            'n_choices': 4,
            'decision_matrix_shape': (2, 2),
            'drt_enabled': True,
            'n_drt_zones': 1,
            'variable_bounds': (0, 3),
            'allowed_headways': np.array([10, 15, 30, 9999]),
            'drt_config': {
                'zones': [
                    {'allowed_fleet_sizes': [0, 5, 10]}
                ]
            },
            'variable_structure': {
                'pt_size': 4,   # 2 routes × 2 intervals
                'drt_size': 2,  # 1 zone × 2 intervals
                'total_size': 6
            }
        }

        # Create base PT+DRT solutions
        base_solutions = [
            {
                'pt': np.array([[0, 1], [2, 0]], dtype=int),
                'drt': np.array([[0, 1]], dtype=int)
            },
            {
                'pt': np.array([[1, 2], [0, 1]], dtype=int),
                'drt': np.array([[1, 2]], dtype=int)
            }
        ]

        # Mock problem with PT+DRT encoding
        class MockPTDRTProblem:
            def __init__(self):
                self.n_var = 6  # 4 PT + 2 DRT variables
                self.xl = np.zeros(6)
                self.xu = np.array([3, 3, 3, 3, 2, 2])  # PT bounds: 0-3, DRT bounds: 0-2

            def encode_solution(self, solution_dict):
                """Encode PT+DRT solution dict to flat vector."""
                pt_flat = solution_dict['pt'].flatten()
                drt_flat = solution_dict['drt'].flatten()
                return np.concatenate([pt_flat, drt_flat])

            def bounds(self):
                return self.xl, self.xu

        problem = MockPTDRTProblem()

        loader = SolutionLoader()
        builder = PopulationBuilder(loader)

        pop_size = 15
        population = builder.build_initial_population(
            problem=problem,
            pop_size=pop_size,
            optimization_data=pt_drt_data,
            base_solutions=base_solutions,
            frac_gaussian_pert=0.6,
            gaussian_sigma=2,
            random_seed=42
        )

        # Validate population
        assert population.shape == (pop_size, problem.n_var)
        assert np.all(population >= problem.xl)
        assert np.all(population <= problem.xu)

        print("   ✅ PT+DRT population created successfully")
        print(f"      Population shape: {population.shape}")
        print(f"      Base solutions used: {len(base_solutions)}")
        print("      Population samples:")
        for i in range(min(10, pop_size)):
            print(f"         {population[i]}")



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
