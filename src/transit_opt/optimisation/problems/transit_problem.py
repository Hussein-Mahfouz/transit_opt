"""
Main pymoo Problem class for transit optimization.

This module provides the core optimization problem class that bridges the domain-specific
transit optimization logic with pymoo's metaheuristic algorithms. It handles solution
encoding/decoding and population-based evaluation.
"""

from typing import Any

import numpy as np
from pymoo.core.problem import Problem

from ..objectives.base import BaseObjective
from .base import BaseConstraintHandler


class TransitOptimizationProblem(Problem):
    """
    Main pymoo Problem class for transit optimization with pluggable objectives and constraints.

    This class serves as the bridge between the domain-specific transit optimization
    components (objectives, constraints, GTFS data) and pymoo's optimization algorithms
    (PSO, GA, NSGA-II, etc.). It handles the conversion between pymoo's flat decision
    vectors and the route×interval solution matrices.

    KEY DESIGN PRINCIPLES:
    - **Algorithm Agnostic**: Works with any pymoo optimization algorithm
    - **Pluggable Components**: Accepts any BaseObjective and BaseConstraintHandler instances
    - **Population-Based**: Efficiently evaluates multiple solutions simultaneously
    - **Solution Encoding**: Handles conversion between flat vectors and 2D matrices
    - **Extensible**: Easy to add new objectives and constraints without modifying this class

    SOLUTION ENCODING:
    - **Pymoo format**: Flat integer array of length (n_routes × n_intervals)
    - **Domain format**: 2D matrix of shape (n_routes, n_intervals)
    - **Values**: Indices into allowed_headways array (0 to n_choices-1)
    - **Example**: [0,1,2,0,1,2] → [[0,1,2], [0,1,2]] for 2 routes, 3 intervals

    CONSTRAINT HANDLING:
    - **Pymoo convention**: g(x) <= 0 means constraint is satisfied
    - **Your handlers**: Return violations where <= 0 means satisfied
    - **Perfect match**: No conversion needed between conventions

    Attributes:
        optimization_data (Dict[str, Any]): Complete GTFS-derived optimization data
        objective (BaseObjective): Single objective function to minimize
        constraints (List[BaseConstraintHandler]): List of constraint handlers
        n_routes (int): Number of transit routes
        n_intervals (int): Number of time intervals per day
        n_choices (int): Number of allowed headway choices (including no-service)

    Args:
        optimization_data: Complete optimization data structure from GTFSDataPreparator
        objective: Single objective function (must inherit from BaseObjective)
        constraints: List of constraint handlers (each inherits from BaseConstraintHandler)
        penalty_config: Dict with keys:
            - 'enabled': bool - Use penalty method instead of hard constraints
            - 'penalty_weight': float - Base penalty weight
            - 'adaptive': bool - Increase penalty weight over generations
            - 'constraint_weights': Dict - Individual constraint penalty weights

    Example:
        ```python
        from transit_opt.optimisation.objectives.service_coverage import HexagonalCoverageObjective
        from transit_opt.optimisation.problems.base import FleetTotalConstraintHandler

        # Create objective
        spatial_equity = HexagonalCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=2.0
        )

        # Create constraints
        fleet_limit = FleetTotalConstraintHandler({
            'baseline': 'current_peak',
            'tolerance': 0.15,
            'measure': 'peak'
        }, opt_data)

        # Create problem
        problem = TransitOptimizationProblem(
            optimization_data=opt_data,
            objective=spatial_equity,
            constraints=[fleet_limit]
        )

        # Use with any pymoo algorithm
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.optimize import minimize

        algorithm = PSO(pop_size=40)
        result = minimize(problem, algorithm, termination=('n_gen', 50))

        # Extract best solution
        best_solution_flat = result.X
        best_solution_matrix = problem.decode_solution(best_solution_flat)
        ```

    Mathematical Formulation:
        minimize f(x)     # Single objective (the BaseObjective.evaluate())
        subject to:
            g_i(x) <= 0   # Constraints (the BaseConstraintHandler.evaluate())
            x_ij ∈ [0, n_choices-1]  # Integer decision variables

        Where:
        - x is solution matrix reshaped as flat vector
        - f(x) is objective function value
        - g_i(x) are constraint violations
        - i ∈ [0, n_routes-1], j ∈ [0, n_intervals-1]
    """

    def __init__(
        self,
        optimization_data: dict[str, Any],
        objective: BaseObjective,
        constraints: list[BaseConstraintHandler] | None = None,
        penalty_config: dict[str, Any] | None = None,
    ):

        print("🏗️  CREATING TRANSIT OPTIMIZATION PROBLEM:")

        # Store components
        self.optimization_data = optimization_data
        self.objective = objective
        self.constraints = constraints or []

        # Extract problem dimensions from optimization data
        self.n_routes = optimization_data["n_routes"]
        self.n_intervals = optimization_data["n_intervals"]
        self.n_choices = optimization_data["n_choices"]

        print("   📊 Problem dimensions:")
        print(f"      Routes: {self.n_routes}")
        print(f"      Time intervals: {self.n_intervals}")
        print(f"      Headway choices: {self.n_choices}")

        # Calculate pymoo problem parameters
        n_var = self.n_routes * self.n_intervals  # Total decision variables
        n_obj = 1  # Single objective optimization
        n_constr = sum(c.n_constraints for c in self.constraints)  # Total constraints

        print("   🔧 Pymoo parameters:")
        print(f"      Decision variables: {n_var}")
        print(f"      Objectives: {n_obj}")
        print(f"      Constraints: {n_constr}")

        # Define variable bounds (indices into allowed_headways)
        xl = np.zeros(n_var, dtype=int)  # Lower bounds (index 0)
        xu = np.full(n_var, self.n_choices - 1, dtype=int)  # Upper bounds (max index)

        #  Penalty method configuration
        self.penalty_config = penalty_config or {'enabled': False}
        self.use_penalty_method = self.penalty_config.get('enabled', False)
        self.penalty_weight = self.penalty_config.get('penalty_weight', 1000.0)
        self.constraint_penalty_weights = self.penalty_config.get('constraint_weights', {})

        # Store constraint info for penalty calculation
        self.constraint_names = [type(c).__name__ for c in (constraints or [])]

        # Initialize pymoo Problem
        if self.use_penalty_method:
            # No hard constraints - handle as penalties
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=0,  # 🔧 Zero constraints for penalty method
                xl=xl, xu=xu,
                vtype=int
            )
            print(f"   🎯 Penalty method enabled: {len(constraints or [])} constraints → objective penalties")
            print(f"   ⚖️ Base penalty weight: {self.penalty_weight}")

        else:
            # Use hard constraints (existing approach)
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=n_constr,
                xl=xl, xu=xu,
                vtype=int
            )
            print(f"   🚦 Hard constraints: {n_constr} constraint(s)")

        # Log constraint details
        if self.constraints:
            print("   📋 Constraint breakdown:")
            for i, constraint in enumerate(self.constraints):
                constraint_info = constraint.get_constraint_info()
                print(
                    f"      {i+1}. {constraint_info['handler_type']}: "
                    f"{constraint_info['n_constraints']} constraint(s)"
                )
        else:
            print("   📋 No constraints specified (unconstrained optimization)")

        print("   ✅ Problem setup complete!")

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate population of solutions for pymoo.

        This is the main method called by pymoo algorithms. It receives a population
        of flat solution vectors and must return objective values and constraint
        violations for each solution.

        EVALUATION PROCESS:
        1. For each solution in population:
           a. Decode flat vector to route×interval matrix
           b. Evaluate objective function
           c. Evaluate all constraint handlers
        2. Pack results into pymoo format

        PERFORMANCE CONSIDERATIONS:
        - Population size typically 20-100 solutions
        - Called every generation (potentially 100+ times)
        - Each evaluation involves fleet calculations and spatial analysis
        - Consider parallel evaluation for large problems

        Args:
            X (np.ndarray): Population matrix of shape (pop_size, n_var)
                           Each row is a flat solution vector
            out (dict): Output dictionary for pymoo results
                       Must contain 'F' (objectives) and 'G' (constraints)

        Pymoo Output Format:
            out["F"]: Objective values, shape (pop_size, 1)
            out["G"]: Constraint violations, shape (pop_size, n_constr)
                     Values <= 0 mean constraint satisfied
        """

        pop_size = len(X)
        F = np.zeros((pop_size, 1))

        # Always initialize G, but only use it for hard constraints
        G = None
        if not self.use_penalty_method and self.n_constr > 0:
            G = np.zeros((pop_size, self.n_constr)) if self.n_constr > 0 else None

        for i in range(pop_size):
            # Decode solution
            solution_matrix = self.decode_solution(X[i])

            # Evaluate base objective
            base_objective = self.objective.evaluate(solution_matrix)

            if self.use_penalty_method and self.constraints:
                # 🔧 PENALTY METHOD: Add constraint violations to objective
                total_penalty = 0.0

                for j, constraint in enumerate(self.constraints):
                    violations = constraint.evaluate(solution_matrix)

                    # Get constraint-specific penalty weight
                    constraint_name = self.constraint_names[j]
                    constraint_weight = self._get_constraint_penalty_weight(constraint_name)

                    # Calculate penalty: sum of squared positive violations
                    constraint_penalty = np.sum(np.maximum(0, violations) ** 2) * constraint_weight
                    total_penalty += constraint_penalty

                # Add penalty to objective
                F[i, 0] = base_objective + total_penalty

            else:
                # Hard constraints (existing approach)
                F[i, 0] = base_objective

                if self.constraints and G is not None:

                    constraint_start_idx = 0

                    for constraint in self.constraints:
                        try:
                            violations = constraint.evaluate(solution_matrix)

                            # Store violations in correct positions
                            constraint_end_idx = constraint_start_idx + len(violations)
                            G[i, constraint_start_idx:constraint_end_idx] = violations
                            constraint_start_idx = constraint_end_idx

                        except Exception as e:
                            print(
                                f"   ⚠️  Constraint evaluation failed for solution {i}: {e}"
                            )
                            # Assign large positive violations (constraint violated)
                            constraint_end_idx = (
                                constraint_start_idx + constraint.n_constraints
                            )
                            G[i, constraint_start_idx:constraint_end_idx] = 1e6
                            constraint_start_idx = constraint_end_idx

        # Pack results for pymoo
        out["F"] = F
        if not self.use_penalty_method and self.n_constr > 0 and G is not None:
            out["G"] = G

        # Log evaluation summary
        if pop_size > 0:
            best_obj = np.min(F[:, 0])
            worst_obj = np.max(F[:, 0])
            avg_obj = np.mean(F[:, 0])

            print("   📊 Evaluation summary:")
            print(f"      Best objective: {best_obj:.4f}")
            print(f"      Worst objective: {worst_obj:.4f}")
            print(f"      Average objective: {avg_obj:.4f}")

            if G is not None:
                feasible_solutions = np.sum(np.all(G <= 0, axis=1))
                print(f"      Feasible solutions: {feasible_solutions}/{pop_size}")

    def _get_constraint_penalty_weight(self, constraint_name: str) -> float:
        """Get penalty weight for specific constraint type."""
        # 1. Check for specific weight first
        if constraint_name in self.constraint_penalty_weights:
            return self.constraint_penalty_weights[constraint_name]

        # 2. Check simplified name patterns
        simplified_patterns = {
            'FleetTotalConstraintHandler': 'fleet_total',
            'FleetPerIntervalConstraintHandler': 'fleet_per_interval',
            'MinimumFleetConstraintHandler': 'minimum_fleet'
        }

        pattern_key = simplified_patterns.get(constraint_name)
        if pattern_key and pattern_key in self.constraint_penalty_weights:
            return self.constraint_penalty_weights[pattern_key]

        # 3. Return base penalty weight as fallback
        return self.penalty_weight

    def update_penalty_weight(self, new_weight: float):
        """Update penalty weight for adaptive penalty scheduling."""
        self.penalty_weight = new_weight

    def _decode_solution(self, x_flat: np.ndarray) -> np.ndarray:
        """
        Convert flat solution vector to route×interval matrix.

        ENCODING DETAILS:
        - Flat vector: [r0i0, r0i1, r0i2, r1i0, r1i1, r1i2, ...]
        - Matrix format: [[r0i0, r0i1, r0i2], [r1i0, r1i1, r1i2], ...]
        - Values: Indices into optimization_data["allowed_headways"]

        Args:
            x_flat: Flat solution vector of length (n_routes × n_intervals)

        Returns:
            Solution matrix of shape (n_routes, n_intervals) with integer indices

        Example:
            >>> x_flat = np.array([0, 1, 2, 3, 1, 0])  # 2 routes, 3 intervals
            >>> matrix = problem._decode_solution(x_flat)
            >>> print(matrix)
            [[0 1 2]
             [3 1 0]]
        """

        # 1. Clip to bounds (PSO can generate out-of-bounds values). Probably redundant
        # as Pymoo should handle this (but just in case)
        x_bounded = np.clip(x_flat, 0, self.n_choices - 1)

        # 2. Round and convert to integers
        x_int = np.round(x_bounded).astype(int)

        return x_int.reshape(self.n_routes, self.n_intervals)

    def _encode_solution(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Convert route×interval matrix to flat solution vector.

        This is the inverse operation of _decode_solution(). Useful for:
        - Converting initial GTFS solution to pymoo format
        - Post-processing optimization results
        - Debugging and validation

        Args:
            solution_matrix: Solution matrix of shape (n_routes, n_intervals)

        Returns:
            Flat solution vector of length (n_routes × n_intervals)

        Example:
            >>> matrix = np.array([[0, 1, 2], [3, 1, 0]])
            >>> x_flat = problem._encode_solution(matrix)
            >>> print(x_flat)
            [0 1 2 3 1 0]
        """
        return solution_matrix.flatten()

    def decode_solution(self, x_flat: np.ndarray) -> np.ndarray:
        """
        Public interface for solution decoding.

        Exposes the internal _decode_solution method for external use.
        Useful for analyzing optimization results and debugging.

        Args:
            x_flat: Flat solution vector from pymoo optimization result

        Returns:
            Solution matrix in domain-specific format
        """
        return self._decode_solution(x_flat)

    def encode_solution(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Public interface for solution encoding.

        Exposes the internal _encode_solution method for external use.
        Useful for providing initial solutions to pymoo algorithms.

        Args:
            solution_matrix: Solution matrix in domain-specific format

        Returns:
            Flat solution vector for pymoo algorithms
        """
        return self._encode_solution(solution_matrix)

    def evaluate_single_solution(self, solution_matrix: np.ndarray) -> dict[str, Any]:
        """
        Evaluate a single solution and return detailed results.

        This is a convenience method for analyzing individual solutions outside
        of the optimization loop. It provides more detailed output than the
        population-based _evaluate() method.

        USEFUL FOR:
        - Analyzing the initial GTFS solution
        - Debugging optimization results
        - Comparing solutions manually
        - Generating detailed reports

        Args:
            solution_matrix: Solution matrix of shape (n_routes, n_intervals)

        Returns:
            Dictionary containing:
            - 'objective': Objective function value
            - 'constraints': Array of constraint violations (if any)
            - 'feasible': Whether all constraints are satisfied
            - 'constraint_details': Individual constraint information

        Example:
            ```python
            # Evaluate initial GTFS solution
            initial_solution = optimization_data["initial_solution"]
            results = problem.evaluate_single_solution(initial_solution)

            print(f"Initial objective: {results['objective']:.4f}")
            print(f"Initial feasible: {results['feasible']}")

            # Compare with optimized solution
            best_matrix = problem.decode_solution(optimization_result.X)
            opt_results = problem.evaluate_single_solution(best_matrix)

            print(f"Improvement: {results['objective'] - opt_results['objective']:.4f}")
            ```
        """

        print("\n🔍 EVALUATING SINGLE SOLUTION:")
        print(f"   Solution shape: {solution_matrix.shape}")

        # Validate solution shape
        expected_shape = (self.n_routes, self.n_intervals)
        if solution_matrix.shape != expected_shape:
            print(
                f"   ❌ Invalid solution shape: expected {expected_shape}, got {solution_matrix.shape}"
            )
            return {
                "objective": np.inf,
                "constraints": np.array([]),
                "feasible": False,
                "constraint_details": [],
                "solution_matrix": solution_matrix.copy(),
            }

        # Evaluate objective
        try:
            objective_value = self.objective.evaluate(solution_matrix)
            print(f"   📊 Objective value: {objective_value:.4f}")
        except Exception as e:
            print(f"   ❌ Objective evaluation failed: {e}")
            objective_value = np.inf

        # Evaluate constraints
        constraint_violations = []
        constraint_details = []

        if self.constraints:
            print("   📋 Constraint evaluation:")

            for i, constraint in enumerate(self.constraints):
                try:
                    violations = constraint.evaluate(solution_matrix)
                    constraint_info = constraint.get_constraint_info()

                    constraint_violations.extend(violations)
                    constraint_details.append(
                        {
                            "handler_type": constraint_info["handler_type"],
                            "n_constraints": len(violations),
                            "violations": violations,
                            "satisfied": bool(np.all(violations <= 0)),
                        }
                    )

                    satisfied_count = np.sum(violations <= 0)
                    print(
                        f"      {i+1}. {constraint_info['handler_type']}: "
                        f"{satisfied_count}/{len(violations)} satisfied"
                    )

                except Exception as e:
                    print(f"      {i+1}. Constraint evaluation failed: {e}")
                    # Add placeholder violations
                    n_constr = constraint.n_constraints
                    failed_violations = np.full(n_constr, 1e6)
                    constraint_violations.extend(failed_violations)
                    constraint_details.append(
                        {
                            "handler_type": "FAILED",
                            "n_constraints": n_constr,
                            "violations": failed_violations,
                            "satisfied": False,
                        }
                    )

        # Determine feasibility
        constraint_violations = (
            np.array(constraint_violations) if constraint_violations else np.array([])
        )
        feasible = len(constraint_violations) == 0 or np.all(constraint_violations <= 0)

        print(f"   ✅ Solution feasible: {feasible}")

        return {
            "objective": objective_value,
            "constraints": constraint_violations,
            "feasible": feasible,
            "constraint_details": constraint_details,
            "solution_matrix": solution_matrix.copy(),
        }

    def get_problem_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the optimization problem.

        USEFUL FOR:
        - Debugging problem setup
        - Logging optimization configurations
        - Generating problem reports
        - Validating problem dimensions

        Returns:
            Dictionary with complete problem information
        """

        # Get objective information
        objective_info = {
            "type": type(self.objective).__name__,
            "class_name": self.objective.__class__.__name__,
        }

        # Get constraint information
        constraint_info = []
        for constraint in self.constraints:
            info = constraint.get_constraint_info()
            constraint_info.append(info)

        return {
            "problem_type": "TransitOptimizationProblem",
            "dimensions": {
                "n_routes": self.n_routes,
                "n_intervals": self.n_intervals,
                "n_choices": self.n_choices,
                "n_variables": self.n_var,
                "n_objectives": self.n_obj,
                "n_constraints": self.n_constr,
            },
            "objective": objective_info,
            "constraints": constraint_info,
            "variable_bounds": {"lower": self.xl, "upper": self.xu, "type": "integer"},
            "optimization_data_keys": list(self.optimization_data.keys()),
        }
