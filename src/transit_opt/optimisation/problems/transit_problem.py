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
from .base import (BaseConstraintHandler, FleetPerIntervalConstraintHandler,
                   FleetTotalConstraintHandler)


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

        # Check if DRT is enabled
        self.drt_enabled = optimization_data.get('drt_enabled', False)

        if self.drt_enabled:
            self.n_drt_zones = optimization_data["n_drt_zones"]
            self.var_structure = optimization_data["variable_structure"]

            print("   📊 Problem dimensions (PT+DRT):")
            print(f"      PT Routes: {self.n_routes}")
            print(f"      DRT Zones: {self.n_drt_zones}")
            print(f"      Time intervals: {self.n_intervals}")
            print(f"      PT Headway choices: {self.n_choices}")

            n_var = optimization_data["total_decision_variables"]
        else:
            self.n_drt_zones = 0
            self.var_structure = None

            print("   📊 Problem dimensions (PT-only):")
            print(f"      Routes: {self.n_routes}")
            print(f"      Time intervals: {self.n_intervals}")
            print(f"      Headway choices: {self.n_choices}")

            # PT-only logic
            n_var = self.n_routes * self.n_intervals

        # Calculate rest of pymoo problem parameters
        n_obj = 1  # Single objective optimization
        n_constr = sum(c.n_constraints for c in self.constraints)  # Total constraints

        print("   🔧 Pymoo parameters:")
        print(f"      Decision variables: {n_var}")
        print(f"      Objectives: {n_obj}")
        print(f"      Constraints: {n_constr}")

        # Define variable bounds (the indices of the minimum and maximum choices)
        xl = np.zeros(n_var, dtype=int)  # Lower bounds (index 0)

        # xu depends on whether DRT is enabled (combined bounds) or PT-only
        # if DRT is enabled and PT headway choices are not the same length as DRT fleet size
        # choices, then we need to use the combined variable bounds
        # e.g.:
            # allowed_headways = [0, 10, 20, 30]  # 4 choices (0-indexed 0-3)
            # allowed_fleet_sizes = [0, 5, 10]    # 3 choices (0-indexed 0-2)
            # combined_variable_bounds = [4, 4, 4, ..., 2, 2, 2]
            # first n_var_pt are headway choices, last n_var_drt are fleet size choices
            # note: the reason for the '4' in combined_variable_bounds for headways is that
            # we have an extra index for no_service
        if self.drt_enabled:
            # Use combined variable bounds for DRT+PT
            combined_bounds = optimization_data['combined_variable_bounds']
            xu = np.array(combined_bounds, dtype=int) - 1  # Convert choices to max indices
            print(f"      Total variables: {n_var}")
            print(f"      PT variables: {optimization_data['pt_decision_variables']}")
            print(f"      DRT variables: {optimization_data['drt_decision_variables']}")
        else:
            # Original PT-only bounds
            xu = np.full(n_var, self.n_choices - 1, dtype=int)  # Upper bounds (max index)

        #  Penalty method configuration
        self.penalty_config = penalty_config or {"enabled": False}
        self.use_penalty_method = self.penalty_config.get("enabled", False)
        self.penalty_weight = self.penalty_config.get("penalty_weight", 1000.0)
        self.constraint_penalty_weights = self.penalty_config.get(
            "constraint_weights", {}
        )

        # Store constraint info for penalty calculation
        self.constraint_names = [type(c).__name__ for c in (constraints or [])]

        # Initialize pymoo Problem
        if self.use_penalty_method:
            # No hard constraints - handle as penalties
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=0,  # 🔧 Zero constraints for penalty method
                xl=xl,
                xu=xu,
                vtype=int,
            )
            print(
                f"   🎯 Penalty method enabled: {len(constraints or [])} constraints → objective penalties"
            )
            print(f"   ⚖️ Base penalty weight: {self.penalty_weight}")

        else:
            # Use hard constraints (existing approach)
            super().__init__(
                n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, vtype=int
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
            solution = self.decode_solution(X[i])

            # Evaluate base objective (PT-only for now) #TODO: extend for DRT
            if self.drt_enabled:
                # For now, only pass PT part to objective functions
                base_objective = self.objective.evaluate(solution['pt'])
            else:
                # PT-only case unchanged
                base_objective = self.objective.evaluate(solution)


            if self.use_penalty_method and self.constraints:
                # 🔧 PENALTY METHOD: Add constraint violations to objective
                total_penalty = 0.0

                for j, constraint in enumerate(self.constraints):
                    # Smart constraint handling based on type and DRT status (same as hard constraints)
                    if isinstance(constraint, FleetTotalConstraintHandler) and self.drt_enabled:
                        # FleetTotalConstraintHandler can handle full PT+DRT solution
                        violations = constraint.evaluate(solution)
                    elif self.drt_enabled:
                        # Other constraints only handle PT part when DRT enabled
                        violations = constraint.evaluate(solution['pt'])
                    else:
                        # PT-only case: pass solution directly
                        violations = constraint.evaluate(solution)

                    # Get constraint-specific penalty weight
                    constraint_name = self.constraint_names[j]
                    constraint_weight = self._get_constraint_penalty_weight(
                        constraint_name
                    )

                    # Calculate penalty: sum of squared positive violations
                    constraint_penalty = (
                        np.sum(np.maximum(0, violations) ** 2) * constraint_weight
                    )
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
                            # Smart constraint handling based on type and DRT status
                            if isinstance(constraint, FleetTotalConstraintHandler) and self.drt_enabled:
                                # FleetTotalConstraintHandler can handle full PT+DRT solution
                                violations = constraint.evaluate(solution)
                            elif self.drt_enabled:
                                # Other constraints only handle PT part when DRT enabled
                                violations = constraint.evaluate(solution['pt'])
                            else:
                                # PT-only case: pass solution directly
                                violations = constraint.evaluate(solution)
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

                # Add per-constraint feasibility breakdown for hard constraints
                if G.shape[1] > 1:  # Multiple constraints
                    print("      Per-constraint feasibility breakdown:")
                    constraint_start_idx = 0

                    # 🔧 NEW: Track interval-specific feasibility for hard constraints too
                    interval_feasibility_hard = {}

                    for constraint_idx, constraint in enumerate(self.constraints):
                        constraint_end_idx = constraint_start_idx + constraint.n_constraints
                        constraint_violations = G[:, constraint_start_idx:constraint_end_idx]
                        constraint_satisfied = np.sum(np.all(constraint_violations <= 1e-6, axis=1))
                        constraint_name = self._get_constraint_name(constraint_idx)
                        print(f"        {constraint_name}: {constraint_satisfied}/{pop_size} solutions")

                        # 🔧 NEW: For FleetPerInterval, track individual interval feasibility
                        if isinstance(constraint, FleetPerIntervalConstraintHandler):
                            for interval_idx in range(constraint.n_constraints):
                                constraint_col_idx = constraint_start_idx + interval_idx
                                interval_satisfied = np.sum(constraint_violations[:, interval_idx] <= 1e-6)
                                interval_name = f"{constraint_name}_Interval_{interval_idx}"
                                interval_feasibility_hard[interval_name] = interval_satisfied

                        constraint_start_idx = constraint_end_idx

                    # 🔧 NEW: Print interval-specific breakdown for hard constraints
                    if interval_feasibility_hard:
                        print("      Per-interval feasibility breakdown:")
                        # Group by interval for cleaner display
                        interval_data = {}
                        for interval_name, satisfied_count in interval_feasibility_hard.items():
                            if 'Interval_' in interval_name:
                                interval_num = interval_name.split('_')[-1]
                                interval_data[int(interval_num)] = satisfied_count

                        # Print in interval order
                        for interval_idx in sorted(interval_data.keys()):
                            satisfied_count = interval_data[interval_idx]
                            interval_label = self._get_interval_label(interval_idx)
                            print(f"        Interval {interval_idx} ({interval_label}): {satisfied_count}/{pop_size} solutions")
            elif self.use_penalty_method and self.constraints:
                # Penalty method: evaluate original constraints to check feasibility
                feasible_count = 0

                # Track per-constraint and per-interval feasibility
                constraint_feasibility = {}
                interval_feasibility = {}

                for constraint_idx, constraint in enumerate(self.constraints):
                    constraint_name = constraint.__class__.__name__.replace('ConstraintHandler', '')
                    constraint_feasibility[constraint_name] = 0

                    # For FleetPerInterval, track each interval separately
                    if isinstance(constraint, FleetPerIntervalConstraintHandler):
                        for interval_idx in range(self.n_intervals):
                            interval_name = f"{constraint_name}_Interval_{interval_idx}"
                            interval_feasibility[interval_name] = 0

                for i in range(pop_size):
                    solution_matrix = self.decode_solution(X[i])
                    is_feasible = True

                    # Check all constraint handlers
                    for constraint_idx, constraint in enumerate(self.constraints):
                        constraint_name = constraint.__class__.__name__.replace('ConstraintHandler', '')
                          # Apply smart constraint handling here (FleetTotal works on full solution,
                          # others on PT only)
                        if isinstance(constraint, FleetTotalConstraintHandler) and self.drt_enabled:
                            # FleetTotalConstraintHandler can handle full PT+DRT solution
                            violations = constraint.evaluate(solution_matrix)
                        elif self.drt_enabled:
                            # Other constraints only handle PT part when DRT enabled
                            violations = constraint.evaluate(solution_matrix['pt'])
                        else:
                            # PT-only case: pass solution directly
                            violations = constraint.evaluate(solution_matrix)


                        # Check overall constraint feasibility
                        constraint_satisfied = np.all(violations <= 1e-6)
                        if constraint_satisfied:
                            constraint_feasibility[constraint_name] += 1
                        else:
                            is_feasible = False

                        # For FleetPerInterval, track individual interval feasibility
                        if isinstance(constraint, FleetPerIntervalConstraintHandler):
                            for interval_idx, interval_violation in enumerate(violations):
                                interval_name = f"{constraint_name}_Interval_{interval_idx}"
                                if interval_violation <= 1e-6:
                                    interval_feasibility[interval_name] += 1

                    if is_feasible:
                        feasible_count += 1

                print(f"      Feasible solutions: {feasible_count}/{pop_size}")

                # Print detailed constraint breakdown
                if len(self.constraints) > 1:
                    print("      Per-constraint feasibility breakdown:")
                    for constraint_name, satisfied_count in constraint_feasibility.items():
                        print(f"        {constraint_name}: {satisfied_count}/{pop_size} solutions")

                # Print interval-specific breakdown for FleetPerInterval
                if interval_feasibility:
                    print("      Per-interval feasibility breakdown:")
                    # Group by interval for cleaner display
                    interval_data = {}
                    for interval_name, satisfied_count in interval_feasibility.items():
                        if 'Interval_' in interval_name:
                            interval_num = interval_name.split('_')[-1]
                            interval_data[int(interval_num)] = satisfied_count

                    # Print in interval order
                    for interval_idx in sorted(interval_data.keys()):
                        satisfied_count = interval_data[interval_idx]
                        interval_label = self._get_interval_label(interval_idx)
                        print(f"        Interval {interval_idx} ({interval_label}): {satisfied_count}/{pop_size} solutions")

    def _get_constraint_penalty_weight(self, constraint_name: str) -> float:
        """Get penalty weight for specific constraint type."""
        # 1. Check for specific weight first
        if constraint_name in self.constraint_penalty_weights:
            return self.constraint_penalty_weights[constraint_name]

        # 2. Check simplified name patterns
        simplified_patterns = {
            "FleetTotalConstraintHandler": "fleet_total",
            "FleetPerIntervalConstraintHandler": "fleet_per_interval",
            "MinimumFleetConstraintHandler": "minimum_fleet",
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
            For PT only: Solution matrix of shape (n_routes, n_intervals) with integer indices
            For PT+DRT: Solution matrix of shape: dict with 'pt' and 'drt' keys

        Example:
            >>> x_flat = np.array([0, 1, 2, 3, 1, 0])  # 2 routes, 3 intervals
            >>> matrix = problem._decode_solution(x_flat)
            >>> print(matrix)
            [[0 1 2]
             [3 1 0]]
        """

        if not self.drt_enabled:
            # 1. Clip to bounds (PSO can generate out-of-bounds values). Probably redundant
            # as Pymoo should handle this (but just in case)
            x_bounded = np.clip(x_flat, 0, self.n_choices - 1)
            # 2. Round and convert to integers
            x_int = np.round(x_bounded).astype(int)

            return x_int.reshape(self.n_routes, self.n_intervals)

        # DRT-enabled case: use proper variable-specific bounds
        pt_size = self.var_structure['pt_size']
        pt_shape = self.var_structure['pt_shape']
        drt_shape = self.var_structure['drt_shape']

        # split the flat vector first
        pt_flat = x_flat[:pt_size]
        drt_flat = x_flat[pt_size:]
        # Apply correct bounds to each part
        pt_bounded = np.clip(pt_flat, 0, self.n_choices - 1) # Pt uses headway bounds
        # DRT uses fleet size bounds from combined_variable_bounds
        drt_bounds = self.optimization_data['combined_variable_bounds'][pt_size:]
        drt_bounded = np.clip(drt_flat, 0, np.array(drt_bounds) - 1)

        # Convert to matrices and reshape
        pt_matrix = np.round(pt_bounded).astype(int).reshape(pt_shape)
        drt_matrix = np.round(drt_bounded).astype(int).reshape(drt_shape)

        return {
            'pt': pt_matrix,
            'drt': drt_matrix
        }


    def _encode_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Convert route×interval matrix or PT+DRT dict to flat solution vector.

        This is the inverse operation of _decode_solution(). Useful for:
        - Converting initial GTFS solution to pymoo format
        - Post-processing optimization results
        - Debugging and validation

        Args:
            solution: 
                - PT only: Solution matrix of shape (n_routes, n_intervals)
                - PT+DRT: dict with 'pt'/'drt' keys


        Returns:
            Flat solution vector of length (n_routes × n_intervals)

        Example:
            >>> matrix = np.array([[0, 1, 2], [3, 1, 0]])
            >>> x_flat = problem._encode_solution(matrix)
            >>> print(x_flat)
            [0 1 2 3 1 0]
        """
        if not self.drt_enabled:
            # PT-only: solution should be a matrix
            if isinstance(solution, dict):
                # Handle case where PT-only solution is passed as dict
                return solution['pt'].flatten()
            else:
                return solution.flatten()
        else:
            # DRT-enabled: solution is a dict
            if not isinstance(solution, dict) or 'pt' not in solution or 'drt' not in solution:
                raise ValueError("DRT-enabled problems require solution dict with 'pt' and 'drt' keys")

            pt_flat = solution['pt'].flatten()
            drt_flat = solution['drt'].flatten()
            return np.concatenate([pt_flat, drt_flat])

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
        # Handle different solution formats
        if self.drt_enabled:
            if isinstance(solution_matrix, dict):
                print(f"   PT solution shape: {solution_matrix['pt'].shape}")
                print(f"   DRT solution shape: {solution_matrix['drt'].shape}")
                # Validate PT shape
                expected_pt_shape = (self.n_routes, self.n_intervals)
                if solution_matrix['pt'].shape != expected_pt_shape:
                    print(f"   ❌ Invalid PT shape: expected {expected_pt_shape}, got {solution_matrix['pt'].shape}")
                    return {"objective": np.inf, "constraints": np.array([]), "feasible": False, "constraint_details": []}
            else:
                print(f"   ❌ DRT-enabled problem expects dict format, got {type(solution_matrix)}")
                return {"objective": np.inf, "constraints": np.array([]), "feasible": False, "constraint_details": []}
        else:
            print(f"   Solution shape: {solution_matrix.shape}")
            # Validate solution shape for PT-only
            expected_shape = (self.n_routes, self.n_intervals)
            if solution_matrix.shape != expected_shape:
                print(f"   ❌ Invalid solution shape: expected {expected_shape}, got {solution_matrix.shape}")
                return {"objective": np.inf, "constraints": np.array([]), "feasible": False, "constraint_details": []}

        # Evaluate objective
        try:
            # For now, objectives only handle PT part #TODO: extend for DRT
            if self.drt_enabled:
                if not isinstance(solution_matrix, dict):
                    raise ValueError("DRT-enabled problems expect dict solution format")
                objective_value = self.objective.evaluate(solution_matrix['pt'])
            else:
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
                    # For now, constraints only handle PT part TODO: extend for DRT
                    if self.drt_enabled:
                        violations = constraint.evaluate(solution_matrix['pt'])
                    else:
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

    def _get_constraint_name(self, constraint_idx: int) -> str:
        """Get readable name for constraint by index."""
        if constraint_idx < len(self.constraints):
            constraint = self.constraints[constraint_idx]
            return constraint.__class__.__name__.replace('ConstraintHandler', '')
        return f"Constraint_{constraint_idx}"

    def _get_interval_label(self, interval_idx: int) -> str:
        """Get readable label for time interval."""
        try:
            if hasattr(self, 'optimization_data') and 'intervals' in self.optimization_data:
                labels = self.optimization_data['intervals']['labels']
                if interval_idx < len(labels):
                    return labels[interval_idx]
        except:
            pass
        return f"Period_{interval_idx}"


    def is_feasible(self, solution_flat: np.ndarray) -> bool:
        """
        Check if a solution satisfies all constraints.
        
        Leverages existing constraint evaluation logic from evaluate_single_solution
        for consistency and code reuse.
        
        Args:
            solution_flat: Flat solution vector from particle
            
        Returns:
            bool: True if solution is feasible (satisfies all constraints)
        """
        if not self.constraints:
            return True  # No constraints means always feasible

        # Decode and use existing evaluation logic
        solution_matrix = self.decode_solution(solution_flat)

        # Use existing constraint evaluation logic
        for constraint in self.constraints:
            try:
                violations = constraint.evaluate(solution_matrix)

                # Check if any constraint is violated
                if np.any(violations > 0):
                    return False


            except Exception:
                # If constraint evaluation fails, consider infeasible
                return False

        return True  # All constraints satisfied
