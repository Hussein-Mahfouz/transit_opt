"""
Main pymoo Problem class for transit optimization.

This module provides the core optimization problem class that bridges the domain-specific
transit optimization logic with pymoo's metaheuristic algorithms. It handles solution
encoding/decoding and population-based evaluation.
"""

import logging
from typing import Any

import numpy as np
from pymoo.core.problem import Problem

from ..objectives.base import BaseObjective
from .base import BaseConstraintHandler, FleetPerIntervalConstraintHandler, FleetTotalConstraintHandler

logger = logging.getLogger(__name__)


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
        from transit_opt.optimisation.objectives.service_coverage import StopCoverageObjective
        from transit_opt.optimisation.problems.base import FleetTotalConstraintHandler

        # Create objective
        spatial_equity = StopCoverageObjective(
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
        fixed_intervals: list[int] | None = None,
    ):
        logger.info("🏗️  CREATING TRANSIT OPTIMIZATION PROBLEM:")

        # Store components
        self.optimization_data = optimization_data
        self.objective = objective
        self.constraints = constraints or []

        # Handle fixed intervals
        # Store as sorted list of unique indices
        self.fixed_intervals = sorted(list(set(fixed_intervals))) if fixed_intervals else []
        self.n_intervals = optimization_data["n_intervals"]

        # Validate fixed intervals
        if self.fixed_intervals:
            invalid = [i for i in self.fixed_intervals if i < 0 or i >= self.n_intervals]
            if invalid:
                msg = f"Invalid fixed_intervals: {invalid}. Must be in range [0, {self.n_intervals - 1}]"
                logger.error(msg)
                raise ValueError(msg)

            # Determine active intervals (indices that WILL be optimized)
            self.active_intervals = [i for i in range(self.n_intervals) if i not in self.fixed_intervals]
            logger.info(f"🔒 Fixed intervals (frozen): {self.fixed_intervals}")
            logger.info(f"🔓 Active intervals (optimizing): {self.active_intervals}")
        else:
            self.active_intervals = list(range(self.n_intervals))

        # Extract problem dimensions from optimization data
        self.n_routes = optimization_data["n_routes"]
        self.n_intervals = optimization_data["n_intervals"]
        self.n_choices = optimization_data["n_choices"]

        # Check if DRT is enabled
        self.drt_enabled = optimization_data.get("drt_enabled", False)

        if self.drt_enabled:
            self.n_drt_zones = optimization_data["n_drt_zones"]
            self.var_structure = optimization_data["variable_structure"]
            # We must set this too for masked variable decoding in combined mode
            self.var_structure["pt_shape"] = (self.n_routes, self.n_intervals)
            self.var_structure["drt_shape"] = (self.n_drt_zones, self.n_intervals)

            logger.info(
                f"""📊 Problem dimensions (PT+DRT):
                • PT routes: {self.n_routes}
                • DRT zones: {self.n_drt_zones}
                • Intervals: {self.n_intervals}
                • Headway choices: {self.n_choices}"""
            )

            if self.fixed_intervals:
                # Calculate reduced variables for both PT and DRT
                # Assuming DRT variables are also per-interval (zones * intervals)
                n_var_pt = self.n_routes * len(self.active_intervals)
                n_var_drt = self.n_drt_zones * len(self.active_intervals)
                n_var = n_var_pt + n_var_drt

                logger.info(
                    f"   🔒 Masking enabled! Optimization variables reduced from {optimization_data['total_decision_variables']} to {n_var}"
                )
            else:
                n_var = optimization_data["total_decision_variables"]
        else:
            self.n_drt_zones = 0
            self.var_structure = None

            logger.info(
                f"""📊 Problem dimensions (PT-only):
            • Routes: {self.n_routes}
            • Time intervals: {self.n_intervals}
            • Headway choices: {self.n_choices}
            """
            )

            # PT-only logic: n_routes * active_intervals
            n_var = self.n_routes * len(self.active_intervals)

            if self.fixed_intervals:
                total_possible = self.n_routes * self.n_intervals
                logger.info(f"   🔒 Masking enabled! Optimization variables reduced from {total_possible} to {n_var}")

        # Calculate rest of pymoo problem parameters
        n_obj = 1  # Single objective optimization
        n_constr = sum(c.n_constraints for c in self.constraints)  # Total constraints

        logger.info(
            f"""🔧 Pymoo parameters:
            • Decision variables: {n_var}
            • Objectives: {n_obj}
            • Constraints: {n_constr}
            """
        )

        # Define variable bounds (the indices of the minimum and maximum choices)
        xl = np.zeros(n_var, dtype=int)  # Lower bounds (index 0)

        # xu depends on whether DRT is enabled (combined bounds) or PT-only
        if self.drt_enabled:
            if self.fixed_intervals:
                # Reconstruct bounds for reduced variables
                # PT part: all same n_choices
                pt_bounds = [self.n_choices] * (self.n_routes * len(self.active_intervals))

                # DRT part: Extract per-zone bounds from combined_bounds
                # combined_bounds structure: [pt_vars..., drt_vars...]
                orig_pt_size = self.var_structure["pt_size"]
                orig_drt_bounds = optimization_data["combined_variable_bounds"][orig_pt_size:]

                # Reshape original DRT bounds to (n_zones, n_intervals)
                orig_drt_bounds_matrix = np.array(orig_drt_bounds).reshape(self.n_drt_zones, self.n_intervals)

                # Slice for active intervals
                active_drt_bounds = orig_drt_bounds_matrix[:, self.active_intervals].flatten()

                combined_bounds = pt_bounds + active_drt_bounds.tolist()
                xu = np.array(combined_bounds, dtype=int) - 1
            else:
                combined_bounds = optimization_data["combined_variable_bounds"]
                xu = np.array(combined_bounds, dtype=int) - 1

            logger.info(f"      Total variables: {n_var}")
            if not self.fixed_intervals:
                logger.info(f"      PT variables: {optimization_data['pt_decision_variables']}")
                logger.info(f"      DRT variables: {optimization_data['drt_decision_variables']}")
        else:
            # Original PT-only bounds
            xu = np.full(n_var, self.n_choices - 1, dtype=int)  # Upper bounds (max index)

        #  Penalty method configuration
        self.penalty_config = penalty_config or {"enabled": False}
        self.use_penalty_method = self.penalty_config.get("enabled", False)
        self.penalty_weight = self.penalty_config.get("penalty_weight", 1000.0)
        self.constraint_penalty_weights = self.penalty_config.get("constraint_weights", {})

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
            logger.info(f"   🎯 Penalty method enabled: {len(constraints or [])} constraints → objective penalties")
            logger.info(f"   ⚖️ Base penalty weight: {self.penalty_weight}")

        else:
            # Use hard constraints (existing approach)
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, vtype=int)
            logger.info(f"   🚦 Hard constraints: {n_constr} constraint(s)")

        # Log constraint details
        if self.constraints:
            logger.info("   📋 Constraint breakdown:")
            for i, constraint in enumerate(self.constraints):
                constraint_info = constraint.get_constraint_info()
                logger.info(
                    f"      {i + 1}. {constraint_info['handler_type']}: "
                    f"{constraint_info['n_constraints']} constraint(s)"
                )
        else:
            logger.info("   📋 No constraints specified (unconstrained optimization)")

        # Store full initial solution for fixed variable reference (masked optimization)
        if self.fixed_intervals:
            if "initial_solution" not in optimization_data:
                msg = "Masked optimization requires 'initial_solution' in optimization_data"
                logger.error(msg)
                raise ValueError(msg)

            # Encode initial solution to flat vector to use as base for reconstruction
            # CRITICAL: We pass apply_mask=False here because we need the FULL solution template
            # for decoding, even if we are optimizing a subset of variables.
            self.initial_solution_flat = self._encode_solution(optimization_data["initial_solution"], apply_mask=False)
            logger.info("   🔒 Stored initial solution for fixed interval reconstruction")

        logger.info("   ✅ Problem setup complete!")

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
            # 1. Decode solution
            solution = self.decode_solution(X[i])

            # 2. Evaluate base objective for this solution
            base_objective = self.objective.evaluate(solution)

            # 3. Handle constraints for this solution
            if self.use_penalty_method and self.constraints:
                # 🔧 PENALTY METHOD: Add constraint violations to objective
                total_penalty = 0.0

                for j, constraint in enumerate(self.constraints):
                    # Smart constraint handling based on type and DRT status (same as hard constraints)
                    if (
                        isinstance(constraint, FleetTotalConstraintHandler | FleetPerIntervalConstraintHandler)
                        and self.drt_enabled
                    ):
                        # Handlers that can evaluate full PT+DRT solutions
                        violations = constraint.evaluate(solution)
                    elif self.drt_enabled:
                        # Other constraints only handle PT part when DRT enabled
                        violations = constraint.evaluate(solution["pt"])
                    else:
                        # PT-only case: pass solution directly
                        violations = constraint.evaluate(solution)

                    # Get constraint-specific penalty weight
                    constraint_name = self.constraint_names[j]
                    constraint_weight = self._get_constraint_penalty_weight(constraint_name)
                    # Calculate penalty: sum of squared positive violations
                    constraint_penalty = np.sum(np.maximum(0, violations)) * constraint_weight
                    total_penalty += constraint_penalty

                # Add penalty to objective
                F[i, 0] = base_objective + total_penalty

            else:
                # Hard constraints
                F[i, 0] = base_objective

                if self.constraints and G is not None:
                    constraint_start_idx = 0

                    for constraint in self.constraints:
                        try:
                            # Smart constraint handling based on type and DRT status
                            if (
                                isinstance(constraint, FleetTotalConstraintHandler | FleetPerIntervalConstraintHandler)
                                and self.drt_enabled
                            ):
                                # Handlers that can evaluate full PT+DRT solutions
                                violations = constraint.evaluate(solution)
                            elif self.drt_enabled:
                                # Other constraints only handle PT part when DRT enabled
                                violations = constraint.evaluate(solution["pt"])
                            else:
                                # PT-only case: pass solution directly
                                violations = constraint.evaluate(solution)
                            # Store violations in correct positions
                            constraint_end_idx = constraint_start_idx + len(violations)
                            G[i, constraint_start_idx:constraint_end_idx] = violations
                            constraint_start_idx = constraint_end_idx

                        except Exception as e:
                            logger.error(f"   ⚠️  Constraint evaluation failed for solution {i}: {e}")
                            # Assign large positive violations (constraint violated)
                            constraint_end_idx = constraint_start_idx + constraint.n_constraints
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

            logger.info(f"""
                        📊 Evaluation summary:
                        • Best objective {best_obj},
                        • Worst objective {worst_obj},
                        • Average objective {avg_obj}
                        """)

            # ===== HARD CONSTRAINTS LOGGING =====
            if G is not None:
                feasible_solutions = np.sum(np.all(G <= 0, axis=1))
                logger.info(f"      Feasible solutions: {feasible_solutions}/{pop_size}")

                # Add per-constraint feasibility breakdown for hard constraints
                if G.shape[1] > 1:  # Multiple constraints
                    logger.info("      Per-constraint feasibility breakdown:")
                    constraint_start_idx = 0

                    # Track interval-specific feasibility for hard constraints too
                    interval_feasibility_hard = {}

                    for constraint_idx, constraint in enumerate(self.constraints):
                        constraint_end_idx = constraint_start_idx + constraint.n_constraints
                        constraint_violations = G[:, constraint_start_idx:constraint_end_idx]
                        constraint_satisfied = np.sum(np.all(constraint_violations <= 1e-6, axis=1))
                        constraint_name = self._get_constraint_name(constraint_idx)
                        logger.info(f"        {constraint_name}: {constraint_satisfied}/{pop_size} solutions")

                        # For FleetPerInterval, track individual interval feasibility
                        if isinstance(constraint, FleetPerIntervalConstraintHandler):
                            # Check if both ceiling and floor constraints exist
                            has_ceiling = constraint.config.get("tolerance") is not None
                            has_floor = constraint.config.get("min_fraction") is not None

                            # Number of actual time intervals (NOT number of constraints!)
                            n_intervals = self.n_intervals

                            # Track ceiling constraints (first n_intervals violations)
                            if has_ceiling:
                                for interval_idx in range(n_intervals):
                                    constraint_col_idx = constraint_start_idx + interval_idx
                                    interval_violations = G[:, constraint_col_idx]
                                    interval_satisfied = np.sum(interval_violations <= 1e-6)
                                    interval_name = f"{constraint_name}_Ceiling_Interval_{interval_idx}"
                                    interval_feasibility_hard[interval_name] = interval_satisfied

                            # Track floor constraints (last n_intervals violations)
                            if has_floor:
                                floor_start_idx = constraint_start_idx + (n_intervals if has_ceiling else 0)
                                for interval_idx in range(n_intervals):
                                    constraint_col_idx = floor_start_idx + interval_idx
                                    interval_violations = G[:, constraint_col_idx]
                                    interval_satisfied = np.sum(interval_violations <= 1e-6)
                                    interval_name = f"{constraint_name}_Floor_Interval_{interval_idx}"
                                    interval_feasibility_hard[interval_name] = interval_satisfied

                        constraint_start_idx = constraint_end_idx

                    # Print interval-specific breakdown for hard constraints
                    if interval_feasibility_hard:
                        logger.info("      Per-interval feasibility breakdown:")

                        # Sort keys to ensure consistent output order
                        sorted_keys = sorted(interval_feasibility_hard.keys())

                        for key in sorted_keys:
                            satisfied_count = interval_feasibility_hard[key]

                            # Try to extract interval index to get label
                            try:
                                parts = key.split("_")
                                interval_idx = int(parts[-1])
                                interval_label = self._get_interval_label(interval_idx)
                                logger.info(f"        {key} ({interval_label}): {satisfied_count}/{pop_size} solutions")
                            except (ValueError, IndexError):
                                # Fallback if format is unexpected or parsing fails
                                logger.info(f"        {key}: {satisfied_count}/{pop_size} solutions")

            # ===== PENALTY METHOD LOGGING =====
            elif self.use_penalty_method and self.constraints:
                # Penalty method: evaluate original constraints to check feasibility
                feasible_count = 0

                # Track per-constraint and per-interval feasibility
                constraint_feasibility = {}
                interval_feasibility = {}

                for constraint_idx, constraint in enumerate(self.constraints):
                    # Use unique name for each instance
                    base_name = constraint.__class__.__name__.replace("ConstraintHandler", "")
                    constraint_name = f"{base_name}_{constraint_idx}"
                    constraint_feasibility[constraint_name] = 0

                    # For FleetPerInterval, track each interval separately
                    if isinstance(constraint, FleetPerIntervalConstraintHandler):
                        has_ceiling = constraint.config.get("tolerance") is not None
                        has_floor = constraint.config.get("min_fraction") is not None

                        # Track each interval for both types
                        for interval_idx in range(self.n_intervals):
                            if has_ceiling:
                                interval_name = f"{constraint_name}_Ceiling_{interval_idx}"
                                interval_feasibility[interval_name] = 0
                            if has_floor:
                                interval_name = f"{constraint_name}_Floor_{interval_idx}"
                                interval_feasibility[interval_name] = 0

                for i in range(pop_size):
                    solution_matrix = self.decode_solution(X[i])
                    is_feasible = True

                    # Check all constraint handlers
                    for constraint_idx, constraint in enumerate(self.constraints):
                        # Use unique name for each instance
                        base_name = constraint.__class__.__name__.replace("ConstraintHandler", "")
                        constraint_name = f"{base_name}_{constraint_idx}"

                        if constraint_name not in constraint_feasibility:
                            constraint_feasibility[constraint_name] = 0

                        # Determine correct input for constraints
                        if hasattr(self, "drt_enabled") and self.drt_enabled:
                            # FleetTotal and FleetPerInterval can handle full PT+DRT solution now
                            if isinstance(constraint, (FleetTotalConstraintHandler, FleetPerIntervalConstraintHandler)):
                                violations = constraint.evaluate(solution_matrix)
                            else:
                                # Older/Other constraints only know about PT part
                                pt_part = (
                                    solution_matrix["pt"] if isinstance(solution_matrix, dict) else solution_matrix
                                )
                                violations = constraint.evaluate(pt_part)
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
                            # The violation array depends on what constraints are configured
                            has_ceiling = constraint.config.get("tolerance") is not None
                            has_floor = constraint.config.get("min_fraction") is not None
                            n_intervals = self.n_intervals

                            # violations array structure: [ceiling_violations..., floor_violations...]

                            offset = 0
                            # Track ceiling constraints (first n_intervals violations if ceiling active)
                            if has_ceiling:
                                for interval_idx in range(n_intervals):
                                    # Ensure we don't go out of bounds
                                    if offset + interval_idx < len(violations):
                                        violation = violations[offset + interval_idx]
                                        interval_key = f"{constraint_name}_Ceiling_{interval_idx}"
                                        if interval_key not in interval_feasibility:
                                            interval_feasibility[interval_key] = 0
                                        if violation <= 1e-6:
                                            interval_feasibility[interval_key] += 1
                                offset += n_intervals

                            # Track floor constraints (next n_intervals violations if floor active)
                            if has_floor:
                                for interval_idx in range(n_intervals):
                                    if offset + interval_idx < len(violations):
                                        violation = violations[offset + interval_idx]
                                        interval_key = f"{constraint_name}_Floor_{interval_idx}"
                                        if interval_key not in interval_feasibility:
                                            interval_feasibility[interval_key] = 0
                                        if violation <= 1e-6:
                                            interval_feasibility[interval_key] += 1

                    if is_feasible:
                        feasible_count += 1

                logger.info(f"      Feasible solutions: {feasible_count}/{pop_size}")

                # Print detailed constraint breakdown
                if len(self.constraints) > 1:
                    logger.info("      Per-constraint feasibility breakdown:")
                    for constraint_name, satisfied_count in constraint_feasibility.items():
                        logger.info(f"        {constraint_name}: {satisfied_count}/{pop_size} solutions")

                # Print interval-specific breakdown for FleetPerInterval
                if interval_feasibility:
                    logger.info("      Per-interval feasibility breakdown:")

                    # Sort keys to ensure consistent output order
                    sorted_keys = sorted(interval_feasibility.keys())

                    for key in sorted_keys:
                        satisfied_count = interval_feasibility[key]

                        # Try to extract interval index to get label
                        # Expected format: Name_Index_Type_IntervalIndex (e.g. FleetPerInterval_0_Ceiling_1)
                        try:
                            parts = key.split("_")
                            interval_idx = int(parts[-1])
                            interval_label = self._get_interval_label(interval_idx)
                            logger.info(f"        {key} ({interval_label}): {satisfied_count}/{pop_size} solutions")
                        except (ValueError, IndexError):
                            # Fallback if format is unexpected or parsing fails
                            logger.info(f"        {key}: {satisfied_count}/{pop_size} solutions")

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

        Handles NaN/Inf from PSO numerical instability by replacing
        with valid defaults instead of crashing.

        Supports masked optimization (fixed intervals):
        - If fixed_intervals are set, expands reduced x_flat to full size using initial_solution.

        ENCODING DETAILS:
        - Flat vector: [r0i0, r0i1, r0i2, r1i0, r1i1, r1i2, ...]
        - Matrix format: [[r0i0, r0i1, r0i2], [r1i0, r1i1, r1i2], ...]
        - Values: Indices into optimization_data["allowed_headways"]

        Args:
            x_flat: Flat solution vector of length (n_routes × n_intervals) or reduced length.

        Returns:
            For PT only: Solution matrix of shape (n_routes, n_intervals) with integer indices
            For PT+DRT: Solution matrix of shape: dict with 'pt' and 'drt' keys
        """

        # ===== STEP 0: HANDLE MASKED VARIABLES (FIXED INTERVALS) =====
        if hasattr(self, "fixed_intervals") and self.fixed_intervals:
            # We are in masked mode. x_flat is REDUCED.
            # We need to reconstruct the full vector using initial_solution values for fixed parts.

            # Create full-size array initialized with the fixed values
            full_x = self.initial_solution_flat.copy()

            # We need to map the reduced x_flat values into the correct positions in full_x.
            # The mapping depends on how we reduced it in __init__.

            # Logic mirrors __init__ reduction:
            # PT: sequential routes, filtered intervals
            # DRT: sequential zones, filtered intervals

            current_x_idx = 0

            # 1. Fill PT part
            if self.drt_enabled:
                n_pt_vars_full = self.var_structure["pt_size"]
                n_pt_routes = self.n_routes
            else:
                n_pt_vars_full = self.n_routes * self.n_intervals
                n_pt_routes = self.n_routes

            # Iterate through routes (PT)
            for r in range(n_pt_routes):
                for i_idx in range(self.n_intervals):
                    # Calculate index in full vector
                    full_idx = r * self.n_intervals + i_idx

                    if i_idx in self.active_intervals:
                        # Safety check
                        if current_x_idx >= len(x_flat):
                            logger.error(f"Decoding Error: current_x_idx {current_x_idx} > len(x_flat) {len(x_flat)}")
                            logger.error(
                                f"Routes: {n_pt_routes}, Intervals: {self.n_intervals}, Active: {len(self.active_intervals)}"
                            )
                            logger.error(f"Current r={r}, i={i_idx}")
                            raise IndexError(f"Decoding out of bounds: {current_x_idx} >= {len(x_flat)}")

                        # This variable is active, take from optimization vector
                        full_x[full_idx] = x_flat[current_x_idx]
                        current_x_idx += 1

            # 2. Fill DRT part (if enabled)
            if self.drt_enabled:
                n_drt_zones = self.n_drt_zones
                pt_offset = self.var_structure["pt_size"]

                for z in range(n_drt_zones):
                    for i_idx in range(self.n_intervals):
                        # Calculate index in full vector
                        # zones are appended after all PT vars
                        full_idx = pt_offset + (z * self.n_intervals + i_idx)

                        if i_idx in self.active_intervals:
                            if current_x_idx >= len(x_flat):
                                logger.error(
                                    f"During DRT decoding, ran out of variables in x_flat "
                                    f"(len={len(x_flat)}). Expected more active variables."
                                )
                                logger.error(f"Current z={z}, i={i_idx}")
                                raise IndexError(f"Decoding out of bounds: {current_x_idx} >= {len(x_flat)}")

                            # This variable is active
                            full_x[full_idx] = x_flat[current_x_idx]
                            current_x_idx += 1

            # Use the reconstructed full vector for the rest of decoding
            x_flat = full_x

        # ===== STEP 1: DETECT AND FIX BAD VALUES FROM PSO =====
        if x_flat is None:
            logger.warning("Values provided to decode_solution are None. Returning None.")
            return None

        if np.any(~np.isfinite(x_flat)):
            bad_indices = np.where(~np.isfinite(x_flat))[0]
            bad_values = x_flat[bad_indices]
            logger.warning(
                f"PSO generated non-finite values at indices {bad_indices}: {bad_values}. "
                f"Replacing with safe defaults to continue optimization."
            )

            # Replace NaN/Inf with middle-of-range values (safer than edges)
            if not self.drt_enabled:
                # PT-ONLY: Replace with no-service index (last index)
                no_service_index = self.n_choices - 1
                x_flat = np.where(np.isfinite(x_flat), x_flat, no_service_index)

                logger.warning(f"   Replaced {len(bad_indices)} NaN/Inf with no-service index {no_service_index}")
            else:
                # PT+DRT: Use variable-specific worst-case defaults
                pt_size = self.var_structure["pt_size"]
                combined_bounds = self.optimization_data["combined_variable_bounds"]

                # Create default array: no-service for PT, min-fleet (0) for DRT
                default_values = np.zeros(len(combined_bounds), dtype=int)
                default_values[:pt_size] = self.n_choices - 1  # PT: no-service index
                # DRT portion already 0 (minimum fleet)

                x_flat = np.where(np.isfinite(x_flat), x_flat, default_values)

                pt_bad = np.sum(bad_indices < pt_size)
                drt_bad = len(bad_indices) - pt_bad
                logger.warning(
                    f"   Replaced {pt_bad} PT NaN/Inf with no-service, {drt_bad} DRT NaN/Inf with min-fleet (0)"
                )

        # ===== STEP 2: PROCEED WITH NORMAL DECODING =====
        if not self.drt_enabled:
            # 1. Clip to bounds (PSO can generate out-of-bounds values). Probably redundant
            # as Pymoo should handle this (but just in case)
            x_bounded = np.clip(x_flat, 0, self.n_choices - 1)
            # 2. Round and convert to integers
            x_int = np.round(x_bounded).astype(int)

            # VALIDATION
            if np.any((x_int < 0) | (x_int >= self.n_choices)):
                invalid_mask = (x_int < 0) | (x_int >= self.n_choices)
                invalid_indices = np.where(invalid_mask)[0]
                raise ValueError(
                    f"Invalid PT solution indices after decoding: {invalid_indices}. "
                    f"Values: {x_int[invalid_mask]}. Valid range: [0, {self.n_choices - 1}]"
                )

            return x_int.reshape(self.n_routes, self.n_intervals)
        else:
            # DRT-enabled case: use proper variable-specific bounds
            pt_size = self.var_structure["pt_size"]
            pt_shape = self.var_structure["pt_shape"]
            drt_shape = self.var_structure["drt_shape"]

            # split the flat vector first
            pt_flat = x_flat[:pt_size]
            drt_flat = x_flat[pt_size:]
            # Apply correct bounds to each part
            pt_bounded = np.clip(pt_flat, 0, self.n_choices - 1)  # Pt uses headway bounds
            # DRT uses fleet size bounds from combined_variable_bounds
            drt_bounds = self.optimization_data["combined_variable_bounds"][pt_size:]
            drt_bounded = np.clip(drt_flat, 0, np.array(drt_bounds) - 1)

            # Convert to matrices and reshape
            pt_matrix = np.round(pt_bounded).astype(int).reshape(pt_shape)
            drt_matrix = np.round(drt_bounded).astype(int).reshape(drt_shape)

            # VALIDATION
            if np.any((pt_matrix < 0) | (pt_matrix >= self.n_choices)):
                invalid_mask = (pt_matrix < 0) | (pt_matrix >= self.n_choices)
                raise ValueError(
                    f"Invalid PT indices after decoding. "
                    f"Invalid count: {np.sum(invalid_mask)}. Valid range: [0, {self.n_choices - 1}]"
                )

            # Validate DRT indices
            for zone_idx in range(drt_matrix.shape[0]):
                max_drt_choice = drt_bounds[zone_idx] - 1
                if np.any(drt_matrix[zone_idx, :] > max_drt_choice):
                    raise ValueError(
                        f"Invalid DRT indices for zone {zone_idx}. "
                        f"Max allowed: {max_drt_choice}, "
                        f"Got: {drt_matrix[zone_idx, drt_matrix[zone_idx, :] > max_drt_choice]}"
                    )

            return {"pt": pt_matrix, "drt": drt_matrix}

    def _encode_solution(self, solution: np.ndarray, apply_mask: bool = True) -> np.ndarray:
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
            apply_mask: Whether to return only active variables (True) or full vector (False).
                       Default True (returns valid pymoo decision vector).


        Returns:
            Flat solution vector (containing only ACTIVE variables if masking is enabled)

        Example:
            >>> matrix = np.array([[0, 1, 2], [3, 1, 0]])
            >>> x_flat = problem._encode_solution(matrix)
            >>> print(x_flat)
            [0 1 2 3 1 0]
        """
        # If masked, we need to extract only the active values
        # This mirrors the decoding process where we ignore fixed values

        if not self.drt_enabled:
            # PT-only: solution should be a matrix
            if isinstance(solution, dict) and "pt" in solution:
                solution_matrix = solution["pt"]
            else:
                solution_matrix = solution

            if self.fixed_intervals and apply_mask:
                # Masked encoding: Extract only active columns
                # solution_matrix[:, active_intervals].flatten() is correct as it follows
                # the same order as decoding.
                return solution_matrix[:, self.active_intervals].flatten()
            else:
                return solution_matrix.flatten()
        else:
            # DRT-enabled
            if not isinstance(solution, dict) or "pt" not in solution or "drt" not in solution:
                # Fallback if just matrix passed in combined mode (sometimes happens in tests)
                # Assuming it's PT part
                if not isinstance(solution, dict):
                    if apply_mask:
                        logger.warning(
                            "Passed array to _encode_solution in DRT mode with mask applied. Assuming PT-only part."
                        )

                    # Simple logic: if matrix passed, treat as PT matrix
                    if self.fixed_intervals and apply_mask:
                        return solution[:, self.active_intervals].flatten()
                    return solution.flatten()

                logger.error("Invalid solution format for DRT-enabled problem.")
                raise ValueError("DRT-enabled problems require solution dict with 'pt' and 'drt' keys")

            pt_matrix = solution["pt"]
            drt_matrix = solution["drt"]

            if self.fixed_intervals and apply_mask:
                # Extract active parts for both
                pt_flat = pt_matrix[:, self.active_intervals].flatten()
                drt_flat = drt_matrix[:, self.active_intervals].flatten()
                return np.concatenate([pt_flat, drt_flat])
            else:
                pt_flat = pt_matrix.flatten()
                drt_flat = drt_matrix.flatten()
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

        logger.debug("\n🔍 EVALUATING SINGLE SOLUTION:")
        # Handle different solution formats
        if self.drt_enabled:
            if isinstance(solution_matrix, dict):
                logger.debug(f"   PT solution shape: {solution_matrix['pt'].shape}")
                logger.debug(f"   DRT solution shape: {solution_matrix['drt'].shape}")
                # Validate PT shape
                expected_pt_shape = (self.n_routes, self.n_intervals)
                if solution_matrix["pt"].shape != expected_pt_shape:
                    logger.error(
                        f"   ❌ Invalid PT shape: expected {expected_pt_shape}, got {solution_matrix['pt'].shape}"
                    )
                    return {
                        "objective": np.inf,
                        "constraints": np.array([]),
                        "feasible": False,
                        "constraint_details": [],
                    }
            else:
                logger.error(f"   ❌ DRT-enabled problem expects dict format, got {type(solution_matrix)}")
                return {"objective": np.inf, "constraints": np.array([]), "feasible": False, "constraint_details": []}
        else:
            logger.debug(f"   Solution shape: {solution_matrix.shape}")
            # Validate solution shape for PT-only
            expected_shape = (self.n_routes, self.n_intervals)
            if solution_matrix.shape != expected_shape:
                logger.error(f"   ❌ Invalid solution shape: expected {expected_shape}, got {solution_matrix.shape}")
                return {"objective": np.inf, "constraints": np.array([]), "feasible": False, "constraint_details": []}

        # Evaluate objective
        try:
            if self.drt_enabled:
                if not isinstance(solution_matrix, dict):
                    logger.error(f"   ❌ DRT-enabled problem expects dict format, got {type(solution_matrix)}")
                    raise ValueError("DRT-enabled problems expect dict solution format")
            objective_value = self.objective.evaluate(solution_matrix)
            logger.info(f"   📊 Objective value: {objective_value:.4f}")
        except Exception as e:
            logger.error(f"   ❌ Objective evaluation failed: {e}")
            objective_value = np.inf

        # Evaluate constraints
        constraint_violations = []
        constraint_details = []

        if self.constraints:
            logger.info("   📋 Constraint evaluation:")

            for i, constraint in enumerate(self.constraints):
                try:
                    # Determine correct input for constraints
                    if self.drt_enabled:
                        if isinstance(constraint, (FleetTotalConstraintHandler, FleetPerIntervalConstraintHandler)):
                            violations = constraint.evaluate(solution_matrix)
                        else:
                            # Older/Other constraints only know about PT part
                            violations = constraint.evaluate(solution_matrix["pt"])
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
                    logger.info(
                        f"      {i + 1}. {constraint_info['handler_type']}: "
                        f"{satisfied_count}/{len(violations)} satisfied"
                    )

                except Exception as e:
                    logger.error(f"      {i + 1}. Constraint evaluation failed: {e}")
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
        constraint_violations = np.array(constraint_violations) if constraint_violations else np.array([])
        feasible = len(constraint_violations) == 0 or np.all(constraint_violations <= 0)

        logger.info(f"   ✅ Solution feasible: {feasible}")

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
            base_name = constraint.__class__.__name__.replace("ConstraintHandler", "")
            return f"{base_name}_{constraint_idx}"
        return f"Constraint_{constraint_idx}"

    def _get_interval_label(self, interval_idx: int) -> str:
        """Get readable label for time interval."""
        try:
            if hasattr(self, "optimization_data") and "intervals" in self.optimization_data:
                labels = self.optimization_data["intervals"]["labels"]
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
        return True  # All constraints satisfied
        return True  # All constraints satisfied
