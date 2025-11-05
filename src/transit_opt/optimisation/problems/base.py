"""
Base classes for optimization problems and constraint handlers.

This module provides the foundational infrastructure for defining flexible
optimization problems with pluggable objectives and configurable constraints.
The design supports various metaheuristic algorithms through pymoo integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class BaseConstraintHandler(ABC):
    """
    Abstract base class for constraint handlers in transit optimization.

    Constraint handlers encapsulate specific constraint logic (fleet limits,
    service requirements, etc.) and can be mixed and matched in different
    optimization problems. Each handler calculates constraint violations
    independently.

    Key Design Principles:
    - Single Responsibility: Each handler manages one constraint type
    - Composable: Handlers can be combined in any configuration
    - Reusable: Same handler works across different objectives
    - Testable: Each handler can be validated independently

    Attributes:
        config (Dict[str, Any]): Configuration parameters for this constraint
        opt_data (Dict[str, Any]): Optimization data from GTFSDataPreparator
        n_constraints (int): Number of constraint values this handler produces

    Args:
        config: Constraint-specific configuration dictionary
        optimization_data: Complete optimization data structure

    Example:
        ```python
        # Configure fleet constraint
        fleet_config = {
            'baseline': 'current_peak',
            'tolerance': 0.15,
            'level': 'system'
        }

        # Create handler
        handler = FleetTotalConstraintHandler(fleet_config, opt_data)

        # Evaluate constraint for a solution
        violations = handler.evaluate(solution_matrix)
        # violations[i] <= 0 means constraint i is satisfied
        ```
    """

    def __init__(self, config: dict[str, Any], optimization_data: dict[str, Any]):
        """
        Initialize constraint handler with configuration and data.

        Args:
            config: Constraint-specific configuration parameters
            optimization_data: Complete optimization data from preparator
        """
        self.config = config
        self.opt_data = optimization_data
        self.n_intervals = optimization_data["n_intervals"]
        self.n_constraints = self._calculate_n_constraints()

        # Extract commonly used data for performance
        self.n_routes = optimization_data["n_routes"]
        self.n_intervals = optimization_data["n_intervals"]
        self.allowed_headways = optimization_data["allowed_headways"]
        self.no_service_index = optimization_data["no_service_index"]
        self.round_trip_times = optimization_data["routes"]["round_trip_times"]

        # Validate configuration during initialization
        self._validate_config()

    @abstractmethod
    def _calculate_n_constraints(self) -> int:
        """
        Calculate the number of constraint values this handler produces.

        Returns:
            Number of constraint equations (e.g., 1 for total fleet,
            n_intervals for per-interval fleet)
        """
        pass

    @abstractmethod
    def evaluate(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint violations for a given solution.

        Args:
            solution_matrix: Decision matrix (n_routes × n_intervals) with
                           headway choice indices

        Returns:
            Array of constraint violation values. Values <= 0 indicate
            the constraint is satisfied; positive values indicate violations.
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Override in subclasses to add specific validation logic.
        Default implementation does basic type checking.
        """
        if not isinstance(self.config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(self.config)}")

    def _calculate_fleet_from_solution(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate fleet requirements using shared GTFSDataPreparator logic.

        This ensures consistency with baseline fleet analysis by using
        the same calculation methods and operational parameters.
        """
        from ..utils.fleet_calculations import calculate_fleet_requirements

        # Extract operational parameters directly from optimization data
        fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]
        operational_buffer = fleet_analysis["operational_buffer"]
        no_service_threshold = fleet_analysis["no_service_threshold_minutes"]

        # Use shared calculation logic
        fleet_results = calculate_fleet_requirements(
            headways_matrix=solution_matrix,
            round_trip_times=self.round_trip_times,
            operational_buffer=operational_buffer,
            no_service_threshold=no_service_threshold,
            allowed_headways=self.allowed_headways,
            no_service_index=self.no_service_index,
        )

        return fleet_results["fleet_per_interval"]

    def _calculate_drt_fleet_from_solution(self, drt_solution_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate DRT fleet requirements from DRT solution matrix.

        Args:
            drt_solution_matrix: DRT decision matrix (n_drt_zones × n_intervals) with
                            fleet size choice indices

        Returns:
            Array of DRT fleet per interval (length n_intervals)
        """
        if not self.opt_data.get('drt_enabled', False):
            # Return zeros if DRT not enabled
            return np.zeros(self.n_intervals)

        # Get DRT configuration
        drt_zones = self.opt_data['drt_config']['zones']
        n_intervals = self.n_intervals

        # Initialize fleet per interval
        drt_fleet_per_interval = np.zeros(n_intervals)

        # Sum fleet across all DRT zones for each interval
        for zone_idx, zone_config in enumerate(drt_zones):
            allowed_fleet_sizes = zone_config['allowed_fleet_sizes']

            for interval_idx in range(n_intervals):
                # Get fleet choice index for this zone and interval
                fleet_choice_idx = drt_solution_matrix[zone_idx, interval_idx]

                # Convert choice index to actual fleet size
                fleet_size = allowed_fleet_sizes[fleet_choice_idx]

                # Add to total for this interval
                drt_fleet_per_interval[interval_idx] += fleet_size

        return drt_fleet_per_interval

    def get_constraint_info(self) -> dict[str, Any]:
        """
        Get human-readable information about this constraint.

        Returns:
            Dictionary with constraint metadata for logging/debugging
        """
        return {
            "handler_type": self.__class__.__name__,
            "n_constraints": self.n_constraints,
            "config": self.config.copy(),
        }


class FleetTotalConstraintHandler(BaseConstraintHandler):
    """
    Constraint handler for total system fleet size limits including PT and DRT.

    This constraint ensures that the total number of vehicles across all PT routes
    and DRT zones and time intervals does not exceed a specified limit. When DRT is
    enabled, it automatically includes both PT and DRT fleet in the total count.

    Configuration Parameters:
        baseline (str): How to determine baseline fleet
            - 'current_peak': Use peak fleet from current GTFS
            - 'current_average': Use average fleet from current GTFS
            - 'manual': Use manually specified value
        baseline_value (float, optional): Manual baseline value (if baseline='manual')
        tolerance (float): Allowed increase above baseline (e.g., 0.15 = 15%)
        measure (str, optional): Which measure to constrain
            - 'peak': Constrain peak interval fleet
            - 'average': Constrain average across intervals
            - 'total': Constrain sum across all intervals (default)

    DRT Integration:
        When DRT is enabled (opt_data['drt_enabled'] = True):
        - Automatically includes DRT fleet in total calculations
        - Baseline includes initial DRT fleet allocation
        - All measures (peak/average/total) include both PT and DRT
        - Solution format expects dict with 'pt' and 'drt' keys

        When DRT is disabled:
        - Operates as PT-only constraint (backward compatible)
        - Solution format can be matrix or dict with 'pt' key

    Example Configuration with DRT:
        ```python
        # Total system fleet limit including PT + DRT
        config = {
            'baseline': 'current_peak',
            'tolerance': 0.20,  # 20% increase allowed
            'measure': 'peak'
        }

        # If current GTFS needs 100 vehicles at peak:
        # - Constraint limit = 100 * (1 + 0.20) = 120 vehicles
        # - Any solution requiring > 120 vehicles violates constraint
        # - Note that initial DRT fleet is always 0, so we do not need to calculate or add it to constraint limit
        handler = FleetTotalConstraintHandler(config, optimization_data)
        ```

        **Fixed Fleet Size:**
        ```python
        # Exactly 80 vehicles available (no increase allowed)
        config = {
            'baseline': 'manual',
            'baseline_value': 80,
            'tolerance': 0.0,
            'measure': 'peak'
        }
        handler = FleetTotalConstraintHandler(config, optimization_data)

        # Test PT+DRT solution
        solution = {
            'pt': np.array([[1, 2, 1], [0, 1, 2]]),  # PT headway choices
            'drt': np.array([[2, 1, 3], [1, 2, 0]])  # DRT fleet choices
        }
        violations = handler.evaluate(solution)
        # violations = [total_pt_fleet + total_drt_fleet - 156]
        # If violations[0] <= 0: constraint satisfied
        # If violations[0] > 0: need to reduce fleet by violations[0] vehicles
        ```

        **Average Fleet Control:**
        ```python
        # Limit average fleet across all time periods
        config = {
            'baseline': 'current_average',
            'tolerance': 0.10,
            'measure': 'average'
        }

        # If current GTFS averages 60 vehicles across time periods:
        # - Limit = 60 * 1.10 = 66 vehicles average
        # - Example solution (without DRT calculation)
        # - Solution with intervals [50, 70, 80, 60] = 65 avg ✓ (satisfied)
        # - Solution with intervals [70, 80, 90, 60] = 75 avg ✗ (violation = 9)
        ```

    Usage in Optimization Problems:
        ```python
        from transit_opt.optimisation.problems import FleetTotalConstraintHandler

        # Create constraint as part of optimization problem
        constraints = [
            FleetTotalConstraintHandler({
                'baseline': 'current_peak',
                'tolerance': 0.15,
                'measure': 'peak'
            }, opt_data)
        ]

        # In pymoo problem class:
        def _evaluate(self, x, out, *args, **kwargs):
            violations = []
            for constraint in constraints:
                violations.extend(constraint.evaluate(x))
            out["G"] = np.array(violations)  # G <= 0 for feasible solutions
        ```
    """

    def _calculate_n_constraints(self) -> int:
        """Single constraint: total fleet <= limit."""
        return 1

    def _validate_config(self) -> None:
        """Validate fleet constraint configuration."""
        super()._validate_config()

        # Check required parameters
        if "baseline" not in self.config:
            raise ValueError("Fleet constraint requires 'baseline' parameter")

        valid_baselines = ["current_peak", "current_average", "manual"]
        if self.config["baseline"] not in valid_baselines:
            raise ValueError(f"baseline must be one of {valid_baselines}")

        if self.config["baseline"] == "manual" and "baseline_value" not in self.config:
            raise ValueError("Manual baseline requires 'baseline_value' parameter")

        # Set defaults
        self.config.setdefault("tolerance", 0.1)
        self.config.setdefault("measure", "average")

    def evaluate(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluate total fleet constraint including both PT and DRT fleet when applicable.

        Args:
            solution_matrix: For PT-only: (n_routes × n_intervals) matrix
                            For PT+DRT: dict with 'pt' and 'drt' keys

        Returns:
            Single-element array: [total_fleet_usage - fleet_limit]
            Negative values indicate constraint satisfaction
        """
        # Check if DRT is enabled
        drt_enabled = self.opt_data.get('drt_enabled', False)

        if drt_enabled:
            # Handle PT+DRT case
            if not isinstance(solution_matrix, dict) or 'pt' not in solution_matrix or 'drt' not in solution_matrix:
                raise ValueError("DRT-enabled problems require solution dict with 'pt' and 'drt' keys")

            # Calculate PT fleet requirements
            pt_fleet_per_interval = self._calculate_fleet_from_solution(solution_matrix['pt'])

            # Calculate DRT fleet requirements
            drt_fleet_per_interval = self._calculate_drt_fleet_from_solution(solution_matrix['drt'])

            # Combine PT and DRT fleet
            total_fleet_per_interval = pt_fleet_per_interval + drt_fleet_per_interval

        else:
            # Handle PT-only case (existing logic)
            if isinstance(solution_matrix, dict):
                # If dict format but DRT not enabled, extract PT part
                pt_solution = solution_matrix.get('pt', solution_matrix)
                total_fleet_per_interval = self._calculate_fleet_from_solution(pt_solution)
            else:
                # Standard PT-only matrix
                total_fleet_per_interval = self._calculate_fleet_from_solution(solution_matrix)

        # Calculate fleet usage based on configured measure (same logic as before)
        if self.config["measure"] == "peak":
            fleet_usage = np.max(total_fleet_per_interval)
        elif self.config["measure"] == "average":
            fleet_usage = np.mean(total_fleet_per_interval)
        elif self.config["measure"] == "total":
            fleet_usage = np.sum(total_fleet_per_interval)
        else:
            raise ValueError(f"Unknown measure: {self.config['measure']}")

        # Get fleet limit based on baseline configuration
        fleet_limit = self._get_fleet_limit()

        # Calculate constraint violation
        violation = fleet_usage - fleet_limit

        return np.array([violation])

    def _get_fleet_limit(self) -> float:
        """
        Calculate the fleet limit based on baseline and tolerance.
        Baseline is based on PT fleet size only (as DRT does not
        exist in baseline scenario)
        """
        if self.config["baseline"] == "manual":
            baseline = self.config["baseline_value"]
        else:
            # Extract from fleet analysis in optimization data
            fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]

            if self.config["baseline"] == "current_peak":
                baseline = fleet_analysis.get("total_current_fleet_peak")
            elif self.config["baseline"] == "current_average":
                baseline = fleet_analysis.get("total_current_fleet_average")
            else:
                raise ValueError(f"Unknown baseline: {self.config['baseline']}")

        # Apply tolerance
        tolerance = self.config["tolerance"]
        fleet_limit = baseline * (1 + tolerance)

        return fleet_limit


class FleetPerIntervalConstraintHandler(BaseConstraintHandler):
    """
    Constraint handler for per-interval fleet size limits.

    This constraint ensures that fleet requirements for each time interval
    individually do not exceed specified limits and/or stay above minimum levels.
    Useful for scenarios where you have different capacity constraints by time of day.

    Configuration Parameters:
        baseline (str): How to determine baseline fleet per interval
            - 'current_by_interval': Use current GTFS fleet by interval
            - 'current_peak': Use current peak fleet for all intervals
            - 'manual': Use manually specified values
        baseline_values (List[float], optional): Manual values per interval
        tolerance (float): Allowed increase above baseline per interval
        min_fraction (float): Allowed reduction below baseline per interval
        allow_borrowing (bool, optional): Whether intervals can borrow unused
                                        capacity from other intervals

    Example Configuration:
        **Peak/Off-Peak Capacity Limits:**
        ```python
        # Different limits based on current service levels
        config = {
            'baseline': 'current_by_interval',
            'tolerance': 0.10  # 10% increase allowed per interval
        }

        # Suppose current GTFS needs: [40, 80, 60, 30] vehicles by interval
        # Constraint limits become: [44, 88, 66, 33] vehicles
        handler = FleetPerIntervalConstraintHandler(config, optimization_data)

        # Test solution
        solution_matrix = np.array([
            [2, 1, 2, 4],  # Route 0 service pattern
            [3, 0, 1, 4],  # Route 1 service pattern
            [1, 1, 3, 3]   # Route 2 service pattern
        ])

        violations = handler.evaluate(solution_matrix)
        # Returns: [interval_0_usage - 44, interval_1_usage - 88,
        #           interval_2_usage - 66, interval_3_usage - 33]

        # Interpretation:
        # violations[0] = -5  → Interval 0: 5 vehicles under limit ✓
        # violations[1] = 2   → Interval 1: 2 vehicles over limit ✗
        # violations[2] = 0   → Interval 2: exactly at limit ✓
        # violations[3] = -10 → Interval 3: 10 vehicles under limit ✓
        ```

        **Uniform Capacity Across Time:**
        ```python
        # Same capacity limit for all time periods
        config = {
            'baseline': 'current_peak',
            'tolerance': 0.0  # No increase allowed
        }

        # If current peak = 100 vehicles, all intervals limited to 100
        # Useful for: fixed depot capacity, uniform driver availability
        ```

        **Manual Time-Specific Limits:**
        ```python
        # Custom limits based on operational constraints
        config = {
            'baseline': 'manual',
            'baseline_values': [50, 120, 80, 40],  # Morning, peak, afternoon, evening
            'tolerance': 0.05  # 5% flexibility
        }

        # Real-world example: Bus depot has different capacity by time
        # - Morning (6-9 AM): Limited drivers → 50 vehicles max
        # - Peak (9-3 PM): Full capacity → 120 vehicles max
        # - Afternoon (3-6 PM): Some drivers off → 80 vehicles max
        # - Evening (6-10 PM): Reduced service → 40 vehicles max

        handler = FleetPerIntervalConstraintHandler(config, optimization_data)

        # Check if solution respects time-varying capacity
        violations = handler.evaluate(candidate_solution)
        if all(v <= 0 for v in violations):
            print("✅ Solution respects all time-period capacity limits")
        else:
            over_capacity = [(i, v) for i, v in enumerate(violations) if v > 0]
            print(f"⚠️  Over capacity in intervals: {over_capacity}")
        ```

        **Practical Usage Pattern:**
        ```python
        # Common pattern: combine with total fleet constraint
        constraints = [
            # System-wide budget limit
            FleetTotalConstraintHandler({
                'baseline': 'current_peak',
                'tolerance': 0.15,
                'measure': 'peak'
            }, opt_data),

            # Time-specific operational limits
            FleetPerIntervalConstraintHandler({
                'baseline': 'current_by_interval',
                'tolerance': 0.20  # More flexibility per interval
            }, opt_data)
        ]

        # This ensures solutions are:
        # 1. Within overall budget (total fleet constraint)
        # 2. Operationally feasible by time period (per-interval constraint)
        ```
    """

    def _calculate_n_constraints(self) -> int:
        """Calculate constraints: intervals × (ceiling + floor constraints)."""
        n_ceiling = self.n_intervals if self.config.get('tolerance') is not None else 0
        n_floor = self.n_intervals if self.config.get('min_fraction') is not None else 0
        total = n_ceiling + n_floor

        logger.debug(f"FleetPerIntervalConstraintHandler constraint count calculation:")
        logger.debug(f"  tolerance specified: {self.config.get('tolerance') is not None} → {n_ceiling} ceiling constraints")
        logger.debug(f"  min_fraction specified: {self.config.get('min_fraction') is not None} → {n_floor} floor constraints")
        logger.debug(f"  TOTAL: {total} constraints")

        return total

    def _validate_config(self) -> None:
        """Validate per-interval fleet constraint configuration."""
        super()._validate_config()

        tolerance = self.config.get('tolerance', None)
        min_fraction = self.config.get('min_fraction', None)

        # At least one constraint must be specified
        if tolerance is None and min_fraction is None:
            raise ValueError(
                "FleetPerIntervalConstraintHandler must specify either 'tolerance' "
                "or 'min_fraction' or both"
            )
        # Validate parameter ranges
        if tolerance is not None and tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        if min_fraction is not None and not (0.0 <= min_fraction <= 1.0):
            raise ValueError("min_fraction must be between 0.0 and 1.0")

        if "baseline" not in self.config:
            raise ValueError("Per-interval fleet constraint requires 'baseline'")

        valid_baselines = ["current_by_interval", "current_peak", "manual"]
        if self.config["baseline"] not in valid_baselines:
            raise ValueError(f"baseline must be one of {valid_baselines}")

        if self.config["baseline"] == "manual" and "baseline_values" not in self.config:
            raise ValueError("Manual baseline requires 'baseline_values'")

        # Set defaults
        self.config.setdefault("allow_borrowing", False)

    def evaluate(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluate both ceiling and floor constraints for fleet per interval.

        Args:
            solution_matrix: Decision matrix (n_routes × n_intervals) - PT only

        Returns:
            Array of constraint violations where <= 0 means satisfied.
            Order: [ceiling_violations_per_interval..., floor_violations_per_interval...]
        """
        # Calculate fleet requirements by interval
        fleet_per_interval = self._calculate_fleet_from_solution(solution_matrix)

        violations = []

        # Check ceiling constraints
        tolerance = self.config.get('tolerance', None)
        if tolerance is not None:
            ceiling_violations = self._check_ceiling_violations(fleet_per_interval)
            violations.extend(ceiling_violations)

        # Check floor constraints
        min_fraction = self.config.get('min_fraction', None)
        if min_fraction is not None:
            floor_violations = self._check_floor_violations(fleet_per_interval)
            violations.extend(floor_violations)

        return np.array(violations)


    def _check_ceiling_violations(self, fleet_per_interval: np.ndarray) -> list[float]:
        """Check ceiling constraint violations"""
        violations = []
        baseline_fleet = self._get_interval_baselines()
        tolerance = self.config['tolerance']

        for interval_idx in range(self.n_intervals):
            current_fleet = fleet_per_interval[interval_idx]
            max_allowed = baseline_fleet[interval_idx] * (1 + tolerance)

            # Positive violation means constraint is violated
            violation = current_fleet - max_allowed
            violations.append(violation)

        return violations

    def _check_floor_violations(self, fleet_per_interval: np.ndarray) -> list[float]:
        """Check floor constraint violations"""
        violations = []
        baseline_fleet = self._get_interval_baselines()
        min_fraction = self.config['min_fraction']

        for interval_idx in range(self.n_intervals):
            current_fleet = fleet_per_interval[interval_idx]
            min_required = baseline_fleet[interval_idx] * min_fraction

            # Positive violation means constraint is violated (current < required)
            violation = min_required - current_fleet
            violations.append(violation)

        return violations

    def _get_interval_baselines(self) -> np.ndarray:
        """Get baseline fleet values for each interval (shared by ceiling/floor logic)."""
        if self.config["baseline"] == "manual":
            baseline_values = np.array(self.config["baseline_values"])
            if len(baseline_values) != self.n_intervals:
                raise ValueError(f"Manual baseline must have {self.n_intervals} values")
            return baseline_values
        else:
            # Extract from fleet analysis (existing logic)
            fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]

            if self.config["baseline"] == "current_by_interval":
                baseline_values = fleet_analysis.get("current_fleet_by_interval")
                return np.array(baseline_values)
            elif self.config["baseline"] == "current_peak":
                peak_fleet = fleet_analysis.get("total_current_fleet_peak")
                return np.full(self.n_intervals, peak_fleet)
            else:
                raise ValueError(f"Unknown baseline: {self.config['baseline']}")

    def get_constraint_info(self) -> dict[str, Any]:
        """Get information about this constraint handler."""
        constraint_types = []
        if self.config.get('tolerance') is not None:
            constraint_types.append(f"ceiling (tolerance: {self.config['tolerance']})")
        if self.config.get('min_fraction') is not None:
            constraint_types.append(f"floor (min_fraction: {self.config['min_fraction']})")

        return {
            "handler_type": f"FleetPerInterval ({', '.join(constraint_types)})",
            "n_constraints": self.n_constraints,
            "baseline": self.config['baseline'],
            "constraint_details": {
                "tolerance": self.config.get('tolerance'),
                "min_fraction": self.config.get('min_fraction'),
                "allow_borrowing": self.config.get('allow_borrowing', False)
            }
        }



    def _get_interval_limits(self) -> np.ndarray:
        """Get fleet limits for each time interval."""
        if self.config["baseline"] == "manual":
            baseline_values = np.array(self.config["baseline_values"])
            if len(baseline_values) != self.n_intervals:
                raise ValueError(f"Manual baseline must have {self.n_intervals} values")
        else:
            # Extract from fleet analysis
            fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]

            if self.config["baseline"] == "current_by_interval":
                # Try to get per-interval data, fall back to peak
                baseline_values = fleet_analysis.get(
                    "current_fleet_by_interval",
                    [fleet_analysis.get("total_current_fleet_peak")]
                    * self.n_intervals,
                )
                baseline_values = np.array(baseline_values)
            elif self.config["baseline"] == "current_peak":
                peak_fleet = fleet_analysis.get("total_current_fleet_peak")
                baseline_values = np.full(self.n_intervals, peak_fleet)
            else:
                raise ValueError(f"Unknown baseline: {self.config['baseline']}")

        # Apply tolerance to get limits
        tolerance = self.config["tolerance"]
        limits = baseline_values * (1 + tolerance)

        return limits

    def _apply_borrowing_logic(
        self, violations: np.ndarray, limits: np.ndarray
    ) -> np.ndarray:
        """
        Apply borrowing logic if enabled.

        This allows intervals with excess capacity to lend to intervals
        that are over their individual limits, as long as the total
        system capacity is not exceeded.
        """
        # Calculate total excess and total shortage
        excess = np.maximum(0, -violations)  # Negative violations = excess capacity
        shortage = np.maximum(0, violations)  # Positive violations = shortage

        total_excess = np.sum(excess)
        total_shortage = np.sum(shortage)

        # If total excess covers total shortage, no violations
        if total_excess >= total_shortage:
            return np.zeros_like(violations)

        # Otherwise, reduce violations proportionally to available excess
        if total_shortage > 0:
            reduction_factor = total_excess / total_shortage
            adjusted_violations = shortage * (1 - reduction_factor)
            return adjusted_violations

        return violations


class MinimumFleetConstraintHandler(BaseConstraintHandler):
    """
    Constraint handler for minimum fleet size requirements.

    Ensures that solutions maintain minimum fleet levels to prevent
    over-reduction of service. Can apply constraints at system level
    (single minimum) or interval level (minimum per time period).


    Configuration Parameters:
        min_fleet_fraction (float): Minimum fleet as fraction of original
                                  (e.g., 0.7 = at least 70% of original fleet)
        level (str): Constraint application level
            - 'system': Single constraint for entire system
            - 'interval': One constraint per time interval
        measure (str): What to measure when level='system'
            - 'peak': Constrain peak interval fleet
            - 'average': Constrain average across intervals
            - 'total': Constrain sum across all intervals
        baseline (str): Which original fleet measure to use as reference
            - 'current_peak': Reference original peak fleet
            - 'current_by_interval': Reference original per-interval fleet
            - 'current_average': Reference original average fleet

    Example Configurations:
        **System-Level Minimum Service:**
        ```python
        # Maintain at least 80% of current peak fleet system-wide
        config = {
            'min_fleet_fraction': 0.8,
            'level': 'system',
            'measure': 'peak',
            'baseline': 'current_peak'
        }

        # Scenario: Current GTFS peak = 150 vehicles
        # Minimum required = 150 * 0.8 = 120 vehicles at peak
        handler = MinimumFleetConstraintHandler(config, optimization_data)

        # Test solution scenarios:
        solution_high_service = create_solution_with_frequent_headways()
        violations_1 = handler.evaluate(solution_high_service)
        # Expected: violations_1 = [negative_value] (constraint satisfied)

        solution_low_service = create_solution_with_sparse_headways()
        violations_2 = handler.evaluate(solution_low_service)
        # Expected: violations_2 = [positive_value] (minimum service violated)

        print(f"High service solution violation: {violations_1[0]}")  # e.g., -20
        print(f"Low service solution violation: {violations_2[0]}")   # e.g., +15
        ```

        **Time-Period Specific Minimums:**
        ```python
        # Each time period must maintain 70% of its current service
        config = {
            'min_fleet_fraction': 0.7,
            'level': 'interval',
            'baseline': 'current_by_interval'
        }

        # Scenario: Current fleet = [60, 120, 90, 45] by time period
        # Minimums become = [42, 84, 63, 31.5] → [42, 84, 63, 32]
        handler = MinimumFleetConstraintHandler(config, optimization_data)

        # Candidate solution results in: [50, 70, 65, 30] fleet
        violations = handler.evaluate(candidate_solution)
        # violations = [42-50, 84-70, 63-65, 32-30] = [-8, 14, -2, 2]

        # Interpretation:
        # Interval 0: 8 vehicles above minimum ✓
        # Interval 1: 14 vehicles below minimum ✗ (service too low)
        # Interval 2: 2 vehicles above minimum ✓
        # Interval 3: 2 vehicles below minimum ✗ (service too low)

        if any(v > 0 for v in violations):
            print("⚠️  Solution reduces service below acceptable levels")
            problematic_intervals = [i for i, v in enumerate(violations) if v > 0]
            print(f"Problematic time periods: {problematic_intervals}")
        ```

        **Real World Service Standard Constraints:**
        ```python
        # Real-world example: City policy requires minimum service levels

        # Policy: "Peak hour service cannot be reduced below 90% of current"
        peak_hour_minimum = {
            'min_fleet_fraction': 0.9,
            'level': 'system',
            'measure': 'peak',
            'baseline': 'current_peak'
        }

        # Policy: "Off-peak service cannot be reduced below 60% of current"
        # (Assuming intervals 0,3 are off-peak, 1,2 are peak)
        manual_minimums = {
            'min_fleet_fraction': 1.0,  # Will be ignored for manual baseline
            'level': 'interval',
            'baseline': 'manual',
            'baseline_values': [
                current_fleet[0] * 0.6,  # Off-peak morning: 60%
                current_fleet[1] * 0.9,  # Peak midday: 90%
                current_fleet[2] * 0.9,  # Peak afternoon: 90%
                current_fleet[3] * 0.6   # Off-peak evening: 60%
            ]
        }

        constraints = [
            MinimumFleetConstraintHandler(peak_hour_minimum, opt_data),
            MinimumFleetConstraintHandler(manual_minimums, opt_data)
        ]

        # This creates 5 constraints total:
        # 1 system-wide peak constraint + 4 interval-specific constraints
        ```

        **Integration with Other Constraints:**
        ```python
        # Typical constraint combination for transit optimization

        def create_comprehensive_constraints(opt_data):
            return [
                # Budget limit: no more than 10% increase in peak fleet
                FleetTotalConstraintHandler({
                    'baseline': 'current_peak',
                    'tolerance': 0.10,
                    'measure': 'peak'
                }, opt_data),

                # Operational limits: respect time-varying capacity
                FleetPerIntervalConstraintHandler({
                    'baseline': 'current_by_interval',
                    'tolerance': 0.15
                }, opt_data),

                # Service standards: maintain at least 70% of current service
                MinimumFleetConstraintHandler({
                    'min_fleet_fraction': 0.70,
                    'level': 'system',
                    'measure': 'average',
                    'baseline': 'current_average'
                }, opt_data)
            ]

        # This creates a feasible solution space that:
        # 1. Stays within budget (total fleet ≤ 110% of current)
        # 2. Respects operational constraints (per-interval limits)
        # 3. Maintains minimum service quality (≥70% current service)

        constraints = create_comprehensive_constraints(opt_data)
        total_constraints = sum(c.n_constraints for c in constraints)
        print(f"Total constraint equations: {total_constraints}")
        # Output: "Total constraint equations: 7" (1 + 4 + 1 + 1)
        ```
    """

    def _calculate_n_constraints(self) -> int:
        """Number of constraints depends on level setting."""
        if self.config.get("level", "system") == "system":
            return 1  # Single system-wide constraint
        else:  # level == 'interval'
            return self.n_intervals  # One constraint per interval

    def _validate_config(self) -> None:
        """Validate minimum fleet constraint configuration."""
        super()._validate_config()

        if "min_fleet_fraction" not in self.config:
            raise ValueError("Minimum fleet constraint requires 'min_fleet_fraction'")

        min_fraction = self.config["min_fleet_fraction"]
        if not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_fleet_fraction must be between 0.0 and 1.0")

        # Set defaults
        self.config.setdefault("level", "system")
        self.config.setdefault("measure", "peak")
        self.config.setdefault("baseline", "current_peak")

        # Validate level
        if self.config["level"] not in ["system", "interval"]:
            raise ValueError("level must be 'system' or 'interval'")

        # Validate measure (only used for system level)
        if self.config["level"] == "system" and self.config["measure"] not in [
            "peak",
            "average",
            "total",
        ]:
            raise ValueError("measure must be 'peak', 'average', or 'total'")

        # Validate baseline
        valid_baselines = ["current_peak", "current_by_interval", "current_average"]
        if self.config["baseline"] not in valid_baselines:
            raise ValueError(f"baseline must be one of {valid_baselines}")

    def evaluate(self, solution_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluate minimum fleet constraint(s).

        Args:
            solution_matrix: Decision matrix (n_routes × n_intervals)

        Returns:
            Array of constraint violations:
            - System level: Single element [min_required - actual_fleet]
            - Interval level: Per-interval [min_required_i - actual_fleet_i, ...]
            Positive values indicate constraint violation (too little fleet)
        """
        # Calculate current fleet usage
        fleet_per_interval = self._calculate_fleet_from_solution(solution_matrix)

        if self.config["level"] == "system":
            return self._evaluate_system_constraint(fleet_per_interval)
        else:  # level == "interval"
            return self._evaluate_interval_constraints(fleet_per_interval)

    def _evaluate_system_constraint(self, fleet_per_interval: np.ndarray) -> np.ndarray:
        """Evaluate system-level minimum fleet constraint."""
        # Calculate current fleet measure
        if self.config["measure"] == "peak":
            current_fleet = np.max(fleet_per_interval)
        elif self.config["measure"] == "average":
            current_fleet = np.mean(fleet_per_interval)
        elif self.config["measure"] == "total":
            current_fleet = np.sum(fleet_per_interval)
        else:
            raise ValueError(f"Unknown measure: {self.config['measure']}")

        # Get minimum required fleet
        min_required = self._get_system_minimum()

        # Calculate violation (positive = violation)
        violation = min_required - current_fleet

        return np.array([violation])

    def _evaluate_interval_constraints(
        self, fleet_per_interval: np.ndarray
    ) -> np.ndarray:
        """Evaluate per-interval minimum fleet constraints."""
        # Get minimum required fleet for each interval
        min_required_per_interval = self._get_interval_minimums()

        # Calculate violations (positive = violation)
        violations = min_required_per_interval - fleet_per_interval

        return violations

    def _get_system_minimum(self) -> float:
        """Calculate system-wide minimum required fleet."""
        fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]

        if self.config["baseline"] == "current_peak":
            original_fleet = fleet_analysis["total_current_fleet_peak"]
        elif self.config["baseline"] == "current_average":
            original_fleet = fleet_analysis["total_current_fleet_average"]
        elif self.config["baseline"] == "current_by_interval":
            # Use appropriate measure from per-interval data
            fleet_by_interval = np.array(fleet_analysis["current_fleet_by_interval"])
            if self.config["measure"] == "peak":
                original_fleet = np.max(fleet_by_interval)
            elif self.config["measure"] == "average":
                original_fleet = np.mean(fleet_by_interval)
            elif self.config["measure"] == "total":
                original_fleet = np.sum(fleet_by_interval)
        else:
            raise ValueError(f"Unknown baseline: {self.config['baseline']}")

        min_fraction = self.config["min_fleet_fraction"]
        return original_fleet * min_fraction

    def _get_interval_minimums(self) -> np.ndarray:
        """Calculate minimum required fleet for each interval."""
        fleet_analysis = self.opt_data["constraints"]["fleet_analysis"]

        if self.config["baseline"] == "current_by_interval":
            original_fleet_by_interval = np.array(
                fleet_analysis["current_fleet_by_interval"]
            )
        elif self.config["baseline"] == "current_peak":
            # Use peak fleet for all intervals
            peak_fleet = fleet_analysis["total_current_fleet_peak"]
            original_fleet_by_interval = np.full(self.n_intervals, peak_fleet)
        elif self.config["baseline"] == "current_average":
            # Use average fleet for all intervals
            avg_fleet = fleet_analysis["total_current_fleet_average"]
            original_fleet_by_interval = np.full(self.n_intervals, avg_fleet)
        else:
            raise ValueError(f"Unknown baseline: {self.config['baseline']}")

        min_fraction = self.config["min_fleet_fraction"]
        return original_fleet_by_interval * min_fraction
