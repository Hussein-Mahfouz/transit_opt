"""
Solution loading utilities for custom PSO sampling.

This module provides utilities to load and validate base solutions from various sources
for use in custom PSO sampling. Supports loading from optimization data, pre-computed
solutions, and multiple GTFS feeds.

The idea is to pass a list of soltions to the sampling argument in pymoo: https://pymoo.org/algorithms/soo/pso.html
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SolutionLoader:
    """
    Handle loading and validation of base solutions from various sources.

    This class provides a unified interface for loading base solutions that can be used
    as starting points in custom PSO sampling. It supports multiple input formats and
    validates solutions before use.

    Supported Input Types:
    - 'from_data': Use initial solution from optimization_data
    - List of pre-computed solutions (matrices or dicts)
    - Multiple GTFS feed paths (requires additional processing)

    Solution Format Handling:
    - PT-only: numpy arrays of shape (n_routes, n_intervals)
    - PT+DRT: dicts with 'pt' and 'drt' keys
    - Mixed formats: automatically detected and validated
    """

    def __init__(self):
        """Initialize solution loader."""
        pass

    def load_solutions(
        self, config_spec: str | list, optimization_data: dict[str, Any]
    ) -> list[np.ndarray | dict[str, np.ndarray]]:
        """
        Load base solutions from configuration specification.

        Args:
            config_spec: Solution specification from configuration:
                        - 'from_data': Use optimization_data['initial_solution']
                        - List: Pre-provided solutions
            optimization_data: Complete optimization data structure

        Returns:
            List of validated solutions in domain format

        Raises:
            ValueError: If config_spec is invalid or solutions are malformed
        """
        if config_spec == "from_data":
            return self._load_from_optimization_data(optimization_data)
        elif isinstance(config_spec, list):
            return self._load_from_list(config_spec, optimization_data)
        else:
            raise ValueError(f"Invalid base_solutions specification: {config_spec}")

    def _load_from_optimization_data(
        self, optimization_data: dict[str, Any]
    ) -> list[np.ndarray | dict[str, np.ndarray]]:
        """Load initial solution from optimization data."""
        if "initial_solution" not in optimization_data:
            raise ValueError("optimization_data missing 'initial_solution' key")

        initial_solution = optimization_data["initial_solution"]
        # Handle flat solutions by decoding them first (same logic as _load_from_list)
        if isinstance(initial_solution, np.ndarray) and initial_solution.ndim == 1:
            # This is a flat solution vector from extract_optimization_data_with_drt()
            # Decode to domain format (handles both PT-only and PT+DRT)
            initial_solution = self._decode_flat_solution(initial_solution, optimization_data)

        validated_solution = self._validate_solution(initial_solution, optimization_data)

        return [validated_solution]

    def _load_from_list(
        self, solution_list: list, optimization_data: dict[str, Any]
    ) -> list[np.ndarray | dict[str, np.ndarray]]:
        """Load and validate solutions from provided list."""
        if not solution_list:
            return []

        validated_solutions = []
        for i, solution in enumerate(solution_list):
            try:
                # Handle flat solutions by decoding them first
                if isinstance(solution, np.ndarray) and solution.ndim == 1:
                    # This is a flat solution vector (e.g., from extract_multiple_gtfs_solutions)
                    # Decode to domain format
                    solution = self._decode_flat_solution(solution, optimization_data)

                # Now validate the solution (existing logic)
                validated = self._validate_solution(solution, optimization_data)
                validated_solutions.append(validated)
            except Exception as e:
                raise ValueError(f"Solution {i} is invalid: {e}") from e

        return validated_solutions

    def _decode_flat_solution(self, flat_solution: np.ndarray, optimization_data: dict[str, Any]):
        """Decode flat solution to domain format using optimization data structure."""
        drt_enabled = optimization_data.get("drt_enabled", False)

        if not drt_enabled:
            # PT-only: reshape to matrix
            shape = optimization_data["decision_matrix_shape"]
            return flat_solution.reshape(shape)
        else:
            # PT+DRT: split and reshape using variable structure
            var_structure = optimization_data["variable_structure"]
            pt_size = var_structure["pt_size"]
            pt_shape = var_structure["pt_shape"]
            drt_shape = var_structure["drt_shape"]

            pt_flat = flat_solution[:pt_size]
            drt_flat = flat_solution[pt_size:]

            return {"pt": pt_flat.reshape(pt_shape), "drt": drt_flat.reshape(drt_shape)}

    def _validate_solution(
        self, solution: np.ndarray | dict[str, np.ndarray], optimization_data: dict[str, Any]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Validate solution format and dimensions against optimization data.

        Args:
            solution: Solution to validate (matrix or dict format)
            optimization_data: Reference optimization data for validation

        Returns:
            Validated solution (same format as input)

        Raises:
            ValueError: If solution format or dimensions are invalid
        """
        drt_enabled = optimization_data.get("drt_enabled", False)
        n_routes = optimization_data["n_routes"]
        n_intervals = optimization_data["n_intervals"]

        if drt_enabled:
            # DRT-enabled: expect dict format
            if not isinstance(solution, dict):
                raise ValueError("DRT-enabled problems require dict solution format")

            if "pt" not in solution or "drt" not in solution:
                raise ValueError("DRT solution dict must have 'pt' and 'drt' keys")

            # Validate PT part
            pt_solution = solution["pt"]
            if not isinstance(pt_solution, np.ndarray):
                raise ValueError("PT solution must be numpy array")

            expected_pt_shape = (n_routes, n_intervals)
            if pt_solution.shape != expected_pt_shape:
                raise ValueError(f"PT solution shape {pt_solution.shape} != expected {expected_pt_shape}")

            # Validate DRT part
            drt_solution = solution["drt"]
            if not isinstance(drt_solution, np.ndarray):
                raise ValueError("DRT solution must be numpy array")

            n_drt_zones = optimization_data.get("n_drt_zones", 0)
            expected_drt_shape = (n_drt_zones, n_intervals)
            if drt_solution.shape != expected_drt_shape:
                raise ValueError(f"DRT solution shape {drt_solution.shape} != expected {expected_drt_shape}")

            # Validate value ranges
            self._validate_pt_values(pt_solution, optimization_data)
            self._validate_drt_values(drt_solution, optimization_data)

            return {"pt": pt_solution.copy(), "drt": drt_solution.copy()}

        else:
            # PT-only: expect array format
            if isinstance(solution, dict):
                # Handle case where dict is provided for PT-only problem
                if "pt" in solution:
                    solution = solution["pt"]
                else:
                    raise ValueError("PT-only problems require array solution format")

            if not isinstance(solution, np.ndarray):
                raise ValueError("PT solution must be numpy array")

            expected_shape = (n_routes, n_intervals)
            if solution.shape != expected_shape:
                raise ValueError(f"Solution shape {solution.shape} != expected {expected_shape}")

            # Validate value ranges
            self._validate_pt_values(solution, optimization_data)

            return solution.copy()

    def _validate_pt_values(self, pt_solution: np.ndarray, optimization_data: dict[str, Any]):
        """Validate PT solution values are valid headway indices."""
        n_choices = optimization_data["n_choices"]
        min_val = np.min(pt_solution)
        max_val = np.max(pt_solution)

        if min_val < 0:
            raise ValueError(f"PT solution contains negative indices: min={min_val}")
        if max_val >= n_choices:
            raise ValueError(f"PT solution contains invalid indices: max={max_val}, n_choices={n_choices}")

        # Check for integer values
        if not np.allclose(pt_solution, np.round(pt_solution)):
            raise ValueError("PT solution must contain integer indices")

    def _validate_drt_values(self, drt_solution: np.ndarray, optimization_data: dict[str, Any]):
        """Validate DRT solution values are valid fleet size indices."""
        if not optimization_data.get("drt_enabled", False):
            return

        drt_zones = optimization_data.get("drt_config", {}).get("zones", [])
        if len(drt_zones) == 0:
            return

        for zone_idx, zone in enumerate(drt_zones):
            max_fleet_choices = len(zone.get("allowed_fleet_sizes", [0]))
            if zone_idx < drt_solution.shape[0]:
                zone_values = drt_solution[zone_idx, :]
                min_val = np.min(zone_values)
                max_val = np.max(zone_values)

                if min_val < 0:
                    raise ValueError(f"DRT zone {zone_idx} contains negative indices: min={min_val}")
                if max_val >= max_fleet_choices:
                    raise ValueError(
                        f"DRT zone {zone_idx} contains invalid indices: max={max_val}, max_choices={max_fleet_choices}"
                    )

        # Check for integer values
        if not np.allclose(drt_solution, np.round(drt_solution)):
            raise ValueError("DRT solution must contain integer indices")

    def resolve_base_solutions_descriptor(self, base_spec: Any, optimization_data: dict[str, Any]) -> list:
        """
        Resolve a YAML-friendly 'base_solutions' descriptor into a list of flat numpy arrays
        suitable for PSO seeding.
        """
        if base_spec is None:
            return []

        # If already a concrete list, return copy
        if isinstance(base_spec, list):
            return list(base_spec)

        # String 'from_data'
        if isinstance(base_spec, str):
            if base_spec == "from_data":
                if "initial_solution" not in optimization_data:
                    raise ValueError("optimization_data missing 'initial_solution' for 'from_data'")
                logger.info("Seeding: Loaded base_solutions from optimization_data['initial_solution']")
                return [optimization_data["initial_solution"]]
            raise ValueError("Unsupported base_solutions string. Use 'from_data' or provide list/dict descriptor.")

        # Dict descriptor
        if isinstance(base_spec, dict):
            gtfs_paths = base_spec.get("gtfs_paths")
            drt_paths = base_spec.get("drt_solution_paths")

            # Directory scan fallback
            if not gtfs_paths and "solutions_dir" in base_spec:
                sol_dir = Path(base_spec["solutions_dir"])
                if not sol_dir.exists():
                    raise FileNotFoundError(f"solutions_dir not found: {sol_dir}")
                gtfs_glob = base_spec.get("gtfs_glob", "*_gtfs.zip")
                drt_glob = base_spec.get("drt_glob", "*_drt.json")
                gtfs_paths = [str(p) for p in sorted(sol_dir.glob(gtfs_glob))]
                drt_paths = [str(p) for p in sorted(sol_dir.glob(drt_glob))]

            if gtfs_paths:
                from transit_opt.preprocessing.prepare_gtfs import \
                    GTFSDataPreparator

                # Calculate interval_hours from n_intervals (the correct way)
                n_intervals = optimization_data.get("n_intervals")
                if n_intervals is None:
                    raise ValueError("optimization_data missing 'n_intervals'")
                interval_hours = 24 // n_intervals

                # Get allowed headways without no-service marker
                allowed_headways_raw = optimization_data.get("allowed_headways", [10, 15, 30, 60, 120, 9999])
                allowed_headways = [h for h in allowed_headways_raw if h < 9000]

                # Get DRT config if enabled
                drt_enabled = optimization_data.get("drt_enabled", False)
                drt_config = optimization_data.get("drt_config") if drt_enabled else None

                logger.info(f"📂 Loading {len(gtfs_paths)} GTFS solutions...")
                logger.info(f"   Interval hours: {interval_hours}")
                logger.info(f"   DRT enabled: {drt_enabled}")
                logger.info(f"   Allowed headways: {allowed_headways}")

                # Get the FILTERED route IDs from the current optimization problem
                target_route_ids = optimization_data.get("routes", {}).get("ids", [])
                n_routes = optimization_data.get("n_routes")

                logger.info(f"   Target routes: {n_routes} (filtered from original GTFS)")

                if not target_route_ids or n_routes is None:
                    raise ValueError("optimization_data missing route information for seed loading")

                # Use the first seed GTFS to create the preparator (just for loading)
                # But we'll pass the target route IDs to extract_multiple_gtfs_solutions
                try:
                    preparator = GTFSDataPreparator(
                        gtfs_path=gtfs_paths[0],
                        interval_hours=interval_hours,
                    )

                    # Pass target_route_ids so the method knows what shape to create
                    solutions = preparator.extract_multiple_gtfs_solutions(
                        gtfs_paths=gtfs_paths,
                        allowed_headways=allowed_headways,
                        drt_config=drt_config,
                        drt_solution_paths=drt_paths if drt_enabled else None,
                        target_route_ids=target_route_ids,
                    )

                    if not solutions:
                        raise ValueError("extract_multiple_gtfs_solutions returned empty list")

                    logger.info(f"✅ Loaded {len(solutions)} base solutions from GTFS paths")
                    return solutions

                except Exception as e:
                    logger.error(f"❌ ERROR in extract_multiple_gtfs_solutions: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

        raise ValueError("Unsupported base_solutions descriptor. Use 'from_data', a list, or dict with gtfs_paths.")


