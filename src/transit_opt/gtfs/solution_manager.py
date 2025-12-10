"""
Solution export management for transit optimization results.

This module coordinates the export of optimization solutions to various formats,
handling both PT-only and combined PT+DRT solutions with minimal metadata embedding.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from transit_opt.gtfs.drt import DRTSolutionExporter
from transit_opt.gtfs.gtfs import SolutionConverter
from transit_opt.optimisation.config.config_manager import \
    SolutionSamplingStrategyConfig

logger = logging.getLogger(__name__)


class SolutionExportManager:
    """
    Coordinates export of optimization solutions to standardized formats.

    Manages the conversion and export of optimization results, supporting:
    - PT-only solutions → GTFS format
    - Combined PT+DRT solutions → GTFS + JSON formats
    - Minimal metadata embedding (relies on directory structure for organization)
    """

    def __init__(self, optimization_data: dict[str, Any]):
        """
        Initialize export manager based on optimization problem configuration.

        Args:
            optimization_data: Complete optimization setup from GTFSDataPreparator
        """
        self.optimization_data = optimization_data
        self.drt_enabled = optimization_data.get("drt_enabled", False)

        # Always initialize PT converter
        self.pt_converter = SolutionConverter(optimization_data)

        # Initialize DRT exporter only if DRT is enabled
        self.drt_exporter = None
        if self.drt_enabled:
            self.drt_exporter = DRTSolutionExporter(optimization_data)

    def export_single_solution(
        self,
        solution: np.ndarray | dict[str, np.ndarray],
        solution_id: str,
        output_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Export a single optimization solution with minimal metadata.

        Args:
            solution: Solution matrix (PT-only) or dict with 'pt'/'drt' keys
            solution_id: Unique identifier for this solution
            output_dir: Directory where files should be created
            metadata: Optional minimal metadata (objective_value, timestamp, etc.)

        Returns:
            Export summary with file paths and minimal metadata
        """
        metadata = metadata or {}
        exports = {}

        # Determine solution type and extract components
        if isinstance(solution, dict):
            # Combined PT+DRT solution
            pt_solution = solution["pt"]
            drt_solution = solution.get("drt")
        else:
            # PT-only solution
            pt_solution = solution
            drt_solution = None

        # Export PT component (always present)
        exports["pt"] = self._export_pt_solution(pt_solution, solution_id, output_dir)

        # Export DRT component if present and enabled
        if drt_solution is not None and self.drt_enabled:
            # Create minimal DRT metadata with PT cross-reference only
            drt_metadata = {"pt_gtfs_reference": Path(exports["pt"]["path"]).name}
            # Add objective value if provided (useful for analysis)
            if "objective_value" in metadata:
                drt_metadata["objective_value"] = metadata["objective_value"]

            exports["drt"] = self._export_drt_solution({"drt": drt_solution}, solution_id, output_dir, drt_metadata)

        # Return minimal result structure
        return {
            "solution_id": solution_id,
            "exports": exports,
            "metadata": {
                "solution_id": solution_id,
                **metadata,  # Include only what was explicitly provided
            },
        }

    def export_solution_set(
        self,
        solutions: list[dict[str, Any]],
        base_output_dir: str,
        solution_prefix: str = "solution",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Export multiple solutions with consistent naming and minimal metadata.

        Args:
            solutions: List of solution dicts with 'solution' and 'objective' keys
            base_output_dir: Base directory for all solution files
            solution_prefix: Prefix for solution IDs (e.g., "best_solution")
            metadata: Optional common metadata (kept minimal)

        Returns:
            List of export results for each solution
        """
        if not solutions:
            return []

        results = []  # results to convert to gtfs / json
        summary_rows = []  # csv with objective value of each run

        for i, solution_data in enumerate(solutions):
            # Use rank directly (already 0-indexed from extract_solutions_for_export)
            rank = solution_data.get("rank", i)
            # Generate consistent solution ID
            solution_id = f"{solution_prefix}_{rank:02d}"

            # Create minimal per-solution metadata
            solution_metadata = {}

            # Only include objective value (essential for analysis)
            if "objective" in solution_data:
                solution_metadata["objective_value"] = solution_data["objective"]

            # Store the rank from extraction
            solution_metadata["rank"] = rank

            # Export the solution
            result = self.export_single_solution(
                solution=solution_data["solution"],
                solution_id=solution_id,
                output_dir=base_output_dir,
                metadata=solution_metadata,
            )

            results.append(result)

            # Prepare row for CSV summary
            summary_rows.append(
                {
                    "solution_id": solution_id,
                    "swarm_id": solution_data.get("run_id", ""),
                    "rank": rank,
                    "objective": solution_data.get("objective", ""),
                    "generation_found": solution_data.get("generation_found", ""),
                    "violations": solution_data.get("violations", ""),
                }
            )
        # write csv summary of results
        self._export_solution_summary_csv(summary_rows, base_output_dir, solution_prefix)

        return results

    def _export_pt_solution(self, pt_solution: np.ndarray, solution_id: str, output_dir: str) -> dict[str, Any]:
        """Export PT solution to GTFS format with no metadata embedding."""
        # Convert to headways and extract templates
        headways_dict = self.pt_converter.solution_to_headways(pt_solution)
        templates = self.pt_converter.extract_route_templates()

        # Create GTFS with solution-specific service ID
        service_id = f"optimized_{solution_id}"

        # Create the output path with the solution_id in the filename
        # This ensures the GTFS ZIP has a predictable name for cross-referencing
        gtfs_output_path = Path(output_dir) / f"{solution_id}_gtfs"

        # Export to GTFS ZIP
        gtfs_path = self.pt_converter.build_complete_gtfs(
            headways_dict=headways_dict,
            templates=templates,
            service_id=service_id,
            output_dir=str(gtfs_output_path),
            zip_output=True,
        )

        return {"type": "gtfs", "path": gtfs_path, "service_id": service_id, "format": "zip"}

    def _export_drt_solution(
        self, drt_solution: dict[str, np.ndarray], solution_id: str, output_dir: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Export DRT solution to JSON format with minimal metadata."""

        # Create the output file path with solution_id in the filename
        output_path = Path(output_dir) / f"{solution_id}_drt.json"

        # Export DRT solution - remove solution_id from direct parameters
        drt_path = self.drt_exporter.export_solution(
            solution=drt_solution, output_path=str(output_path), metadata=metadata
        )

        return {"type": "drt", "path": drt_path, "format": "json", "pt_reference": metadata.get("pt_gtfs_reference")}

    def extract_solutions_for_export(self, result, output_cfg: dict) -> list[dict]:
        """
        Extract solutions for export using sampling strategy from config.

        Args:
            result: OptimizationResult or MultiRunResult
            output_cfg: Output config dict with 'sampling_strategy' key

        Returns:
            List of solution dicts with 'solution', 'objective', and 'rank' keys
        """
        from transit_opt.gtfs.solution_sampling import generate_sampling_ranks

        best_run = output_cfg.get("best_run", True)

        # Get solutions based on result type
        if hasattr(result, "num_runs_completed"):
            sols = result.best_result.best_feasible_solutions if best_run else result.best_feasible_solutions_all_runs
        else:
            sols = result.best_feasible_solutions

        # Remove duplicates
        if len(sols) > 1:
            original_count = len(sols)
            sols = self._remove_duplicate_solutions(sols)
            logger.info(f"📊 Removed {original_count - len(sols)} duplicate solutions")

        # Sort by objective (best first)
        sols = sorted(sols, key=lambda s: s.get("objective", float("inf")))

        # Get sampling strategy config
        sampling_cfg = output_cfg.get("sampling_strategy", {})

        # Set max_rank if not specified (default to all tracked solutions)
        if "max_rank" not in sampling_cfg or sampling_cfg["max_rank"] is None:
            sampling_cfg["max_rank"] = len(sols)

        # Create config object
        strategy_config = SolutionSamplingStrategyConfig(
            type=sampling_cfg.get("type", "uniform"),
            max_to_save=sampling_cfg.get("max_to_save", 10),
            max_rank=sampling_cfg["max_rank"],
            power_exponent=sampling_cfg.get("power_exponent", 2.0),
            geometric_base=sampling_cfg.get("geometric_base", 2.0),
            manual_ranks=sampling_cfg.get("manual_ranks", []),
        )

        # Generate ranks using strategy
        ranks = generate_sampling_ranks(strategy_config)

        logger.info("📊 Sampling strategy:")
        logger.info(f"   Type: {strategy_config.type}")
        logger.info(f"   Tracked solutions: {len(sols)}")
        logger.info(f"   Selected ranks: {ranks}")

        # Extract selected solutions
        selected = []
        for rank in ranks:
            if rank < len(sols):
                sol = sols[rank].copy()
                sol["rank"] = rank
                selected.append(sol)

        if len(selected) < len(ranks):
            logger.warning(f"⚠️  Only {len(selected)}/{len(ranks)} ranks available (tracked {len(sols)} solutions)")

        return selected

    def _remove_duplicate_solutions(self, solutions: list[dict]) -> list[dict]:
        """
        Remove duplicate solutions using efficient fingerprinting.

        Uses numpy array comparison with shape validation for fast deduplication.

        Args:
            solutions: List of solution dicts with 'solution' and 'objective' keys

        Returns:
            List of unique solutions (preserves order, keeps first occurrence)
        """
        if not solutions:
            return []

        logger.info(f"📊 Deduplicating {len(solutions)} solutions...")

        unique_solutions = []
        seen_fingerprints = set()

        for sol in solutions:
            solution_matrix = sol["solution"]

            # Create simple fingerprint: hash of (bytes, shape)
            if isinstance(solution_matrix, dict):
                # PT+DRT case
                fingerprint = hash((
                    solution_matrix["pt"].tobytes(),
                    solution_matrix["pt"].shape,
                    solution_matrix["drt"].tobytes(),
                    solution_matrix["drt"].shape
                ))
            else:
                # PT-only case
                fingerprint = hash((
                    solution_matrix.tobytes(),
                    solution_matrix.shape
                ))

            # O(1) set lookup
            if fingerprint not in seen_fingerprints:
                unique_solutions.append(sol)
                seen_fingerprints.add(fingerprint)
            else:
                logger.debug(f"   Duplicate found: objective={sol.get('objective', 'N/A'):.4f}")

        logger.info(f"   Deduplication: {len(solutions)} → {len(unique_solutions)} unique solutions")
        return unique_solutions

    def _export_solution_summary_csv(self, rows, output_dir, solution_prefix):
        """Export a CSV summary of solutions with their objectives and ranks."""
        csv_path = Path(output_dir) / f"{solution_prefix}_summary.csv"
        fieldnames = ["solution_id", "rank", "swarm_id", "objective", "generation_found", "violations"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("✅ Solution summary CSV written: %s", csv_path)















