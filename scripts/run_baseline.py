# filepath: scripts/run_baseline.py
"""
Evaluate baseline GTFS performance across all objective configurations.

This script processes all config files and evaluates the baseline (original GTFS)
objective value for each configuration. Results are saved to a single CSV for
easy comparison with optimization results.

Usage:
    # Evaluate all configs in configs/ directory
    python scripts/run_baseline.py

    # Evaluate specific configs (e.g. all service coverage configs)
    python scripts/run_baseline.py --configs configs/sc_*.yaml

    # Evaluate all configs that start with sc or wt:
    python scripts/run_baseline.py --configs configs/sc_*.yaml configs/wt_*.yaml

    # Evaluate PT-only baseline (recommended: ignores DRT even if enabled)
    python scripts/run_baseline.py --configs configs/sc_*.yaml --pt-only-baseline

    # Custom output location
    python scripts/run_baseline.py --output results/baselines.csv

Notes:
    - Always use the configs/ prefix for patterns when running from the project root.
    - --pt-only-baseline disables DRT for baseline evaluation, even if enabled in the config.
      This ensures the baseline reflects the original GTFS (PT-only) and is directly comparable
      to optimization results.
"""

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml

# Add src to path
project_root = Path(__file__).resolve().parent.parent
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from transit_opt.logging import setup_logger
from transit_opt.optimisation.objectives import StopCoverageObjective, WaitingTimeObjective
from transit_opt.optimisation.problems.base import (
    FleetPerIntervalConstraintHandler,
    FleetTotalConstraintHandler,
    MinimumFleetConstraintHandler,
)
from transit_opt.optimisation.problems.transit_problem import TransitOptimizationProblem
from transit_opt.optimisation.spatial.boundaries import StudyAreaBoundary
from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def inject_boundary(cfg: dict) -> None:
    """Add boundary object if config specifies boundary path."""
    obj_cfg = cfg.setdefault("problem", {}).setdefault("objective", {})
    boundary_path = cfg.get("input", {}).get("boundary_geojson")

    if boundary_path and Path(boundary_path).exists():
        gdf = gpd.read_file(boundary_path)
        obj_cfg["boundary"] = StudyAreaBoundary(
            boundary_gdf=gdf,
            crs=obj_cfg.get("crs", "EPSG:3857"),
            buffer_km=cfg.get("input", {}).get("boundary_buffer_km", 2.0),
        )


def prepare_opt_data(cfg: dict) -> dict:
    """Prepare optimization data from GTFS."""
    inp = cfg.get("input", {})
    gtfs_path = inp.get("gtfs_path")

    if isinstance(gtfs_path, (list, tuple)):
        gtfs_path = gtfs_path[0]

    preparator = GTFSDataPreparator(gtfs_path=gtfs_path, interval_hours=inp.get("interval_hours"))

    allowed_headways = inp.get("allowed_headways")
    drt_cfg = inp.get("drt", {})

    if drt_cfg.get("enabled", False):
        return preparator.extract_optimization_data_with_drt(allowed_headways=allowed_headways, drt_config=drt_cfg)
    else:
        return preparator.extract_optimization_data(allowed_headways=allowed_headways)


def evaluate_baseline(cfg: dict, opt_data: dict) -> dict:
    logger = logging.getLogger("transit_opt.scripts.run_baseline")

    """Evaluate baseline GTFS with configured objective."""
    obj_cfg = cfg.get("problem", {}).get("objective", {})
    objective_type = obj_cfg.get("type", "StopCoverageObjective")

    # Extract configuration name from results_dir
    config_name = Path(cfg.get("output", {}).get("results_dir", "unknown")).name

    # Extract boundary from obj_cfg (already injected by inject_boundary())
    boundary = obj_cfg.get("boundary")

    # Create appropriate objective
    if objective_type == "StopCoverageObjective":
        objective = StopCoverageObjective(
            optimization_data=opt_data,
            spatial_resolution_km=obj_cfg.get("spatial_resolution_km", 2.0),
            crs=obj_cfg.get("crs", "EPSG:3857"),
            boundary=boundary,
            time_aggregation=obj_cfg.get("time_aggregation", "average"),
            spatial_lag=obj_cfg.get("spatial_lag", False),
            alpha=obj_cfg.get("alpha", 0.0),
            population_weighted=obj_cfg.get("population_weighted", False),
            population_layer=obj_cfg.get("population_layer"),
            population_power=obj_cfg.get("population_power", 1.0),
            demand_weighted=obj_cfg.get("demand_weighted", False),
            trip_data_path=obj_cfg.get("trip_data_path"),
            trip_data_crs=obj_cfg.get("trip_data_crs"),
            demand_power=obj_cfg.get("demand_power", 1.0),
            min_trip_distance_m=obj_cfg.get("min_trip_distance_m"),
        )
    elif objective_type == "WaitingTimeObjective":
        objective = WaitingTimeObjective(
            optimization_data=opt_data,
            spatial_resolution_km=obj_cfg.get("spatial_resolution_km", 2.0),
            crs=obj_cfg.get("crs", "EPSG:3857"),
            boundary=boundary,
            metric=obj_cfg.get("metric", "total"),
            time_aggregation=obj_cfg.get("time_aggregation", "average"),
            population_weighted=obj_cfg.get("population_weighted", False),
            population_layer=obj_cfg.get("population_layer"),
            population_power=obj_cfg.get("population_power", 1.0),
            demand_weighted=obj_cfg.get("demand_weighted", False),
            trip_data_path=obj_cfg.get("trip_data_path"),
            trip_data_crs=obj_cfg.get("trip_data_crs"),
            demand_power=obj_cfg.get("demand_power", 1.0),
            min_trip_distance_m=obj_cfg.get("min_trip_distance_m"),
        )
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    # Setup Constraints
    constraints = []
    for c_cfg in cfg.get("problem", {}).get("constraints", []):
        c_type = c_cfg.get("type")
        if c_type == "FleetTotalConstraintHandler":
            constraints.append(FleetTotalConstraintHandler(c_cfg, opt_data))
        elif c_type == "MinimumFleetConstraintHandler":
            constraints.append(MinimumFleetConstraintHandler(c_cfg, opt_data))
        elif c_type == "FleetPerIntervalConstraintHandler":
            constraints.append(FleetPerIntervalConstraintHandler(c_cfg, opt_data))

    # Setup Penalty Config
    algo_cfg = cfg.get("optimization", {}).get("algorithm", {})
    penalty_config = {
        "enabled": algo_cfg.get("use_penalty_method", False),
        "penalty_weight": algo_cfg.get("penalty_weight", 1000.0),
        "constraint_weights": cfg.get("problem", {}).get("penalty_weights", {}),
    }

    # Create problem instance with the real objective
    problem = TransitOptimizationProblem(
        optimization_data=opt_data, objective=objective, constraints=constraints, penalty_config=penalty_config
    )

    # Decode initial solution
    decoded_solution = problem.decode_solution(opt_data["initial_solution"])

    # Evaluate baseline raw objective
    raw_objective_value = objective.evaluate(decoded_solution)

    # Calculate Penalties
    violations = [c.evaluate(decoded_solution) for c in constraints]

    # Use the problem's penalty weight lookup (which handles name mapping)
    constraint_names = problem.constraint_names

    total_penalty = 0.0
    violation_details = {}

    for v, name in zip(violations, constraint_names):
        # Calculate sum of positive violations
        v_sum = np.sum(np.maximum(0, v))

        if v_sum > 0:
            # Use problem's method which handles name mapping correctly
            p_weight = problem._get_constraint_penalty_weight(name)
            penalty = v_sum * p_weight
            total_penalty += penalty
            violation_details[name] = {"violation_sum": v_sum, "weight": p_weight, "penalty": penalty}

    penalized_objective_value = raw_objective_value + total_penalty

    logger.info(f"📊 BASELINE EVALUATION:")
    logger.info(f"   Raw Objective:       {raw_objective_value:,.2f}")
    logger.info(f"   Total Penalty:       {total_penalty:,.2f}")
    logger.info(f"   Penalized Objective: {penalized_objective_value:,.2f}")

    if violation_details:
        logger.info("   ⚠️  Constraint Violations:")
        for name, details in violation_details.items():
            logger.info(
                f"      - {name}: violation={details['violation_sum']:.2f}, weight={details['weight']:.0f}, penalty={details['penalty']:.2f}"
            )
    else:
        logger.info("   ✅ Baseline is feasible (no violations)")

    # Return result dictionary
    return {
        "config_name": config_name,
        "objective_type": objective_type,
        "metric": obj_cfg.get("metric", "N/A"),
        "time_aggregation": obj_cfg.get("time_aggregation", "N/A"),
        "spatial_resolution_km": obj_cfg.get("spatial_resolution_km", 2.0),
        "demand_weighted": obj_cfg.get("demand_weighted", False),
        "population_weighted": obj_cfg.get("population_weighted", False),
        "baseline_objective_value": raw_objective_value,
        "penalized_objective_value": penalized_objective_value,
        "is_feasible": total_penalty == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline GTFS across all configs")
    parser.add_argument(
        "--configs", nargs="+", default=None, help="Config files to evaluate (default: all configs/*.yaml)"
    )
    parser.add_argument(
        "--output",
        default="output/base_objective_values.csv",
        help="Output CSV path (default: output/base_objective_values.csv)",
    )
    parser.add_argument(
        "--pt-only-baseline",
        action="store_true",
        help="Force baseline evaluation to ignore DRT (PT only, disables drt_enabled in opt_data)",
    )
    args = parser.parse_args()

    # Create output directory and set up logging properly
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging in the same directory as the output CSV
    setup_logger(
        name="transit_opt",
        log_dir=str(output_path.parent),
        log_file="baseline_evaluation.log",
        console_level="INFO",
        file_level="DEBUG",
    )

    logger = logging.getLogger("transit_opt.scripts.evaluate_baselines")

    # Find config files
    if args.configs:
        config_paths = []
        for pattern in args.configs:
            # Handle glob patterns if passed as strings
            if "*" in pattern:
                config_paths.extend(list(Path(".").glob(pattern)))
            else:
                config_paths.append(Path(pattern))
    else:
        # Default: all YAML files in configs/ directory
        config_paths = list(Path("configs").glob("*.yaml"))

    logger.info(f"📋 Found {len(config_paths)} config files to process")

    results = []
    failed = []

    for config_path in sorted(config_paths):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📋 Processing: {config_path}")
        logger.info(f"{'=' * 70}")

        try:
            # Load config
            cfg = load_config(str(config_path))

            # Inject boundary if needed
            inject_boundary(cfg)

            # Prepare data
            opt_data = prepare_opt_data(cfg)

            # Force PT-only baseline if requested
            if args.pt_only_baseline and opt_data.get("drt_enabled"):
                logger.info("ℹ️  Forcing PT-only baseline (ignoring DRT configuration)")
                opt_data["drt_enabled"] = False
                # We might need to reshape initial solution if it was combined
                if isinstance(opt_data["initial_solution"], dict):
                    opt_data["initial_solution"] = opt_data["initial_solution"]["pt"]

            # Evaluate
            result = evaluate_baseline(cfg, opt_data)
            results.append(result)

        except Exception as e:
            logger.error(f"❌ Failed to process {config_path}: {e}", exc_info=True)
            failed.append(str(config_path))

    # Write results to CSV
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ Saved {len(results)} baseline evaluations to: {output_path}")
        logger.info(f"{'=' * 70}")

        # Print summary table
        logger.info("\n📊 BASELINE SUMMARY:")
        logger.info(f"{'Config':<20} {'Raw Obj':<15} {'Penalized Obj':<15} {'Feasible':<10}")
        logger.info("-" * 70)
        for r in results:
            logger.info(
                f"{r['config_name']:<20} {r['baseline_objective_value']:<15.2f} {r['penalized_objective_value']:<15.2f} {str(r['is_feasible']):<10}"
            )

    else:
        logger.warning("❌ No results to save!")

    # Report failures
    if failed:
        logger.error(f"\n❌ Failed configs ({len(failed)}):")
        for f in failed:
            logger.error(f"  - {f}")


if __name__ == "__main__":
    main()
