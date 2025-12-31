# Run script for entire pipeline
# Run from repo root:  python3 scripts/run.py --config configs/config_template.yaml
# Replace config_template with your config name

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import yaml

from transit_opt.logging import setup_logger

# put src on path
project_root = Path(__file__).resolve().parent
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


from transit_opt.gtfs.solution_manager import SolutionExportManager
from transit_opt.optimisation.config.config_manager import OptimizationConfigManager
from transit_opt.optimisation.runners.pso_runner import PSORunner
from transit_opt.optimisation.spatial.boundaries import StudyAreaBoundary
from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def inject_boundary_if_path(cfg: dict) -> None:
    logger = logging.getLogger("transit_opt.scripts.run")
    # Check if Boundary exists
    obj = cfg.setdefault("problem", {}).setdefault("objective", {})
    bpath = cfg.get("input", {}).get("boundary_geojson")
    if bpath:
        gdf = gpd.read_file(bpath)
        obj["boundary"] = StudyAreaBoundary(
            boundary_gdf=gdf,
            crs=obj.get("crs", cfg.get("problem", {}).get("objective", {}).get("crs", "EPSG:3857")),
            buffer_km=obj.get("boundary_buffer_km", cfg.get("input", {}).get("boundary_buffer_km", 2.0)),
        )
        logger.info("Injected StudyAreaBoundary from %s", bpath)


def prepare_opt_data(cfg: dict) -> dict:
    inp = cfg.get("input", {})
    gtfs_path = inp.get("gtfs_path")
    # allow single string or list
    if isinstance(gtfs_path, (list, tuple)):
        gtfs_path = gtfs_path[0]
    preparator = GTFSDataPreparator(gtfs_path=gtfs_path, interval_hours=inp.get("interval_hours"))
    allowed_headways = inp.get("allowed_headways")
    drt_cfg = inp.get("drt") or inp.get("drt_config") or {}
    if drt_cfg.get("enabled", False):
        return preparator.extract_optimization_data_with_drt(allowed_headways=allowed_headways, drt_config=drt_cfg)
    else:
        return preparator.extract_optimization_data(allowed_headways=allowed_headways)


def export_results(opt_data: dict, res, cfg: dict, iteration_dir: Path) -> None:
    logger = logging.getLogger("transit_opt.scripts.run")

    out_cfg = cfg.get("output", {})
    if not out_cfg.get("save_results", False):
        return

    # Use iteration-specific pso_results directory
    out_dir = iteration_dir / "pso_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_manager = SolutionExportManager(opt_data)

    # Use SolutionExportManager to extract solutions ---
    to_export = export_manager.extract_solutions_for_export(res, out_cfg)

    if not to_export:
        logger.warning("No solution available to export")
        return

    prefix = out_cfg.get("solution_prefix", "solution")
    logger.info("Exporting %d solution(s) to %s (prefix=%s)", len(to_export), out_dir, prefix)
    export_results = export_manager.export_solution_set(
        solutions=to_export, base_output_dir=str(out_dir), solution_prefix=prefix, metadata=None
    )

    logger.info("Export completed. Files written:")
    for r in export_results:
        for k, v in r["exports"].items():
            logger.info("  %s -> %s", k, v["path"])


def main(config_path: str, iteration: int = 1, overwrite: bool = False):
    # 1. Load config and set up logging
    cfg = load_config(config_path)

    # Get base results directory from config
    out_cfg = cfg.get("output", {})
    base_results_dir = Path(out_cfg.get("results_dir", "output/run"))

    # Create iteration-specific directory
    iteration_dir = base_results_dir / f"iteration_{iteration:02d}"
    pso_results_dir = iteration_dir / "pso_results"

    # overwrite check
    if pso_results_dir.exists() and not overwrite:
        print(f"\n❌ ERROR: Iteration directory already exists: {iteration_dir}")
        print(f"   Use --overwrite flag to replace existing results, or:")
        print(f"   • Run with different --iteration number")
        print(f"   • Manually delete: rm -rf {iteration_dir}")
        print(f"   • Move existing results: mv {iteration_dir} {iteration_dir}_backup")
        sys.exit(1)

    if pso_results_dir.exists() and overwrite:
        import shutil

        print(f"\n⚠️  WARNING: Overwriting existing iteration directory: {iteration_dir}")
        shutil.rmtree(iteration_dir)

    pso_results_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging in iteration directory
    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("log_file", "run.log")
    console_level = log_cfg.get("console_level", "INFO")
    file_level = log_cfg.get("file_level", "DEBUG")

    setup_logger(
        name="transit_opt",
        log_dir=str(pso_results_dir),
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
    )

    logger = logging.getLogger("transit_opt.scripts.run")
    logger.info(f"🔄 Starting optimization iteration {iteration}")
    logger.info(f"📁 Iteration directory: {iteration_dir}")
    logger.info(f"📁 PSO results directory: {pso_results_dir}")
    logger.info("📋 Config file:\n%s", yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    if out_cfg.get("save_results", False):
        import shutil

        out_dir = Path(out_cfg.get("results_dir"))
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_dest = pso_results_dir / "config.yaml"

        shutil.copy2(config_path, cfg_dest)
        logger.info(f"💾 Saved config to {cfg_dest}")

    # 2. Add study area boundary for clipping gtfs / zones
    inject_boundary_if_path(cfg)
    # 3. Prepare optimization data
    opt_data = prepare_opt_data(cfg)

    cfg_manager = OptimizationConfigManager(config_dict=cfg)

    # 4. Seed solutions if seeds are provided in config [seeding]
    # Resolve sampling descriptors to concrete seeds (if provided)
    try:
        cfg_manager.resolve_sampling_base_solutions(opt_data)
        logger.info("Resolved sampling.base_solutions (if descriptor present)")
    except Exception as e:
        logger.warning("Failed to resolve sampling.base_solutions: %s", e)
    # 5. Run optimization
    runner = PSORunner(cfg_manager)

    run_cfg = cfg.get("optimization", {}).get("run", {})
    if run_cfg.get("multi_swarm", False):
        res = runner.optimize_multi_run(
            optimization_data=opt_data,
            num_runs=run_cfg.get("num_runs"),
            parallel=run_cfg.get("parallel", False),
            track_best_n=run_cfg.get("track_best_n"),
        )
    else:
        res = runner.optimize(opt_data, track_best_n=run_cfg.get("track_best_n"))

    # 6. Export results to file
    export_results(opt_data, res, cfg, iteration_dir)

    logger.info("✅ Optimization complete!")
    logger.info(f"📊 Results saved to: {iteration_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run transit_opt PSO from config")
    p.add_argument("--config", "-c", default="configs/config_basic.yaml", help="YAML config path")
    p.add_argument("--iteration", "-i", type=int, default=1, help="Iteration number (default: 1)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing iteration directory")
    args = p.parse_args()
    main(args.config, iteration=args.iteration, overwrite=args.overwrite)
