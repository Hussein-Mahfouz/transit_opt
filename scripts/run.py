import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import yaml

# put src on path
project_root = Path(__file__).resolve().parent
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


from transit_opt.gtfs.solution_manager import SolutionExportManager
from transit_opt.optimisation.config.config_manager import \
    OptimizationConfigManager
from transit_opt.optimisation.runners.pso_runner import PSORunner
from transit_opt.optimisation.spatial.boundaries import StudyAreaBoundary
from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

def inject_boundary_if_path(cfg: dict) -> None:
    # Check if Boundary exists
    obj = cfg.setdefault("problem", {}).setdefault("objective", {})
    bpath = cfg.get("input", {}).get("boundary_geojson")
    if bpath:
        gdf = gpd.read_file(bpath)
        obj["boundary"] = StudyAreaBoundary(
            boundary_gdf=gdf,
            crs=obj.get("crs", cfg.get("problem", {}).get("objective", {}).get("crs", "EPSG:3857")),
            buffer_km=obj.get("boundary_buffer_km", cfg.get("input", {}).get("boundary_buffer_km", 2.0))
        )
        logger.info("Injected StudyAreaBoundary from %s", bpath)

def prepare_opt_data(cfg: dict) -> dict:
    inp = cfg.get("input", {})
    gtfs_path = inp.get("gtfs_path")
    # allow single string or list
    if isinstance(gtfs_path, (list, tuple)):
        gtfs_path = gtfs_path[0]
    preparator = GTFSDataPreparator(
        gtfs_path=gtfs_path,
        interval_hours=inp.get("interval_hours"),
        log_level=cfg.get("logging", {}).get("console_level", "INFO"),
    )
    allowed_headways = inp.get("allowed_headways")
    drt_cfg = inp.get("drt") or inp.get("drt_config") or {}
    if drt_cfg.get("enabled", False):
        return preparator.extract_optimization_data_with_drt(
            allowed_headways=allowed_headways,
            drt_config=drt_cfg
        )
    else:
        return preparator.extract_optimization_data(allowed_headways=allowed_headways)


def export_results(opt_data: dict, res, cfg: dict) -> None:
    out_cfg = cfg.get("output", {})
    if not out_cfg.get("save_results", False):
        return

    out_dir = Path(out_cfg.get("results_dir"))
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
        solutions=to_export,
        base_output_dir=str(out_dir),
        solution_prefix=prefix,
        metadata=None
    )

    logger.info("Export completed. Files written:")
    for r in export_results:
        for k, v in r["exports"].items():
            logger.info("  %s -> %s", k, v["path"])


def main(config_path: str):
    # 1. Load config
    cfg = load_config(config_path)
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
        res = runner.optimize(
            opt_data,
            track_best_n=run_cfg.get("track_best_n")
        )
    # 6. Export results to file
    export_results(opt_data, res, cfg)

    print("Done. Best objective:", getattr(res, "best_objective", None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run transit_opt PSO from config")
    p.add_argument("--config", "-c", default="configs/config_basic.yaml", help="YAML config path")
    args = p.parse_args()
    main(args.config)
