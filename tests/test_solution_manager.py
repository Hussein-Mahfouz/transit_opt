import tempfile
from pathlib import Path

import numpy as np
import pytest

from transit_opt.gtfs.solution_manager import SolutionExportManager
from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


class TestSolutionExportManager:
    """Test solution export coordination with minimal metadata approach."""

    @pytest.fixture
    def pt_only_opt_data(self, sample_gtfs_path):
        """Create real PT-only optimization data."""
        print("🔧 Creating PT-only optimization data from real GTFS")
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)
        return preparator.extract_optimization_data([15, 30, 60, 120])

    @pytest.fixture
    def drt_opt_data(self, sample_gtfs_path):
        """Create real PT+DRT optimization data."""
        print("🔧 Creating PT+DRT optimization data from real GTFS")
        test_data_dir = Path(__file__).parent / "data"

        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)

        drt_config = {
            "enabled": True,
            "target_crs": "EPSG:3857",
            "zones": [
                {
                    "zone_id": "drt_duke_1",
                    "service_area_path": str(test_data_dir / "drt" / "drt_duke_1.shp"),
                    "allowed_fleet_sizes": [0, 5, 10, 15, 20],
                    "zone_name": "Duke Area 1",
                },
                {
                    "zone_id": "drt_duke_2",
                    "service_area_path": str(test_data_dir / "drt" / "drt_duke_2.shp"),
                    "allowed_fleet_sizes": [0, 8, 16, 24],
                    "zone_name": "Duke Area 2",
                },
            ],
        }

        return preparator.extract_optimization_data_with_drt([15, 30, 60, 120], drt_config)

    def test_initialization_for_different_problem_types(self, pt_only_opt_data, drt_opt_data):
        """
        Test SolutionExportManager initialization for both PT-only and PT+DRT problems.

        🎯 Purpose: Verify that the manager correctly identifies problem type and initializes
        the appropriate converters based on whether DRT is enabled.
        """
        print("\n" + "=" * 70)
        print("🧪 TESTING SOLUTION EXPORT MANAGER INITIALIZATION")
        print("=" * 70)

        # === Test PT-only initialization ===
        print("\n🚌 Testing PT-only problem initialization...")
        print(f"   Input problem type: {pt_only_opt_data['problem_type']}")
        print(f"   DRT enabled: {pt_only_opt_data.get('drt_enabled', False)}")

        pt_manager = SolutionExportManager(pt_only_opt_data)

        print(f"   ✅ Manager DRT enabled: {pt_manager.drt_enabled}")
        print(f"   ✅ DRT exporter: {pt_manager.drt_exporter}")
        print(f"   ✅ PT converter created: {pt_manager.pt_converter is not None}")

        assert pt_manager.drt_enabled is False
        assert pt_manager.drt_exporter is None
        assert pt_manager.pt_converter is not None

        # === Test PT+DRT initialization ===
        print("\n🚁 Testing PT+DRT problem initialization...")
        print(f"   Input problem type: {drt_opt_data['problem_type']}")
        print(f"   DRT enabled: {drt_opt_data.get('drt_enabled', False)}")
        print(f"   Number of DRT zones: {drt_opt_data.get('n_drt_zones', 0)}")

        drt_manager = SolutionExportManager(drt_opt_data)

        print(f"   ✅ Manager DRT enabled: {drt_manager.drt_enabled}")
        print(f"   ✅ DRT exporter created: {drt_manager.drt_exporter is not None}")
        print(f"   ✅ PT converter created: {drt_manager.pt_converter is not None}")

        assert drt_manager.drt_enabled is True
        assert drt_manager.drt_exporter is not None
        assert drt_manager.pt_converter is not None

        print("🎉 Initialization works correctly for both problem types!")

    def test_export_single_pt_solution(self, pt_only_opt_data):
        """
        Test export of a single PT-only solution with minimal metadata.

        🎯 Purpose: Verify that PT-only solution export creates GTFS files
        without unnecessary metadata embedding.
        """
        print("\n" + "=" * 70)
        print("🧪 TESTING SINGLE PT-ONLY SOLUTION EXPORT")
        print("=" * 70)

        # === Create realistic PT solution ===
        print("\n📊 Creating realistic PT solution...")
        n_routes = pt_only_opt_data["n_routes"]
        n_intervals = pt_only_opt_data["n_intervals"]

        # Create solution with valid headway indices
        pt_solution = np.random.randint(0, len(pt_only_opt_data["allowed_headways"]), size=(n_routes, n_intervals))

        print(f"   📈 Solution shape: {pt_solution.shape} ({n_routes} routes × {n_intervals} intervals)")
        print(f"   📋 Allowed headways: {pt_only_opt_data['allowed_headways']} minutes")
        print(f"   📋 Solution matrix:\n{pt_solution}")

        # === Initialize manager and export ===
        print("\n🏗️  Exporting PT solution...")
        manager = SolutionExportManager(pt_only_opt_data)

        # Only essential metadata
        essential_metadata = {"objective_value": 0.456}

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   📁 Output directory: {temp_dir}")

            result = manager.export_single_solution(
                solution=pt_solution, solution_id="pt_test", output_dir=temp_dir, metadata=essential_metadata
            )

            print("   ✅ Export completed")

            # === Verify minimal result structure ===
            print("\n📋 Verifying export result...")

            print(f"   🏷️  Solution ID: {result['solution_id']}")
            assert result["solution_id"] == "pt_test"

            print(f"   📊 Minimal metadata keys: {list(result['metadata'].keys())}")
            # Should only contain solution_id and what we explicitly provided
            assert "solution_id" in result["metadata"]
            assert result["metadata"]["objective_value"] == 0.456
            # Should NOT contain test framework metadata
            assert "algorithm" not in result["metadata"]
            assert "experiment_name" not in result["metadata"]

            print(f"   📦 Export types: {list(result['exports'].keys())}")
            assert "pt" in result["exports"]
            assert "drt" not in result["exports"]

            # === Verify GTFS file was created ===
            print("\n🚌 Verifying GTFS file creation...")
            pt_export = result["exports"]["pt"]

            print(f"   📄 GTFS file path: {pt_export['path']}")
            print(f"   🏷️  Service ID: {pt_export['service_id']}")
            print(f"   📁 File format: {pt_export['format']}")

            assert pt_export["type"] == "gtfs"
            assert pt_export["format"] == "zip"
            assert pt_export["service_id"] == "optimized_pt_test"
            assert Path(pt_export["path"]).exists()
            assert pt_export["path"].endswith(".zip")

            print("   ✅ GTFS ZIP file created successfully")
        print("🎉 PT-only solution export works correctly with minimal metadata!")

    def test_export_combined_pt_drt_solution(self, drt_opt_data):
        """
        Test export of a combined PT+DRT solution with minimal metadata.

        🎯 Purpose: Verify that combined solutions create both GTFS and DRT JSON files
        with only essential cross-references.
        """
        print("\n" + "=" * 70)
        print("🧪 TESTING COMBINED PT+DRT SOLUTION EXPORT")
        print("=" * 70)

        # === Create realistic combined solution ===
        print("\n📊 Creating realistic combined solution...")
        n_routes = drt_opt_data["n_routes"]
        n_intervals = drt_opt_data["n_intervals"]
        n_drt_zones = drt_opt_data["n_drt_zones"]

        pt_solution = np.random.randint(0, len(drt_opt_data["allowed_headways"]), size=(n_routes, n_intervals))

        # Create valid DRT solution (choice indices within allowed ranges)
        drt_solution = np.array(
            [
                [1, 3, 4, 2],  # Duke Area 1: valid indices for allowed_fleet_sizes
                [0, 2, 3, 1],  # Duke Area 2: valid indices for allowed_fleet_sizes
            ]
        )

        combined_solution = {"pt": pt_solution, "drt": drt_solution}

        print(f"   📈 PT shape: {pt_solution.shape} ({n_routes} routes × {n_intervals} intervals)")
        print(f"   📈 DRT shape: {drt_solution.shape} ({n_drt_zones} zones × {n_intervals} intervals)")
        print(f"   📋 PT matrix:\n{pt_solution}")
        print(f"   📋 DRT matrix:\n{drt_solution}")

        # === Initialize manager and export ===
        print("\n🏗️  Exporting combined solution...")
        manager = SolutionExportManager(drt_opt_data)

        # Only essential metadata
        essential_metadata = {"objective_value": 0.123}

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   📁 Output directory: {temp_dir}")

            result = manager.export_single_solution(
                solution=combined_solution,
                solution_id="combined_test",
                output_dir=temp_dir,
                metadata=essential_metadata,
            )

            print("   ✅ Combined export completed")

            # === Verify minimal result structure ===
            print("\n📋 Verifying combined export result...")

            print(f"   🏷️  Solution ID: {result['solution_id']}")
            assert result["solution_id"] == "combined_test"

            print(f"   📊 Objective value: {result['metadata']['objective_value']}")
            assert result["metadata"]["objective_value"] == 0.123

            print(f"   📦 Export types: {list(result['exports'].keys())}")
            assert "pt" in result["exports"]
            assert "drt" in result["exports"]

            # === Verify PT export ===
            print("\n🚌 Verifying PT export...")
            pt_export = result["exports"]["pt"]

            print(f"   📄 PT file: {pt_export['path']}")
            assert pt_export["type"] == "gtfs"
            assert pt_export["format"] == "zip"
            assert Path(pt_export["path"]).exists()
            print("   ✅ PT GTFS file created")

            # === Verify DRT export ===
            print("\n🚁 Verifying DRT export...")
            drt_export = result["exports"]["drt"]

            print(f"   📄 DRT file: {drt_export['path']}")
            print(f"   🔗 PT reference: {drt_export['pt_reference']}")

            assert drt_export["type"] == "drt"
            assert drt_export["format"] == "json"
            assert Path(drt_export["path"]).exists()
            assert drt_export["pt_reference"] == "combined_test_gtfs.zip"
            print("   ✅ DRT JSON file created with essential PT cross-reference only")

            # === Verify cross-references are consistent ===
            print("\n🔗 Verifying cross-reference consistency...")
            pt_filename = Path(pt_export["path"]).name
            drt_pt_ref = drt_export["pt_reference"]

            print(f"   📄 Actual PT filename: {pt_filename}")
            print(f"   🔗 DRT PT reference: {drt_pt_ref}")

            assert pt_filename == drt_pt_ref
            print("   ✅ Cross-references are consistent")

        print("🎉 Combined PT+DRT solution export works correctly with minimal metadata!")

    def test_export_solution_set_uses_rank_field(self, sample_optimization_data):
        """Test that export_solution_set properly uses the rank field in filenames."""
        print("\n🧪 TESTING EXPORT WITH RANK FIELD")

        manager = SolutionExportManager(sample_optimization_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create solutions with rank field (as extract_solutions_for_export would)
            solutions = [
                {"solution": np.random.randint(0, 3, (11, 8)), "objective": 100.0, "rank": 0},
                {"solution": np.random.randint(0, 3, (11, 8)), "objective": 110.0, "rank": 5},
                {"solution": np.random.randint(0, 3, (11, 8)), "objective": 120.0, "rank": 10},
            ]

            # Export solutions
            results = manager.export_solution_set(
                solutions=solutions, base_output_dir=temp_dir, solution_prefix="combined_solution"
            )

            # Verify filenames use rank values
            print("   📋 Exported files:")
            for result in results:
                solution_id = result["solution_id"]
                print(f"      {solution_id}")

                # Solution IDs should be combined_solution_00, combined_solution_05, combined_solution_10
                assert solution_id in ["combined_solution_00", "combined_solution_05", "combined_solution_10"]

            print("✅ Solution export uses rank field correctly")

    #########################
    # tests for different sampling approaches
    #########################

    def test_extract_with_power_law_sampling(self, sample_optimization_data):
        """Test power law sampling extraction strategy."""
        print("\n🧪 TESTING POWER LAW SAMPLING EXTRACTION")

        manager = SolutionExportManager(sample_optimization_data)

        mock_result = type(
            "obj",
            (object,),
            {
                "best_feasible_solutions": [
                    {"solution": np.random.randint(0, 3, (10, 4)), "objective": i * 10.0} for i in range(100)
                ]
            },
        )()

        output_cfg = {
            "best_run": True,
            "sampling_strategy": {
                "type": "power",
                "max_to_save": 10,
                "max_rank": 100,
                "power_exponent": 2.5,
            },
        }

        extracted = manager.extract_solutions_for_export(mock_result, output_cfg)

        ranks = [sol["rank"] for sol in extracted]
        print(f"   Power law ranks: {ranks}")
        print(f"   Expected: ~10, Got: {len(extracted)}")

        # Power law may return 8-10 due to rounding
        assert 8 <= len(extracted) <= 10
        assert ranks[0] == 0

        print("✅ Power law sampling works correctly")

    def test_extract_with_manual_ranks(self, sample_optimization_data):
        """Test manual rank specification."""
        print("\n🧪 TESTING MANUAL RANK SAMPLING")

        manager = SolutionExportManager(sample_optimization_data)

        mock_result = type(
            "obj",
            (object,),
            {
                "best_feasible_solutions": [
                    {"solution": np.random.randint(0, 3, (10, 4)), "objective": i * 10.0} for i in range(50)
                ]
            },
        )()

        manual_ranks = [0, 1, 5, 10, 25, 49]
        output_cfg = {
            "best_run": True,
            "sampling_strategy": {
                "type": "manual",
                "max_to_save": len(manual_ranks),
                "max_rank": 50,
                "manual_ranks": manual_ranks,
            },
        }

        extracted = manager.extract_solutions_for_export(mock_result, output_cfg)

        actual_ranks = [sol["rank"] for sol in extracted]
        print(f"   Expected ranks: {manual_ranks}")
        print(f"   Actual ranks: {actual_ranks}")

        assert actual_ranks == manual_ranks
        print("✅ Manual rank sampling works correctly")

    def test_extract_with_geometric_sampling(self, sample_optimization_data):
        """Test geometric (logarithmic) sampling."""
        print("\n🧪 TESTING GEOMETRIC SAMPLING")

        manager = SolutionExportManager(sample_optimization_data)

        mock_result = type(
            "obj",
            (object,),
            {
                "best_feasible_solutions": [
                    {"solution": np.random.randint(0, 3, (10, 4)), "objective": i * 10.0} for i in range(128)
                ]
            },
        )()

        output_cfg = {
            "best_run": True,
            "sampling_strategy": {
                "type": "geometric",
                "max_to_save": 8,
                "max_rank": 128,
                "geometric_base": 2.0,
            },
        }

        extracted = manager.extract_solutions_for_export(mock_result, output_cfg)

        actual_ranks = [sol["rank"] for sol in extracted]
        print(f"   Geometric ranks: {actual_ranks}")
        print(f"   Count: {len(actual_ranks)} (max 8)")

        assert len(actual_ranks) <= 8
        assert actual_ranks[0] == 0

        print("✅ Geometric sampling works correctly")

    def test_max_rank_defaults_to_all_solutions(self, sample_optimization_data):
        """Test that max_rank defaults to number of tracked solutions when not specified."""
        print("\n🧪 TESTING MAX_RANK DEFAULT BEHAVIOR")

        manager = SolutionExportManager(sample_optimization_data)

        mock_result = type(
            "obj",
            (object,),
            {
                "best_feasible_solutions": [
                    {"solution": np.random.randint(0, 3, (10, 4)), "objective": i * 10.0} for i in range(75)
                ]
            },
        )()

        output_cfg = {
            "best_run": True,
            "sampling_strategy": {
                "type": "uniform",
                "max_to_save": 10,
                # max_rank NOT specified - should default to 75
            },
        }

        extracted = manager.extract_solutions_for_export(mock_result, output_cfg)

        ranks = [sol["rank"] for sol in extracted]
        print(f"   Sampled ranks: {ranks}")
        print(f"   Last rank: {ranks[-1]} (should be ≥60)")

        assert len(extracted) == 10
        assert ranks[-1] >= 60

        print("✅ max_rank correctly defaults to all tracked solutions")

    def test_sampling_with_insufficient_solutions(self, sample_optimization_data):
        """Test sampling when fewer solutions available than requested."""
        print("\n🧪 TESTING SAMPLING WITH INSUFFICIENT SOLUTIONS")

        manager = SolutionExportManager(sample_optimization_data)

        mock_result = type(
            "obj",
            (object,),
            {
                "best_feasible_solutions": [
                    {"solution": np.random.randint(0, 3, (10, 4)), "objective": i * 10.0} for i in range(5)
                ]
            },
        )()

        output_cfg = {
            "best_run": True,
            "sampling_strategy": {"type": "uniform", "max_to_save": 10, "max_rank": 5},
        }

        extracted = manager.extract_solutions_for_export(mock_result, output_cfg)

        ranks = [sol["rank"] for sol in extracted]
        print(f"   Available: 5, Requested: 10, Got: {len(extracted)}")
        print(f"   Ranks: {ranks}")

        assert len(extracted) == 5
        assert set(ranks) == {0, 1, 2, 3, 4}

        print("✅ Correctly handles insufficient solutions")
