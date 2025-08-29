"""Test fixtures for spatial optimization tests."""

from pathlib import Path

import numpy as np
import pytest

from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


@pytest.fixture
def sample_gtfs_path():
    """Path to sample GTFS feed for testing."""
    # Store the GTFS file in tests/data/
    test_data_dir = Path(__file__).parent.parent / "data"
    gtfs_file = test_data_dir / "duke-nc-us.zip"

    # if not gtfs_file.exists():
    #     pytest.skip(f"Test GTFS file not found: {gtfs_file}")

    return str(gtfs_file)


@pytest.fixture
def sample_optimization_data(sample_gtfs_path):
    """Create optimization data for spatial testing."""
    preparator = GTFSDataPreparator(
        gtfs_path=sample_gtfs_path,
        interval_hours=3,
        log_level="ERROR",  # Quiet during tests
    )

    allowed_headways = [5, 10, 15, 30, 60, 120]
    return preparator.extract_optimization_data(allowed_headways)


@pytest.fixture
def sample_solutions(sample_optimization_data):
    """
    Create test solution matrices for constraint validation.

    WHAT THIS FIXTURE DOES:
    Creates 4 different service level scenarios as solution matrices where each element
    represents a headway choice index (not actual minutes).

    SOLUTION MATRIX FORMAT:
    - Shape: (n_routes, n_intervals)
    - Values: Indices into allowed_headways array
    - Example: If allowed_headways = [5, 10, 15, 30, 60, 120], then:
        * Index 0 = 5-minute headways
        * Index 1 = 10-minute headways
        * Index 2 = 15-minute headways
        * Index 3 = 30-minute headways
        * Index 6 = no service (9999 minutes)

    SOLUTION TYPES:
    - high_service: All zeros (index 0 = 5min headways) - MOST EXPENSIVE
    - medium_service: All ones (index 1 = 10min headways) - MODERATE COST
    - low_service: All twos (index 3 = 30min headways) - LOWER COST
    - no_service: All no_service_index (9999min) - ZERO COST

    Args:
        sample_optimization_data: GTFS-derived optimization data structure

    Returns:
        dict: Solution matrices for different service levels
    """
    n_routes = sample_optimization_data["n_routes"]
    n_intervals = sample_optimization_data["n_intervals"]
    no_service_index = sample_optimization_data["no_service_index"]
    allowed_headways = sample_optimization_data["allowed_headways"]

    print("\n📊 CREATING TEST SOLUTIONS:")
    print(f"   Routes: {n_routes}, Intervals: {n_intervals}")
    print(f"   Allowed headways: {allowed_headways}")
    print(f"   No-service index: {no_service_index}")

    solutions = {
        # High service solution (frequent headways - use index 0)
        "high_service": np.zeros((n_routes, n_intervals), dtype=int),
        # Medium service solution (moderate headways - use index 1)
        "medium_service": np.ones((n_routes, n_intervals), dtype=int),
        # Low service solution (sparse headways - use index 3)
        "low_service": np.full((n_routes, n_intervals), 3, dtype=int),
        # Minimal service solution (mostly no service)
        "no_service": np.full((n_routes, n_intervals), no_service_index, dtype=int),
    }

    # Print solution details for debugging
    for name, matrix in solutions.items():
        print(f"   {name}: shape={matrix.shape}, unique_values={np.unique(matrix)}")

    return solutions


@pytest.fixture
def precalculated_fleet_data(sample_optimization_data):
    """
    Calculate expected fleet requirements for test solutions using real GTFS parameters.

    WHAT THIS FIXTURE DOES:
    1. Extracts baseline fleet data already calculated by GTFSDataPreparator
    2. Creates test solution matrices (same as sample_solutions fixture)
    3. Calculates exact fleet requirements for each solution using calculate_fleet_requirements()
    4. Returns both baseline data and calculated fleet data for deterministic testing

    WHY WE NEED THIS:
    - Constraint tests need to know EXACT expected values, not just relationships
    - Uses same fleet calculation logic as the actual constraint handlers
    - Provides ground truth for test assertions

    BASELINE DATA (from GTFS analysis):
    - current_peak_fleet: Maximum vehicles needed system-wide from current schedule
    - current_fleet_by_interval: Vehicles needed per time interval from current schedule
    - current_fleet_per_route: Peak vehicles needed per route from current schedule

    SOLUTION DATA (calculated for test solutions):
    - peak_fleet: Maximum vehicles needed across all intervals for this solution
    - fleet_by_interval: Vehicles needed per interval for this solution
    - average_fleet: Average vehicles across all intervals for this solution

    Args:
        sample_optimization_data: GTFS-derived optimization data structure

    Returns:
        dict: {"baseline": baseline_data, "solutions": solution_fleet_data}
    """
    print("\n🔧 CALCULATING FLEET REQUIREMENTS FOR TEST SOLUTIONS:")

    # Extract already-calculated baseline data from GTFSDataPreparator
    fleet_analysis = sample_optimization_data["constraints"]["fleet_analysis"]
    baseline_data = {
        "current_peak_fleet": fleet_analysis["total_current_fleet_peak"],
        "current_fleet_by_interval": fleet_analysis["current_fleet_by_interval"],
        "current_fleet_per_route": fleet_analysis["current_fleet_per_route"],
    }

    print("   📊 BASELINE (from GTFS current schedule):")
    print(f"      System peak fleet: {baseline_data['current_peak_fleet']} vehicles")
    print(f"      Fleet by interval: {baseline_data['current_fleet_by_interval']}")

    # Get parameters for fleet calculations (same as GTFSDataPreparator used)
    from transit_opt.optimisation.utils.fleet_calculations import calculate_fleet_requirements

    allowed_headways = sample_optimization_data["allowed_headways"]
    round_trip_times = sample_optimization_data["routes"]["round_trip_times"]
    operational_buffer = fleet_analysis["operational_buffer"]
    no_service_threshold = fleet_analysis["no_service_threshold_minutes"]
    no_service_index = sample_optimization_data["no_service_index"]

    n_routes = sample_optimization_data["n_routes"]
    n_intervals = sample_optimization_data["n_intervals"]

    print("   🔧 FLEET CALCULATION PARAMETERS:")
    print(f"      Operational buffer: {operational_buffer}")
    print(f"      No-service threshold: {no_service_threshold} minutes")
    print(f"      Round-trip times: {round_trip_times}")

    # Create same test solutions as sample_solutions fixture
    test_solutions = {
        "high_service": np.zeros((n_routes, n_intervals), dtype=int),
        "medium_service": np.ones((n_routes, n_intervals), dtype=int),
        "low_service": np.full((n_routes, n_intervals), 3, dtype=int),
        "no_service": np.full((n_routes, n_intervals), no_service_index, dtype=int),
    }

    solution_fleet_data = {}
    print("\n   🚌 CALCULATING FLEET FOR EACH TEST SOLUTION:")

    for solution_name, solution_matrix in test_solutions.items():
        print(f"\n      → {solution_name.upper()}:")

        # Convert solution indices to actual headway minutes
        # This is the key step - solution_matrix contains indices, we need actual headway values
        headways_matrix = np.array(
            [
                [allowed_headways[solution_matrix[i, j]] for j in range(n_intervals)]
                for i in range(n_routes)
            ]
        )

        print(f"        Solution indices: {np.unique(solution_matrix)}")
        print(f"        Actual headways: {np.unique(headways_matrix)}")

        # Calculate fleet requirements using same function as constraint handlers
        fleet_results = calculate_fleet_requirements(
            headways_matrix=headways_matrix,
            round_trip_times=round_trip_times,
            operational_buffer=operational_buffer,
            no_service_threshold=no_service_threshold,
            allowed_headways=allowed_headways,
            no_service_index=no_service_index,
        )

        # Store calculated results
        fleet_data = {
            "peak_fleet": fleet_results["total_peak_fleet"],
            "fleet_by_interval": fleet_results["fleet_per_interval"],
            "fleet_per_route": fleet_results["fleet_per_route"],
            "average_fleet": np.mean(fleet_results["fleet_per_interval"]),
        }

        solution_fleet_data[solution_name] = fleet_data

        # Print calculated results for debugging
        print(f"        Peak fleet: {fleet_data['peak_fleet']} vehicles")
        print(f"        Average fleet: {fleet_data['average_fleet']:.1f} vehicles")
        print(f"        Fleet by interval: {fleet_data['fleet_by_interval']}")

    result = {"baseline": baseline_data, "solutions": solution_fleet_data}

    print("\n✅ PRECALCULATED FLEET DATA READY")
    return result

