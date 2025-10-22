"""
Unit tests for HexagonalZoneSystem calculation methods.

This test suite validates the core spatial calculation logic used in 
transit optimization objectives. Tests focus on mathematical correctness
and edge case handling rather than integration.

Test Coverage:
- DRT vehicle activity calculations
- PT+DRT data combination logic  
- Spatial intersection mapping
- Vehicle counting and aggregation
- Edge cases and error handling
"""

from pathlib import Path

import numpy as np
import pytest

from transit_opt.optimisation.spatial.zoning import HexagonalZoneSystem
from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


class TestHexagonalZoneSystemCalculations:
    """Unit tests for HexagonalZoneSystem calculation methods."""

    @pytest.fixture
    def drt_test_setup(self):
        """
        Unified fixture providing complete DRT test setup.
        
        Returns a dictionary with:
        - optimization_data: Complete PT+DRT optimization data
        - zone_system: HexagonalZoneSystem with DRT mappings already applied
        - test_data_dir: Path to test data directory
        """
        test_data_dir = Path(__file__).parent / "data"
        gtfs_path = str(test_data_dir / "duke-nc-us.zip")

        # Create preparator
        preparator = GTFSDataPreparator(gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60, 120]

        # DRT configuration
        drt_config = {
            'enabled': True,
            'target_crs': 'EPSG:3857',
            'default_drt_speed_kmh': 25.0,
            'zones': [
                {
                    'zone_id': 'test_zone_1',
                    'service_area_path': str(test_data_dir / "drt" / "drt_duke_1.shp"),
                    'allowed_fleet_sizes': [0, 10, 20, 30],
                    'zone_name': 'Test Zone 1',
                    'drt_speed_kmh': 20.0
                },
                {
                    'zone_id': 'test_zone_2',
                    'service_area_path': str(test_data_dir / "drt" / "drt_duke_2.shp"),
                    'allowed_fleet_sizes': [0, 5, 15, 25],
                    'zone_name': 'Test Zone 2'
                }
            ]
        }

        # Extract optimization data with DRT
        optimization_data = preparator.extract_optimization_data_with_drt(allowed_headways, drt_config)

        # Create zone system with DRT mappings
        zone_system = HexagonalZoneSystem(
            gtfs_feed=preparator.feed,
            hex_size_km=1.0,
            crs="EPSG:3857",
            drt_config=optimization_data['drt_config']  # DRT mappings applied during init
        )

        return {
            'optimization_data': optimization_data,
            'zone_system': zone_system,
            'test_data_dir': test_data_dir,
            'preparator': preparator  # In case tests need access to raw preparator
        }

    @pytest.fixture
    def pt_only_setup(self):
        """Fixture for PT-only tests (for backward compatibility testing)."""
        test_data_dir = Path(__file__).parent / "data"
        gtfs_path = str(test_data_dir / "duke-nc-us.zip")

        preparator = GTFSDataPreparator(gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60, 120]

        # PT-only optimization data
        optimization_data = preparator.extract_optimization_data(allowed_headways)

        # PT-only zone system
        zone_system = HexagonalZoneSystem(
            gtfs_feed=preparator.feed,
            hex_size_km=1.0,
            crs="EPSG:3857"
        )

        return {
            'optimization_data': optimization_data,
            'zone_system': zone_system,
            'test_data_dir': test_data_dir
        }


    def test_drt_vehicle_calculation_basic_math(self, drt_test_setup):
        """Test DRT vehicle activity calculation with known inputs."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Create controlled DRT solution
        n_intervals = optimization_data['n_intervals']
        drt_solution = np.array([
            [2, 1, 3, 0],  # Zone 1: [20, 10, 30, 0] vehicles
            [1, 2, 1, 0],  # Zone 2: [5, 15, 5, 0] vehicles
        ])
        assert drt_solution.shape == (2, n_intervals)

        # Calculate DRT vehicles
        drt_vehicles = zone_system._calculate_drt_vehicles_by_interval(
            drt_solution, optimization_data
        )

        # Verify structure
        n_hex_zones = len(zone_system.hex_grid)
        assert drt_vehicles['intervals'].shape == (n_intervals, n_hex_zones)
        assert len(drt_vehicles['average']) == n_hex_zones
        assert len(drt_vehicles['peak']) == n_hex_zones
        assert len(drt_vehicles['sum']) == n_hex_zones

        # Verify mathematical properties
        assert np.all(drt_vehicles['intervals'] >= 0), "Vehicle activity should be non-negative"
        assert np.all(drt_vehicles['average'] >= 0), "Average should be non-negative"
        assert np.all(drt_vehicles['peak'] >= 0), "Peak should be non-negative"

        # Test aggregation consistency
        calculated_average = np.mean(drt_vehicles['intervals'], axis=0)
        np.testing.assert_array_almost_equal(
            drt_vehicles['average'], calculated_average, decimal=10
        )

        calculated_peak = np.max(drt_vehicles['intervals'], axis=0)
        np.testing.assert_array_equal(drt_vehicles['peak'], calculated_peak)

        # Verify we actually have some activity (since mappings are guaranteed)
        total_activity = np.sum(drt_vehicles['average'])
        print(f"✅ DRT calculation structure verified: {drt_vehicles['intervals'].shape}")
        print(f"✅ Total DRT activity: {total_activity:.2f}")

        # Should have positive activity since DRT zones are properly mapped
        assert total_activity > 0, "Should have positive DRT activity with proper spatial mappings"

    def test_drt_vehicle_calculation_edge_cases(self, drt_test_setup):
        """Test DRT calculation edge cases."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        n_intervals = optimization_data['n_intervals']
        n_hex_zones = len(zone_system.hex_grid)

        # Test 1: Zero fleet everywhere
        zero_drt_solution = np.zeros((2, n_intervals), dtype=int)
        zero_result = zone_system._calculate_drt_vehicles_by_interval(
            zero_drt_solution, optimization_data
        )

        assert np.all(zero_result['intervals'] == 0), "Zero fleet should produce zero activity"
        assert np.all(zero_result['average'] == 0), "Zero fleet should produce zero average"
        assert np.all(zero_result['peak'] == 0), "Zero fleet should produce zero peak"

        # Test 2: Maximum fleet everywhere
        max_drt_solution = np.array([
            [3, 3, 3, 3],  # Zone 1: max fleet (index 3 = 30 vehicles)
            [3, 3, 3, 3],  # Zone 2: max fleet (index 3 = 25 vehicles)
        ])
        max_result = zone_system._calculate_drt_vehicles_by_interval(
            max_drt_solution, optimization_data
        )

        # Check if any zones are actually affected
        total_affected_zones = sum(len(zone.get('affected_hex_zones', []))
                                for zone in optimization_data['drt_config']['zones'])

        if total_affected_zones > 0:
            assert np.sum(max_result['average']) > 0, "Maximum fleet should produce positive activity when zones are mapped"
        else:
            print("⚠️ No hex zones affected by DRT - this indicates a spatial mapping issue")

        # Test 3: Single interval with fleet
        single_interval_solution = np.zeros((2, n_intervals), dtype=int)
        single_interval_solution[0, 1] = 2  # 20 vehicles in zone 1, interval 1
        single_result = zone_system._calculate_drt_vehicles_by_interval(
            single_interval_solution, optimization_data
        )

        if total_affected_zones > 0:
            # Activity should only be in interval 1
            interval_0_activity = np.sum(single_result['intervals'][0, :])
            interval_1_activity = np.sum(single_result['intervals'][1, :])
            interval_2_activity = np.sum(single_result['intervals'][2, :])

            assert interval_0_activity == 0, "No activity in interval 0"
            assert interval_1_activity > 0, "Activity in interval 1"
            assert interval_2_activity == 0, "No activity in interval 2"

        print("✅ Edge case testing completed")


    def test_combine_vehicle_data_math(self, drt_test_setup):
        """Test _combine_vehicle_data() mathematical correctness using simple data."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Simple test data - 4 intervals, 5 zones
        pt_data = {
            'intervals': np.array([
                [1, 2, 0, 1, 0],  # Interval 0: total = 4
                [3, 4, 2, 3, 1],  # Interval 1: total = 13 ← PEAK
                [2, 3, 1, 2, 0],  # Interval 2: total = 8
                [1, 1, 0, 1, 0],  # Interval 3: total = 3
            ], dtype=float),
            'average': np.array([1.75, 2.5, 0.75, 1.75, 0.25]),
            'peak': np.array([3, 4, 2, 3, 1]),  # From interval 1
            'sum': np.array([7, 10, 3, 7, 1]),
            'interval_labels': ['Morning', 'Midday', 'Evening', 'Night']
        }

        drt_data = {
            'intervals': np.array([
                [0, 1, 1, 0, 1],  # Interval 0: total = 3
                [1, 2, 2, 1, 2],  # Interval 1: total = 8 ← PEAK
                [1, 1, 1, 1, 1],  # Interval 2: total = 5
                [0, 0, 1, 0, 0],  # Interval 3: total = 1
            ], dtype=float),
            'average': np.array([0.5, 1.0, 1.25, 0.5, 1.0]),
            'peak': np.array([1, 2, 2, 1, 2]),  # From interval 1
            'sum': np.array([2, 4, 5, 2, 4]),
            'interval_labels': ['Morning', 'Midday', 'Evening', 'Night']
        }

        # Test the combination
        combined = zone_system._combine_vehicle_data(pt_data, drt_data)

        # Check addition for intervals, average, sum
        expected_intervals = pt_data['intervals'] + drt_data['intervals']
        np.testing.assert_array_equal(combined['intervals'], expected_intervals)
        np.testing.assert_array_equal(combined['average'], pt_data['average'] + drt_data['average'])
        np.testing.assert_array_equal(combined['sum'], pt_data['sum'] + drt_data['sum'])

        # Check NEW peak logic: should be from system peak interval (interval 1)
        combined_totals = np.sum(expected_intervals, axis=1)  # [4, 21, 13, 4]
        peak_interval = np.argmax(combined_totals)  # Should be 1
        expected_peak = expected_intervals[peak_interval, :]  # Values from interval 1

        np.testing.assert_array_equal(combined['peak'], expected_peak)
        assert peak_interval == 1, f"Peak should be interval 1, got {peak_interval}"

        print("✅ Data combination verified")
        print(f"   System peak: interval {peak_interval} (total: {combined_totals[peak_interval]})")

    def test_spatial_intersection_logic(self, drt_test_setup):
        """Test DRT zone spatial intersection calculations."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Get DRT zones with spatial mappings (should be included now)
        drt_zones = optimization_data['drt_config']['zones']

        # Verify that affected_hex_zones were calculated during zone system creation
        for drt_zone in drt_zones:
            assert 'affected_hex_zones' in drt_zone, f"Zone {drt_zone['zone_id']} missing spatial mapping"

            affected_zones = drt_zone['affected_hex_zones']
            assert isinstance(affected_zones, list), "affected_hex_zones should be a list"

            # Verify all indices are valid
            n_hex_zones = len(zone_system.hex_grid)
            for hex_idx in affected_zones:
                assert isinstance(hex_idx, (int, np.integer)), f"Index should be integer: {hex_idx}"
                assert 0 <= hex_idx < n_hex_zones, f"Invalid hex zone index: {hex_idx}"

            print(f"   Zone {drt_zone['zone_id']}: {len(affected_zones)} affected hex zones")

        # Test that spatial geometry exists
        for drt_zone in drt_zones:
            assert 'geometry' in drt_zone, f"Zone {drt_zone['zone_id']} missing geometry"
            assert hasattr(drt_zone['geometry'], 'bounds'), "Geometry should have bounds"

        print("✅ Spatial intersection logic verified")

    def test_vehicles_per_zone_pt_only_compatibility(self, drt_test_setup):
        """Test that _vehicles_per_zone works correctly for PT-only solutions."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Create PT-only solution
        n_routes = optimization_data['n_routes']
        n_intervals = optimization_data['n_intervals']
        pt_solution = np.random.randint(0, 4, size=(n_routes, n_intervals))

        # Test PT-only calculation (should work even with DRT-enabled system)
        pt_vehicles = zone_system._vehicles_per_zone(pt_solution, optimization_data)

        # Verify structure
        n_hex_zones = len(zone_system.hex_grid)
        expected_keys = ['intervals', 'average', 'peak', 'sum', 'interval_labels']

        for key in expected_keys:
            assert key in pt_vehicles, f"Missing key: {key}"

        assert pt_vehicles['intervals'].shape == (n_intervals, n_hex_zones)
        assert len(pt_vehicles['average']) == n_hex_zones
        assert len(pt_vehicles['peak']) == n_hex_zones

        # Test mathematical consistency
        calculated_avg = np.mean(pt_vehicles['intervals'], axis=0)
        np.testing.assert_array_almost_equal(pt_vehicles['average'], calculated_avg, decimal=10)

        calculated_peak = np.max(pt_vehicles['intervals'], axis=0)
        np.testing.assert_array_equal(pt_vehicles['peak'], calculated_peak)

        print("✅ PT-only compatibility verified")

    def test_vehicles_per_zone_pt_drt_combined(self, drt_test_setup):
        """Test _vehicles_per_zone with PT+DRT combined solutions."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Create combined solution
        n_routes = optimization_data['n_routes']
        n_intervals = optimization_data['n_intervals']

        pt_solution = np.random.randint(0, 4, size=(n_routes, n_intervals))
        drt_solution = np.array([
            [2, 1, 3, 0],  # Zone 1
            [1, 2, 1, 0],  # Zone 2
        ])

        combined_solution = {'pt': pt_solution, 'drt': drt_solution}

        # Calculate combined vehicles
        combined_vehicles = zone_system._vehicles_per_zone(combined_solution, optimization_data)

        # Calculate components separately for verification
        pt_only_vehicles = zone_system._calculate_pt_vehicles_by_interval(pt_solution, optimization_data)
        drt_only_vehicles = zone_system._calculate_drt_vehicles_by_interval(drt_solution, optimization_data)

        # Verify combination is correct
        expected_intervals = pt_only_vehicles['intervals'] + drt_only_vehicles['intervals']
        np.testing.assert_array_almost_equal(combined_vehicles['intervals'], expected_intervals)

        expected_average = pt_only_vehicles['average'] + drt_only_vehicles['average']
        np.testing.assert_array_almost_equal(combined_vehicles['average'], expected_average)

        # Peak should come from the interval with highest total vehicles
        combined_intervals = expected_intervals
        total_vehicles_by_interval = np.sum(combined_intervals, axis=1)
        peak_interval = np.argmax(total_vehicles_by_interval)
        expected_peak = combined_intervals[peak_interval, :]
        np.testing.assert_array_almost_equal(combined_vehicles['peak'], expected_peak)

        print("✅ PT+DRT combination verified")
        print(f"   Combined intervals shape: {combined_vehicles['intervals'].shape}")

    def test_drt_calculation_parameter_sensitivity(self, drt_test_setup):
        """Test that DRT calculations respond correctly to parameter changes."""

        setup = drt_test_setup
        zone_system = setup['zone_system']
        optimization_data = setup['optimization_data']

        # Base solution
        base_solution = np.array([
            [1, 1, 1, 1],  # Zone 1: consistent moderate fleet
            [1, 1, 1, 1],  # Zone 2: consistent moderate fleet
        ])

        base_result = zone_system._calculate_drt_vehicles_by_interval(
            base_solution, optimization_data
        )

        # Double the fleet size
        high_fleet_solution = np.array([
            [2, 2, 2, 2],  # Zone 1: higher fleet
            [2, 2, 2, 2],  # Zone 2: higher fleet
        ])

        high_fleet_result = zone_system._calculate_drt_vehicles_by_interval(
            high_fleet_solution, optimization_data
        )

        # Higher fleet should generally produce higher activity
        base_total_activity = np.sum(base_result['average'])
        high_total_activity = np.sum(high_fleet_result['average'])

        print("✅ Parameter sensitivity verified")
        print(f"   Base activity: {base_total_activity:.2f}")
        print(f"   High fleet activity: {high_total_activity:.2f}")

        # Only test if we have spatial mappings
        total_affected_zones = sum(len(zone.get('affected_hex_zones', []))
                                for zone in optimization_data['drt_config']['zones'])

        if total_affected_zones > 0 and base_total_activity > 0:
            assert high_total_activity >= base_total_activity, \
                "Higher fleet should not decrease total activity"
            print("   ✅ Sensitivity relationship verified")
        else:
            print("   ⚠️ No spatial activity detected - check DRT zone mappings")

    def test_drt_disabled_fallback(self, pt_only_setup):
        """Test DRT calculation when DRT is disabled."""

        setup = pt_only_setup
        zone_system = setup['zone_system']

        # Create fake PT-only optimization data
        pt_only_data = {
            'drt_enabled': False,
            'n_intervals': 4,
            'intervals': {
                'labels': ['Morning', 'Midday', 'Evening', 'Night'],
                'duration_minutes': 360
            }
        }

        # Any DRT solution matrix
        fake_drt_solution = np.array([[1, 2, 1, 0]])

        # Should return all zeros
        drt_result = zone_system._calculate_drt_vehicles_by_interval(
            fake_drt_solution, pt_only_data
        )

        n_hex_zones = len(zone_system.hex_grid)
        assert drt_result['intervals'].shape == (4, n_hex_zones)
        assert np.all(drt_result['intervals'] == 0), "DRT disabled should produce zero activity"
        assert np.all(drt_result['average'] == 0), "DRT disabled should produce zero average"
        assert np.all(drt_result['peak'] == 0), "DRT disabled should produce zero peak"

        print("✅ DRT disabled fallback verified")
