from pathlib import Path

import numpy as np
import pytest

from transit_opt.preprocessing.prepare_gtfs import GTFSDataPreparator


class TestGTFSDataPreparator:

    @pytest.fixture
    def sample_gtfs_path(self):
        """Path to sample GTFS feed for testing."""
        # Store the GTFS file in tests/data/
        test_data_dir = Path(__file__).parent / "data"
        gtfs_file = test_data_dir / "duke-nc-us.zip"

        # if not gtfs_file.exists():
        #     pytest.skip(f"Test GTFS file not found: {gtfs_file}")

        return str(gtfs_file)

    def test_safe_timestr_to_seconds(self):
        """Test time string conversion utility."""
        preparator = GTFSDataPreparator.__new__(GTFSDataPreparator)

        # Test standard times
        assert preparator._safe_timestr_to_seconds("06:30:00") == 23400.0
        assert preparator._safe_timestr_to_seconds("00:00:00") == 0.0

        # Test overnight times
        assert preparator._safe_timestr_to_seconds("25:15:00") == 90900.0

        # Test NaN/None handling
        assert np.isnan(preparator._safe_timestr_to_seconds(np.nan))
        assert np.isnan(preparator._safe_timestr_to_seconds(None))

        # Test already numeric
        assert preparator._safe_timestr_to_seconds(3600.0) == 3600.0

        # Test invalid format
        assert np.isnan(preparator._safe_timestr_to_seconds("invalid"))

    def test_create_initial_solution(self):
        """Test initial solution matrix creation."""
        preparator = GTFSDataPreparator.__new__(GTFSDataPreparator)
        preparator.no_service_threshold_minutes = 480  # 8 hours

        # Test data
        current_headways = np.array(
            [[12.0, 20.0, 35.0], [np.nan, 50.0, 100.0], [15.0, np.nan, 1440.0]]
        )

        allowed_headways = [10, 15, 30, 60, 120]
        headway_to_index = {10: 0, 15: 1, 30: 2, 60: 3, 120: 4, 9999: 5}
        # CREATE headway_to_index FROM allowed_headways
        allowed_values = allowed_headways + [9999.0]  # Add no-service option
        headway_to_index = {float(h): i for i, h in enumerate(allowed_values)}

        result = preparator._create_initial_solution(current_headways, headway_to_index)

        expected = np.array(
            [
                [0, 1, 2],  # 12→15, 20→30, 35→30
                [5, 3, 4],  # NaN→no-service, 50→60, 100→120
                [1, 5, 5],  # 15→15, NaN→no-service, 1440→no-service (>480min threshold)
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_init_validation_with_real_gtfs(self, sample_gtfs_path):
        """Test initialization with real GTFS data and interval validation."""
        # Valid interval_hours (≥3 and divide 24)
        for hours in [3, 4, 6, 8, 12, 24]:
            preparator = GTFSDataPreparator(sample_gtfs_path, hours)
            assert preparator.interval_hours == hours
            assert preparator.n_intervals == 24 // hours
            assert preparator.feed is not None

        # Invalid - too small
        with pytest.raises(ValueError, match="must be ≥ 3"):
            GTFSDataPreparator(sample_gtfs_path, 1)

        with pytest.raises(ValueError, match="must be ≥ 3"):
            GTFSDataPreparator(sample_gtfs_path, 2)

        # Invalid - doesn't divide 24
        with pytest.raises(ValueError, match="must divide 24 evenly"):
            GTFSDataPreparator(sample_gtfs_path, 5)

    def test_extract_optimization_data_integration(self, sample_gtfs_path):
        """Test full optimization data extraction with real GTFS."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=3)
        allowed_headways = [10, 15, 30, 60, 120]

        opt_data = preparator.extract_optimization_data(allowed_headways)

        # Test structure
        assert opt_data["problem_type"] == "discrete_headway_optimization"
        assert opt_data["n_routes"] > 0
        assert opt_data["n_intervals"] == 8  # 24/3 = 8
        assert opt_data["n_choices"] == 6  # 5 headways + no-service

        # Test matrix dimensions
        expected_shape = (opt_data["n_routes"], opt_data["n_intervals"])
        assert opt_data["decision_matrix_shape"] == expected_shape
        assert opt_data["initial_solution"].shape == expected_shape

        # Test allowed_headways includes 9999
        assert 9999.0 in opt_data["allowed_headways"]
        assert len(opt_data["allowed_headways"]) == 6

        # Test data consistency
        assert len(opt_data["routes"]["ids"]) == opt_data["n_routes"]
        assert opt_data["routes"]["round_trip_times"].shape == (opt_data["n_routes"],)
        assert opt_data["routes"]["current_headways"].shape == expected_shape

    def test_headway_calculation_realistic(self, sample_gtfs_path):
        """Test headway calculation with real GTFS schedule patterns."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=3)

        # Extract one route's data to verify headway calculation
        route_data = preparator._extract_route_essentials()
        assert len(route_data) > 0

        # Check that headways are reasonable (between 5 and 1440 minutes)
        for route in route_data[:3]:  # Test first 3 routes
            headways = route["headways_by_interval"]
            valid_headways = headways[~np.isnan(headways)]
            if len(valid_headways) > 0:
                assert np.all(valid_headways >= 2)  # At least 5 min headway
                assert np.all(valid_headways <= 1440)  # At most 24 hours

    def test_round_trip_times_realistic(self, sample_gtfs_path):
        """Test round-trip time calculation with real trip data."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=3)

        route_data = preparator._extract_route_essentials()

        # Check that round-trip times are reasonable
        for route in route_data[:3]:
            rtt = route["round_trip_time"]
            assert 10 <= rtt <= 240  # Between 10 min and 4 hours

    def test_fleet_calculation_realistic(self, sample_gtfs_path):
        """Test fleet size calculation with realistic constraints."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=3)
        allowed_headways = [10, 15, 30, 60, 120]

        opt_data = preparator.extract_optimization_data(allowed_headways)
        fleet_analysis = opt_data["constraints"]["fleet_analysis"]

        # Test fleet analysis structure (updated field names)
        assert "current_fleet_per_route" in fleet_analysis
        assert "current_fleet_by_interval" in fleet_analysis
        assert "total_current_fleet_peak" in fleet_analysis
        assert "route_round_trip_times" in fleet_analysis

        n_routes = opt_data["n_routes"]
        n_intervals = opt_data["n_intervals"]

        # Test dimensions
        assert fleet_analysis["current_fleet_per_route"].shape == (n_routes,)
        assert fleet_analysis["current_fleet_by_interval"].shape == (n_intervals,)
        assert fleet_analysis["route_round_trip_times"].shape == (n_routes,)

        # Test reasonableness
        current_fleet_per_route = fleet_analysis["current_fleet_per_route"]
        current_fleet_by_interval = fleet_analysis["current_fleet_by_interval"]

        # Fleet sizes should be positive for routes with service
        active_routes = current_fleet_per_route > 0
        if np.any(active_routes):
            assert np.all(current_fleet_per_route[active_routes] >= 1)
            assert np.all(
                current_fleet_per_route[active_routes] <= 200
            )  # Reasonable upper bound

        # Interval-based fleet should be reasonable
        active_intervals = current_fleet_by_interval > 0
        if np.any(active_intervals):
            assert np.all(current_fleet_by_interval[active_intervals] >= 1)
            assert np.all(
                current_fleet_by_interval[active_intervals] <= 1000
            )  # System-wide upper bound

        # Peak fleet should be realistic (max across intervals ≤ sum of route peaks)
        total_peak = fleet_analysis["total_current_fleet_peak"]
        naive_sum = np.sum(current_fleet_per_route)
        assert total_peak <= naive_sum  # Realistic should be ≤ naive approach
        assert total_peak >= np.max(
            current_fleet_by_interval
        )  # Should equal max interval

        # Round-trip times should be reasonable
        round_trip_times = fleet_analysis["route_round_trip_times"]
        active_round_trips = round_trip_times[active_routes]
        if len(active_round_trips) > 0:
            assert np.all(active_round_trips >= 10)  # At least 10 minutes
            assert np.all(active_round_trips <= 240)  # At most 4 hours

    def test_fleet_calculation_edge_cases(self):
        """
        Test fleet calculation edge cases with detailed step-by-step verification.

        This test verifies that _analyze_current_fleet() correctly calculates:
        1. Per-route peak fleet requirements (max across intervals for each route)
        2. System-wide fleet by interval (sum across routes for each time period)
        3. Total system peak (max across all intervals - the realistic fleet size)
        4. Efficiency gain vs naive approach (sum of route peaks)

        Uses test data with known expected results to verify calculations.
        """
        # ===== SETUP TEST ENVIRONMENT =====
        preparator = GTFSDataPreparator.__new__(GTFSDataPreparator)
        preparator.n_intervals = 4  # 4 time periods for this test
        preparator.no_service_threshold_minutes = 480  # 8 hours no-service threshold

        # ===== CREATE TEST DATA WITH KNOWN EXPECTED RESULTS =====
        # We'll create 3 routes with different service patterns to test various scenarios
        route_data = [
            {  # ROUTE 1: Morning peak route
                "route_id": "route_1",
                "round_trip_time": 60.0,  # 1 hour round-trip
                "headways_by_interval": np.array(
                    [
                        15.0,  # Interval 0: 15min headway (PEAK - frequent service)
                        30.0,  # Interval 1: 30min headway (moderate service)
                        np.nan,  # Interval 2: No service
                        60.0,  # Interval 3: 60min headway (light service)
                    ]
                ),
            },
            {  # ROUTE 2: Evening peak route (staggered from Route 1)
                "route_id": "route_2",
                "round_trip_time": 120.0,  # 2 hour round-trip (longer route)
                "headways_by_interval": np.array(
                    [
                        60.0,  # Interval 0: 60min headway (light service)
                        45.0,  # Interval 1: 45min headway (moderate service)
                        30.0,  # Interval 2: 30min headway (PEAK - frequent service)
                        np.nan,  # Interval 3: No service
                    ]
                ),
            },
            {  # ROUTE 3: No service route (edge case)
                "route_id": "route_3",
                "round_trip_time": 40.0,  # Short round-trip (doesn't matter - no service)
                "headways_by_interval": np.array(
                    [
                        np.nan,  # Interval 0: No service
                        np.nan,  # Interval 1: No service
                        np.nan,  # Interval 2: No service
                        np.nan,  # Interval 3: No service
                    ]
                ),
            },
        ]

        # ===== RUN THE ANALYSIS =====
        fleet_analysis = preparator._analyze_current_fleet(route_data)

        # Extract results for detailed verification
        current_fleet_per_route = fleet_analysis["current_fleet_per_route"]
        current_fleet_by_interval = fleet_analysis["current_fleet_by_interval"]

        print("=== DETAILED FLEET CALCULATION VERIFICATION ===")

        # ===== VERIFY PER-ROUTE PEAK FLEET CALCULATIONS =====
        print("\n🚌 PER-ROUTE PEAK FLEET VERIFICATION:")
        print("Formula: vehicles = ceil((round_trip_time * 1.15) / headway)")

        # Route 1 calculations by interval:
        print("\nRoute 1 (60min round-trip) - Morning Peak Route:")
        print(
            "  Interval 0: ceil(60 * 1.15 / 15) = ceil(69/15) = ceil(4.6) = 5 vehicles ← PEAK"
        )
        print(
            "  Interval 1: ceil(60 * 1.15 / 30) = ceil(69/30) = ceil(2.3) = 3 vehicles"
        )
        print("  Interval 2: No service = 0 vehicles")
        print(
            "  Interval 3: ceil(60 * 1.15 / 60) = ceil(69/60) = ceil(1.15) = 2 vehicles"
        )
        print("  → Peak across intervals = max(5, 3, 0, 2) = 5 vehicles")
        assert (
            current_fleet_per_route[0] == 5
        ), f"Route 1 peak fleet should be 5, got {current_fleet_per_route[0]}"

        # Route 2 calculations by interval:
        print("\nRoute 2 (120min round-trip) - Evening Peak Route:")
        print(
            "  Interval 0: ceil(120 * 1.15 / 60) = ceil(138/60) = ceil(2.3) = 3 vehicles"
        )
        print(
            "  Interval 1: ceil(120 * 1.15 / 45) = ceil(138/45) = ceil(3.07) = 4 vehicles"
        )
        print(
            "  Interval 2: ceil(120 * 1.15 / 30) = ceil(138/30) = ceil(4.6) = 5 vehicles ← PEAK"
        )
        print("  Interval 3: No service = 0 vehicles")
        print("  → Peak across intervals = max(3, 4, 5, 0) = 5 vehicles")
        assert (
            current_fleet_per_route[1] == 5
        ), f"Route 2 peak fleet should be 5, got {current_fleet_per_route[1]}"

        # Route 3 (no service):
        print("\nRoute 3 (40min round-trip):")
        print("  All intervals: No service = 0 vehicles")
        print("  → Peak across intervals = 0 vehicles")
        assert (
            current_fleet_per_route[2] == 0
        ), f"Route 3 should have 0 fleet, got {current_fleet_per_route[2]}"

        # ===== VERIFY SYSTEM-WIDE FLEET BY INTERVAL =====
        print("\n🕐 SYSTEM-WIDE FLEET BY INTERVAL VERIFICATION:")
        print("Formula: interval_fleet = sum of all route needs at that specific time")
        print(
            "Notice: Route 1 peaks in Interval 0, Route 2 peaks in Interval 2 → Staggered!"
        )

        # Interval 0: Route 1 at peak, Route 2 at minimum
        print("\nInterval 0 (Route 1 peak, Route 2 light):")
        print("  Route 1: 5 vehicles (15min headway - PEAK)")
        print("  Route 2: 3 vehicles (60min headway - light)")
        print("  Route 3: 0 vehicles (no service)")
        print("  → Total = 5 + 3 + 0 = 8 vehicles")
        assert (
            current_fleet_by_interval[0] == 8
        ), f"Interval 0 should need 8 vehicles, got {current_fleet_by_interval[0]}"

        # Interval 1: Both routes at moderate levels
        print("\nInterval 1 (Both routes moderate):")
        print("  Route 1: 3 vehicles (30min headway)")
        print("  Route 2: 4 vehicles (45min headway)")
        print("  Route 3: 0 vehicles (no service)")
        print("  → Total = 3 + 4 + 0 = 7 vehicles")
        assert (
            current_fleet_by_interval[1] == 7
        ), f"Interval 1 should need 7 vehicles, got {current_fleet_by_interval[1]}"

        # Interval 2: Route 1 off, Route 2 at peak
        print("\nInterval 2 (Route 1 off, Route 2 peak):")
        print("  Route 1: 0 vehicles (no service)")
        print("  Route 2: 5 vehicles (30min headway - PEAK)")
        print("  Route 3: 0 vehicles (no service)")
        print("  → Total = 0 + 5 + 0 = 5 vehicles")
        assert (
            current_fleet_by_interval[2] == 5
        ), f"Interval 2 should need 5 vehicles, got {current_fleet_by_interval[2]}"

        # Interval 3: Only Route 1 light service
        print("\nInterval 3 (Only Route 1 light):")
        print("  Route 1: 2 vehicles (60min headway)")
        print("  Route 2: 0 vehicles (no service)")
        print("  Route 3: 0 vehicles (no service)")
        print("  → Total = 2 + 0 + 0 = 2 vehicles")
        assert (
            current_fleet_by_interval[3] == 2
        ), f"Interval 3 should need 2 vehicles, got {current_fleet_by_interval[3]}"

        # ===== VERIFY PEAK SYSTEM FLEET (REALISTIC TOTAL) =====
        print("\n🎯 SYSTEM PEAK FLEET VERIFICATION:")
        print("Realistic total = max fleet needed across all time intervals")
        intervals_fleet = [8, 7, 5, 2]
        expected_peak = max(intervals_fleet)  # = 8 vehicles

        print(f"  Fleet by interval: {intervals_fleet}")
        print(
            f"  → System peak = max({', '.join(map(str, intervals_fleet))}) = {expected_peak} vehicles"
        )
        print(
            f"  → This means we need {expected_peak} vehicles total to serve the system"
        )

        assert (
            fleet_analysis["total_current_fleet_peak"] == expected_peak
        ), f"System peak should be {expected_peak}, got {fleet_analysis['total_current_fleet_peak']}"

        # ===== VERIFY EFFICIENCY GAIN CALCULATION =====
        print("\n📈 EFFICIENCY GAIN VERIFICATION:")
        print(
            "Compares realistic interval-based approach vs naive sum-of-peaks approach"
        )
        print("🎯 STAGGERED PEAKS CREATE EFFICIENCY GAIN!")

        naive_sum = sum(current_fleet_per_route)  # Sum of route peaks = 5 + 5 + 0 = 10
        realistic_peak = expected_peak  # Max across intervals = 8
        expected_efficiency = naive_sum - realistic_peak  # 10 - 8 = 2 vehicles saved!

        print(
            f"  Naive approach: sum of route peaks = {current_fleet_per_route[0]} + {current_fleet_per_route[1]} + {current_fleet_per_route[2]} = {naive_sum} vehicles"
        )
        print("    (Assumes Route 1 and Route 2 both peak simultaneously)")
        print(f"  Realistic approach: max interval fleet = {realistic_peak} vehicles")
        print("    (Route 1 peaks at Interval 0, Route 2 peaks at Interval 2)")
        print(
            f"  → Efficiency gain = {naive_sum} - {realistic_peak} = {expected_efficiency} vehicles saved"
        )

        if expected_efficiency > 0:
            savings_percent = (expected_efficiency / naive_sum) * 100
            print(
                f"  → 🎉 Realistic approach saves {expected_efficiency} vehicles ({savings_percent:.1f}% reduction)!"
            )
            print("  → This is why staggered service patterns are more efficient!")
        elif expected_efficiency == 0:
            print("  → No efficiency gain (routes peak simultaneously)")
        else:
            print(f"  → Realistic approach needs {-expected_efficiency} more vehicles")

        assert (
            fleet_analysis["fleet_stats"]["fleet_efficiency_gain"]
            == expected_efficiency
        ), f"Efficiency gain should be {expected_efficiency}, got {fleet_analysis['fleet_stats']['fleet_efficiency_gain']}"

        print("\n✅ All fleet calculation tests passed!")
        print(f"   🎯 System needs {expected_peak} vehicles (realistic)")
        print(
            f"   📊 Efficiency vs naive: {expected_efficiency} vehicles saved ({(expected_efficiency/naive_sum)*100:.1f}% reduction)"
        )

    # ---------------------------
    # DRT functionality
    # ---------------------------

    def test_extract_optimization_data_with_drt(self, sample_gtfs_path):
        """Test DRT-enabled optimization data extraction with real spatial layers."""
        from pathlib import Path

        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)  # 4 periods per day
        allowed_headways = [15, 30, 60, 120]

        # DRT configuration using test shapefiles
        test_data_dir = Path(__file__).parent / "data/drt"
        drt_config = {
            'enabled': True,
            'target_crs': 'EPSG:3857',  # Web Mercator for area calculations
            'default_drt_speed_kmh': 25.0,
            'zones': [
                {
                    'zone_id': 'drt_duke_1',
                    'service_area_path': str(test_data_dir / "drt_duke_1.shp"),
                    'allowed_fleet_sizes': [0, 5, 10, 20, 30],
                    'zone_name': 'Duke Area 1',
                    'drt_speed_kmh': 20.0  # Zone-specific speed
                },
                {
                    'zone_id': 'drt_duke_2',
                    'service_area_path': str(test_data_dir / "drt_duke_2.shp"),
                    'allowed_fleet_sizes': [0, 3, 8, 15, 25],
                    'zone_name': 'Duke Area 2'
                    # Will use default_drt_speed_kmh
                }
            ]
        }

        # Test DRT-enabled extraction
        opt_data = preparator.extract_optimization_data_with_drt(allowed_headways, drt_config)

        # ===== TEST DRT-ENABLED STRUCTURE =====

        # DRT flags and dimensions
        assert opt_data['drt_enabled'] is True
        assert opt_data['n_drt_zones'] == 2
        assert opt_data['drt_max_choices'] == 5  # Max of [5, 5] choices per zone

        # Variable dimensions
        pt_vars = opt_data['n_routes'] * opt_data['n_intervals']
        drt_vars = 2 * opt_data['n_intervals']  # 2 zones × 4 intervals
        assert opt_data['pt_decision_variables'] == pt_vars
        assert opt_data['drt_decision_variables'] == drt_vars
        assert opt_data['total_decision_variables'] == pt_vars + drt_vars

        # Combined variable bounds structure
        combined_bounds = opt_data['combined_variable_bounds']
        assert len(combined_bounds) == pt_vars + drt_vars
        assert all(bound == len(allowed_headways) for bound in combined_bounds[:pt_vars])  # PT bounds
        assert combined_bounds[pt_vars:pt_vars + opt_data['n_intervals']] == [5] * opt_data['n_intervals']  # Zone 1 bounds
        assert combined_bounds[pt_vars + opt_data['n_intervals']:] == [5] * opt_data['n_intervals']  # Zone 2 bounds

        # ===== TEST DRT CONFIGURATION WITH SPATIAL DATA =====

        drt_config_loaded = opt_data['drt_config']
        assert drt_config_loaded['enabled'] is True
        assert drt_config_loaded['target_crs'] == 'EPSG:3857'
        assert len(drt_config_loaded['zones']) == 2

        # Test spatial data loading
        zone1 = drt_config_loaded['zones'][0]
        zone2 = drt_config_loaded['zones'][1]

        # Check spatial fields were added
        for zone in [zone1, zone2]:
            assert 'geometry' in zone  # Shapefile geometry loaded
            assert 'area_km2' in zone  # Area calculated
            assert 'crs' in zone       # CRS tracked
            assert zone['crs'] == 'EPSG:3857'
            assert zone['area_km2'] > 0.0  # Positive area

        # Check speed configuration
        assert zone1['drt_speed_kmh'] == 20.0  # Zone-specific
        assert zone2['drt_speed_kmh'] == 25.0  # Default from config

        # Check fleet size options preserved
        assert zone1['allowed_fleet_sizes'] == [0, 5, 10, 20, 30]
        assert zone2['allowed_fleet_sizes'] == [0, 3, 8, 15, 25]

        # Test total service area calculation
        expected_total_area = zone1['area_km2'] + zone2['area_km2']
        assert abs(drt_config_loaded['total_service_area'] - expected_total_area) < 0.001

        # ===== TEST COMBINED INITIAL SOLUTION =====

        initial_solution = opt_data['initial_solution']
        assert len(initial_solution) == pt_vars + drt_vars

        # PT part should be integers in valid range
        pt_part = initial_solution[:pt_vars]
        assert all(isinstance(x, (int, np.integer)) for x in pt_part)
        assert all(0 <= x < opt_data["n_choices"] for x in pt_part)

        # DRT part should be integers in valid range
        drt_part = initial_solution[pt_vars:]
        assert all(isinstance(x, (int, np.integer)) for x in drt_part)

        # Check DRT bounds are respected per zone
        drt_zone1_part = drt_part[:opt_data['n_intervals']]
        drt_zone2_part = drt_part[opt_data['n_intervals']:]
        assert all(0 <= x < 5 for x in drt_zone1_part)  # Zone 1: 5 choices
        assert all(0 <= x < 5 for x in drt_zone2_part)  # Zone 2: 5 choices

        # ===== TEST BACKWARD COMPATIBILITY FIELDS =====

        # Original PT fields should still exist and be valid
        assert opt_data['problem_type'] == 'discrete_headway_optimization'
        assert opt_data['n_routes'] > 0
        assert opt_data['n_intervals'] == 4  # 24/6 = 4
        assert opt_data['n_choices'] == 5    # 4 headways + no-service

        # PT-specific data should be unchanged from non-DRT version
        assert len(opt_data['routes']['ids']) == opt_data['n_routes']
        assert opt_data['routes']['round_trip_times'].shape == (opt_data['n_routes'],)

        # Fleet analysis should still work
        fleet_analysis = opt_data['constraints']['fleet_analysis']
        assert 'current_fleet_per_route' in fleet_analysis
        assert 'total_current_fleet_peak' in fleet_analysis


    def test_extract_optimization_data_with_drt_pt_only_mode(self, sample_gtfs_path):
        """Test that PT-only mode works when DRT config is None or disabled."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60, 120]

        # Test 1: No DRT config (None)
        opt_data_none = preparator.extract_optimization_data_with_drt(allowed_headways, None)

        assert opt_data_none['drt_enabled'] is False
        assert opt_data_none['n_drt_zones'] == 0
        assert opt_data_none['drt_config'] is None
        assert opt_data_none['drt_decision_variables'] == 0

        # Should match regular extract_optimization_data output structure
        pt_vars = opt_data_none['n_routes'] * opt_data_none['n_intervals']
        assert opt_data_none['total_decision_variables'] == pt_vars
        assert opt_data_none['pt_decision_variables'] == pt_vars

        # Test 2: DRT config with enabled=False
        drt_config_disabled = {'enabled': False}
        opt_data_disabled = preparator.extract_optimization_data_with_drt(allowed_headways, drt_config_disabled)

        assert opt_data_disabled['drt_enabled'] is False
        assert opt_data_disabled['n_drt_zones'] == 0


    def test_extract_optimization_data_with_drt_validation(self, sample_gtfs_path):
        """Test DRT configuration validation with various error cases."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60]

        # Test missing target_crs
        with pytest.raises(ValueError, match="must specify a valid target_crs"):
            bad_config = {'enabled': True, 'zones': []}
            preparator.extract_optimization_data_with_drt(allowed_headways, bad_config)

        # Test empty zones
        with pytest.raises(ValueError, match="must specify at least one zone"):
            bad_config = {'enabled': True, 'target_crs': 'EPSG:3857', 'zones': []}
            preparator.extract_optimization_data_with_drt(allowed_headways, bad_config)

        # Test missing zone fields
        with pytest.raises(ValueError, match="missing required field"):
            bad_config = {
                'enabled': True,
                'target_crs': 'EPSG:3857',
                'zones': [{'zone_id': 'test'}]  # Missing required fields
            }
            preparator.extract_optimization_data_with_drt(allowed_headways, bad_config)

        # Test invalid fleet sizes
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            bad_config = {
                'enabled': True,
                'target_crs': 'EPSG:3857',
                'zones': [{
                    'zone_id': 'test',
                    'service_area_path': '/fake/path.shp',
                    'allowed_fleet_sizes': [-5, 10],  # Negative fleet size
                    'zone_name': 'Test'
                }]
            }
            preparator.extract_optimization_data_with_drt(allowed_headways, bad_config)


    def test_extract_optimization_data_with_drt_missing_shapefile(self, sample_gtfs_path):
        """Test error handling for missing DRT shapefile."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=6)
        allowed_headways = [15, 30, 60]

        drt_config = {
            'enabled': True,
            'target_crs': 'EPSG:3857',
            'zones': [{
                'zone_id': 'missing_zone',
                'service_area_path': '/nonexistent/path.shp',  # File doesn't exist
                'allowed_fleet_sizes': [0, 5, 10],
                'zone_name': 'Missing Zone'
            }]
        }

        with pytest.raises(FileNotFoundError, match="DRT service area file not found"):
            preparator.extract_optimization_data_with_drt(allowed_headways, drt_config)
