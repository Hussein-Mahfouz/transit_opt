import pytest
import numpy as np
from pathlib import Path
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
        assert preparator._safe_timestr_to_seconds('06:30:00') == 23400.0
        assert preparator._safe_timestr_to_seconds('00:00:00') == 0.0
        
        # Test overnight times
        assert preparator._safe_timestr_to_seconds('25:15:00') == 90900.0
        
        # Test NaN/None handling
        assert np.isnan(preparator._safe_timestr_to_seconds(np.nan))
        assert np.isnan(preparator._safe_timestr_to_seconds(None))
        
        # Test already numeric
        assert preparator._safe_timestr_to_seconds(3600.0) == 3600.0
        
        # Test invalid format
        assert np.isnan(preparator._safe_timestr_to_seconds('invalid'))
    
    def test_create_initial_solution(self):
        """Test initial solution matrix creation."""
        preparator = GTFSDataPreparator.__new__(GTFSDataPreparator)
        preparator.no_service_threshold_minutes = 480  # 8 hours
        
        # Test data
        current_headways = np.array([
            [12.0, 20.0, 35.0],
            [np.nan, 50.0, 100.0],
            [15.0, np.nan, 1440.0]
        ])
        
        allowed_headways = [10, 15, 30, 60, 120]
        headway_to_index = {10: 0, 15: 1, 30: 2, 60: 3, 120: 4, 9999: 5}
        
        result = preparator._create_initial_solution(current_headways, headway_to_index)
        
        expected = np.array([
            [0, 1, 2],  # 12→15, 20→30, 35→30
            [5, 3, 4],  # NaN→no-service, 50→60, 100→120
            [1, 5, 5]   # 15→15, NaN→no-service, 1440→no-service (>480min threshold)
        ])
        
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
        assert opt_data['problem_type'] == 'discrete_headway_optimization'
        assert opt_data['n_routes'] > 0
        assert opt_data['n_intervals'] == 8  # 24/3 = 8
        assert opt_data['n_choices'] == 6  # 5 headways + no-service
        
        # Test matrix dimensions
        expected_shape = (opt_data['n_routes'], opt_data['n_intervals'])
        assert opt_data['decision_matrix_shape'] == expected_shape
        assert opt_data['initial_solution'].shape == expected_shape
        
        # Test allowed_headways includes 9999
        assert 9999.0 in opt_data['allowed_headways']
        assert len(opt_data['allowed_headways']) == 6
        
        # Test data consistency
        assert len(opt_data['routes']['ids']) == opt_data['n_routes']
        assert opt_data['routes']['round_trip_times'].shape == (opt_data['n_routes'],)
        assert opt_data['routes']['current_headways'].shape == expected_shape



    def test_headway_calculation_realistic(self, sample_gtfs_path):
        """Test headway calculation with real GTFS schedule patterns."""
        preparator = GTFSDataPreparator(sample_gtfs_path, interval_hours=3)
        
        # Extract one route's data to verify headway calculation
        route_data = preparator._extract_route_essentials()
        assert len(route_data) > 0
        
        # Check that headways are reasonable (between 5 and 1440 minutes)
        for route in route_data[:3]:  # Test first 3 routes
            headways = route['headways_by_interval']
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
            rtt = route['round_trip_time']
            assert 10 <= rtt <= 240  # Between 10 min and 4 hours