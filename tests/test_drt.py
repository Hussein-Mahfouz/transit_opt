import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from transit_opt.gtfs.drt import DRTSolutionExporter


class TestDRTSolutionExporter:
    """Test DRT solution export functionality - focused on export logic only."""

    @pytest.fixture
    def mock_drt_opt_data(self):
        """Create minimal DRT optimization data for testing."""
        return {
            'drt_enabled': True,
            'n_intervals': 4,
            'intervals': {'labels': ['00-06h', '06-12h', '12-18h', '18-24h']},
            'drt_config': {
                'zones': [
                    {
                        'zone_id': 'zone_A',
                        'zone_name': 'Test Zone A',
                        'allowed_fleet_sizes': [0, 5, 10, 15],
                        'area_km2': 2.5,
                        'drt_speed_kmh': 25.0
                    },
                    {
                        'zone_id': 'zone_B',
                        'zone_name': 'Test Zone B',
                        'allowed_fleet_sizes': [0, 8, 16],
                        'area_km2': 3.8,
                        'drt_speed_kmh': 30.0
                    }
                ]
            }
        }

    def test_initialization_success(self, mock_drt_opt_data):
        """Test successful initialization with valid DRT data."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        assert exporter.opt_data == mock_drt_opt_data
        assert len(exporter.zones) == 2
        assert exporter.n_intervals == 4

    def test_initialization_fails_without_drt(self):
        """Test initialization fails when DRT not enabled."""
        invalid_data = {'drt_enabled': False}

        with pytest.raises(ValueError, match="Cannot export DRT solutions: DRT not enabled"):
            DRTSolutionExporter(invalid_data)

    def test_export_solution_basic(self, mock_drt_opt_data):
        """Test basic DRT solution export to JSON."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        # Create test solution (2 zones × 4 intervals)
        drt_matrix = np.array([
            [1, 3, 2, 0],  # Zone A: 5, 15, 10, 0 vehicles
            [0, 2, 1, 2]   # Zone B: 0, 16, 8, 16 vehicles
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_drt.json"

            result_path = exporter.export_solution(
                solution={'drt': drt_matrix},
                output_path=str(output_path)
            )

            # Verify file created
            assert Path(result_path).exists()

            # Load and verify content
            with open(result_path) as f:
                data = json.load(f)

            assert 'solution_metadata' in data
            assert 'drt_solutions' in data
            assert len(data['drt_solutions']) == 2

    def test_choice_index_to_fleet_size_conversion(self, mock_drt_opt_data):
        """Test that choice indices are correctly converted to fleet sizes."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        drt_matrix = np.array([
            [1, 3, 0, 2],  # Zone A: indices 1,3,0,2 → fleet sizes 5,15,0,10
            [2, 0, 1, 2]   # Zone B: indices 2,0,1,2 → fleet sizes 16,0,8,16
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_conversion.json"

            result_path = exporter.export_solution(
                solution={'drt': drt_matrix},
                output_path=str(output_path)
            )

            with open(result_path) as f:
                data = json.load(f)

            # Check Zone A conversions
            zone_a = data['drt_solutions']['zone_A']['fleet_deployment']
            assert zone_a['00-06h']['fleet_choice_idx'] == 1
            assert zone_a['00-06h']['fleet_size'] == 5
            assert zone_a['06-12h']['fleet_choice_idx'] == 3
            assert zone_a['06-12h']['fleet_size'] == 15

            # Check Zone B conversions
            zone_b = data['drt_solutions']['zone_B']['fleet_deployment']
            assert zone_b['00-06h']['fleet_choice_idx'] == 2
            assert zone_b['00-06h']['fleet_size'] == 16
            assert zone_b['18-24h']['fleet_choice_idx'] == 2
            assert zone_b['18-24h']['fleet_size'] == 16

    def test_export_with_metadata(self, mock_drt_opt_data):
        """Test that metadata is properly included in export."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        drt_matrix = np.array([[1, 2, 0, 1], [0, 1, 2, 0]])
        metadata = {'run_id': 'test_123', 'objective_value': 0.456}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_metadata.json"

            result_path = exporter.export_solution(
                solution={'drt': drt_matrix},
                output_path=str(output_path),
                metadata=metadata
            )

            with open(result_path) as f:
                data = json.load(f)

            solution_meta = data['solution_metadata']
            assert solution_meta['run_id'] == 'test_123'
            assert solution_meta['objective_value'] == 0.456
            assert 'created_at' in solution_meta

    def test_validation_matrix_shape(self, mock_drt_opt_data):
        """Test matrix shape validation."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        # Wrong shape matrix (should be 2×4)
        wrong_matrix = np.array([[1, 2, 3]])  # 1×3 instead of 2×4

        validation = exporter.validate_solution_matrix(wrong_matrix)

        assert validation['valid'] is False
        assert 'Shape' in validation['errors'][0]

    def test_validation_value_ranges(self, mock_drt_opt_data):
        """Test value range validation."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        # Matrix with invalid values
        invalid_matrix = np.array([
            [-1, 3, 2, 5],  # -1 negative, 5 too large for Zone A (max 3)
            [0, 4, 1, 2]    # 4 too large for Zone B (max 2)
        ])

        validation = exporter.validate_solution_matrix(invalid_matrix)

        assert validation['valid'] is False
        assert any('negative' in error for error in validation['errors'])
        assert any('invalid indices' in error for error in validation['errors'])

    def test_invalid_solution_format(self, mock_drt_opt_data):
        """Test error handling for invalid solution format."""
        exporter = DRTSolutionExporter(mock_drt_opt_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.json"

            # Missing 'drt' key
            with pytest.raises(ValueError, match="Solution must be a dictionary containing 'drt' key"):
                exporter.export_solution(
                    solution={'pt': np.array([[1, 2]])},
                    output_path=str(output_path)
                )
