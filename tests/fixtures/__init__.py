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

    allowed_headways = [10, 15, 30, 60, 120]
    return preparator.extract_optimization_data(allowed_headways)



