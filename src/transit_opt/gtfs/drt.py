"""
DRT solution export utilities.

This module provides functionality to export DRT solutions to JSON format
for reuse in future optimization runs or integration with external simulation frameworks
(i.e. MATSim).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class DRTSolutionExporter:
    """
    Export DRT optimization solutions to human-readable JSON format.
    
    This class handles the conversion of DRT solution matrices (containing fleet size indices)
    back to actual fleet deployment schedules with metadata for reuse and external integration.
    
    Key Features:
    - Converts indices back to actual fleet sizes
    - Preserves some zone metadata and operational parameters
    - Validation of solution data
    - Support for metadata attachment
    
    Usage:
        >>> exporter = DRTSolutionExporter(optimization_data)
        >>> path = exporter.export_solution(
        ...     solution={'pt': pt_matrix, 'drt': drt_matrix},
        ...     output_path="results/drt_solution.json",
        ...     metadata={'objective_value': 0.0245, 'run_id': 'pso_001'}
        ... )
    """

    def __init__(self, optimization_data: dict[str, Any]):
        """
        Initialize DRT solution exporter.
        
        Args:
            optimization_data: Complete optimization data structure from GTFSDataPreparator
                              Must contain DRT configuration and interval information
        """
        self.opt_data = optimization_data

        # Validate that DRT is enabled
        if not optimization_data.get('drt_enabled', False):
            raise ValueError("Cannot export DRT solutions: DRT not enabled in optimization data")

        self.drt_config = optimization_data['drt_config']
        self.zones = self.drt_config['zones']
        self.interval_labels = optimization_data['intervals']['labels']
        self.n_intervals = optimization_data['n_intervals']

        if not self.zones:
            raise ValueError("No DRT zones found in optimization data")

    def export_solution(
        self,
        solution: dict[str, np.ndarray],
        output_path: str,
        metadata: dict[str, Any] = None
    ) -> str:
        """
        Export DRT solution to JSON file for future reuse.
        
        Args:
            solution: Combined PT+DRT solution dict with 'drt' key containing the DRT matrix
            output_path: Path for JSON output file (can be relative or absolute)
            metadata: Optional metadata to include in the output file
            
        Returns:
            Absolute path to created JSON file
            
        Raises:
            ValueError: If solution format is invalid or DRT data is missing
            IOError: If file cannot be written
        """
        # Validate solution format
        if not isinstance(solution, dict) or 'drt' not in solution:
            raise ValueError("Solution must be a dictionary containing 'drt' key for DRT export")

        drt_matrix = solution['drt']

        # Validate DRT matrix dimensions
        expected_shape = (len(self.zones), self.n_intervals)
        if drt_matrix.shape != expected_shape:
            raise ValueError(f"DRT matrix shape {drt_matrix.shape} != expected {expected_shape}")

        # Build DRT solution structure
        drt_solutions = {}

        for zone_idx, zone in enumerate(self.zones):
            zone_id = zone['zone_id']
            allowed_fleet_sizes = zone.get('allowed_fleet_sizes', [])

            # Extract fleet deployment for this zone
            fleet_deployment = {}
            for interval_idx, interval_label in enumerate(self.interval_labels):
                choice_idx = int(drt_matrix[zone_idx, interval_idx])

                # Validate choice index and convert to fleet size
                if 0 <= choice_idx < len(allowed_fleet_sizes):
                    fleet_size = allowed_fleet_sizes[choice_idx]
                else:
                    raise ValueError(
                        f"Invalid fleet size index {choice_idx} for zone {zone_id} "
                        f"at interval {interval_label}."
                        f"Index must be in [0, {len(allowed_fleet_sizes)-1}]"
                    )

                fleet_deployment[interval_label] = {
                    "fleet_choice_idx": choice_idx,
                    "fleet_size": fleet_size
                }

            # Store zone solution with metadata
            drt_solutions[zone_id] = {
                "zone_info": {
                    "zone_id": zone_id,
                    "zone_name": zone.get('zone_name', zone_id),
                    "service_area_km2": zone.get('area_km2', 0.0),
                    "drt_speed_kmh": zone.get('drt_speed_kmh', 25.0)
                },
                "fleet_deployment": fleet_deployment
            }

        # Build complete export structure
        export_data = {
            "solution_metadata": {
                "created_at": datetime.now().isoformat(),
                "optimization_type": "PT+DRT",
                "total_zones": len(self.zones),
                "total_intervals": self.n_intervals,
                "interval_labels": self.interval_labels,
                **(metadata or {})
            },
            "drt_solutions": drt_solutions
        }

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        except OSError as e:
            raise OSError(f"Failed to write DRT solution file {output_file}: {e}") from e

        # Summary logging
        total_deployments = sum(len(z['fleet_deployment']) for z in drt_solutions.values())
        total_vehicles = sum(
            deployment['fleet_size']
            for zone_data in drt_solutions.values()
            for deployment in zone_data['fleet_deployment'].values()
        )

        print(f"✅ Exported DRT solution to: {output_file.absolute()}")
        print(f"   Zones: {len(drt_solutions)}")
        print(f"   Total deployments: {total_deployments}")
        print(f"   Total vehicle-intervals: {total_vehicles}")

        return str(output_file.absolute())

    def validate_solution_matrix(self, drt_matrix: np.ndarray) -> dict[str, Any]:
        """
        Validate DRT solution matrix format and values.
        
        Args:
            drt_matrix: DRT solution matrix to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check shape
        expected_shape = (len(self.zones), self.n_intervals)
        if drt_matrix.shape != expected_shape:
            validation['valid'] = False
            validation['errors'].append(f"Shape {drt_matrix.shape} != expected {expected_shape}")
            return validation

        # Check value ranges for each zone
        for zone_idx, zone in enumerate(self.zones):
            zone_id = zone['zone_id']
            allowed_fleet_sizes = zone.get('allowed_fleet_sizes', [])
            max_choice_idx = len(allowed_fleet_sizes) - 1

            zone_values = drt_matrix[zone_idx, :]
            min_val = np.min(zone_values)
            max_val = np.max(zone_values)

            if min_val < 0:
                validation['valid'] = False
                validation['errors'].append(f"Zone {zone_id} has negative indices: min={min_val}")

            if max_val > max_choice_idx:
                validation['valid'] = False
                validation['errors'].append(
                    f"Zone {zone_id} has invalid indices: max={max_val}, max_allowed={max_choice_idx}"
                )

        # Calculate statistics
        total_deployments = np.sum(drt_matrix > 0)  # Non-zero fleet deployments
        zero_deployments = np.sum(drt_matrix == 0)  # Zero fleet deployments

        validation['statistics'] = {
            'total_cells': drt_matrix.size,
            'active_deployments': int(total_deployments),
            'zero_deployments': int(zero_deployments),
            'utilization_rate': float(total_deployments / drt_matrix.size),
            'zones': len(self.zones),
            'intervals': self.n_intervals
        }

        # Warnings for unusual patterns
        utilization_rate = validation['statistics']['utilization_rate']
        if utilization_rate < 0.1:
            validation['warnings'].append(f"Very low DRT utilization: {utilization_rate:.1%}")
        elif utilization_rate > 0.9:
            validation['warnings'].append(f"Very high DRT utilization: {utilization_rate:.1%}")

        return validation

