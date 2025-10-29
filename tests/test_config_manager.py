"""
Basic tests for optimization configuration management.

These tests validate that the configuration system works correctly with
simple, realistic configurations. Focus on core functionality rather than
edge cases.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from transit_opt.optimisation.config import OptimizationConfigManager, PSOConfig, TerminationConfig


class TestConfigDataClasses:
    """Test basic configuration data class creation and validation."""

    def test_pso_config_defaults(self):
        """Test PSOConfig creates with reasonable defaults."""
        config = PSOConfig(pop_size=50)

        assert config.pop_size == 50
        assert config.inertia_weight == 0.9
        assert config.cognitive_coeff == 2.0
        assert config.social_coeff == 2.0
        assert config.adaptive

        print(
            f"✅ PSOConfig defaults: pop_size={config.pop_size}, w={config.inertia_weight}"
        )

    def test_pso_config_validation(self):
        """Test PSOConfig validates parameters reasonably."""
        # This should work
        config = PSOConfig(pop_size=30, inertia_weight=0.7)
        assert config.pop_size == 30
        assert config.inertia_weight == 0.7

        # This should fail
        with pytest.raises(ValueError):
            PSOConfig(pop_size=2)  # Too small

        print("✅ PSOConfig validation works")

    def test_termination_config_defaults(self):
        """Test TerminationConfig creates with reasonable defaults."""
        config = TerminationConfig(max_generations=100)

        assert config.max_generations == 100
        assert config.max_time_minutes is None
        assert config.convergence_tolerance == 1e-6
        assert config.convergence_patience == 50

        print(f"✅ TerminationConfig defaults: max_gen={config.max_generations}")


class TestConfigManager:
    """Test OptimizationConfigManager core functionality."""

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""

        # Create a simple test YAML config
        test_config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.5,
                },
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 75, "inertia_weight": 0.8},
                "termination": {"max_generations": 150},
                "monitoring": {"progress_frequency": 20},
            },
        }

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            # Load config from file
            manager = OptimizationConfigManager(temp_path)

            # Verify values were loaded correctly
            pso_config = manager.get_pso_config()
            assert pso_config.pop_size == 75
            assert pso_config.inertia_weight == 0.8

            term_config = manager.get_termination_config()
            assert term_config.max_generations == 150

            mon_config = manager.get_monitoring_config()
            assert mon_config.progress_frequency == 20

            print("✅ YAML config loading works")

        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_dict_config_loading(self):
        """Test loading configuration from dictionary."""
        test_config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 40},
                "termination": {"max_generations": 80},
            },
        }

        manager = OptimizationConfigManager(config_dict=test_config)

        pso_config = manager.get_pso_config()
        assert pso_config.pop_size == 40

        term_config = manager.get_termination_config()
        assert term_config.max_generations == 80

        print("✅ Dictionary config loading works")

    def test_config_validation(self):
        """Test configuration validation catches basic errors."""

        # Missing required section should fail
        bad_config = {
            "problem": {"objective": {"type": "StopCoverageObjective"}}
            # Missing 'optimization' section
        }

        with pytest.raises(ValueError, match="Missing required configuration section"):
            OptimizationConfigManager(config_dict=bad_config)

        print("✅ Config validation works")

    def test_config_summary_printing(self):
        """Test config summary prints without errors."""
        # Create minimal config instead of relying on defaults
        minimal_config = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 50},
                "termination": {"max_generations": 100},
            },
        }

        manager = OptimizationConfigManager(config_dict=minimal_config)

        # This should not raise any exceptions
        try:
            manager.print_summary()
            print("✅ Config summary printing works")
        except Exception as e:
            pytest.fail(f"Config summary printing failed: {e}")

    def test_required_parameters(self):
        """Test that required parameters are enforced."""

        # Missing pop_size should fail
        config_no_pop = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO"
                    # Missing pop_size
                },
                "termination": {"max_generations": 100},
            },
        }

        with pytest.raises(ValueError, match="Missing required parameter 'pop_size'"):
            OptimizationConfigManager(config_dict=config_no_pop)

        # Missing max_generations should fail
        config_no_gen = {
            "problem": {
                "objective": {"type": "StopCoverageObjective"},
                "constraints": [],
            },
            "optimization": {
                "algorithm": {"type": "PSO", "pop_size": 50},
                "termination": {
                    # Missing max_generations
                },
            },
        }

        with pytest.raises(
            ValueError, match="Missing required parameter 'max_generations'"
        ):
            OptimizationConfigManager(config_dict=config_no_gen)

        # No configuration at all should fail
        with pytest.raises(ValueError, match="Configuration is required"):
            OptimizationConfigManager()

        print("✅ Required parameters properly enforced")


class TestRealisticConfigExample:
    """Test with realistic configuration that matches your constraint tests."""

    def test_transit_optimization_config(self):
        """Test configuration that matches your existing constraint setup."""

        # Configuration similar to what you use in constraint tests
        config = {
            "problem": {
                "objective": {
                    "type": "StopCoverageObjective",
                    "spatial_resolution_km": 2.0,
                    "crs": "EPSG:3857",
                    "time_aggregation": "average",
                },
                "constraints": [
                    {
                        "type": "FleetTotalConstraintHandler",
                        "baseline": "current_peak",
                        "tolerance": 0.15,
                        "measure": "peak",
                    },
                    {
                        "type": "FleetPerIntervalConstraintHandler",
                        "baseline": "current_by_interval",
                        "tolerance": 0.20,
                    },
                ],
            },
            "optimization": {
                "algorithm": {
                    "type": "PSO",
                    "pop_size": 50,
                    "inertia_weight": 0.9,
                    "cognitive_coeff": 2.0,
                    "social_coeff": 2.0,
                },
                "termination": {
                    "max_generations": 100,
                    "convergence_tolerance": 1e-6,
                    "convergence_patience": 30,
                },
                "monitoring": {
                    "progress_frequency": 10,
                    "save_history": True,
                    "detailed_logging": False,
                },
                "multi_run": {"enabled": False},
            },
        }

        manager = OptimizationConfigManager(config_dict=config)

        # Verify constraint configuration
        problem_config = manager.get_problem_config()
        constraints = problem_config["constraints"]
        assert len(constraints) == 2
        assert constraints[0]["type"] == "FleetTotalConstraintHandler"
        assert constraints[1]["type"] == "FleetPerIntervalConstraintHandler"

        # Verify PSO configuration
        pso_config = manager.get_pso_config()
        assert pso_config.pop_size == 50
        assert pso_config.inertia_weight == 0.9

        # Verify termination configuration
        term_config = manager.get_termination_config()
        assert term_config.max_generations == 100
        assert term_config.convergence_patience == 30

        print("✅ Realistic transit optimization config works")
        print("✅ All configuration components validate correctly")


if __name__ == "__main__":
    """Run tests directly for quick validation."""
    print("🧪 Running configuration manager tests...")
    pytest.main([__file__, "-v", "-s"])
