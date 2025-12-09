"""
Configuration data classes and management for optimization.

This module defines structured configuration classes for optimization
and provides validation and loading capabilities. It integrates with the
existing constraint configuration system while adding optimization-specific
parameters.

The configuration system supports:
- PSO algorithm parameters (population, coefficients, etc.)
- Termination criteria (generations, time limits, convergence)
- Progress monitoring and logging options
- Multi-run statistical analysis
- Parameter sweep capabilities
- Integration with existing constraint YAML format

Example YAML Configuration:
```yaml
problem:
  objective:
    type: "StopCoverageObjective"
    spatial_resolution_km: 2.0
    crs: "EPSG:3857"
  constraints:
    - type: "FleetTotalConstraintHandler"
      baseline: "current_peak"
      tolerance: 0.15

optimization:
  algorithm:
    type: "PSO"
    pop_size: 100
    inertia_weight: 0.9
    cognitive_coeff: 2.0
    social_coeff: 2.0
  termination:
    max_generations: 200
    convergence_tolerance: 1e-6
  monitoring:
    progress_frequency: 10
  multi_run:
    enabled: true
    num_runs: 10
```

Usage:
```python
config_manager = OptimizationConfigManager('config.yaml')
pso_config = config_manager.get_pso_config()
runner = ProductionPSORunner(config_manager)
```
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from transit_opt.optimisation.utils.solution_loader import SolutionLoader

logger = logging.getLogger(__name__)

# Replace the PSOConfig class definition around line 50

@dataclass
class PSOConfig:
    """
    Particle Swarm Optimization algorithm configuration.

    This class encapsulates all PSO-specific parameters that control the
    algorithm behavior. Uses PyMOO's native adaptive PSO implementation.

    ALGORITHM MODES:
    ===============

    **Standard PSO Mode:**
    - Fixed inertia weight, cognitive, and social coefficients
    - Traditional PSO behavior (adaptive=False)

    **Adaptive PSO Mode:**
    - Dynamically adjusts w, c1, c2 based on swarm spread
    - Uses PyMOO's built-in adaptive algorithm (adaptive=True)
    - Initial values serve as starting points

    Attributes:
    ===========

    pop_size : int (REQUIRED)
        Population size (number of particles in swarm).
        Typical range: 20-200, default 50 works well for most problems.

    inertia_weight : float, default=0.9
        Inertia weight (w) for velocity update.
        - For adaptive=False: fixed value throughout optimization
        - For adaptive=True: initial value, will be adjusted by PyMOO
        - Typical range: 0.4-0.9

    cognitive_coeff : float, default=2.0
        Cognitive coefficient (c1) - attraction to personal best.
        - For adaptive=False: fixed value throughout optimization
        - For adaptive=True: initial value, will be adjusted by PyMOO
        - Typical range: 1.5-2.5, default 2.0 is standard.

    social_coeff : float, default=2.0
        Social coefficient (c2) - attraction to global best.
        - For adaptive=False: fixed value throughout optimization
        - For adaptive=True: initial value, will be adjusted by PyMOO
        - Typical range: 1.5-2.5, default 2.0 is standard.

    adaptive : bool, default=True
        Whether to use PyMOO's adaptive PSO algorithm.
        - True: w, c1, c2 change dynamically based on swarm diversity
        - False: w, c1, c2 remain constant throughout optimization

    **Penalty Method Parameters:**

    use_penalty_method : bool, default=False
        Whether to apply penalty method for constraints:
        - True: Soft constraints with penalties added to objective
        - False: Hard constraints (infeasible solutions rejected)

    penalty_weight : float, default=1000.0
        Default penalty weight for constraint violations.
        - Higher values penalize violations more strongly
        - Typical range: 100-10000
        - Used as fallback for constraints without specific weights

    adaptive_penalty : bool, default=False
        Whether to adaptively increase penalty weight:
        - True: Increase penalty weight over generations
        - False: Fixed penalty weight throughout optimization

    penalty_increase_rate : float, default=2.0
        Multiplicative factor for adaptive penalty increases.
        - Must be > 1.0 for meaningful adaptation
        - Formula: new_weight = current_weight × (rate ^ progress)
        - Typical range: 1.1-3.0

    Usage Examples:
    ==============

    ```python
    # Adaptive PSO (recommended)
    config = PSOConfig(
        pop_size=50,
        inertia_weight=0.9,        # Initial value
        cognitive_coeff=2.0,       # Initial value
        social_coeff=2.0,          # Initial value
        adaptive=True              # PyMOO handles adaptation
    )

    # Traditional fixed PSO
    config = PSOConfig(
        pop_size=30,
        inertia_weight=0.7,        # Fixed value
        cognitive_coeff=2.0,       # Fixed value
        social_coeff=2.0,          # Fixed value
        adaptive=False             # No adaptation
    )

    # Penalty method PSO
    config = PSOConfig(
        pop_size=50,
        adaptive=True,
        use_penalty_method=True,
        penalty_weight=1500.0,
        adaptive_penalty=True
    )
    ```
    """

    pop_size: int  # REQUIRED - no default
    inertia_weight: float = 0.9  # Initial or fixed inertia weight
    cognitive_coeff: float = 2.0  # Initial or fixed cognitive coefficient
    social_coeff: float = 2.0  # Initial or fixed social coefficient
    adaptive: bool = True  # Use PyMOO's adaptive PSO

    use_penalty_method: bool = False
    penalty_weight: float = 1000.0
    adaptive_penalty: bool = False
    penalty_increase_rate: float = 2.0

    def __post_init__(self):
        """Validate PSO configuration parameters."""
        if self.pop_size < 5:
            raise ValueError("Population size must be at least 5")
        if self.pop_size > 1000:
            raise ValueError("Population size should not exceed 1000 (memory/time)")

        if not 0.0 <= self.inertia_weight <= 2.0:
            raise ValueError("Inertia weight should be in range [0.0, 2.0]")

        if not 0.0 <= self.cognitive_coeff <= 5.0:
            raise ValueError("Cognitive coefficient should be in range [0.0, 5.0]")

        if not 0.0 <= self.social_coeff <= 5.0:
            raise ValueError("Social coefficient should be in range [0.0, 5.0]")

        if self.penalty_weight <= 0:
            raise ValueError("Penalty weight must be positive")
        if self.penalty_increase_rate <= 1.0:
            raise ValueError("Penalty increase rate must be > 1.0 for adaptive penalty")

@dataclass
class TerminationConfig:
    """
    Termination criteria configuration for optimization.

    Multiple termination criteria can be active simultaneously - optimization
    stops when ANY criterion is met. This provides robust stopping conditions
    that prevent runaway optimization while ensuring adequate search time.

    Attributes:
        max_generations: Maximum number of generations to run
            - Primary stopping criterion, prevents infinite optimization
            - Should be set based on problem complexity and available time
            - Typical range: 50-500, larger for complex problems

        max_time_minutes: Maximum wall-clock time in minutes
            - Useful for time-limited scenarios (cluster jobs, real-time systems)
            - None = no time limit (only generation limit applies)
            - Useful range: 10-120 minutes depending on urgency

        convergence_tolerance: Objective improvement threshold for convergence
            - If best objective improves by less than this over patience generations,
              optimization terminates (converged)
            - Smaller values require more precise convergence
            - Typical range: 1e-4 to 1e-8 depending on objective scale

        convergence_patience: Generations to wait for improvement
            - Number of generations without significant improvement before declaring convergence
            - Prevents premature termination due to temporary stagnation
            - Typical range: 20-100, larger for noisy objective functions

        target_objective: Target objective value for early termination
            - Stop if optimization reaches this objective value
            - Useful when you know the theoretical optimum or acceptable threshold
            - None = no target-based stopping

    Termination Logic:
        The optimization terminates when the FIRST of these conditions is met:
        1. max_generations reached
        2. max_time_minutes exceeded (if specified)
        3. Convergence detected (improvement < tolerance for patience generations)
        4. target_objective achieved (if specified)

    Example Configurations:
        ```python
        # Quick exploration (testing)
        config = TerminationConfig(max_generations=50, max_time_minutes=5)

        # Production optimization
        config = TerminationConfig(
            max_generations=500,
            max_time_minutes=60,
            convergence_tolerance=1e-6,
            convergence_patience=50
        )

        # Target-driven optimization
        config = TerminationConfig(
            max_generations=1000,
            target_objective=10.0,  # Stop when objective <= 10.0
            convergence_tolerance=1e-8
        )
        ```
    """

    max_generations: int  # REQUIRED - no default
    max_time_minutes: int | None = None
    convergence_tolerance: float = 1e-6  # Sensible default
    convergence_patience: int = 50  # Sensible default
    target_objective: float | None = None  # Optional

    def __post_init__(self):
        """Validate termination configuration."""
        if self.max_generations < 1:
            raise ValueError("Max generations must be positive")

        if self.max_time_minutes is not None and self.max_time_minutes < 1:
            raise ValueError("Max time must be positive")

        if self.convergence_tolerance <= 0:
            raise ValueError("Convergence tolerance must be positive")

        if self.convergence_patience < 1:
            raise ValueError("Convergence patience must be positive")


@dataclass
class MonitoringConfig:
    """
    Progress monitoring and logging configuration.

    Controls how optimization progress is reported and logged during execution.
    Balances between informative progress reporting and performance overhead.

    Attributes:
        progress_frequency: Report progress every N generations
            - Lower values = more frequent updates, higher overhead
            - Higher values = less frequent updates, lower overhead
            - Typical range: 5-50, default 10 provides good balance

        save_history: Whether to save complete optimization history
            - True: Enables detailed post-optimization analysis but uses more memory
            - False: Saves memory but limits post-optimization analysis
            - Always True for production analysis, False only for memory-limited scenarios

        checkpoint_frequency: Save checkpoint every N generations
            - Enables recovery from crashes for long optimizations
            - 0 = no checkpoints (memory-efficient)
            - Typical range: 20-100 for long optimizations

        detailed_logging: Enable detailed per-generation logging
            - True: Log detailed statistics, population diversity, constraint violations
            - False: Log only basic progress information
            - Use True for debugging, False for production

        log_level: Logging verbosity level
            - 'ERROR': Only log errors and failures
            - 'WARNING': Log warnings and errors
            - 'INFO': Log progress and informational messages (default)
            - 'DEBUG': Log detailed debugging information

    Memory and Performance Considerations:
        - save_history=True: ~1MB per 100 generations for typical problems
        - detailed_logging=True: ~10x more log output
        - checkpoint_frequency>0: ~500KB per checkpoint

    Example Configurations:
        ```python
        # Development/debugging configuration
        config = MonitoringConfig(
            progress_frequency=5,
            save_history=True,
            detailed_logging=True,
            log_level='DEBUG'
        )

        # Production configuration
        config = MonitoringConfig(
            progress_frequency=20,
            save_history=True,
            checkpoint_frequency=50,
            detailed_logging=False,
            log_level='INFO'
        )

        # Memory-limited configuration
        config = MonitoringConfig(
            progress_frequency=50,
            save_history=False,
            checkpoint_frequency=0,
            detailed_logging=False,
            log_level='WARNING'
        )
        ```
    """

    progress_frequency: int = 10
    save_history: bool = True
    checkpoint_frequency: int = 0  # Sensible default (no checkpoints)
    detailed_logging: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.progress_frequency < 1:
            raise ValueError("Progress frequency must be positive")

        if self.checkpoint_frequency < 0:
            raise ValueError("Checkpoint frequency cannot be negative")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")


@dataclass
class MultiRunConfig:
    """
    Multi-run analysis and parameter sweep configuration.

    Supports two types of multi-run analysis:
    1. Statistical analysis: Run same configuration multiple times for reliability
    2. Parameter sweep: Test different PSO parameters to find optimal settings

    Attributes:
        enabled: Whether to enable multi-run analysis
            - False: Single optimization run (fastest, minimal analysis)
            - True: Enable statistical analysis or parameter sweep

        num_runs: Number of runs for statistical analysis
            - Only used when parameter_sweep is None
            - More runs = better statistics but longer execution time
            - Typical range: 5-30, default 5 for basic analysis

        parameter_sweep: Dictionary defining parameter sweep ranges
            - None: Perform statistical analysis with fixed parameters
            - Dict: Perform parameter sweep testing different PSO settings
            - Keys: Parameter names ('inertia_weight', 'cognitive_coeff', 'social_coeff')
            - Values: List of parameter values to test

        parallel: Whether to run multiple optimizations in parallel
            - True: Use multiprocessing for faster execution (if available)
            - False: Sequential execution (more predictable, easier debugging)
            - Note: Parallel execution may not be implemented initially

        statistical_analysis: Types of statistical analysis to perform
            - List of analysis types: ['basic', 'detailed', 'comparison']
            - 'basic': Mean, std, min, max of final objectives
            - 'detailed': Distribution analysis, confidence intervals
            - 'comparison': Statistical significance testing

    Parameter Sweep Example:
        ```yaml
        multi_run:
          enabled: true
          parameter_sweep:
            inertia_weight: [0.5, 0.7, 0.9]
            cognitive_coeff: [1.5, 2.0, 2.5]
            social_coeff: [1.5, 2.0, 2.5]
        ```
        This creates 3×3×3 = 27 different PSO configurations to test.

    Statistical Analysis Example:
        ```yaml
        multi_run:
          enabled: true
          num_runs: 10
          statistical_analysis: ['basic', 'detailed']
        ```
        This runs the same configuration 10 times for statistical reliability.

    Example Configurations:
        ```python
        # No multi-run (single optimization)
        config = MultiRunConfig(enabled=False)

        # Basic statistical analysis
        config = MultiRunConfig(
            enabled=True,
            num_runs=10,
            statistical_analysis=['basic']
        )

        # Parameter sweep for PSO tuning
        config = MultiRunConfig(
            enabled=True,
            parameter_sweep={
                'inertia_weight': [0.5, 0.7, 0.9],
                'pop_size': [30, 50, 100]
            },
            parallel=False
        )
        ```
    """

    enabled: bool = False
    num_runs: int = 5
    parameter_sweep: dict[str, list[int | float]] | None = None
    parallel: bool = False
    statistical_analysis: list[str] = field(default_factory=lambda: ["basic"])

    def __post_init__(self):
        """Validate multi-run configuration."""
        if self.enabled:
            if self.num_runs < 2:
                raise ValueError(
                    "Number of runs must be at least 2 for multi-run analysis"
                )

            if self.num_runs > 100:
                raise ValueError(
                    "Number of runs should not exceed 100 (time/resource limits)"
                )

            valid_analyses = ["basic", "detailed", "comparison"]
            for analysis in self.statistical_analysis:
                if analysis not in valid_analyses:
                    raise ValueError(
                        f"Statistical analysis '{analysis}' not in {valid_analyses}"
                    )

            if self.parameter_sweep:
                valid_params = [
                    "pop_size",
                    "inertia_weight",
                    "cognitive_coeff",
                    "social_coeff",
                ]
                for param in self.parameter_sweep:
                    if param not in valid_params:
                        raise ValueError(f"Parameter '{param}' not in {valid_params}")

                    if not isinstance(self.parameter_sweep[param], list):
                        raise ValueError(
                            f"Parameter sweep values for '{param}' must be a list"
                        )

                    if len(self.parameter_sweep[param]) < 2:
                        raise ValueError(
                            f"Parameter sweep for '{param}' must have at least 2 values"
                        )

@dataclass
class SamplingConfig:
    """
    Configuration for custom sampling in PSO optimization.
    Attributes:
        enabled: Whether custom sampling is enabled
        base_solutions: List of base solutions or 'from_data' to sample from
        frac_gaussian_pert: Fraction of new solutions from Gaussian perturbation around base solutions
        gaussian_sigma: Standard deviation fraction for Gaussian perturbation
        random_seed: Random seed for reproducibility

    Note: frac_lhs is calculated automatically as (1.0 - frac_gaussian_pert)
"""
    enabled: bool = False
    base_solutions: str | list = "from_data"  # "from_data" or list of solutions
    frac_gaussian_pert: float = 0.7
    gaussian_sigma: float = 1.0
    random_seed: int | None = None

    @property
    def frac_lhs(self) -> float:
        """Get fraction of Latin Hypercube Sampling."""
        return 1.0 - self.frac_gaussian_pert

    def __post_init__(self):
        """Validate sampling configuration after initialization."""
        if self.enabled:
            # Validate fraction is between 0 and 1
            if not 0.0 <= self.frac_gaussian_pert <= 1.0:
                raise ValueError(f"frac_gaussian_pert must be between 0.0 and 1.0, got {self.frac_gaussian_pert}")

            # Check sigma is positive
            if self.gaussian_sigma <= 0:
                raise ValueError("gaussian_sigma must be positive")

@dataclass
class SolutionSamplingStrategyConfig:
    """
    Configuration for solution sampling strategy.

    Determines which solutions from tracked pool get exported for downstream analysis.
    """
    type: str = "uniform"  # "uniform", "power", "geometric", "fibonacci", "manual"
    max_to_save: int = 10
    max_rank: int | None = None  # Defaults to track_best_n if not specified

    # Type-specific parameters
    power_exponent: float = 2.0
    geometric_base: float = 2.0
    manual_ranks: list[int] = field(default_factory=list)

    def __post_init__(self):
        # Validate type
        valid_types = ["uniform", "power", "geometric", "fibonacci", "manual"]
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")

        # Validate max_to_save
        if self.max_to_save < 1:
            raise ValueError(f"max_to_save must be >= 1, got {self.max_to_save}")

        # Validate max_rank when specified
        if self.max_rank is not None and self.max_rank < 1:
            raise ValueError(
                f"max_rank must be positive when specified, got {self.max_rank}"
            )

        # Validate type-specific parameters
        if self.type == "power" and self.power_exponent < 1.0:
            raise ValueError(f"power_exponent must be >= 1.0, got {self.power_exponent}")

        if self.type == "geometric" and self.geometric_base <= 1.0:
            raise ValueError(f"geometric_base must be > 1.0, got {self.geometric_base}")

        if self.type == "manual":
            if not self.manual_ranks:
                raise ValueError("manual_ranks must be provided when type='manual'")
            # Ensure manual ranks are integers
            self.manual_ranks = [int(r) for r in self.manual_ranks]
            # Validate no duplicates
            if len(self.manual_ranks) != len(set(self.manual_ranks)):
                raise ValueError("manual_ranks must contain unique values")


class OptimizationConfigManager:
    """
    Comprehensive configuration manager for transit optimization.

    This class handles loading, validation, and management of optimization
    configurations from YAML files. It integrates with the existing constraint
    configuration system while adding PSO-specific parameters and capabilities.

    Key Features:
        - YAML configuration loading with validation
        - Integration with existing constraint configurations
        - Default configuration management
        - Configuration merging and overrides
        - Structured access to all configuration sections

    Configuration Structure:
        ```yaml
        problem:
          objective: {...}      # Objective function configuration
          constraints: [...]    # List of constraint configurations

        optimization:
          algorithm: {...}      # PSO algorithm parameters
          termination: {...}    # Stopping criteria
          monitoring: {...}     # Progress reporting
          multi_run: {...}      # Statistical/parameter sweep analysis
        ```

    Usage Pattern:
        ```python
        # Load configuration from file
        config_manager = OptimizationConfigManager('optimization_config.yaml')

        # Access structured configurations
        pso_config = config_manager.get_pso_config()
        termination_config = config_manager.get_termination_config()

        # Create optimization components
        runner = ProductionPSORunner(config_manager)
        problem = config_manager.create_problem(optimization_data)
        ```
    """

    def __init__(self, config_path: str | None = None, config_dict: dict | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)

        Raises:
            FileNotFoundError: If config_path doesn't exist
            ValueError: If both, neither, or invalid config sources provided
            yaml.YAMLError: If YAML file is malformed
            ValueError: If configuration validation fails
        """
        if config_path and config_dict:
            raise ValueError("Provide either config_path or config_dict, not both")

        if not config_path and not config_dict:
            raise ValueError(
                "Configuration is required. Provide either:\n"
                "  - config_path: Path to YAML configuration file\n"
                "  - config_dict: Configuration dictionary\n"
                "Example: OptimizationConfigManager('my_config.yaml')"
            )

        if config_path:
            self.config = self._load_yaml_config(config_path)
            print("📋 Using loaded configuration file")
        else:
            self.config = config_dict
            print("📋 Using provided configuration dictionary")

        # Validate and setup structured configs
        self._validate_config()
        self._setup_structured_configs()

    def _load_yaml_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file with error handling."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")

        print(f"📂 Loaded configuration from {config_path}")
        return config

    def _validate_config(self):
        """Validate configuration structure and required fields."""
        # Check required top-level sections
        required_sections = ["problem", "optimization"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: '{section}'")

        # Validate problem section
        problem_config = self.config["problem"]
        if "objective" not in problem_config:
            raise ValueError("Missing 'objective' in problem configuration")

        # Validate optimization section
        opt_config = self.config["optimization"]
        required_opt_sections = ["algorithm"]
        for section in required_opt_sections:
            if section not in opt_config:
                raise ValueError(f"Missing required optimization section: '{section}'")

        # Validate algorithm type
        alg_config = opt_config["algorithm"]
        if alg_config.get("type", "PSO") != "PSO":
            raise ValueError("Only PSO algorithm supported currently")

    def _setup_structured_configs(self):
        """Setup structured configuration objects with validation."""
        opt_config = self.config["optimization"]

        # Setup PSO configuration - check required parameters
        alg_config = opt_config["algorithm"]

        if "pop_size" not in alg_config:
            raise ValueError(
                "Missing required parameter 'pop_size' in algorithm configuration.\n"
                "Example:\n"
                "optimization:\n"
                "  algorithm:\n"
                "    pop_size: 50"
            )

        self.pso_config = PSOConfig(
            pop_size=alg_config["pop_size"],  # REQUIRED
            inertia_weight=alg_config.get("inertia_weight", 0.9),  # Optional with default
            cognitive_coeff=alg_config.get("cognitive_coeff", 2.0),
            social_coeff=alg_config.get("social_coeff", 2.0),
            adaptive=alg_config.get("adaptive", True),  # Optional with default
            # Read penalty method parameters from configuration
            use_penalty_method=alg_config.get("use_penalty_method", False),
            penalty_weight=alg_config.get("penalty_weight", 1000.0),
            adaptive_penalty=alg_config.get("adaptive_penalty", False),
            penalty_increase_rate=alg_config.get("penalty_increase_rate", 2.0),
        )

        # Setup sampling configuration - all optional with defaults
        sampling_config = self.config.get("optimization", {}).get("sampling", {})
        self.sampling_config = SamplingConfig(
            enabled=sampling_config.get("enabled", False),
            base_solutions=sampling_config.get("base_solutions", []),
            frac_gaussian_pert=sampling_config.get("frac_gaussian_pert", 0.3),
            gaussian_sigma=sampling_config.get("gaussian_sigma", 1.0),
            random_seed=sampling_config.get("random_seed", None)
        )

        # Setup termination configuration - check required parameters
        term_config = opt_config.get("termination", {})

        if "max_generations" not in term_config:
            raise ValueError(
                "Missing required parameter 'max_generations' in termination configuration.\n"
                "Example:\n"
                "optimization:\n"
                "  termination:\n"
                "    max_generations: 100"
            )

        self.termination_config = TerminationConfig(
            max_generations=term_config["max_generations"],  # REQUIRED
            max_time_minutes=term_config.get("max_time_minutes"),  # Optional
            convergence_tolerance=term_config.get(
                "convergence_tolerance", 1e-6
            ),  # Optional with default
            convergence_patience=term_config.get(
                "convergence_patience", 50
            ),  # Optional with default
            target_objective=term_config.get("target_objective"),  # Optional
        )

        # Setup monitoring configuration - all optional with defaults
        mon_config = opt_config.get("monitoring", {})
        self.monitoring_config = MonitoringConfig(
            progress_frequency=mon_config.get("progress_frequency", 10),
            save_history=mon_config.get("save_history", True),
            checkpoint_frequency=mon_config.get("checkpoint_frequency", 0),
            detailed_logging=mon_config.get("detailed_logging", False),
            log_level=mon_config.get("log_level", "INFO"),
        )

        # Setup multi-run configuration - all optional with defaults
        multi_config = opt_config.get("multi_run", {})
        self.multi_run_config = MultiRunConfig(
            enabled=multi_config.get("enabled", False),
            num_runs=multi_config.get("num_runs", 5),
            parameter_sweep=multi_config.get("parameter_sweep"),
            parallel=multi_config.get("parallel", False),
            statistical_analysis=multi_config.get("statistical_analysis", ["basic"]),
        )

        # Setup output sampling strategy
        output_cfg = self.config.get("output", {})
        sampling_strategy_cfg = output_cfg.get("sampling_strategy", {})

        self.sampling_strategy_config = SolutionSamplingStrategyConfig(
            type=sampling_strategy_cfg.get("type", "uniform"),
            max_to_save=sampling_strategy_cfg.get("max_to_save", 10),
            max_rank=sampling_strategy_cfg.get("max_rank"),  # None is OK
            power_exponent=sampling_strategy_cfg.get("power_exponent", 2.0),
            geometric_base=sampling_strategy_cfg.get("geometric_base", 2.0),
            manual_ranks=sampling_strategy_cfg.get("manual_ranks", [])
        )

    def get_pso_config(self) -> PSOConfig:
        """Get PSO algorithm configuration."""
        return self.pso_config

    def get_sampling_config(self) -> SamplingConfig:
        """Get sampling configuration for custom PSO initialization."""
        return self.sampling_config

    def get_termination_config(self) -> TerminationConfig:
        """Get termination criteria configuration."""
        return self.termination_config

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring and logging configuration."""
        return self.monitoring_config

    def get_multi_run_config(self) -> MultiRunConfig:
        """Get multi-run analysis configuration."""
        return self.multi_run_config

    def get_problem_config(self) -> dict[str, Any]:
        """Get problem configuration (objective + constraints)."""
        return self.config["problem"]

    def get_sampling_strategy_config(self) -> SolutionSamplingStrategyConfig:
        """Get output sampling strategy configuration."""
        return self.sampling_strategy_config

    def get_full_config(self) -> dict[str, Any]:
        """Get complete configuration dictionary."""
        return self.config.copy()

    def resolve_sampling_base_solutions(self, optimization_data: dict) -> None:
        """
        Resolve sampling.base_solutions descriptor (if provided) into concrete list
        of flat solution arrays and inject into sampling config.
        """
        sampling = getattr(self, "sampling_config", None)
        if sampling is None or not sampling.enabled:
            logger.info("Sampling is not enabled")
            return

        base = sampling.base_solutions

        # If already a concrete list of arrays, do nothing
        if isinstance(base, list) and base and hasattr(base[0], "shape"):
            logger.info("Sampling base_solutions already resolved")
            return

        loader = SolutionLoader()
        resolved = loader.resolve_base_solutions_descriptor(base, optimization_data)

        # Inject resolved list back into sampling_config and config dict
        self.sampling_config.base_solutions = resolved
        self.config.setdefault("optimization", {}).setdefault("sampling", {})["base_solutions"] = resolved
        logger.info(f"Resolved sampling base_solutions: {len(resolved)} solutions loaded")

    def print_summary(self):
        """Print configuration summary for verification."""
        print("\n📋 OPTIMIZATION CONFIGURATION SUMMARY:")

        print("   🎯 Problem Configuration:")
        print(f"      Objective: {self.config['problem']['objective']['type']}")
        print(
            f"      Constraints: {len(self.config['problem'].get('constraints', []))}"
        )

        print("   🔄 Algorithm Configuration:")
        print("      Type: PSO")
        print(f"      Population size: {self.pso_config.pop_size}")
        print(f"      Inertia weight: {self.pso_config.inertia_weight} ({'adaptive' if self.pso_config.adaptive else 'fixed'})")
        print(f"      Cognitive/Social coeffs: {self.pso_config.cognitive_coeff}/{self.pso_config.social_coeff}")
        print(f"      Adaptive algorithm: {'Enabled' if self.pso_config.adaptive else 'Disabled'}")


        if self.pso_config.adaptive:
            print("      Strategy: PyMOO adaptive PSO (parameters adjust based on swarm spread)")
        else:
            print("      Strategy: Fixed parameters throughout optimization")

        if self.pso_config.use_penalty_method:
            print("      Constraint handling: Penalty method")
            print(f"      Initial penalty weight: {self.pso_config.penalty_weight}")
            if self.pso_config.adaptive_penalty:
                print("      Adaptive penalty: Enabled")
                print(
                    f"      Penalty increase rate: {self.pso_config.penalty_increase_rate}"
                )
            else:
                print("      Adaptive penalty: Disabled")
        print("   ⏰ Termination Configuration:")
        print(f"      Max generations: {self.termination_config.max_generations}")
        if self.termination_config.max_time_minutes:
            print(f"      Max time: {self.termination_config.max_time_minutes} minutes")
        print(
            f"      Convergence tolerance: {self.termination_config.convergence_tolerance}"
        )

        print("   📊 Monitoring Configuration:")
        print(f"      Progress frequency: {self.monitoring_config.progress_frequency}")
        print(f"      Save history: {self.monitoring_config.save_history}")
        print(f"      Log level: {self.monitoring_config.log_level}")

        print("   🔢 Multi-run Configuration:")
        print(f"      Enabled: {self.multi_run_config.enabled}")
        if self.multi_run_config.enabled:
            if self.multi_run_config.parameter_sweep:
                print(
                    f"      Parameter sweep: {list(self.multi_run_config.parameter_sweep.keys())}"
                )
            else:
                print(f"      Statistical runs: {self.multi_run_config.num_runs}")

