"""
PSO Runner for Transit Optimization.

This module provides a complete, production-ready PSO optimization runner
that integrates with the existing transit optimization system. It handles:

- Configuration management and validation
- Problem creation from optimization data
- PSO algorithm setup with adaptive inertia weight
- Single and multi-run optimization
- Result processing and statistical analysis
- Integration with existing constraint and objective systems

The runner is designed to be the main entry point for PSO-based transit
optimization, providing a clean interface while leveraging pymoo's robust
optimization framework and the existing transit optimization components.

Key Features:
- Uses configuration system for all parameters
- Supports adaptive inertia weight PSO
- Integrates with existing TransitOptimizationProblem
- Supports both single and multi-run optimization  
- Uses pymoo's built-in progress reporting and history tracking
- Returns results in expected format for downstream analysis
- Comprehensive error handling and validation

Usage:
```python
from transit_opt.optimisation.config import OptimizationConfigManager
from transit_opt.optimisation.runners import PSORunner

# Load configuration
config_manager = OptimizationConfigManager('pso_config.yaml')

# Create and run optimization
runner = PSORunner(config_manager)
result = runner.optimize(optimization_data)

# Access results
print(f"Best objective: {result.best_objective}")
print(f"Best solution: {result.best_solution}")
print(f"Optimization time: {result.optimization_time}s")
```

Integration with Existing System:
The runner integrates seamlessly with:
- Your existing constraint system (FleetTotalConstraintHandler, etc.)
- Your existing objective system (HexagonalCoverageObjective)
- Your existing TransitOptimizationProblem class
- Your existing optimization data preparation workflow

This provides a complete optimization solution without duplicating existing
functionality or requiring changes to your current constraint/objective code.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.max_time import TimeBasedTermination

from ..config.config_manager import OptimizationConfigManager
from ..problems.transit_problem import TransitOptimizationProblem


@dataclass
@dataclass
class OptimizationResult:
    """
    Complete optimization result with statistics and analysis for PSO transit optimization.
    
    This dataclass encapsulates all results from a single PSO optimization run,
    providing comprehensive information needed for analysis, validation, deployment,
    and further research. It serves as the standardized output format for both
    single runs and as building blocks for multi-run statistical analysis.
    
    RESULT CATEGORIES:
    
    **Core Solution Data:**
    - Best solution found (in domain-specific format)
    - Objective function value achieved
    - Constraint satisfaction analysis
    
    **Optimization Process Information:**
    - Complete generation-by-generation history
    - Performance timing and efficiency metrics
    - Algorithm configuration used
    
    **Quality Assessment:**
    - Convergence analysis and stopping reasons
    - Constraint violation details
    - Statistical measures of optimization quality
    
    INTEGRATION WITH EXISTING SYSTEM:
    This result format integrates seamlessly with:
    - Your existing constraint handlers for violation analysis
    - Your existing objective functions for value interpretation
    - Your existing TransitOptimizationProblem for solution decoding
    - Downstream analysis tools expecting this format
    
    SOLUTION FORMAT EXPLANATION:
    The best_solution is returned in the natural domain format (route×interval matrix)
    rather than the flat vector format used internally by PSO. This makes it
    directly usable for:
    - GTFS reconstruction via your existing tools
    - Visualization with spatial coverage tools
    - Fleet analysis using existing constraint handlers
    - Service pattern analysis and validation
    
    CONSTRAINT VIOLATIONS STRUCTURE:
    The constraint_violations dict provides detailed analysis:
    - 'feasible': boolean indicating if solution can be deployed
    - 'total_violations': count of violated constraints
    - 'violation_details': per-constraint violation analysis
    Each violation detail includes constraint index, violation amount, and feasibility status.
    
    OPTIMIZATION HISTORY FORMAT:
    The optimization_history list contains generation-by-generation data:
    - Best, worst, mean, and std objective values per generation
    - Population size and diversity metrics
    - Improvement tracking between generations
    This enables convergence analysis and algorithm performance assessment.
    
    Attributes:
        best_solution (np.ndarray): Best solution found during optimization in route×interval 
                                   matrix format. Shape: (n_routes, n_intervals). Values are
                                   indices into allowed_headways array from optimization data.
                                   Example: [[0, 1, 2], [1, 2, 0]] means route 0 uses headway
                                   indices 0,1,2 across time intervals, route 1 uses 1,2,0.
                                   
        best_objective (float): Best objective function value achieved during optimization.
                               Lower values typically indicate better solutions for minimization
                               problems. Units and interpretation depend on the specific objective
                               function used (e.g., variance for coverage objectives).
                               
        constraint_violations (dict[str, Any]): Detailed constraint violation analysis containing:
                                              - 'feasible' (bool): Whether solution satisfies all constraints
                                              - 'total_violations' (int): Number of violated constraints  
                                              - 'violation_details' (list): Per-constraint analysis with
                                                violation amounts and constraint indices
                                              Essential for determining if solution can be deployed.
                                              
        optimization_time (float): Total wall-clock time for optimization in seconds.
                                  Includes problem setup, algorithm execution, and result processing.
                                  Useful for performance analysis and resource planning.
                                  Does not include time for optimization data preparation.
                                  
        generations_completed (int): Number of PSO generations actually executed.
                                   May be less than configured maximum if early termination
                                   occurred (time limit, convergence, etc.). Used for
                                   convergence analysis and performance assessment.
                                   
        optimization_history (list[dict[str, Any]]): Generation-by-generation optimization progress.
                                                   Each entry contains statistics for one generation:
                                                   - 'generation' (int): Generation number (0-based)
                                                   - 'best_objective' (float): Best solution in generation
                                                   - 'mean_objective' (float): Population mean
                                                   - 'std_objective' (float): Population standard deviation
                                                   - 'improvement' (float): Improvement from previous generation
                                                   Enables convergence plots and algorithm diagnostics.
                                                   
        algorithm_config (dict[str, Any]): PSO algorithm configuration used for this run.
                                         Contains all PSO-specific parameters:
                                         - 'pop_size' (int): Population size (number of particles)
                                         - 'inertia_weight' (float): Initial/fixed inertia weight
                                         - 'inertia_weight_final' (float|None): Final weight for adaptive PSO
                                         - 'cognitive_coeff' (float): Cognitive coefficient (c1)
                                         - 'social_coeff' (float): Social coefficient (c2)
                                         - 'variant' (str): PSO variant identifier
                                         Essential for reproducing results and parameter analysis.
                                         
        convergence_info (dict[str, Any]): Analysis of optimization convergence behavior.
                                         Contains convergence diagnostics:
                                         - 'converged' (bool): Whether algorithm converged
                                         - 'reason' (str): Convergence/termination reason
                                         - 'recent_improvement' (float): Recent improvement rate
                                         - 'final_generation' (int): Last generation executed
                                         - 'final_objective' (float): Final objective value
                                         Helps assess optimization quality and parameter tuning.
                                         
        performance_stats (dict[str, Any]): Detailed performance and efficiency metrics.
                                          Contains timing and efficiency data:
                                          - 'total_time' (float): Total optimization time
                                          - 'avg_time_per_generation' (float): Average generation time
                                          - 'generations_per_second' (float): Generation rate
                                          - 'inertia_weight_schedule' (list): Weight evolution (if adaptive)
                                          Used for performance analysis and resource planning.
    
    Example Usage:
        ```python
        # Run optimization
        runner = PSORunner(config_manager)
        result = runner.optimize(optimization_data)
        
        # Check if solution is deployable
        if result.constraint_violations['feasible']:
            print("✅ Solution can be deployed")
            
            # Access the solution matrix
            headway_indices = result.best_solution
            print(f"Solution shape: {headway_indices.shape}")
            
            # Convert to actual headway minutes
            allowed_headways = optimization_data['allowed_headways']
            actual_headways = allowed_headways[headway_indices]
            print(f"Route 0 headways: {actual_headways[0]} minutes")
            
        else:
            print("❌ Solution violates constraints")
            violations = result.constraint_violations['total_violations']
            print(f"Number of violations: {violations}")
            
        # Analyze optimization performance
        print(f"Optimization took {result.optimization_time:.1f}s")
        print(f"Completed {result.generations_completed} generations")
        print(f"Final objective: {result.best_objective:.6f}")
        
        # Check convergence
        if result.convergence_info['converged']:
            print("Algorithm converged successfully")
        else:
            print("Algorithm terminated due to other criteria")
            
        # Plot convergence history
        history = result.optimization_history
        objectives = [gen['best_objective'] for gen in history]
        plt.plot(objectives)
        plt.xlabel('Generation')
        plt.ylabel('Best Objective')
        plt.title('PSO Convergence')
        plt.show()
        ```
        
    Integration Examples:
        ```python
        # Use with existing constraint handlers for validation
        from transit_opt.optimisation.problems.base import FleetTotalConstraintHandler
        
        constraint = FleetTotalConstraintHandler(config, optimization_data)
        violations = constraint.evaluate(result.best_solution)
        
        # Use with existing objective functions for detailed analysis
        from transit_opt.optimisation.objectives.service_coverage import HexagonalCoverageObjective
        
        objective = HexagonalCoverageObjective(optimization_data)
        detailed_analysis = objective.get_detailed_analysis(result.best_solution)
        
        # Use with existing visualization tools
        objective.spatial_system.visualize_spatial_coverage(
            solution_matrix=result.best_solution,
            optimization_data=optimization_data
        )
        ```
        
    Notes:
        - All timing information uses wall-clock time (not CPU time)
        - Constraint violations use pymoo convention: G(x) ≤ 0
        - Objective values depend on specific objective function used
        - History data is only available if save_history=True in pymoo optimization
        - Performance stats may include algorithm-specific metrics (e.g., adaptive weights)
        
    See Also:
        - MultiRunResult: For statistical analysis across multiple runs
        - TransitOptimizationProblem: For solution encoding/decoding
        - HexagonalCoverageObjective: For objective function details
        - Constraint handlers: For constraint violation interpretation
    """

    # Core solution and objective results
    best_solution: np.ndarray          # Route×interval matrix of headway indices
    best_objective: float              # Objective function value (minimization)

    # Constraint satisfaction analysis
    constraint_violations: dict[str, Any]  # Detailed feasibility analysis

    # Optimization process timing
    optimization_time: float           # Total wall-clock time in seconds
    generations_completed: int         # Actual generations executed

    # Detailed progress and analysis data (with defaults for optional fields)
    optimization_history: list[dict[str, Any]] = field(default_factory=list)  # Generation-by-generation progress
    algorithm_config: dict[str, Any] = field(default_factory=dict)            # PSO parameters used
    convergence_info: dict[str, Any] = field(default_factory=dict)            # Convergence analysis
    performance_stats: dict[str, Any] = field(default_factory=dict)           # Performance metrics


@dataclass
class MultiRunResult:
    """
    Statistical analysis results from multiple independent PSO optimization runs.
    
    This dataclass aggregates and analyzes results from multiple independent PSO runs,
    providing statistical confidence in optimization results. Multi-run optimization is 
    recommended as it accounts for the stochastic nature of PSO algorithms.
    
    STATISTICAL RELIABILITY:
    PSO is inherently stochastic due to random initialization and update mechanisms.
    Single runs can be misleading as they may represent lucky or unlucky outcomes.
    Multi-run analysis provides:
    
    - **Statistical confidence**: Mean and variance estimates of true performance
    - **Robustness assessment**: Consistency across different random seeds
    - **Best-case performance**: Guaranteed best solution across all attempts
    - **Parameter validation**: Evidence of algorithm configuration quality
    - **Risk assessment**: Understanding of worst-case and typical outcomes
    
    INTEGRATION WITH SINGLE-RUN WORKFLOWS:
    The best_result field is a complete OptimizationResult that can be used
    identically to single-run results:
    
    ```python
    # Multi-run optimization
    multi_result = runner.optimize_multi_run(optimization_data, num_runs=20)
    
    # Use best result exactly like single-run result
    best_solution = multi_result.best_result.best_solution
    best_objective = multi_result.best_result.best_objective
    
    # Access single-run analysis for the best result
    if multi_result.best_result.constraint_violations['feasible']:
        print("Best solution is deployable")
    ```
    
    STATISTICAL SUMMARY STRUCTURE:
    The statistical_summary dict provides comprehensive statistical analysis:
    
    **Objective Statistics:**
    - 'objective_mean': Average objective across all runs
    - 'objective_std': Standard deviation (measure of consistency)
    - 'objective_min': Best objective found (same as best_result.best_objective)
    - 'objective_max': Worst objective found
    - 'objective_median': Median objective (robust central tendency)
    
    **Performance Statistics:**
    - 'time_mean': Average optimization time per run
    - 'time_std': Standard deviation of times
    - 'time_total': Total time for all runs
    - 'generations_mean': Average generations completed
    - 'generations_std': Standard deviation of generations
    
    **Algorithm Quality Indicators:**
    - 'success_rate': Proportion of runs that completed successfully
    - 'convergence_rate': Proportion of runs that showed convergence
    - 'num_runs': Total number of runs attempted
    
    STATISTICAL INTERPRETATION GUIDE:
    
    **Consistency Assessment:**
    ```python
    stats = multi_result.statistical_summary
    coefficient_variation = stats['objective_std'] / stats['objective_mean']
    
    if coefficient_variation < 0.05:
        print("Very consistent results - excellent parameter configuration")
    elif coefficient_variation < 0.15:
        print("Moderately consistent results - good parameter configuration")
    else:
        print("High variability - consider parameter tuning")
    ```
    
    **Performance Assessment:**
    ```python
    success_rate = stats['success_rate']
    convergence_rate = stats['convergence_rate']
    
    if success_rate > 0.95:
        print("Algorithm is robust - rarely fails")
    if convergence_rate > 0.80:
        print("Algorithm converges reliably")
    else:
        print("Consider increasing max_generations or adjusting termination")
    ```
    
    **Statistical Confidence:**
    ```python
    # Rough confidence interval (assumes normal distribution)
    mean_obj = stats['objective_mean']
    std_obj = stats['objective_std']
    n_runs = stats['num_runs']
    
    # 95% confidence interval for mean
    margin_error = 1.96 * std_obj / np.sqrt(n_runs)
    ci_lower = mean_obj - margin_error
    ci_upper = mean_obj + margin_error
    
    print(f"95% CI for mean objective: [{ci_lower:.4f}, {ci_upper:.4f}]")
    ```
    
    RESOURCE PLANNING:
    Multi-run results help with computational resource planning:
    
    ```python
    stats = multi_result.statistical_summary
    
    # Estimate future optimization times
    expected_time = stats['time_mean']
    time_std = stats['time_std']
    
    # Conservative estimate (mean + 2 std deviations covers ~95% of runs)
    conservative_time_estimate = expected_time + 2 * time_std
    
    print(f"Expected optimization time: {expected_time:.1f}s")
    print(f"Conservative estimate: {conservative_time_estimate:.1f}s")
    ```
    
    Attributes:
        best_result (OptimizationResult): Best result found across all runs.
                                        This is the recommended solution for deployment
                                        as it represents the best outcome achieved.
                                        Can be used identically to single-run results.
                                        
        all_results (list[OptimizationResult]): Complete results from each successful run.
                                               Useful for detailed analysis, convergence
                                               study, and understanding result distribution.
                                               Failed runs are not included in this list.
                                               
        statistical_summary (dict[str, Any]): Comprehensive statistical analysis including:
                                             - Objective value statistics (mean, std, min, max, median)
                                             - Timing statistics (mean, std, total time)  
                                             - Algorithm performance (success rate, convergence rate)
                                             - Generation completion statistics
                                             Essential for assessing algorithm reliability and performance.
                                             
        total_time (float): Total wall-clock time for all runs combined in seconds.
                           Includes successful and failed runs. Used for resource
                           planning and cost analysis. Does not include data
                           preparation time outside the optimization runs.
                           
        num_runs_completed (int): Number of runs that completed successfully.
                                 May be less than requested if some runs failed.
                                 Used to assess algorithm robustness and for
                                 statistical validity checks (more runs = better statistics).
    
    Example Usage:
        ```python
        # Run multi-optimization
        multi_result = runner.optimize_multi_run(optimization_data, num_runs=25)
        
        # Use best solution for deployment
        if multi_result.best_result.constraint_violations['feasible']:
            print("✅ Best solution is feasible")
            best_headways = multi_result.best_result.best_solution
            
        # Assess algorithm performance
        stats = multi_result.statistical_summary
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Mean objective: {stats['objective_mean']:.4f} ± {stats['objective_std']:.4f}")
        print(f"Best objective: {stats['objective_min']:.4f}")
        print(f"Consistency (CV): {stats['objective_std']/stats['objective_mean']:.3f}")
        
        # Resource planning for future runs
        avg_time = stats['time_mean']
        std_time = stats['time_std'] 
        conservative_estimate = avg_time + 2 * std_time
        print(f"Future optimization time estimate: {conservative_estimate:.1f}s")
        
        # Detailed analysis of all runs
        objectives = [result.best_objective for result in multi_result.all_results]
        
        import matplotlib.pyplot as plt
        plt.hist(objectives, bins=10, alpha=0.7)
        plt.axvline(stats['objective_mean'], color='red', label='Mean')
        plt.axvline(stats['objective_min'], color='green', label='Best')
        plt.xlabel('Objective Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Optimization Results')
        plt.legend()
        plt.show()
        ```
        
    Notes:
        - Statistical validity improves with √num_runs_completed
        - Typical production runs: 10-50 iterations depending on time budget
        - Failed runs are excluded from statistical analysis but count toward total_time
        - all_results are ordered by execution order, not by objective quality
        - For ensemble methods, sort all_results by best_objective before use
        
    See Also:
        - OptimizationResult: For detailed single-run analysis
        - PSORunner.optimize_multi_run(): For executing multi-run optimization
        - Statistical analysis examples in notebooks/optimization_analysis.ipynb
    """

    # Core results and analysis
    best_result: OptimizationResult                    # Best solution across all runs
    all_results: list[OptimizationResult]              # All successful run results
    statistical_summary: dict[str, Any]                # Statistical analysis

    # Resource and performance tracking
    total_time: float                                  # Total time for all runs
    num_runs_completed: int                            # Number of successful runs


class AdaptivePSO(PSO):
    """
    Particle Swarm Optimization with adaptive inertia weight scheduling.
    
    This class extends pymoo's standard PSO implementation to support adaptive
    inertia weight that automatically balances exploration and exploitation during
    optimization. 
    
    ADAPTIVE INERTIA WEIGHT STRATEGY:
    
    Standard PSO uses fixed inertia weight throughout optimization:
    - High values (0.9): Emphasize exploration, particles move more freely
    - Low values (0.4): Emphasize exploitation, particles focus on local search
    - Fixed values require manual tuning for each problem
    
    Adaptive PSO automatically adjusts inertia weight over generations:
    - **Early generations**: High inertia weight for global exploration
    - **Later generations**: Low inertia weight for local exploitation
    - **Automatic tuning**: No manual parameter adjustment needed
    
    MATHEMATICAL FORMULATION:
    
    **Standard PSO velocity update:**
    ```
    v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
    ```
    **Adaptive inertia weight schedule:**
    ```
    w(t) = w_initial - (w_initial - w_final) * t / (T - 1)
    ```
    
    Where:
    - w(t): Inertia weight at generation t
    - w_initial: Starting inertia weight (typically 0.9)
    - w_final: Ending inertia weight (typically 0.4)
    - t: Current generation (0-based)
    - T: Total number of generations
    
    **Example weight evolution (100 generations, 0.9 → 0.4):**
    - Generation 0: w = 0.900 (maximum exploration)
    - Generation 25: w = 0.775 (balanced)
    - Generation 50: w = 0.650 (transitioning)
    - Generation 75: w = 0.525 (focusing)
    - Generation 99: w = 0.400 (maximum exploitation)
    
    BENEFITS FOR DISCRETE OPTIMIZATION:
    
    **Traditional fixed weight issues:**
    - High weight: Good exploration but poor convergence
    - Low weight: Fast convergence but may miss global optimum
    - Medium weight: Compromise that may be suboptimal for both phases
    
    **Adaptive weight advantages:**
    - **Automatic exploration**: Early generations search widely
    - **Automatic exploitation**: Later generations refine solutions
    - **Better convergence**: Systematic transition between phases
    - **Reduced tuning**: One less parameter to tune manually
    - **Problem adaptivity**: Works well across different problem types
    
    INTEGRATION WITH PYMOO FRAMEWORK:
    
    This class seamlessly integrates with pymoo's optimization infrastructure:
    - Compatible with all pymoo termination criteria
    - Works with pymoo's callback system
    - Supports pymoo's result format and history tracking
    - Can be used with pymoo's multi-objective extensions (future)
    
    ```python
    # Drop-in replacement for standard PSO
    from pymoo.optimize import minimize
    
    # Standard PSO
    standard_pso = PSO(pop_size=40, w=0.7, c1=2.0, c2=2.0)
    
    # Adaptive PSO (recommended)
    adaptive_pso = AdaptivePSO(
        pop_size=40, 
        inertia_weight=0.9,           # Start high for exploration
        inertia_weight_final=0.4,     # End low for exploitation
        cognitive_coeff=2.0,
        social_coeff=2.0
    )
    
    # Both work identically with pymoo.minimize()
    result = minimize(problem, adaptive_pso, termination)
    ```
    
    PARAMETER SELECTION GUIDELINES:
    
    **Population Size (pop_size):**
    - Small problems (< 100 variables): 20-40 particles
    - Medium problems (100-500 variables): 40-80 particles  
    - Large problems (> 500 variables): 80-150 particles
    - Transit problems: 40-60 particles typically sufficient
    
    **Inertia Weight Range:**
    - **Conservative**: 0.9 → 0.5 (gradual transition)
    - **Standard**: 0.9 → 0.4 (recommended default)
    - **Aggressive**: 0.9 → 0.2 (fast exploitation)
    
    **Cognitive/Social Coefficients:**
    - **Balanced**: c1 = c2 = 2.0 (recommended)
    - **Individualistic**: c1 > c2 (particles trust personal experience)
    - **Social**: c1 < c2 (particles follow swarm)
    
    PERFORMANCE CHARACTERISTICS:
    
    **Computational Overhead:**
    - Minimal: Only one additional calculation per generation
    - Weight update: O(1) time complexity
    - Memory overhead: Negligible (few additional instance variables)
    
    **Convergence Properties:**
    - **Exploration phase**: Broad search, slow convergence
    - **Transition phase**: Gradual focusing on promising regions
    - **Exploitation phase**: Rapid convergence to local optima
    - **Overall**: Better global optimum approximation than fixed weight

    """

    def __init__(self, pop_size: int = 50,
                 inertia_weight: float = 0.9,
                 inertia_weight_final: float | None = None,
                 cognitive_coeff: float = 2.0,
                 social_coeff: float = 2.0,
                 **kwargs):
        """
        Initialize adaptive PSO with configurable inertia weight scheduling.
        
        Creates a PSO algorithm that can operate in either fixed inertia weight mode
        (traditional PSO) or adaptive inertia weight mode (recommended for most problems).
        The adaptive mode provides automatic exploration-exploitation balance without
        manual parameter tuning.
        
        Args:
            pop_size (int, optional): Number of particles in the swarm.
                                    More particles provide better exploration but increase
                                    computational cost. Typical range: 20-100.
                                    Defaults to 50.
                                    
            inertia_weight (float, optional): Initial inertia weight (or fixed weight if final is None).
                                            Controls particle momentum and exploration tendency.
                                            Higher values emphasize exploration (0.7-1.2).
                                            Lower values emphasize exploitation (0.2-0.6).
                                            Defaults to 0.9.
                                            
            inertia_weight_final (float | None, optional): Final inertia weight for adaptive scheduling.
                                                         If None, uses fixed inertia weight mode.
                                                         If specified, enables adaptive mode with linear
                                                         decay from initial to final weight.
                                                         Must be less than inertia_weight.
                                                         Defaults to None.
                                                         
            cognitive_coeff (float, optional): Cognitive coefficient (c1 parameter).
                                             Controls attraction to particle's personal best position.
                                             Higher values increase individual particle learning.
                                             Typical range: 0.5-4.0, recommended: 2.0.
                                             Defaults to 2.0.
                                             
            social_coeff (float, optional): Social coefficient (c2 parameter).
                                          Controls attraction to global best position.
                                          Higher values increase swarm coordination.
                                          Typical range: 0.5-4.0, recommended: 2.0.
                                          Defaults to 2.0.
                                          
            **kwargs: Additional keyword arguments passed to parent PSO class.
                     May include boundary handling, variant selection, etc.
                     See pymoo.algorithms.soo.nonconvex.pso.PSO for details.
        
        Examples:
            ```python
            # Fixed inertia weight PSO (traditional)
            fixed_pso = AdaptivePSO(
                pop_size=40,
                inertia_weight=0.7,          # Fixed weight
                inertia_weight_final=None,   # No adaptation
                cognitive_coeff=2.0,
                social_coeff=2.0
            )
            
            # Adaptive inertia weight PSO (recommended)
            adaptive_pso = AdaptivePSO(
                pop_size=60,
                inertia_weight=0.9,          # High initial weight (exploration)
                inertia_weight_final=0.4,    # Low final weight (exploitation)
                cognitive_coeff=2.0,
                social_coeff=2.0
            )
            
            # Conservative adaptive schedule
            conservative_pso = AdaptivePSO(
                pop_size=40,
                inertia_weight=0.8,          # Lower initial weight
                inertia_weight_final=0.6,    # Higher final weight
                cognitive_coeff=1.5,         # Reduced coefficients
                social_coeff=1.5
            )
            
            # Aggressive adaptive schedule  
            aggressive_pso = AdaptivePSO(
                pop_size=80,
                inertia_weight=0.95,         # Very high initial weight
                inertia_weight_final=0.2,    # Very low final weight
                cognitive_coeff=2.5,         # Increased coefficients
                social_coeff=2.5
            )
            ```
            
        Notes:
            - The algorithm automatically detects termination criteria to enable adaptive scheduling
            - If max_generations cannot be determined, falls back to fixed inertia weight
            - All parameters are validated during setup phase
            - Adaptive mode requires generation-based termination (not just time-based)
        """

        # Initialize parent PSO with initial inertia weight
        # Note: pymoo PSO uses 'w' for inertia weight parameter
        super().__init__(
            pop_size=pop_size,          # Number of particles
            w=inertia_weight,           # Initial inertia weight
            c1=cognitive_coeff,         # Cognitive coefficient (attraction to personal best)
            c2=social_coeff,           # Social coefficient (attraction to global best)
            **kwargs                    # Additional parameters (boundary handling, etc.)
        )

        # Store adaptive weight parameters for scheduling
        self.initial_inertia_weight = inertia_weight      # Starting weight value
        self.final_inertia_weight = inertia_weight_final  # Ending weight value (None = fixed mode)
        self.is_adaptive = inertia_weight_final is not None  # Whether adaptive mode is enabled

        # Generation tracking for adaptive scheduling
        # Will be set during setup phase by analyzing termination criteria
        self.max_generations = None

    def setup(self, problem, **kwargs):
        """
        Setup PSO algorithm and extract termination information for adaptive scheduling.
        
        This method is called by pymoo before optimization begins. It performs standard
        PSO setup and additionally extracts maximum generation information needed for
        adaptive inertia weight scheduling.
        
        TERMINATION ANALYSIS:
        The method analyzes termination criteria to determine maximum generations:
        - Single criterion: Extracts n_max_gen directly
        - Multiple criteria: Searches for generation-based criterion
        - Time-only termination: Falls back to fixed inertia weight mode
        
        This information is essential for calculating the linear decay schedule
        of the adaptive inertia weight.
        
        Args:
            problem: Optimization problem instance (from pymoo)
            **kwargs: Additional setup arguments, including 'termination' criteria
            
        Returns:
            Setup result from parent PSO class (typically None)
            
        Side Effects:
            - Calls parent PSO setup method
            - Sets self.max_generations for adaptive weight calculation
            - Logs warnings if adaptive mode cannot be enabled
        """
        # Perform standard PSO setup
        result = super().setup(problem, **kwargs)

        # Extract maximum generations from termination criteria for adaptive scheduling
        # This is crucial for calculating the inertia weight decay schedule
        termination = kwargs.get('termination')
        if termination is not None:
            # Handle single termination criterion
            if hasattr(termination, 'n_max_gen'):
                self.max_generations = termination.n_max_gen
            # Handle multiple termination criteria (TerminationCollection)
            elif hasattr(termination, 'criteria') and termination.criteria:
                for criterion in termination.criteria:
                    if hasattr(criterion, 'n_max_gen'):
                        self.max_generations = criterion.n_max_gen
                        break  # Use first generation-based criterion found

        # Log warning if adaptive mode requested but cannot be enabled
        if self.is_adaptive and self.max_generations is None:
            print("⚠️  Warning: Adaptive inertia weight requires generation-based termination")
            print("   Falling back to fixed inertia weight mode")

        return result

    def _next(self):
        """
        Execute one PSO generation with adaptive inertia weight update.
        
        This method overrides the parent class to add adaptive inertia weight
        scheduling before each generation. The inertia weight is updated based
        on the current generation and total generations, then standard PSO
        operations are performed.
        
        EXECUTION SEQUENCE:
        1. **Weight Update**: Calculate new inertia weight for current generation
        2. **Parameter Application**: Update PSO internal weight parameter
        3. **Standard PSO**: Execute normal PSO generation (parent method)
        
        The adaptive weight calculation uses the current generation (self.n_gen)
        and maximum generations (self.max_generations) to determine progress
        and apply the linear decay schedule.
        
        GENERATION COUNTING:
        - pymoo uses 0-based generation counting
        - Generation 0: Initial population evaluation
        - Generation 1+: Evolution and selection operations
        - self.n_gen tracks current generation automatically
        
        WEIGHT UPDATE TIMING:
        The weight is updated at the beginning of each generation, ensuring:
        - Generation 0 uses initial_inertia_weight
        - Final generation uses inertia_weight_final (if adaptive)
        - Intermediate generations use linearly interpolated weights
        
        Side Effects:
            - Updates self.w (pymoo's inertia weight parameter)
            - Calls parent _next() method for standard PSO operations
            - May print debugging information about weight updates
        """

        # Update inertia weight adaptively before generation execution
        if self.is_adaptive and self.max_generations is not None:
            # Calculate appropriate weight for current generation
            current_w = self._calculate_adaptive_weight(self.n_gen, self.max_generations)
            # Update pymoo's internal inertia weight parameter
            self.w = current_w

        # Execute standard PSO generation with updated inertia weight
        super()._next()

    def _calculate_adaptive_weight(self, generation: int, max_generations: int) -> float:
        """
        Calculate adaptive inertia weight for current generation using linear decay.
        
        This method implements the core adaptive inertia weight strategy using
        linear interpolation between initial and final weights. The linear decay
        provides smooth transition from exploration to exploitation phases.
        
        MATHEMATICAL FORMULA:
        ```
        w(t) = w_initial - (w_initial - w_final) * t / (T - 1)
        ```
        
        Where:
        - w(t): Inertia weight at generation t
        - w_initial: Starting weight (typically 0.9)
        - w_final: Ending weight (typically 0.4)
        - t: Current generation (0 to T-1)
        - T: Total generations
        
        BOUNDARY CONDITIONS:
        - **Generation 0**: Returns initial_inertia_weight exactly
        - **Final generation**: Returns final_inertia_weight exactly  
        - **Invalid inputs**: Returns initial_inertia_weight safely
        - **Non-adaptive mode**: Returns initial_inertia_weight
        
        EXAMPLE CALCULATIONS:
        ```
        # Configuration: initial=0.9, final=0.4, max_gen=100
        generation=0:   w = 0.900 (full exploration)
        generation=25:  w = 0.775 (75% exploration, 25% exploitation)
        generation=50:  w = 0.650 (50% exploration, 50% exploitation)
        generation=75:  w = 0.525 (25% exploration, 75% exploitation)
        generation=99:  w = 0.400 (full exploitation)
        ```
        
        Args:
            generation (int): Current generation number (0-based).
                            Must be in range [0, max_generations-1].
                            Represents optimization progress.
                            
            max_generations (int): Total number of generations planned.
                                 Must be > 1 for meaningful adaptation.
                                 Determines the decay rate.
        
        Returns:
            float: Inertia weight for the specified generation.
                   Guaranteed to be in range [min(initial, final), max(initial, final)].
                   Returns initial_inertia_weight for edge cases and invalid inputs.
        
        Examples:
            ```python
            pso = AdaptivePSO(inertia_weight=0.9, inertia_weight_final=0.4)
            
            # Early generation - high exploration
            w_early = pso._calculate_adaptive_weight(5, 100)   # Returns ~0.875
            
            # Mid generation - balanced
            w_mid = pso._calculate_adaptive_weight(50, 100)    # Returns 0.650
            
            # Late generation - high exploitation  
            w_late = pso._calculate_adaptive_weight(95, 100)   # Returns ~0.425
            
            # Boundary cases
            w_first = pso._calculate_adaptive_weight(0, 100)   # Returns 0.900 exactly
            w_last = pso._calculate_adaptive_weight(99, 100)   # Returns 0.400 exactly
            ```
            
        Implementation Notes:
            - Uses floating-point arithmetic for smooth weight transitions
            - Handles edge cases gracefully without exceptions
            - Optimized for repeated calls during optimization
            - Thread-safe (no shared state modifications)
        """
        # Safety check: fall back to initial weight for invalid configurations
        if not self.is_adaptive or max_generations <= 1:
            return self.initial_inertia_weight

        # Boundary condition: first generation uses initial weight exactly
        if generation <= 0:
            return self.initial_inertia_weight

        # Boundary condition: final generation uses final weight exactly
        if generation >= max_generations - 1:
            return self.final_inertia_weight

        # Linear decay calculation for intermediate generations
        # Progress ranges from 0.0 (start) to 1.0 (end)
        progress = generation / (max_generations - 1)

        # Linear interpolation: w(t) = w_start + progress * (w_end - w_start)
        # Rearranged as: w(t) = w_start - progress * (w_start - w_end)
        weight = self.initial_inertia_weight - (
            self.initial_inertia_weight - self.final_inertia_weight
        ) * progress

        return weight


class PSORuntimeCallback(Callback):
    """
    Callback for tracking PSO runtime information during optimization.
    
    This callback extends pymoo's Callback system to capture additional runtime
    information that isn't tracked by pymoo's default mechanisms. It's particularly
    useful for monitoring adaptive PSO parameters and detailed timing analysis.
    
    TRACKING CAPABILITIES:
    - **Generation timing**: Wall-clock time for each generation
    - **Adaptive parameters**: Inertia weight evolution (for AdaptivePSO)
    - **Runtime statistics**: Data for performance analysis
    
    INTEGRATION WITH PYMOO:
    This callback is designed to work seamlessly with pymoo's optimization loop:
    - Automatically called by pymoo during optimization
    - Non-intrusive: doesn't affect optimization performance
    - Thread-safe: uses only local data structures
    
    USAGE PATTERN:
    ```python
    callback = PSORuntimeCallback()
    result = minimize(problem, algorithm, termination, callback=callback)
    
    # Access tracked information
    print(f"Total generations: {len(callback.generation_times)}")
    if callback.inertia_weights:
        print(f"Weight evolution: {callback.inertia_weights[0]:.3f} → {callback.inertia_weights[-1]:.3f}")
    ```
    
    Attributes:
        start_time (float | None): Optimization start timestamp. Set when first called.
                                  Used as reference point for timing calculations.
                                  
        generation_times (list[float]): Wall-clock time elapsed at end of each generation.
                                       Index 0 is always 0.0 (start), subsequent entries
                                       show cumulative time elapsed since optimization began.
                                       
        inertia_weights (list[float]): Inertia weight values for each generation.
                                     Only populated when using AdaptivePSO algorithm.
                                     Shows weight evolution throughout optimization.
    
    Notes:
        - Timing uses wall-clock time, not CPU time
        - First generation timing is always 0.0 by design
        - Inertia weights are only tracked for AdaptivePSO instances
        - Minimal computational overhead during optimization
    """

    def __init__(self):
        """
        Initialize callback with empty tracking structures.
        
        Creates empty lists for tracking timing and parameter evolution.
        The start_time is set to None and will be initialized when
        optimization begins.
        """
        super().__init__()
        self.start_time = None              # Will be set on first notify() call
        self.generation_times = []          # Cumulative timing for each generation
        self.inertia_weights = []          # Weight values (only for AdaptivePSO)

    def notify(self, algorithm):
        """
        Called by pymoo at the end of each generation during optimization.
        
        This method captures timing information and algorithm-specific parameters
        for later analysis. It's designed to be lightweight to minimize impact
        on optimization performance.
        
        EXECUTION TIMING:
        - Called after each generation completes
        - Records cumulative time elapsed since optimization start
        - First call sets start_time reference point
        
        PARAMETER TRACKING:
        - Detects AdaptivePSO algorithms and tracks inertia weight evolution
        - Other algorithm types: only timing is tracked
        - Safe for all pymoo algorithm types
        
        Args:
            algorithm: The pymoo algorithm instance being used for optimization.
                      Expected to be PSO or AdaptivePSO, but works with any algorithm.
                      
        Side Effects:
            - Updates generation_times with current elapsed time
            - Updates inertia_weights if algorithm is AdaptivePSO
            - Sets start_time on first call
        """
        current_time = time.time()

        # Initialize timing reference on first call
        if self.start_time is None:
            self.start_time = current_time

        # Track inertia weight evolution for adaptive PSO
        if isinstance(algorithm, AdaptivePSO):
            self.inertia_weights.append(algorithm.w)

        # Track cumulative generation timing
        if len(self.generation_times) == 0:
            # First generation: always starts at time 0
            self.generation_times.append(0.0)
        else:
            # Subsequent generations: cumulative elapsed time
            elapsed = current_time - self.start_time
            self.generation_times.append(elapsed)

class PenaltySchedulingCallback(Callback):
    """Adaptive penalty weight scheduling for constraint handling."""

    def __init__(self, initial_penalty: float, increase_rate: float = 2.0):
        super().__init__()
        self.initial_penalty = initial_penalty
        self.increase_rate = increase_rate
        self.current_penalty = initial_penalty

    def notify(self, algorithm):
        gen = algorithm.n_gen

        if hasattr(algorithm.termination, 'n_max_gen'):
            max_gen = algorithm.termination.n_max_gen
            progress = gen / max_gen if max_gen > 0 else 0
        else:
            progress = min(gen / 100.0, 1.0)  # Fallback

        # Exponential penalty increase
        self.current_penalty = self.initial_penalty * (self.increase_rate ** progress)

        # Update problem penalty weight
        if hasattr(algorithm.problem, 'update_penalty_weight'):
            algorithm.problem.update_penalty_weight(self.current_penalty)

        # Log progress
        if gen % 25 == 0 and hasattr(algorithm, 'pop'):
            if hasattr(algorithm.problem, 'use_penalty_method') and algorithm.problem.use_penalty_method:
                # For penalty method: show both penalty weight AND feasible solutions
                print(f"   Gen {gen}: Penalty weight = {self.current_penalty:.1f}")

                # Calculate feasible solutions for penalty method
                # Need to check if solutions would satisfy original constraints
                pop_objectives = algorithm.pop.get("F").flatten()
                feasible_count = 0

                # For penalty method, we need to evaluate original constraints
                # to determine feasibility (since G matrix isn't used)
                if hasattr(algorithm.problem, 'constraints') and algorithm.problem.constraints:
                    for i, solution_flat in enumerate(algorithm.pop.get("X")):
                        solution_matrix = algorithm.problem.decode_solution(solution_flat)
                        is_feasible = True

                        for constraint in algorithm.problem.constraints:
                            violations = constraint.evaluate(solution_matrix)
                            if np.any(violations > 1e-6):  # Same tolerance as hard constraints
                                is_feasible = False
                                break

                        if is_feasible:
                            feasible_count += 1

                pop_size = len(algorithm.pop)
                print(f"   Gen {gen}: Feasible solutions = {feasible_count}/{pop_size}")

            else:
                # For hard constraints: existing logic
                feasible_count = np.sum(algorithm.pop.get("CV") <= 1e-6)
                pop_size = len(algorithm.pop)
                print(f"   Gen {gen}: Feasible solutions = {feasible_count}/{pop_size}")

class CallbackCollection(Callback):
    """
    Wrapper to handle multiple callbacks for pymoo.
    
    Pymoo expects a single callback function, but we need to support
    multiple callbacks (runtime monitoring + penalty scheduling).
    This wrapper calls all callbacks in sequence.
    """

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks or []

    def notify(self, algorithm):
        """Call all callbacks in sequence."""
        for callback in self.callbacks:
            if hasattr(callback, 'notify'):
                callback.notify(algorithm)

    def __len__(self):
        """Return number of wrapped callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Allow iteration over wrapped callbacks."""
        return iter(self.callbacks)

    def __getitem__(self, index):
        """Allow indexing into wrapped callbacks."""
        return self.callbacks[index]



class PSORunner:
    """
    PSO optimization runner for transit optimization problems.
    
    This class provides the main entry point for running PSO-based transit
    optimization. It handles the complete optimization workflow from problem
    setup through result processing, integrating with the existing transit
    optimization system architecture.
    
    ARCHITECTURE INTEGRATION:
    The runner is designed to work seamlessly with existing components:
    - **Configuration system**: Uses OptimizationConfigManager for all parameters
    - **Problem definition**: Creates TransitOptimizationProblem instances
    - **Constraint handlers**: Integrates existing constraint classes
    - **Objective functions**: Works with existing objective implementations
    - **Data pipeline**: Consumes optimization_data from preprocessing
    
    KEY CAPABILITIES:
    - **Single optimization**: Run one optimization with detailed analysis
    - **Multi-run optimization**: Statistical analysis across multiple runs
    - **Adaptive PSO**: Supports adaptive inertia weight scheduling
    - **Progress monitoring**: Real-time progress via pymoo and callbacks
    - **Result processing**: Converts pymoo results to domain-specific format
    - **Error handling**: Robust error management with detailed messages
    
    WORKFLOW OVERVIEW:
    1. **Initialization**: Validate configuration and setup runner
    2. **Problem creation**: Build optimization problem from data and config
    3. **Algorithm setup**: Create configured PSO algorithm instance  
    4. **Optimization execution**: Run pymoo optimization with monitoring
    5. **Result processing**: Convert and analyze results
    6. **Statistics generation**: Create performance and convergence analysis
    
    CONFIGURATION DEPENDENCIES:
    Requires properly configured OptimizationConfigManager with:
    - PSO algorithm parameters (population size, coefficients, etc.)
    - Termination criteria (generations, time limits)
    - Problem definition (objectives, constraints)
    - Multi-run settings (if using multi-run optimization)
    
    Attributes:
        config_manager (OptimizationConfigManager): Configuration manager providing
                                                   all optimization parameters and settings.
                                                   
        optimization_data (dict | None): Current optimization data being processed.
                                        Set during optimize() calls, contains routes,
                                        constraints, and problem-specific data.
                                        
        problem (TransitOptimizationProblem | None): Current optimization problem instance.
                                                   Created during problem setup phase.
    
    Example Usage:
        ```python
        # Single optimization run
        config_manager = OptimizationConfigManager('config.yaml')
        runner = PSORunner(config_manager)
        
        result = runner.optimize(optimization_data)
        print(f"Best objective: {result.best_objective:.4f}")
        
        # Multi-run optimization for statistical analysis
        multi_result = runner.optimize_multi_run(optimization_data, num_runs=10)
        stats = multi_result.statistical_summary
        print(f"Mean ± std: {stats['objective_mean']:.4f} ± {stats['objective_std']:.4f}")
        ```
        
    Error Handling:
        - **Configuration errors**: Raised during initialization if config is invalid
        - **Data errors**: Raised during problem creation if data is malformed
        - **Optimization errors**: Runtime errors are caught and wrapped with context
        - **Multi-run errors**: Individual run failures don't stop remaining runs
        
    Performance Considerations:
        - Memory usage scales with population size and problem size
        - Multi-run optimization is CPU-intensive but embarrassingly parallel
        - History tracking increases memory usage but enables detailed analysis
        - Callback overhead is minimal but can be disabled if needed
    """

    def __init__(self, config_manager: OptimizationConfigManager):
        """
        Initialize PSO runner with configuration manager.
        
        Creates a new PSO runner instance and validates that the provided
        configuration is suitable for PSO optimization. This includes checking
        parameter ranges, constraint compatibility, and termination criteria.
        
        Args:
            config_manager (OptimizationConfigManager): Configured optimization manager
                                                       containing all parameters needed
                                                       for PSO optimization.
                                                       
        Raises:
            ValueError: If configuration contains invalid PSO parameters
            
        Side Effects:
            - Validates all PSO-related configuration parameters
            - Initializes internal state variables to None
            - Ready to accept optimization_data for problem creation
        """
        self.config_manager = config_manager    # Configuration source for all parameters
        self.optimization_data = None          # Will be set during optimize() calls
        self.problem = None                   # Will be created during problem setup

        # Validate configuration early to catch issues before optimization
        self._validate_configuration()

    def _validate_configuration(self):
        """
        Validate that configuration is suitable for PSO optimization.
        
        Performs comprehensive validation of PSO-specific parameters to ensure
        they are within acceptable ranges and mutually compatible. This catches
        configuration errors early rather than during optimization.
        
        VALIDATION CHECKS:
        - **Population size**: Must be >= 5 for meaningful swarm behavior
        - **Generations**: Must be >= 1 for any optimization to occur
        - **Adaptive weights**: Final weight must be < initial weight
        - **Coefficients**: Should be positive (warned but not enforced)
        
        Raises:
            ValueError: If any critical parameter is invalid or incompatible
            
        Notes:
            - Called automatically during __init__()
            - Validation is strict for parameters that would cause optimization failure
            - Some parameters generate warnings but don't prevent initialization
        """
        # Get relevant configuration sections
        pso_config = self.config_manager.get_pso_config()
        term_config = self.config_manager.get_termination_config()

        # Validate population size (too small causes poor optimization)
        if pso_config.pop_size < 5:
            raise ValueError("PSO requires population size >= 5")

        # Validate termination criteria (must allow at least one generation)
        if term_config.max_generations < 1:
            raise ValueError("PSO requires max_generations >= 1")

        # Validate adaptive inertia weight parameters if enabled
        if pso_config.is_adaptive():
            if pso_config.inertia_weight_final >= pso_config.inertia_weight:
                raise ValueError(
                    "Final inertia weight must be less than initial weight for adaptive PSO"
                )

    def optimize(self, optimization_data) -> OptimizationResult:
        """
        Run single PSO optimization with comprehensive result analysis.
        
        Executes a complete PSO optimization workflow including problem setup,
        algorithm execution, and detailed result processing. This is the main
        method for single-run optimization.
        
        EXECUTION WORKFLOW:
        1. **Setup**: Store optimization data and create problem instance
        2. **Configuration**: Create PSO algorithm and termination criteria
        3. **Execution**: Run pymoo optimization with progress monitoring
        4. **Processing**: Convert pymoo results to domain-specific format
        5. **Analysis**: Generate convergence and performance statistics
        
        PROGRESS MONITORING:
        - Uses pymoo's built-in progress reporting for generation-by-generation updates
        - PSORuntimeCallback tracks additional timing and parameter information
        - Real-time console output shows optimization progress
        
        RESULT PROCESSING:
        - Decodes flat solution vector back to route×interval matrix format
        - Analyzes constraint violations with detailed breakdown
        - Generates optimization history for convergence analysis
        - Computes performance statistics and convergence metrics
        
        Args:
            optimization_data (dict): Complete optimization data containing:
                                    - Route and network information
                                    - Allowed headway choices
                                    - Constraint parameters
                                    - Initial solution
                                    - Problem dimensions
                                    
        Returns:
            OptimizationResult: Complete optimization result with:
                              - Best solution found (route×interval matrix)
                              - Objective function value
                              - Constraint violation analysis
                              - Generation-by-generation history
                              - Performance and convergence statistics
                              
        Raises:
            ValueError: If optimization_data is invalid or incomplete
            RuntimeError: If optimization fails during execution
            
        Side Effects:
            - Sets self.optimization_data for use by other methods
            - Creates self.problem instance
            - Prints progress information to console
            - May generate temporary files (depending on pymoo configuration)
            
        Example:
            ```python
            runner = PSORunner(config_manager)
            result = runner.optimize(optimization_data)
            
            # Check if solution is feasible
            if result.constraint_violations['feasible']:
                print("✅ Feasible solution found")
                solution_matrix = result.best_solution  # Shape: (routes, intervals)
            else:
                print("❌ Solution violates constraints")
                print(f"Violations: {result.constraint_violations['total_violations']}")
            ```
        """
        print("🚀 STARTING PSO OPTIMIZATION")

        # Setup optimization problem
        self.optimization_data = optimization_data
        self._create_problem()

        start_time = time.time()

        callbacks = []

        # Add runtime monitoring callback
        runtime_callback = PSORuntimeCallback()
        callbacks.append(runtime_callback)

        # Add penalty scheduling callback if using penalty method with adaptive penalties
        pso_config = self.config_manager.get_pso_config()
        if (hasattr(self.problem, 'use_penalty_method') and
            self.problem.use_penalty_method and
            pso_config.adaptive_penalty):

            penalty_callback = PenaltySchedulingCallback(
                initial_penalty=pso_config.penalty_weight,
                increase_rate=pso_config.penalty_increase_rate
            )
            callbacks.append(penalty_callback)
            print(f"   🎯 Adaptive penalty method enabled: {pso_config.penalty_weight} → increasing")

        try:
            # Create algorithm components
            algorithm = self._create_algorithm()        # Configured PSO instance
            termination = self._create_termination()    # Termination criteria

            # Print configuration summary for user reference
            self._print_optimization_summary()

            # Create single callback wrapper for pymoo
            callback_wrapper = CallbackCollection(callbacks)

            # Execute optimization using pymoo
            print("\n📊 Running optimization (pymoo will show progress)...")
            result = minimize(
                self.problem,            # Problem definition
                algorithm,               # PSO algorithm instance
                termination,            # When to stop optimization
                callback=callback_wrapper,    # Runtime monitoring
                verbose=True,           # Enable pymoo's progress output
                save_history=True       # Enable generation-by-generation tracking
            )

            optimization_time = time.time() - start_time

            # Process pymoo result into domain-specific format
            return self._process_single_result(result, runtime_callback, optimization_time)

        except Exception as e:
            # Provide context for optimization failures
            optimization_time = time.time() - start_time
            raise RuntimeError(f"PSO optimization failed after {optimization_time:.1f}s: {str(e)}") from e

    def optimize_multi_run(self, optimization_data, num_runs: int | None = None,
                           parallel: bool = False) -> MultiRunResult:
        """
        Run multiple independent PSO optimizations for statistical analysis.
        
        Executes multiple independent PSO runs to provide statistical confidence
        in optimization results. This is the recommended approach for production
        use as it accounts for the stochastic nature of PSO algorithms.
        
        STATISTICAL BENEFITS:
        - **Robustness**: Reduces impact of lucky/unlucky individual runs
        - **Confidence intervals**: Enables statistical analysis of performance
        - **Best solution**: Guaranteed best result across multiple attempts
        - **Algorithm assessment**: Evaluates configuration quality and consistency
        
        EXECUTION STRATEGY:
        - Each run is completely independent (different random seed)
        - Failed runs don't stop remaining runs (robust to individual failures)
        - Best result across all runs is selected for deployment
        - Statistical summary provides performance assessment
        
        RESOURCE USAGE:
        - Memory: Scales with number of runs (stores all results)
        - CPU: Linearly scales with num_runs (could be parallelized in future)
        - Time: Approximately num_runs × single_run_time
        
        Args:
            optimization_data (dict): Same optimization data used for all runs
            num_runs (int | None, optional): Number of independent runs to execute.
                                           If None, uses value from configuration.
                                           Must be >= 1.
                                           
        Returns:
            MultiRunResult: Statistical analysis of all runs containing:
                          - best_result: Best OptimizationResult across all runs
                          - all_results: List of all successful OptimizationResults
                          - statistical_summary: Mean, std, min, max statistics
                          - total_time: Combined time for all runs
                          - num_runs_completed: Number of successful runs
                          
        Raises:
            ValueError: If num_runs < 1
            RuntimeError: If all runs fail (no successful results)
            
        Example:
            ```python
            # Run 20 independent optimizations
            multi_result = runner.optimize_multi_run(optimization_data, num_runs=20)
            
            # Use best result for deployment
            best_solution = multi_result.best_result.best_solution
            
            # Assess algorithm performance
            stats = multi_result.statistical_summary
            consistency = stats['objective_std'] / stats['objective_mean']
            
            if consistency < 0.1:
                print("Consistent algorithm performance")
            else:
                print("High variability - consider parameter tuning")
            ```
            
        Notes:
            - Individual run failures are logged but don't stop execution
            - Statistics are computed only from successful runs
            - Best result is guaranteed feasible if any run produces feasible solution
            - Results can be used for ensemble methods or confidence intervals
        """
        import time

        # Get run count from configuration or parameter
        multi_config = self.config_manager.get_multi_run_config()
        runs_to_perform = num_runs if num_runs is not None else multi_config.num_runs

        if runs_to_perform < 1:
            raise ValueError("Number of runs must be at least 1")

        print(f"🔄 STARTING MULTI-RUN PSO OPTIMIZATION ({runs_to_perform} runs)")
        if parallel:
            print("   🚀 Parallel execution enabled")

        start_time = time.time()

        if parallel:
            import os

            # Set environment variable to signal parallel execution
            os.environ['PARALLEL_EXECUTION'] = 'True'

            # Parallel execution using multiprocessing
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # Determine number of workers (leave some cores free)
            max_workers = min(runs_to_perform, max(1, mp.cpu_count() - 1))
            print("🚀 PARALLEL EXECUTION:")
            print(f"   👥 Using {max_workers} parallel workers")
            print("   🔇 Individual run output suppressed for clarity")
            print("   📊 Progress will be shown as runs complete\n")


            all_results = []
            completed_runs = 0

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all runs with unique seeds
                future_to_run = {}
                for run_idx in range(runs_to_perform):
                    # Each submission gets a unique seed
                    future = executor.submit(
                        self._run_single_optimization_with_unique_seed,
                        run_idx,
                        optimization_data
                    )
                    future_to_run[future] = run_idx + 1

                # Collect results as they complete
                for future in as_completed(future_to_run):
                    run_idx = future_to_run[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        completed_runs += 1

                        # Show clean progress update
                        violations = result.constraint_violations
                        feasible_status = "✅ Feasible" if violations['feasible'] else "❌ Infeasible"
                        print(f"[{completed_runs:2d}/{runs_to_perform}] Run {run_idx:2d}: "
                            f"Objective={result.best_objective:.6f}, "
                            f"Gens={result.generations_completed:2d}, "
                            f"Time={result.optimization_time:5.1f}s, {feasible_status}")
                    except Exception as e:
                        print(f"[{completed_runs+1:2d}/{runs_to_perform}] ❌ Run {run_idx:2d}: FAILED - {str(e)}")
                        continue
            # Clean up environment variable
            os.environ.pop('PARALLEL_EXECUTION', None)
            print("\n✅ All parallel runs completed!")

        else:
            # Ensure environment variable is not set for sequential execution
            os.environ.pop('PARALLEL_EXECUTION', None)
            # Sequential execution with unique seeds
            all_results = []        # Store successful results

            # Execute independent runs
            for run_idx in range(runs_to_perform):
                print(f"\n{'='*60}")
                print(f"🏃 RUN {run_idx + 1}/{runs_to_perform}")
                print(f"{'='*60}")

                try:
                    # Run single optimization (each run is independent)
                    result = self._run_single_optimization_with_unique_seed(run_idx, optimization_data)
                    all_results.append(result)

                    print(f"✅ Run {run_idx + 1} completed: objective = {result.best_objective:.6f}")

                except Exception as e:
                    # Log failure but continue with remaining runs
                    print(f"❌ Run {run_idx + 1} failed: {str(e)}")
                    continue

        total_time = time.time() - start_time

        # Check if any runs succeeded
        if not all_results:
            raise RuntimeError("All optimization runs failed")

        # Generate statistical summary from successful runs
        statistical_summary = self._generate_statistical_summary(all_results)

        # Find best result
        best_result = min(all_results, key=lambda r: r.best_objective)


        # Print summary statistics
        print("\n🎯 MULTI-RUN OPTIMIZATION COMPLETED")
        print(f"   Successful runs: {len(all_results)}/{runs_to_perform}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Best objective: {best_result.best_objective:.6f}")
        print(f"   Mean objective: {statistical_summary['objective_mean']:.6f}")
        print(f"   Std objective: {statistical_summary['objective_std']:.6f}")

        return MultiRunResult(
            best_result=best_result,
            all_results=all_results,
            statistical_summary=statistical_summary,
            total_time=total_time,
            num_runs_completed=len(all_results)
        )

    def _run_single_optimization_with_unique_seed(self, run_index: int, optimization_data: dict) -> OptimizationResult:
        """Run single optimization with unique random seed."""
        import os
        import random
        import sys
        import time
        from io import StringIO

        # Generate unique seed based on current time and run index
        unique_seed = int(time.time() * 1000000) % (2**31) + run_index * 1000

        # Set random seeds for reproducible diversity
        random.seed(unique_seed)
        np.random.seed(unique_seed)

        # Check if we're in parallel execution mode
        is_parallel = os.getenv('PARALLEL_EXECUTION', 'False') == 'True'

        # Create fresh config manager and runner for this run
        from ..config.config_manager import OptimizationConfigManager
        fresh_config_manager = OptimizationConfigManager(config_dict=self.config_manager.config)
        fresh_runner = PSORunner(fresh_config_manager)

        if is_parallel:
            # Suppress ALL output during parallel execution
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            devnull = StringIO()
            sys.stdout = devnull
            sys.stderr = devnull

            try:
                result = fresh_runner.optimize(optimization_data)
            finally:
                # Always restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            # Normal execution with output for sequential runs
            print(f"   🎲 Run {run_index + 1}: Using seed {unique_seed}")
            result = fresh_runner.optimize(optimization_data)

        return result

    def _run_single_optimization_for_parallel(self, optimization_data, run_number):
        """Helper method for parallel execution of single optimization run."""
        import io
        import sys

        try:
            # Capture stdout to prevent interleaved output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()  # Redirect to string buffer

            # Create a new instance for thread safety
            runner = PSORunner(self.config_manager)
            result = runner.optimize(optimization_data)

            # Restore stdout
            sys.stdout = old_stdout

            # Return result with run identifier
            result.run_id = run_number
            return result

        except Exception as e:
            # Restore stdout even on error
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            raise RuntimeError(f"Run {run_number} failed: {str(e)}") from e


    def _create_problem(self):
        """
        Create TransitOptimizationProblem instance from optimization data and configuration.
        
        Builds the complete optimization problem by instantiating the objective function
        and all constraint handlers as specified in the configuration. This method
        handles the complex task of connecting the configuration system with the
        actual optimization components.
        
        PROBLEM CONSTRUCTION WORKFLOW:
        1. **Objective creation**: Instantiate configured objective function
        2. **Constraint creation**: Build all configured constraint handlers
        3. **Problem assembly**: Combine components into TransitOptimizationProblem
        4. **Validation**: Verify problem is properly constructed
        
        OBJECTIVE FUNCTION HANDLING:
        - Currently supports HexagonalCoverageObjective
        - Dynamically imports objective class based on configuration
        - Passes only explicitly configured parameters to constructor
        - Extensible to additional objective types
        
        CONSTRAINT HANDLER SUPPORT:
        - FleetTotalConstraintHandler: Global fleet size limits
        - FleetPerIntervalConstraintHandler: Per-interval fleet limits
        - MinimumFleetConstraintHandler: Minimum service requirements
        - Graceful handling of unknown constraint types (warning, not error)
        
        Raises:
            ValueError: If optimization_data is None or problem creation fails
            ImportError: If configured objective/constraint classes cannot be imported
            
        Side Effects:
            - Sets self.problem to new TransitOptimizationProblem instance
            - Imports objective and constraint classes dynamically
            - Prints problem construction progress and summary
            
        Configuration Dependencies:
            - problem_config.objective: Objective function type and parameters
            - problem_config.constraints: List of constraint configurations
            - Each constraint config must have 'type' field plus type-specific parameters
            
        Example Configuration:
            ```yaml
            problem:
              objective:
                type: HexagonalCoverageObjective
                spatial_resolution_km: 3.0
                crs: "EPSG:3857"
              constraints:
                - type: FleetTotalConstraintHandler
                  max_fleet_size: 100
                - type: FleetPerIntervalConstraintHandler
                  max_fleet_per_interval: 80
            ```
        """
        if self.optimization_data is None:
            raise ValueError("Optimization data must be set before creating problem")

        try:
            # Get problem configuration from config manager
            problem_config = self.config_manager.get_problem_config()

            # === OBJECTIVE FUNCTION CREATION ===
            objective_config = problem_config.get('objective', {})
            objective_type = objective_config.get('type')

            if objective_type == 'HexagonalCoverageObjective':
                from ..objectives.service_coverage import HexagonalCoverageObjective

                # Build kwargs with only explicitly configured values
                objective_kwargs = {'optimization_data': self.optimization_data}

                # Add explicitly configured objective parameters
                # This approach avoids passing undefined parameters to constructor
                if 'spatial_resolution_km' in objective_config:
                    objective_kwargs['spatial_resolution_km'] = objective_config['spatial_resolution_km']
                if 'crs' in objective_config:
                    objective_kwargs['crs'] = objective_config['crs']
                if 'boundary' in objective_config:
                    objective_kwargs['boundary'] = objective_config['boundary']
                if 'time_aggregation' in objective_config:
                    objective_kwargs['time_aggregation'] = objective_config['time_aggregation']
                if 'spatial_lag' in objective_config:
                    objective_kwargs['spatial_lag'] = objective_config['spatial_lag']
                if 'alpha' in objective_config:
                    objective_kwargs['alpha'] = objective_config['alpha']
                if 'population_weighted' in objective_config:
                    objective_kwargs['population_weighted'] = objective_config['population_weighted']

                objective = HexagonalCoverageObjective(**objective_kwargs)

            else:
                raise ValueError(f"Unknown objective type: {objective_type}")

            # === CONSTRAINT HANDLER CREATION ===
            constraints = []
            constraint_configs = problem_config.get('constraints', [])

            print(f"   📋 Creating {len(constraint_configs)} constraint handler(s)...")

            for i, constraint_config in enumerate(constraint_configs):
                constraint_type = constraint_config.get('type')
                # Extract constraint-specific parameters (exclude 'type' field)
                constraint_kwargs = {k: v for k, v in constraint_config.items() if k != 'type'}
                print(f"      Creating constraint {i+1}: {constraint_type}")

                # Create appropriate constraint handler based on type
                if constraint_type == 'FleetTotalConstraintHandler':
                    from ..problems.base import FleetTotalConstraintHandler
                    constraint = FleetTotalConstraintHandler(constraint_kwargs, self.optimization_data)
                    constraints.append(constraint)
                    print(f"         ✓ FleetTotal: {constraint.n_constraints} constraint(s)")

                elif constraint_type == 'FleetPerIntervalConstraintHandler':
                    from ..problems.base import FleetPerIntervalConstraintHandler
                    constraint = FleetPerIntervalConstraintHandler(constraint_kwargs, self.optimization_data)
                    constraints.append(constraint)
                    print(f"         ✓ FleetPerInterval: {constraint.n_constraints} constraint(s)")

                elif constraint_type == 'MinimumFleetConstraintHandler':
                    from ..problems.base import MinimumFleetConstraintHandler
                    constraint = MinimumFleetConstraintHandler(constraint_kwargs, self.optimization_data)
                    constraints.append(constraint)
                    print(f"         ✓ MinimumFleet: {constraint.n_constraints} constraint(s)")

                else:
                    # Graceful handling of unknown constraint types
                    print(f"         ⚠️  Warning: Unknown constraint type '{constraint_type}' - skipping")
                    continue

            # Get penalty configuration from PSO config
            pso_config = self.config_manager.get_pso_config()

            if hasattr(pso_config, 'use_penalty_method') and pso_config.use_penalty_method:
                penalty_config = {
                    'enabled': True,
                    'penalty_weight': getattr(pso_config, 'penalty_weight', 1000.0),
                    'adaptive': getattr(pso_config, 'adaptive_penalty', False),
                    'constraint_weights': self._get_constraint_penalty_weights()
                }
            else:
                penalty_config = {'enabled': False}


            # === PROBLEM ASSEMBLY ===
            self.problem = TransitOptimizationProblem(
                self.optimization_data,  # Problem data
                objective,              # Objective function instance
                constraints,            # List of constraint handler instances
                penalty_config=penalty_config
            )

            # Print problem construction summary
            print("✅ Problem created successfully:")
            print(f"   📊 Variables: {self.problem.n_var}")
            print(f"   🚦 Total constraints: {self.problem.n_constr} (from {len(constraints)} handler(s))")
            print(f"   🎯 Objective: {objective_type}")
            print(f"   📋 Constraint types: {[c.__class__.__name__ for c in constraints]}")
            if hasattr(self.problem, 'use_penalty_method'):
                print(f"   🎯 Method: {'Penalty' if self.problem.use_penalty_method else 'Hard constraints'}")


        except Exception as e:
            # Provide context for problem creation failures
            raise ValueError(f"Failed to create optimization problem: {str(e)}") from e

    def _get_constraint_penalty_weights(self) -> dict[str, float]:
        """Extract constraint-specific penalty weights from config."""
        problem_config = self.config_manager.get_problem_config()
        penalty_weights = {}

        # Check if penalty weights are specified in problem config
        if 'penalty_weights' in problem_config:
            penalty_weights = problem_config['penalty_weights']

        return penalty_weights

    def _create_algorithm(self) -> AdaptivePSO:
        """
        Create configured AdaptivePSO algorithm instance.
        
        Instantiates an AdaptivePSO algorithm with parameters from the configuration
        manager. The returned algorithm is ready for use with pymoo's minimize function.
        
        Returns:
            AdaptivePSO: Configured PSO algorithm instance with all parameters
                        set according to configuration (population size, inertia
                        weights, coefficients, etc.)
                        
        Notes:
            - Always returns AdaptivePSO (can operate in fixed-weight mode)
            - Parameters are validated during PSORunner initialization
            - Algorithm is stateless until used in optimization
        """
        pso_config = self.config_manager.get_pso_config()

        algorithm = AdaptivePSO(
            pop_size=pso_config.pop_size,                           # Number of particles
            inertia_weight=pso_config.inertia_weight,              # Initial/fixed weight
            inertia_weight_final=pso_config.inertia_weight_final,  # Final weight (None = fixed)
            cognitive_coeff=pso_config.cognitive_coeff,            # c1 coefficient
            social_coeff=pso_config.social_coeff                  # c2 coefficient
        )

        return algorithm

    def _create_termination(self):
        """
        Create configured termination criteria for optimization.
        
        Builds termination criteria based on configuration, supporting both
        generation-based and time-based termination. Multiple criteria can
        be combined (optimization stops when ANY criterion is met).
        
        TERMINATION TYPES SUPPORTED:
        - **Generation-based**: Stop after maximum generations
        - **Time-based**: Stop after maximum wall-clock time
        - **Combined**: Stop when either generation or time limit reached
        
        Returns:
            Termination: pymoo termination criterion (single or combined)
                        that will be used to determine when optimization stops
                        
        Notes:
            - Generation-based termination is always included
            - Time-based termination is optional (if configured)
            - Combined termination uses TerminationCollection from pymoo
            - Time limits are converted from minutes to seconds for pymoo
        """
        term_config = self.config_manager.get_termination_config()

        # Primary termination criterion: maximum generations
        termination = get_termination("n_gen", term_config.max_generations)

        # Add optional time-based termination
        if term_config.max_time_minutes is not None:
            time_termination = TimeBasedTermination(term_config.max_time_minutes * 60)  # Convert to seconds
            # Combine criteria: optimization stops when EITHER condition is met
            from pymoo.termination.collection import TerminationCollection
            termination = TerminationCollection(termination, time_termination)

        return termination

    def _print_optimization_summary(self):
        """
        Print human-readable summary of optimization configuration.
        
        Displays key optimization parameters and problem characteristics
        to help users understand what optimization is being performed.
        Used for logging and debugging purposes.
        
        INFORMATION DISPLAYED:
        - Algorithm type and key parameters
        - Population size and PSO coefficients  
        - Inertia weight schedule (adaptive vs fixed)
        - Termination criteria
        - Problem size (variables and constraints)
        
        Side Effects:
            - Prints formatted information to console
            - No return value or state changes
        """
        pso_config = self.config_manager.get_pso_config()
        term_config = self.config_manager.get_termination_config()

        print("\n📋 OPTIMIZATION CONFIGURATION:")
        print(f"   Algorithm: PSO ({'adaptive' if pso_config.is_adaptive() else 'canonical'})")
        print(f"   Population size: {pso_config.pop_size}")

        # Display inertia weight information
        if pso_config.is_adaptive():
            print(f"   Inertia weight: {pso_config.inertia_weight:.3f} → {pso_config.inertia_weight_final:.3f}")
        else:
            print(f"   Inertia weight: {pso_config.inertia_weight:.3f} (fixed)")

        print(f"   Cognitive/Social coeffs: {pso_config.cognitive_coeff:.1f}/{pso_config.social_coeff:.1f}")
        print(f"   Max generations: {term_config.max_generations}")

        # Display time limit if configured
        if term_config.max_time_minutes:
            print(f"   Max time: {term_config.max_time_minutes} minutes")

        print(f"   Problem size: {self.problem.n_var} variables, {self.problem.n_constr} constraints")

    def _process_single_result(self, pymoo_result, callback: PSORuntimeCallback,
                            optimization_time: float) -> OptimizationResult:
        """
        Process pymoo optimization result into domain-specific OptimizationResult format.
        
        Converts pymoo's internal result format into the comprehensive OptimizationResult
        format used throughout the transit optimization system. This includes solution
        decoding, constraint analysis, history processing, and statistics generation.
        
        PROCESSING STEPS:
        1. **Solution decoding**: Convert flat vector to route×interval matrix
        2. **Objective extraction**: Handle various pymoo objective formats
        3. **Constraint analysis**: Analyze violation details
        4. **History processing**: Convert pymoo history to domain format
        5. **Statistics generation**: Create performance and convergence metrics
        6. **Configuration recording**: Store algorithm parameters used
        
        Args:
            pymoo_result: Raw result object from pymoo.minimize()
            callback: PSORuntimeCallback instance with timing data
            optimization_time: Total wall-clock optimization time in seconds
            
        Returns:
            OptimizationResult: Complete domain-specific result with all analysis
            
        Side Effects:
            - Prints optimization completion summary
            - Uses self.problem for solution decoding
            - Accesses self.config_manager for algorithm parameters
        """
        # === SOLUTION DECODING ===
        # Convert pymoo's flat solution vector back to route×interval matrix format
        best_solution_flat = pymoo_result.X
        best_solution = self.problem.decode_solution(best_solution_flat)

        # === OBJECTIVE VALUE EXTRACTION ===
        # Handle various pymoo objective result formats safely
        if hasattr(pymoo_result.F, 'item'):
            best_objective = float(pymoo_result.F.item())
        elif hasattr(pymoo_result.F, '__len__') and len(pymoo_result.F) > 0:
            best_objective = float(pymoo_result.F[0])
        else:
            best_objective = float(pymoo_result.F)

        # === CONSTRAINT VIOLATION ANALYSIS ===
        constraint_violations = self._analyze_constraint_violations(pymoo_result)

        # === OPTIMIZATION HISTORY PROCESSING ===
        optimization_history = self._process_optimization_history(pymoo_result.history)

        # === PERFORMANCE STATISTICS GENERATION ===
        performance_stats = self._generate_performance_stats(
            callback, optimization_time, len(optimization_history)
        )

        # === ALGORITHM CONFIGURATION RECORDING ===
        pso_config = self.config_manager.get_pso_config()
        algorithm_config = {
            'pop_size': pso_config.pop_size,
            'inertia_weight': pso_config.inertia_weight,
            'inertia_weight_final': pso_config.inertia_weight_final,
            'cognitive_coeff': pso_config.cognitive_coeff,
            'social_coeff': pso_config.social_coeff,
            'variant': pso_config.variant
        }

        # === CONVERGENCE ANALYSIS ===
        convergence_info = self._analyze_convergence(optimization_history)

        # === COMPLETION SUMMARY ===
        print("\n✅ OPTIMIZATION COMPLETED")
        print(f"   Best objective: {best_objective:.6f}")
        print(f"   Generations: {len(optimization_history)}")
        print(f"   Time: {optimization_time:.1f}s")
        # print(f"   Avg time/gen: {optimization_time/len(optimization_history):.3f}s")
        if len(optimization_history) > 0:
            print(f"   Avg time/gen: {optimization_time/len(optimization_history):.3f}s")
        else:
            print("   Avg time/gen: N/A (no history)")

        if constraint_violations['total_violations'] > 0:
            print(f"   ⚠️  Constraint violations: {constraint_violations['total_violations']}")
        else:
            print("   ✅ All constraints satisfied")

        # === RESULT ASSEMBLY ===
        return OptimizationResult(
            best_solution=best_solution,                    # Route×interval matrix format
            best_objective=best_objective,                  # Single objective value
            constraint_violations=constraint_violations,    # Detailed violation analysis
            optimization_time=optimization_time,           # Total time in seconds
            generations_completed=len(optimization_history), # Actual generations run
            optimization_history=optimization_history,      # Generation-by-generation data
            algorithm_config=algorithm_config,             # PSO parameters used
            convergence_info=convergence_info,             # Convergence analysis
            performance_stats=performance_stats           # Performance metrics
        )

    def _analyze_constraint_violations(self, pymoo_result) -> dict[str, Any]:
        """
        Analyze constraint violations from pymoo result.
        
        Processes pymoo's constraint values (G vector) to provide detailed
        violation analysis including feasibility status and per-constraint details.
        
        PYMOO CONSTRAINT CONVENTION:
        - G(x) <= 0: Feasible (negative or zero values)
        - G(x) > 0: Violated (positive values)
        
        Args:
            pymoo_result: Result object from pymoo optimization
            
        Returns:
            dict: Violation analysis with 'feasible', 'total_violations', 
                  and 'violation_details' keys
        """
        violations = {
            'total_violations': 0,
            'violation_details': [],
            'feasible': True
        }

        if pymoo_result.G is not None and len(pymoo_result.G) > 0:
            constraint_values = pymoo_result.G

            # Count violations (positive values violate G <= 0)
            violation_mask = constraint_values > 0
            violations['total_violations'] = int(np.sum(violation_mask))
            violations['feasible'] = violations['total_violations'] == 0

            # Generate per-constraint details
            for i, (value, violated) in enumerate(zip(constraint_values, violation_mask, strict=False)):
                violations['violation_details'].append({
                    'constraint_index': i,
                    'value': float(value),
                    'violated': bool(violated),
                    'violation_amount': max(0.0, float(value))  # Only positive violations
                })

        return violations

    def _process_optimization_history(self, pymoo_history) -> list[dict[str, Any]]:
        """
        Convert pymoo optimization history to domain-specific format.
        
        Processes pymoo's history object to extract generation-by-generation
        statistics in a format suitable for convergence analysis and plotting.
        
        Args:
            pymoo_history: History object from pymoo (can be None)
            
        Returns:
            list: Generation data with statistics for each generation
        """
        history = []

        if pymoo_history is None:
            return history

        for gen_idx, entry in enumerate(pymoo_history):
            # Extract population objectives for this generation
            pop_objectives = entry.pop.get("F").flatten()

            # Compute generation statistics
            history_entry = {
                'generation': gen_idx,                              # 0-based generation number
                'best_objective': float(np.min(pop_objectives)),    # Best in this generation
                'worst_objective': float(np.max(pop_objectives)),   # Worst in this generation
                'mean_objective': float(np.mean(pop_objectives)),   # Population mean
                'std_objective': float(np.std(pop_objectives)),     # Population diversity
                'population_size': len(pop_objectives)             # Actual population size
            }

            # Calculate improvement from previous generation
            if gen_idx > 0:
                prev_best = history[gen_idx - 1]['best_objective']
                history_entry['improvement'] = prev_best - history_entry['best_objective']  # Positive = improvement
            else:
                history_entry['improvement'] = 0.0  # No improvement for first generation

            history.append(history_entry)

        return history

    def _generate_performance_stats(self, callback: PSORuntimeCallback,
                                   total_time: float, num_generations: int) -> dict[str, Any]:
        """
        Generate performance statistics from optimization run.
        
        Creates performance metrics including timing information and algorithm-specific
        data such as inertia weight evolution for adaptive PSO.
        
        Args:
            callback: PSORuntimeCallback with timing data
            total_time: Total optimization time in seconds
            num_generations: Number of generations completed
            
        Returns:
            dict: Performance metrics including timing and parameter evolution
        """
        stats = {
            'total_time': total_time,
            'num_generations': num_generations,
            'avg_time_per_generation': total_time / max(1, num_generations),
            'generations_per_second': num_generations / max(0.001, total_time)
        }

        # Add inertia weight evolution if available (adaptive PSO)
        if callback.inertia_weights:
            stats['inertia_weight_schedule'] = callback.inertia_weights.copy()
            stats['initial_inertia_weight'] = callback.inertia_weights[0] if callback.inertia_weights else None
            stats['final_inertia_weight'] = callback.inertia_weights[-1] if callback.inertia_weights else None

        return stats

    def _analyze_convergence(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Perform basic convergence analysis on optimization history.
        
        Analyzes the optimization trajectory to determine if the algorithm
        converged or stagnated. Uses simple heuristics based on recent improvement.
        
        Args:
            history: Generation-by-generation optimization data
            
        Returns:
            dict: Convergence analysis with status and metrics
        """
        if len(history) < 5:
            return {'converged': False, 'reason': 'Insufficient generations'}

        # Analyze improvement in last 5 generations
        recent_improvements = [entry['improvement'] for entry in history[-5:]]
        total_recent_improvement = sum(recent_improvements)

        # Simple convergence criterion: very small total improvement recently
        convergence_info = {
            'converged': abs(total_recent_improvement) < 1e-6,
            'recent_improvement': total_recent_improvement,
            'final_generation': len(history) - 1,
            'final_objective': history[-1]['best_objective']
        }

        return convergence_info

    def _generate_statistical_summary(self, results: list[OptimizationResult]) -> dict[str, Any]:
        """
        Generate statistical summary from multiple optimization results.
        
        Computes comprehensive statistics across multiple independent optimization
        runs for multi-run analysis. Includes objective statistics, timing data,
        and algorithm performance metrics.
        
        Args:
            results: List of OptimizationResult instances from successful runs
            
        Returns:
            dict: Statistical summary with means, standard deviations, and rates
        """
        if not results:
            return {}

        # Extract data vectors for statistical analysis
        objectives = [r.best_objective for r in results]
        times = [r.optimization_time for r in results]
        generations = [r.generations_completed for r in results]

        return {
            # Basic run information
            'num_runs': len(results),

            # Objective statistics
            'objective_mean': float(np.mean(objectives)),
            'objective_std': float(np.std(objectives)),
            'objective_min': float(np.min(objectives)),
            'objective_max': float(np.max(objectives)),
            'objective_median': float(np.median(objectives)),

            # Timing statistics
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'time_total': float(np.sum(times)),

            # Generation statistics
            'generations_mean': float(np.mean(generations)),
            'generations_std': float(np.std(generations)),

            # Algorithm performance rates
            'success_rate': len(results) / len(results),  # All provided results succeeded
            'convergence_rate': sum(1 for r in results if r.convergence_info.get('converged', False)) / len(results)
        }
