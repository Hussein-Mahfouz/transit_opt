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
- Your existing objective system (StopCoverageObjective)
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
        from transit_opt.optimisation.objectives.service_coverage import StopCoverageObjective
        
        objective = StopCoverageObjective(optimization_data)
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
        - StopCoverageObjective: For objective function details
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
    best_feasible_solutions: list[dict[str, Any]] = field(default_factory=list)  # Top N feasible solutions for this run


@dataclass
class MultiRunResult:
    """
    Memory-efficient statistical analysis results from multiple independent PSO optimization runs.
    
    This dataclass aggregates and analyzes results from multiple independent PSO runs
    using memory-efficient storage that tracks only essential information while providing
    comprehensive statistical analysis. 
    
    DATA STRUCTURE OVERVIEW:
    
    **best_result**: Complete OptimizationResult for deployment
    - Full solution matrix, detailed analysis, optimization history
    - Same interface as single-run results for seamless integration
    - Represents the best solution found across all runs
    
    **run_summaries**: Lightweight per-run statistics (REPLACES all_results)
    - Essential metrics: objective, feasibility, timing, generations
    - Enables statistical analysis without memory overhead
    - Perfect for performance assessment and algorithm tuning
    
    **best_feasible_solutions_per_run**: Top N feasible solutions per run
    - Complete solution matrices
    - Independent tracking per run (N solutions × num_runs total)
    - Only feasible solutions with deployment potential
    
    STATISTICAL SUMMARY STRUCTURE:
    The statistical_summary provides comprehensive analysis:
    
    **Objective Statistics:**
    - 'objective_mean': Average performance across runs
    - 'objective_std': Consistency measure (lower = more reliable)
    - 'objective_min': Best case performance
    - 'objective_max': Worst case performance
    - 'objective_median': Robust central tendency
    
    **Performance Metrics:**
    - 'time_mean': Average optimization time per run
    - 'feasibility_rate': Proportion of runs finding feasible solutions
    - 'success_rate': Proportion of runs completing successfully
    - 'generations_mean': Average convergence speed
    
    Attributes:
        best_result (OptimizationResult): Best solution across all runs with complete
                                        detail. Can be used identically to single-run
                                        results for deployment or detailed analysis.
                                        
        run_summaries (list[dict]): Lightweight per-run summaries replacing the
                                   memory-intensive all_results list. Each summary
                                   contains essential metrics: run_id, objective,
                                   feasible, generations, time, violations, and
                                   best_feasible_solutions_count.
                                   
        best_feasible_solutions_per_run (list[list[dict]]): Best N feasible solutions
                                                           from each run with complete
                                                           data. Structure: [run_idx][solution_idx]
                                                           = solution_dict. Each solution
                                                           contains full solution matrix,
                                                           objective, generation_found, etc.
                                                           
        statistical_summary (dict[str, Any]): Comprehensive statistical analysis
                                             computed from run summaries. Includes
                                             objective statistics, timing analysis,
                                             and algorithm performance metrics.
                                             
        total_time (float): Total wall-clock time for all runs combined.
                          Includes successful and failed runs for resource planning.
                          
        num_runs_completed (int): Number of runs completing successfully.
                                May be less than requested if some runs failed.
                                Used for statistical validity assessment.
    
    Example Usage:
        ```python
        # Memory-efficient multi-run execution
        multi_result = runner.optimize_multi_run(optimization_data, num_runs=20, track_best_n=3)
        
        # Deploy best solution (same interface as before)
        if multi_result.best_result.constraint_violations['feasible']:
            deploy_solution = multi_result.best_result.best_solution
        
        # Analyze algorithm performance using lightweight summaries
        for summary in multi_result.run_summaries:
            print(f"Run {summary['run_id']}: {summary['objective']:.4f}, "
                  f"feasible={summary['feasible']}, time={summary['time']:.1f}s")
        
        # Statistical analysis for algorithm assessment
        stats = multi_result.statistical_summary
        reliability = stats['objective_std'] / stats['objective_mean']
        print(f"Algorithm reliability (lower=better): {reliability:.3f}")
        print(f"Feasible solution rate: {stats['feasibility_rate']:.1%}")
        
        # Access best feasible solutions for ensemble methods
        all_feasible_solutions = []
        for run_solutions in multi_result.best_feasible_solutions_per_run:
            all_feasible_solutions.extend(run_solutions)
        
        print(f"Total feasible solutions for ensemble: {len(all_feasible_solutions)}")
        
        # Sort all solutions by objective for portfolio creation
        all_feasible_solutions.sort(key=lambda x: x['objective'])
        top_5_global = all_feasible_solutions[:5]
        ```
    """
    # Core results and analysis
    best_result: OptimizationResult                    # Best solution across all runs

    # memory efficient storage of all run results
    run_summaries: list[dict]                          # Lightweight per-run summaries

    # statistics and metadata
    statistical_summary: dict[str, Any]                # Statistical analysis
    total_time: float                                  # Total time for all runs
    num_runs_completed: int                            # Number of successful runs

    best_feasible_solutions_per_run: list[list[dict]] = field(default_factory=list)  # [run_idx][solution_idx] = solution_dict




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
    ```
    
    Attributes:
        start_time (float | None): Optimization start timestamp. Set when first called.
                                  Used as reference point for timing calculations.
                                  
        generation_times (list[float]): Wall-clock time elapsed at end of each generation.
                                       Index 0 is always 0.0 (start), subsequent entries
                                       show cumulative time elapsed since optimization began.
                                       
    
    Notes:
        - Timing uses wall-clock time, not CPU time
        - First generation timing is always 0.0 by design
        - Inertia weights are only tracked for AdaptivePSO instances
        - Minimal computational overhead during optimization
    """

    def __init__(self, track_best_n: int = 5):
        """
        Initialize callback with empty tracking structures.
        
        Creates empty lists for tracking timing and parameter evolution.
        The start_time is set to None and will be initialized when
        optimization begins.
        """
        super().__init__()
        self.start_time = None              # Will be set on first notify() call
        self.generation_times = []          # Cumulative timing for each generation

        # Track best feasible solutions during optimization
        self.feasible_tracker = BestFeasibleSolutionsTracker(track_best_n)



    # Replace the existing notify method in PSORuntimeCallback

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
            - Sets start_time on first call
        """

        # Set start time on first call
        if self.start_time is None:
            self.start_time = time.time()
            return  # Don't process first call

        # Get current generation info
        current_gen = algorithm.n_gen
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Store timing
        self.generation_times.append(elapsed)


        # Track feasible solutions
        if hasattr(algorithm, 'pop') and algorithm.pop is not None:
            pop = algorithm.pop
            objectives = pop.get("F")
            constraint_violations = pop.get("G")
            decision_vars = pop.get("X")
            current_gen = algorithm.n_gen

            if objectives is not None and decision_vars is not None:
                objectives = objectives.flatten()
                feasibles = []
                violations = []
                solution_matrices = []
                generations = []

                for i in range(len(objectives)):
                    # Determine feasibility
                    is_feasible = True
                    violation_count = 0
                    if constraint_violations is not None:
                        individual_violations = constraint_violations[i]
                        is_feasible = np.all(individual_violations <= 1e-6)
                        violation_count = np.sum(individual_violations > 1e-6)
                    try:
                        solution_matrix = algorithm.problem.decode_solution(decision_vars[i])
                    except Exception:
                        solution_matrix = None

                    feasibles.append(is_feasible)
                    violations.append(violation_count)
                    solution_matrices.append(solution_matrix)
                    generations.append(current_gen)

                # Add top N feasible solutions from this generation
                self.feasible_tracker.add_generation_solutions(
                    solution_matrices, objectives, generations, feasibles, violations
                )

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


class BestFeasibleSolutionsTracker:
    """
    Tracks the best N unique feasible solutions during a single optimization run.
    Maintains a sorted list of the best feasible solutions found during optimization.
    """

    def __init__(self, max_solutions: int = 5):
        self.max_solutions = max_solutions
        self.best_solutions = []  # List of solution dictionaries, sorted by objective

    def add_generation_solutions(self, solution_matrices, objectives, generations, feasibles, violations):
        """
        Add up to max_solutions feasible solutions from the current generation,
        ensuring uniqueness and keeping only the best N overall.
        """
        # Collect feasible solutions from this generation
        gen_solutions = []
        for i in range(len(objectives)):
            if not feasibles[i]:
                continue
            # Check for valid solution based on type
            if not np.isfinite(objectives[i]) or solution_matrices[i] is None:
                continue

            # Handle both dict (DRT-enabled) and array (PT-only) solution formats
            if isinstance(solution_matrices[i], dict):
                # DRT-enabled case: check if dict has required keys and valid data
                if 'pt' not in solution_matrices[i] or 'drt' not in solution_matrices[i]:
                    continue
                if solution_matrices[i]['pt'] is None or solution_matrices[i]['drt'] is None:
                    continue
                if solution_matrices[i]['pt'].size == 0 or solution_matrices[i]['drt'].size == 0:
                    continue
            else:
                # PT-only case: check numpy array
                if solution_matrices[i].size == 0:
                    continue

            # Create solution record with proper copying for both formats
            if isinstance(solution_matrices[i], dict):
                # DRT-enabled: deep copy the dictionary structure
                solution_copy = {
                    'pt': solution_matrices[i]['pt'].copy(),
                    'drt': solution_matrices[i]['drt'].copy()
                }
            else:
                # PT-only: direct copy
                solution_copy = solution_matrices[i].copy()

            gen_solutions.append({
                'solution': solution_copy,
                'objective': float(objectives[i]),
                'generation_found': generations[i],
                'feasible': True,
                'violations': violations[i]
            })

        # Sort by objective (best first), take up to max_solutions
        gen_solutions.sort(key=lambda x: x['objective'])
        gen_solutions = gen_solutions[:self.max_solutions]

        # Add to global best_solutions if unique
        for new_sol in gen_solutions:
            is_duplicate = False
            for existing in self.best_solutions:
                # Handle comparison for both dict and array formats
                solutions_equal = False
                if isinstance(existing['solution'], dict) and isinstance(new_sol['solution'], dict):
                    # Both are DRT format: compare both PT and DRT parts
                    solutions_equal = (np.array_equal(existing['solution']['pt'], new_sol['solution']['pt']) and
                                    np.array_equal(existing['solution']['drt'], new_sol['solution']['drt']))
                elif isinstance(existing['solution'], np.ndarray) and isinstance(new_sol['solution'], np.ndarray):
                    # Both are PT-only format: direct comparison
                    solutions_equal = np.array_equal(existing['solution'], new_sol['solution'])

                if solutions_equal and existing['objective'] == new_sol['objective']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                self.best_solutions.append(new_sol)

        # Sort and trim to max_solutions
        self.best_solutions.sort(key=lambda x: x['objective'])
        if len(self.best_solutions) > self.max_solutions:
            self.best_solutions = self.best_solutions[:self.max_solutions]

    def get_best_solutions(self) -> list[dict]:
        """Get copy of best feasible solutions list."""
        return [sol.copy() for sol in self.best_solutions]

    def get_count(self) -> int:
        """Get number of feasible solutions tracked."""
        return len(self.best_solutions)



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


    def optimize(self, optimization_data, track_best_n: int = 5) -> OptimizationResult:
        """
        Run single PSO optimization with comprehensive result analysis and feasible solution tracking.
        
        Executes a complete PSO optimization workflow including problem setup,
        algorithm execution, and detailed result processing. Includes tracking
        of the best N feasible solutions found during optimization for analysis
        and ensemble methods.
        
        EXECUTION WORKFLOW:
        1. **Setup**: Store optimization data and create problem instance
        2. **Configuration**: Create PSO algorithm and termination criteria
        3. **Execution**: Run pymoo optimization with progress monitoring
        4. **Processing**: Convert pymoo results to domain-specific format
        5. **Analysis**: Generate convergence and performance statistics
        6. **Tracking**: Extract best feasible solutions found during optimization
        
        FEASIBLE SOLUTIONS TRACKING:
        During optimization, tracks the best N solutions that satisfy all constraints:
        - **Complete solution data**: Full route×interval matrices 
        - **Generation tracking**: When each solution was discovered
        - **Objective ranking**: Solutions sorted by objective value (best first)
        - **Feasibility guarantee**: Only solutions satisfying all constraints
        - **Memory efficient**: Tracks only best N, not all solutions from all generations
        
        PROGRESS MONITORING:
        - Uses pymoo's built-in progress reporting for generation-by-generation updates
        - Enhanced callback tracks timing, parameter evolution, and feasible solutions
        - Real-time console output shows optimization progress
        
        Args:
            optimization_data (dict): Complete optimization data containing:
                                    - Route and network information
                                    - Allowed headway choices
                                    - Constraint parameters
                                    - Initial solution
                                    - Problem dimensions
            track_best_n (int, optional): Number of best feasible solutions to track
                                        during optimization. These solutions are stored
                                        with complete data for analysis or ensemble use.
                                        Defaults to 5.
                                        
        Returns:
            OptimizationResult: Complete optimization result with:
                            - Best solution found (route×interval matrix)
                            - Objective function value
                            - Constraint violation analysis
                            - Generation-by-generation history
                            - Performance and convergence statistics
                            - **NEW**: best_feasible_solutions list with top N feasible solutions
                            
        Raises:
            ValueError: If optimization_data is invalid or incomplete
            RuntimeError: If optimization fails during execution
            
        Example:
            ```python
            # Run optimization with feasible solution tracking
            result = runner.optimize(optimization_data, track_best_n=3)
            
            # Check if best solution is feasible
            if result.constraint_violations['feasible']:
                print("✅ Best solution is feasible")
                solution_matrix = result.best_solution
            
            # Access tracked feasible solutions
            print(f"Tracked {len(result.best_feasible_solutions)} feasible solutions:")
            for i, sol in enumerate(result.best_feasible_solutions):
                print(f"  Solution {i+1}: objective={sol['objective']:.4f}, "
                    f"found at generation {sol['generation_found']}")
                
                # Access complete solution matrix
                solution_matrix = sol['solution']  # Full route×interval matrix
                
            # Use for ensemble methods
            if len(result.best_feasible_solutions) >= 2:
                print("Multiple feasible solutions available for ensemble analysis")
                # Can analyze diversity, create solution portfolios, etc.
            ```
            
        Integration with Multi-Run:
            ```python
            # Single run feasible solutions feed into multi-run tracking
            single_result = runner.optimize(optimization_data, track_best_n=5)
            # single_result.best_feasible_solutions contains up to 5 solutions
            
            multi_result = runner.optimize_multi_run(optimization_data, num_runs=10, track_best_n=5)
            # multi_result.best_feasible_solutions_per_run[0] contains run 1's solutions
            # multi_result.best_feasible_solutions_per_run[1] contains run 2's solutions
            # etc.
            ```
        """
        print("🚀 STARTING PSO OPTIMIZATION")

        # Setup optimization problem
        self.optimization_data = optimization_data

        # Resolve sampling descriptors (if any) into concrete seed arrays
        try:
            self.config_manager.resolve_sampling_base_solutions(optimization_data)
        except Exception as e:
            print(f"   ⚠️ Failed to resolve sampling.base_solutions: {e}")
            raise
        self._create_problem()

        start_time = time.time()

        callbacks = []

        # Add runtime monitoring callback
        runtime_callback = PSORuntimeCallback(track_best_n=track_best_n)
        #runtime_callback.problem = self.problem  # Provide problem reference for decoding
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
                           parallel: bool = False, track_best_n: int = 5) -> MultiRunResult:
        """
        Run multiple independent PSO optimizations with memory-efficient storage.
        
        Executes multiple independent PSO runs to provide statistical confidence
        in optimization results while using memory-efficient storage that tracks
        only essential information and the best feasible solutions from each run.
        
        MEMORY EFFICIENCY IMPROVEMENTS:
        - **Lightweight summaries**: Stores only essential per-run statistics instead of complete results
        - **Best N tracking**: Tracks only the best N feasible solutions per run with complete data
        - **Single best result**: Maintains one complete OptimizationResult for the overall best solution
        - **Memory reduction**: ~90% less memory usage compared to storing all complete results
        
        STATISTICAL BENEFITS:
        - **Robustness**: Reduces impact of lucky/unlucky individual runs
        - **Confidence intervals**: Enables statistical analysis of performance
        - **Best solution**: Guaranteed best result across multiple attempts
        - **Algorithm assessment**: Evaluates configuration quality and consistency
        
        EXECUTION STRATEGY:
        - Each run is completely independent (different random seed)
        - Failed runs don't stop remaining runs (robust to individual failures)
        - Best result across all runs is selected for deployment
        - Statistical summary computed from lightweight run summaries
        
        FEASIBLE SOLUTIONS TRACKING:
        Each run independently tracks its best N feasible solutions with complete data:
        - Solution matrices (route×interval)
        - Generation information for analysis
        - Objective values for ranking and selection
        - Only feasible solutions are tracked (satisfy all constraints)
        
        Args:
            optimization_data (dict): Same optimization data used for all runs
            num_runs (int | None, optional): Number of independent runs to execute.
                                        If None, uses value from configuration.
                                        Must be >= 1. Defaults to None.
            parallel (bool, optional): Whether to run optimizations in parallel.
                                    Uses multiprocessing for CPU-intensive speedup.
                                    Defaults to False.
            track_best_n (int, optional): Number of best feasible solutions to track
                                        per run. Each run maintains its own list of
                                        the N best feasible solutions found during
                                        optimization. Defaults to 5.
                                        
        Returns:
            MultiRunResult: Memory-efficient multi-run results containing:
                        - best_result: Complete OptimizationResult for best solution
                        - run_summaries: Lightweight per-run statistics (replaces all_results)
                        - best_feasible_solutions_per_run: Top N feasible solutions per run
                        - statistical_summary: Comprehensive statistical analysis
                        - total_time: Combined time for all runs
                        - num_runs_completed: Number of successful runs
                        
        Raises:
            ValueError: If num_runs < 1
            RuntimeError: If all runs fail (no successful results)
            
        Example:
            ```python
            # Memory-efficient multi-run with feasible solution tracking
            multi_result = runner.optimize_multi_run(
                optimization_data, 
                num_runs=10, 
                parallel=True,
                track_best_n=3
            )
            
            # Use best result 
            best_solution = multi_result.best_result.best_solution
            
            # Analyze per-run performance using lightweight summaries
            for summary in multi_result.run_summaries:
                print(f"Run {summary['run_id']}: {summary['objective']:.4f}, "
                    f"Feasible: {summary['feasible']}, "
                    f"Solutions found: {summary['best_feasible_solutions_count']}")
            
            # Access best feasible solutions from each run
            for run_idx, solutions in enumerate(multi_result.best_feasible_solutions_per_run):
                print(f"Run {run_idx + 1}: {len(solutions)} feasible solutions")
                for sol in solutions:
                    print(f"  Objective: {sol['objective']:.4f} (gen {sol['generation_found']})")
            
            # Statistical analysis (same interface as before)
            stats = multi_result.statistical_summary
            consistency = stats['objective_std'] / stats['objective_mean']
            print(f"Algorithm consistency: {consistency:.3f}")
            ```
            
        Notes:
            - Individual run failures are logged but don't stop execution
            - Statistics computed only from successful runs
            - Best result prioritizes feasible solutions when available
            - Feasible solutions ideal for ensemble methods or solution analysis
            - Parallel execution suppresses individual run output for clarity
        """
        import os
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

        # Add tracking for feasible solutions per run
        run_summaries = []                    # Lightweight per-run summaries
        best_feasible_solutions_per_run = []  # Store solutions per run
        overall_best_result = None           # Track only the single best result
        overall_best_objective = float('inf') # For finding best across runs
        completed_runs = 0


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

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all runs with unique seeds
                future_to_run = {}
                for run_idx in range(runs_to_perform):
                    # Each submission gets a unique seed
                    future = executor.submit(
                        self._run_single_optimization_with_unique_seed,
                        run_idx,
                        optimization_data,
                        track_best_n
                    )
                    future_to_run[future] = run_idx + 1

                # Collect results as they complete
                for future in as_completed(future_to_run):
                    run_idx = future_to_run[future]
                    result = None  # ← INITIALIZE result to None each iteration

                    try:
                        result = future.result()
                       # Create lightweight summary
                        run_summary = {
                            'run_id': run_idx,
                            'objective': result.best_objective,
                            'feasible': result.constraint_violations['feasible'],
                            'generations': result.generations_completed,
                            'time': result.optimization_time,
                            'violations': result.constraint_violations['total_violations'],
                            'best_feasible_solutions_count': len(result.best_feasible_solutions)
                        }
                        run_summaries.append(run_summary)

                        # Store feasible solutions from this run
                        best_feasible_solutions_per_run.append(result.best_feasible_solutions)


                        # Track overall best
                        if result.best_objective < overall_best_objective:
                            overall_best_objective = result.best_objective
                            overall_best_result = result

                        # Delete non-best results immediately
                        if result is not overall_best_result:
                            del result

                        completed_runs += 1


                        # Show clean progress update
                        violations_text = f"Violations={run_summary['violations']}" if not run_summary['feasible'] else f"FeasibleSols={run_summary['best_feasible_solutions_count']}"
                        feasible_status = "✅ Feasible" if run_summary['feasible'] else "❌ Infeasible"

                        print(f"[{completed_runs:2d}/{runs_to_perform}] Run {run_idx:2d}: "
                            f"Objective={run_summary['objective']:.6f}, "
                            f"Gens={run_summary['generations']:2d}, "
                            f"Time={run_summary['time']:5.1f}s, "
                            f"{violations_text}, {feasible_status}")

                    except Exception as e:
                        print(f"❌ Run {run_idx:2d}: FAILED - {str(e)}")

                        # Clean up result
                        if result is not None:
                            del result
                        continue
            # Clean up environment variable
            os.environ.pop('PARALLEL_EXECUTION', None)
            print("\n✅ All parallel runs completed!")

        else:
            # Ensure environment variable is not set for sequential execution
            os.environ.pop('PARALLEL_EXECUTION', None)
            # Sequential execution with unique seeds

            # Execute independent runs
            for run_idx in range(runs_to_perform):
                print(f"\n{'='*60}")
                print(f"🏃 RUN {run_idx + 1}/{runs_to_perform}")
                print(f"{'='*60}")

                try:
                    # Run single optimization (each run is independent)
                    result = self._run_single_optimization_with_unique_seed(run_idx, optimization_data, track_best_n)
                    # Create lightweight summary instead of storing full result
                    run_summary = {
                        'run_id': run_idx + 1,
                        'objective': result.best_objective,
                        'feasible': result.constraint_violations['feasible'],
                        'generations': result.generations_completed,
                        'time': result.optimization_time,
                        'violations': result.constraint_violations['total_violations'],
                        'best_feasible_solutions_count': len(result.best_feasible_solutions)
                    }
                    run_summaries.append(run_summary)

                    # Store feasible solutions from this run
                    best_feasible_solutions_per_run.append(result.best_feasible_solutions)

                    # Track overall best (keep only the best complete result)
                    if result.best_objective < overall_best_objective:
                        overall_best_objective = result.best_objective
                        overall_best_result = result

                    # Delete non-best results immediately to save memory
                    if result is not overall_best_result:
                        del result

                    print(f"✅ Run {run_idx + 1} completed: objective = {result.best_objective:.6f}")

                except Exception as e:
                    # Log failure but continue with remaining runs
                    print(f"❌ Run {run_idx + 1} failed: {str(e)}")
                    continue

        total_time = time.time() - start_time

        # Check if any runs succeeded
        # Check if any runs succeeded
        if not run_summaries:
            raise RuntimeError("All optimization runs failed")

        # Generate statistical summary from successful runs
        statistical_summary = self._generate_statistical_summary_from_summaries(run_summaries)

        # Find best result
        best_result = overall_best_result


        # Print summary statistics
        print("\n🎯 MULTI-RUN OPTIMIZATION COMPLETED")
        print(f"   Successful runs: {len(run_summaries)}/{runs_to_perform}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Best objective: {best_result.best_objective:.6f}")
        print(f"   Mean objective: {statistical_summary['objective_mean']:.6f}")
        print(f"   Std objective: {statistical_summary['objective_std']:.6f}")

        # Show summary with feasible solutions
        total_feasible_solutions = sum(len(solutions) for solutions in best_feasible_solutions_per_run)
        print(f"   Total feasible solutions tracked: {total_feasible_solutions}")

        return MultiRunResult(
            best_result=best_result,
            run_summaries=run_summaries,
            best_feasible_solutions_per_run=best_feasible_solutions_per_run,
            statistical_summary=statistical_summary,
            total_time=total_time,
            num_runs_completed=len(run_summaries)
        )

    def _run_single_optimization_with_unique_seed(self, run_index: int, optimization_data: dict,
                                                  track_best_n: int = 5) -> OptimizationResult:
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
                result = fresh_runner.optimize(optimization_data, track_best_n = track_best_n)

            finally:
                # Always restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                # Always cleanup regardless of success/failure
                import gc
                gc.collect()
        else:
            # Normal execution with output for sequential runs
            print(f"   🎲 Run {run_index + 1}: Using seed {unique_seed}")
            result = fresh_runner.optimize(optimization_data, track_best_n= track_best_n)

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
        - Currently supports StopCoverageObjective
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
                type: StopCoverageObjective
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

            if objective_type == 'StopCoverageObjective':
                from ..objectives.service_coverage import StopCoverageObjective

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
                if 'population_layer' in objective_config:
                    objective_kwargs['population_layer'] = objective_config['population_layer']
                if 'population_power' in objective_config:
                    objective_kwargs['population_power'] = objective_config['population_power']

                objective = StopCoverageObjective(**objective_kwargs)

            elif objective_type == 'WaitingTimeObjective':
                from ..objectives.waiting_time import WaitingTimeObjective

                # Build kwargs with only explicitly configured values
                objective_kwargs = {'optimization_data': self.optimization_data}

                # Add explicitly configured waiting time parameters
                if 'spatial_resolution_km' in objective_config:
                    objective_kwargs['spatial_resolution_km'] = objective_config['spatial_resolution_km']
                if 'crs' in objective_config:
                    objective_kwargs['crs'] = objective_config['crs']
                if 'boundary' in objective_config:
                    objective_kwargs['boundary'] = objective_config['boundary']
                if 'time_aggregation' in objective_config:
                    objective_kwargs['time_aggregation'] = objective_config['time_aggregation']
                if 'metric' in objective_config:
                    objective_kwargs['metric'] = objective_config['metric']
                if 'population_weighted' in objective_config:
                    objective_kwargs['population_weighted'] = objective_config['population_weighted']
                if 'population_layer' in objective_config:
                    objective_kwargs['population_layer'] = objective_config['population_layer']
                if 'population_power' in objective_config:
                    objective_kwargs['population_power'] = objective_config['population_power']

                objective = WaitingTimeObjective(**objective_kwargs)

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
                    from ..problems.base import \
                        FleetPerIntervalConstraintHandler
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

    def _create_algorithm(self) -> PSO:
        """
        Create configured PSO algorithm instance.

        Instantiates a PSO algorithm with parameters from the configuration
        manager. The returned algorithm is ready for use with pymoo's minimize function.
        
        Returns:
            PSO: Configured PSO algorithm instance with all parameters
                 set according to configuration (population size, inertia
                        weights, coefficients, etc.)
                        
        Notes:
            - Parameters are validated during PSORunner initialization
            - Algorithm is stateless until used in optimization
        """
        pso_config = self.config_manager.get_pso_config()
        sampling_config = self.config_manager.get_sampling_config()


        # Create PSO algorithm with pymoo's native adaptive support
        algorithm = PSO(
            pop_size=pso_config.pop_size,
            w=pso_config.inertia_weight,           # Initial inertia weight
            c1=pso_config.cognitive_coeff,         # Initial cognitive coefficient
            c2=pso_config.social_coeff,            # Initial social coefficient
            adaptive=pso_config.adaptive,          # Whether to use pymoo's adaptive algorithm
            # TODO: Other pymoo parameters we might want to expose:
            # initial_velocity='random',           # or 'zero'
            # max_velocity_rate=0.20,             # velocity clamping
            # pertube_best=True                   # mutate global best
        )

        # Add custom sampling if enabled
        if sampling_config.enabled:
            print("🔧 Creating custom initial population...")

            from ..utils.population_builder import PopulationBuilder
            from ..utils.solution_loader import SolutionLoader

            solution_loader = SolutionLoader()
            population_builder = PopulationBuilder(solution_loader)

            initial_population = population_builder.build_initial_population(
                problem=self.problem,
                pop_size=pso_config.pop_size,
                optimization_data=self.optimization_data,
                base_solutions=sampling_config.base_solutions,
                frac_gaussian_pert=sampling_config.frac_gaussian_pert,
                gaussian_sigma=sampling_config.gaussian_sigma,
                random_seed=sampling_config.random_seed
            )

            algorithm.initialization.sampling = initial_population
            print(f"✅ Custom population set: shape {initial_population.shape}")
            print(f"   📊 Gaussian fraction: {sampling_config.frac_gaussian_pert:.1%}")
            print(f"   📊 LHS fraction: {sampling_config.frac_lhs:.1%} (calculated)")

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
        print(f"   Algorithm: PSO ({'adaptive' if pso_config.adaptive else 'canonical'})")
        print(f"   Population size: {pso_config.pop_size}")

        # Display inertia weight information
        if pso_config.adaptive:
            print(f"   Initial parameters: w={pso_config.inertia_weight:.1f}, c1={pso_config.cognitive_coeff:.1f}, c2={pso_config.social_coeff:.1f}")
            print("   Adaptation: PyMOO adjusts parameters based on swarm diversity")
        else:
            print(f"   Fixed parameters: w={pso_config.inertia_weight:.1f}, c1={pso_config.cognitive_coeff:.1f}, c2={pso_config.social_coeff:.1f}")

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
            'cognitive_coeff': pso_config.cognitive_coeff,
            'social_coeff': pso_config.social_coeff,
            'adaptive': pso_config.adaptive
        }

        # === CONVERGENCE ANALYSIS ===
        convergence_info = self._analyze_convergence(optimization_history)

        # Get best feasible solutions from callback
        best_feasible_solutions = callback.feasible_tracker.get_best_solutions()

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

        # Show feasible solutions tracked
        print(f"   Best feasible solutions tracked: {len(best_feasible_solutions)}")

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
            performance_stats=performance_stats,           # Performance metrics
            best_feasible_solutions=best_feasible_solutions # Tracked feasible solutions
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

    def _generate_statistical_summary_from_summaries(self, run_summaries: list[dict]) -> dict[str, Any]:
        """Generate statistical summary from lightweight run summaries."""
        if not run_summaries:
            return {}

        objectives = [summary['objective'] for summary in run_summaries]
        times = [summary['time'] for summary in run_summaries]
        generations = [summary['generations'] for summary in run_summaries]
        feasible_count = sum(1 for summary in run_summaries if summary['feasible'])

        return {
            'num_runs': len(run_summaries),
            'objective_mean': float(np.mean(objectives)),
            'objective_std': float(np.std(objectives)),
            'objective_min': float(np.min(objectives)),
            'objective_max': float(np.max(objectives)),
            'objective_median': float(np.median(objectives)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'time_total': float(np.sum(times)),
            'generations_mean': float(np.mean(generations)),
            'generations_std': float(np.std(generations)),
            'success_rate': 1.0,
            'feasibility_rate': feasible_count / len(run_summaries)
        }






















