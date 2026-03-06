"""
Population builder for custom PSO sampling.

This module implements the pre-built population array approach for PyMOO sampling.
It creates complete (pop_size, n_var) arrays that can be passed directly to PSO algorithms.

It is used when we want to pass an initial population to the PSO problem, instead of relying
exclusively on Latin Hypercube Sampling
"""

import logging
from typing import Any

import numpy as np

from .solution_loader import SolutionLoader

logger = logging.getLogger(__name__)


class PopulationBuilder:
    """
    Build complete initial populations for PSO optimization using base solutions + perturbations + LHS.

    This class implements the "Pre-built Full Population Array" approach where we construct
    the complete (pop_size, n_var) initial population before passing it to PyMOO algorithms.
    This approach is more reliable and explicit than custom Sampling classes.

    Population Composition:
    1. Base solutions (loaded from various sources)
    2. Gaussian perturbations around base solutions
    3. LHS samples to fill remaining population slots

    Example:
        pop_size=40, base_solutions=2, frac_gaussian_pert=0.6
        → 2 base + 24 gaussian perturbations + 14 LHS samples = 40 total
    """

    def __init__(self, solution_loader: SolutionLoader):
        """Initialize population builder with solution loader."""
        self.solution_loader = solution_loader

    def build_initial_population(
        self,
        problem,  # TransitOptimizationProblem instance
        pop_size: int,
        optimization_data: dict[str, Any],
        base_solutions: str | list,
        frac_gaussian_pert: float = 0.6,
        gaussian_sigma: float = 1.0,
        frac_reductions: float = 0.0,
        reduction_sigma: float = 1.0,
        random_seed: int = None,
    ) -> np.ndarray:
        """
        Build complete (pop_size, n_var) initial population array.

        Args:
            problem: TransitOptimizationProblem instance for encoding/bounds
            pop_size: Total population size required
            optimization_data: Complete optimization data structure
            base_solutions: Solution specification ('from_data' or list)
            frac_gaussian_pert: Fraction of population for gaussian perturbations
            gaussian_sigma: Standard deviation for gaussian perturbations
            frac_reductions: Fraction of population for service reductions
            reduction_sigma: Standard deviation for service reductions (how much to shift index)
            random_seed: Random seed for reproducibility

        Returns:
            Complete population array of shape (pop_size, n_var)
        """

        logger.info("🔧 Building initial population (size=%d):", pop_size)

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info("   🎲 Random seed set to: %d", random_seed)

        # Step 1: Load and encode base solutions
        base_solutions = self.solution_loader.load_solutions(base_solutions, optimization_data)
        encoded_base_solutions = []

        for i, solution in enumerate(base_solutions):
            encoded = problem.encode_solution(solution)
            encoded_base_solutions.append(encoded)
            logger.info("   📋 Base solution %d: shape %s", i + 1, encoded.shape)

        n_base = len(encoded_base_solutions)
        logger.info("   ✅ Loaded %d base solutions", n_base)

        # Step 2: Calculate population distribution
        n_gaussian = int(frac_gaussian_pert * pop_size)
        n_reductions = int(frac_reductions * pop_size)
        n_lhs = pop_size - n_base - n_gaussian - n_reductions

        if n_lhs < 0:
            raise ValueError(
                f"Population size ({pop_size}) too small for {n_base} base + {n_gaussian} gaussian + {n_reductions} reductions"
            )

        logger.info(f"""
           📊 Population distribution:
            • Base solutions: {n_base}
            • Gaussian perturbations: {n_gaussian}
            • Service reductions: {n_reductions}
            • LHS samples: {n_lhs}
        """)

        # Step 3: Build population components
        population_parts = []

        # Add base solutions
        population_parts.extend(encoded_base_solutions)

        # Generate gaussian perturbations
        if n_gaussian > 0:
            gaussian_samples = self._generate_gaussian_perturbations(
                encoded_base_solutions, n_gaussian, problem, gaussian_sigma
            )
            population_parts.extend(gaussian_samples)

        # Generate service reductions
        if n_reductions > 0:
            reduction_samples = self._generate_service_reductions(
                encoded_base_solutions, n_reductions, problem, reduction_sigma
            )
            population_parts.extend(reduction_samples)

        # Generate LHS samples
        if n_lhs > 0:
            lhs_samples = self._generate_lhs_samples(n_lhs, problem)
            population_parts.extend(lhs_samples)

        # Step 4: Stack into final array
        final_population = np.vstack(population_parts)

        logger.info("Final population shape: %s", final_population.shape)
        logger.info("Ready for PSO algorithm")

        return final_population

    def _generate_gaussian_perturbations(
        self, base_solutions: list[np.ndarray], n_perturbations: int, problem, sigma: float
    ) -> list[np.ndarray]:
        """Generate gaussian perturbations around base solutions."""
        perturbations = []

        if not base_solutions:
            return perturbations

        # Distribute perturbations across base solutions
        perturbations_per_base = n_perturbations // len(base_solutions)
        extra_perturbations = n_perturbations % len(base_solutions)

        for i, base_solution in enumerate(base_solutions):
            # Calculate perturbations for this base solution
            n_for_this_base = perturbations_per_base
            if i < extra_perturbations:
                n_for_this_base += 1

            for _ in range(n_for_this_base):
                # Generate gaussian noise
                noise = np.random.normal(0, sigma, size=base_solution.shape)
                perturbed = base_solution + noise

                # Apply bounds and convert to integers
                perturbed = np.clip(perturbed, problem.xl, problem.xu)
                perturbed = np.round(perturbed).astype(int)

                perturbations.append(perturbed)

        return perturbations

    def _generate_service_reductions(
        self, base_solutions: list[np.ndarray], n_reductions: int, problem, sigma: float = 1.0
    ) -> list[np.ndarray]:
        """
        Generate service reduction samples (biased perturbations).

        This moves headway indices to higher values (lower frequency) to help
        satisfy tight fleet constraints.

        Args:
            base_solutions: List of base solution arrays
            n_reductions: Number of reduction samples to generate
            problem: TransitOptimizationProblem instance
            sigma: Standard deviation for positive perturbation magnitude
        """
        reductions = []

        if not base_solutions:
            return reductions

        # 1. Determine number of PT variables
        # We only want to reduce PT service (increase headways), not DRT
        n_vars = base_solutions[0].shape[0]
        n_pt_vars = n_vars  # Default to all if not specified

        # If problem exposes structure, use it to isolate PT variables
        # (Assuming PT variables are always first in the flattened array, which is standard in this codebase)
        if hasattr(problem, "drt_enabled") and problem.drt_enabled:
            if hasattr(problem, "active_intervals"):
                n_intervals_active = len(problem.active_intervals)
                n_pt_vars = problem.n_routes * n_intervals_active
            else:
                # Fallback if attribute missing
                n_pt_vars = problem.n_routes * problem.n_intervals

        # Distribute reductions across base solutions
        reductions_per_base = n_reductions // len(base_solutions)
        extra_reductions = n_reductions % len(base_solutions)

        for i, base_solution in enumerate(base_solutions):
            # Calculate count for this base
            n_for_this_base = reductions_per_base
            if i < extra_reductions:
                n_for_this_base += 1

            # Prepare noise array sizes
            n_drt_vars = n_vars - n_pt_vars

            for _ in range(n_for_this_base):
                # Generate POSITIVE gaussian noise (absolute value) for PT variables ONLY
                noise_pt = np.abs(np.random.normal(0, sigma, size=n_pt_vars))

                # Zero noise for DRT variables (keep DRT as is)
                noise_drt = np.zeros(n_drt_vars)

                # Combine noise
                noise = np.concatenate([noise_pt, noise_drt])

                # Round to integers
                noise = np.round(noise).astype(int)

                # Apply noise: Original + Positive Noise
                perturbed = base_solution + noise

                # Apply bounds (only upper bound matters here as we are increasing)
                perturbed = np.clip(perturbed, problem.xl, problem.xu)

                # Ensure integer type
                perturbed = perturbed.astype(int)

                reductions.append(perturbed)

        return reductions

    def _generate_lhs_samples(self, n_samples: int, problem) -> list[np.ndarray]:
        """Generate Latin Hypercube samples to fill remaining population."""
        if n_samples <= 0:
            return []

        # Use pymoo's LHS implementation
        from pymoo.operators.sampling.lhs import LHS

        lhs_sampler = LHS()

        # Get problem bounds
        xl, xu = problem.bounds()

        # Generate LHS samples - this returns a 2D numpy array directly
        lhs_samples = lhs_sampler._do(problem=problem, n_samples=n_samples, random_state=np.random.RandomState())

        # Round to integers since we need discrete choices
        lhs_samples = np.round(lhs_samples).astype(int)

        # Ensure bounds are respected after rounding
        lhs_samples = np.clip(lhs_samples, xl, xu)

        # Convert to list of arrays for consistency with other methods
        return [lhs_samples[i] for i in range(n_samples)]
