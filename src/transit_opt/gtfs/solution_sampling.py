
import numpy as np

from transit_opt.optimisation.config.config_manager import \
    SolutionSamplingStrategyConfig


def generate_sampling_ranks(config: SolutionSamplingStrategyConfig) -> list[int]:
    """
    Generate list of integer ranks to sample based on strategy.

    Args:
        config: Sampling strategy configuration

    Returns:
        List of integer ranks (0-indexed, guaranteed unique and sorted)

    Raises:
        ValueError: If configuration is invalid
    """
    if config.type == "uniform":
        return _uniform_sampling(config.max_to_save, config.max_rank)

    elif config.type == "power":
        return _power_law_sampling(config.max_to_save, config.max_rank, config.power_exponent)

    elif config.type == "geometric":
        return _geometric_sampling(config.max_to_save, config.max_rank, config.geometric_base)

    elif config.type == "fibonacci":
        return _fibonacci_sampling(config.max_to_save, config.max_rank)

    elif config.type == "manual":
        return _manual_sampling(config.manual_ranks, config.max_rank)

    else:
        raise ValueError(f"Unknown sampling type: {config.type}")


def _uniform_sampling(max_to_save: int, max_rank: int) -> list[int]:
    """Uniform spacing."""
    if max_to_save == 1:
        return [0]

    ranks = np.linspace(0, max_rank - 1, max_to_save)
    return sorted(set(np.round(ranks).astype(int).tolist()))[:max_to_save]


def _power_law_sampling(max_to_save: int, max_rank: int, exponent: float) -> list[int]:
    """Power law - clusters samples near rank 0 (best solutions)."""
    if max_to_save == 1:
        return [0]

    # Generate normalized positions [0, 1]
    positions = np.linspace(0, 1, max_to_save)

    # Apply power law: positions^exponent gives clustering at 0
    # Higher exponent = more clustering at head
    ranks = (max_rank - 1) * np.power(positions, exponent)

    # Round to integers and ensure uniqueness
    ranks = np.round(ranks).astype(int)
    ranks = np.unique(ranks)

    return ranks.tolist()[:max_to_save]


def _geometric_sampling(max_to_save: int, max_rank: int, base: float) -> list[int]:
    """
    Geometric progression sampling (e.g., 0, 1, 2, 4, 8, 16, ...).

    Args:
        max_to_save: Number of solutions to select
        max_rank: Maximum rank available (exclusive upper bound)
        base: Geometric base (e.g., 2.0 for doubling)

    Returns:
        List of integer ranks following geometric sequence
    """
    if max_to_save == 1:
        return [0]

    ranks = []

    # Generate geometric sequence: base^0, base^1, base^2, ...
    for i in range(max_to_save):
        rank = int(round(base**i)) - 1  # -1 to make it 0-indexed

        if rank >= max_rank:
            break

        ranks.append(rank)

    # Ensure rank 0 is included and deduplicate
    if 0 not in ranks:
        ranks.insert(0, 0)

    return sorted(set(ranks))[:max_to_save]


def _fibonacci_sampling(max_to_save: int, max_rank: int) -> list[int]:
    """Fibonacci sequence sampling - exponential-like growth."""
    if max_to_save == 1:
        return [0]

    ranks = [0, 1]  # Start with 0 and 1

    # Generate Fibonacci sequence
    while len(ranks) < max_to_save:
        next_rank = ranks[-1] + ranks[-2]

        if next_rank >= max_rank:
            break

        ranks.append(next_rank)

    return sorted(set(ranks))[:max_to_save]


def _manual_sampling(manual_ranks: list[int], max_rank: int) -> list[int]:
    """
    Manual rank specification with validation.

    Args:
        manual_ranks: List of specific ranks to select
        max_rank: Maximum valid rank (for filtering)

    Returns:
        Validated, sorted list of integer ranks
    """
    if not manual_ranks:
        raise ValueError("manual_ranks cannot be empty when type='manual'")

    # Ensure integers and filter to valid range
    valid_ranks = [int(r) for r in manual_ranks if 0 <= r < max_rank]

    if not valid_ranks:
        raise ValueError(f"No valid ranks in manual_ranks (max_rank={max_rank})")

    # Return unique, sorted ranks
    return sorted(set(valid_ranks))
