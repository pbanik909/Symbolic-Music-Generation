"""Rhythm and repetition metrics for evaluating generated music."""
import numpy as np


def rhythm_diversity(binary):
    """Fraction of distinct onset patterns across timesteps (higher = more varied)."""
    onsets = [tuple(np.where(binary[t] == 1)[0]) for t in range(binary.shape[0])]
    if not onsets:
        return 0.0
    return len(set(onsets)) / len(onsets)


def repetition_ratio(binary):
    """Fraction of consecutive timesteps that are identical (higher = more repetitive)."""
    if binary.shape[0] < 2:
        return 0.0
    same = sum(np.array_equal(binary[t], binary[t + 1]) for t in range(binary.shape[0] - 1))
    return same / (binary.shape[0] - 1)
