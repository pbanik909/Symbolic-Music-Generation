"""Pitch-distribution metrics for evaluating generated music."""
import numpy as np


def pitch_histogram(binary):
    """Return the normalized distribution of note pitches in a piano-roll."""
    counts = binary.sum(axis=0).astype(float)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    return counts / total


def pitch_histogram_distance(a, b):
    """L1 distance between two pitch histograms (lower = more similar)."""
    ha, hb = pitch_histogram(a), pitch_histogram(b)
    return float(np.abs(ha - hb).sum())
