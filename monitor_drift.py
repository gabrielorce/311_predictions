# monitor_drift.py
import numpy as np
import pandas as pd

def psi(expected, actual, buckets=10):
    """
    Calculate PSI for numerical or categorical features.
    For categorical, compare frequency distribution.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Detect categorical vs numerical
    if expected.dtype == 'object' or expected.nunique() < buckets:
        return _psi_categorical(expected, actual)
    else:
        return _psi_numerical(expected, actual, buckets)

def _psi_numerical(expected, actual, buckets):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)
    return np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))

def _psi_categorical(expected, actual):
    expected_dist = expected.value_counts(normalize=True)
    actual_dist = actual.value_counts(normalize=True)
    categories = set(expected_dist.index).union(actual_dist.index)
    psi_val = 0
    for cat in categories:
        e_perc = expected_dist.get(cat, 0.0001)
        a_perc = actual_dist.get(cat, 0.0001)
        psi_val += (a_perc - e_perc) * np.log(a_perc / e_perc)
    return psi_val
