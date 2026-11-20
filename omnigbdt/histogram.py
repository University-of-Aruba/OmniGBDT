from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import ceil
from numbers import Integral
from operator import index

import numpy as np


def _construct_bin_column_impl(x: np.array, max_bins: int) -> np.array:
    """Choose increasing cuts between distinct finite feature values.

    Args:
        x: Nonempty, finite float64 feature column.
        max_bins: Maximum number of bins in the native histogram.

    Returns:
        Array of at most ``max_bins - 1`` inclusive upper bounds.
    """
    sum_cnt = len(x)
    x, cnt = np.unique(x, return_counts=True)
    if len(x) == 1 or max_bins == 1:
        return np.array([], 'float64')
    if len(x) <= max_bins:
        lower, upper = x[:-1], x[1:]
        if len(x) == 2:
            weight = cnt[1] / sum_cnt
            if np.max(np.abs(x)) < np.finfo(np.float64).max / sum_cnt:
                cuts = (lower * cnt[0] + upper * cnt[1]) / sum_cnt
            elif np.signbit(x[0]) == np.signbit(x[1]):
                cuts = lower + (upper - lower) * weight
            else:
                cuts = lower * (cnt[0] / sum_cnt) + upper * weight
        else:
            cuts = lower * 0.5 + upper * 0.5
        return np.clip(cuts, lower, np.nextafter(upper, lower))

    if len(x) == sum_cnt:
        # Distinct values are divided into equal-count bins, with larger bins first.
        size, extra = divmod(len(x), max_bins)
        edges = np.arange(1, max_bins)
        return x[size * edges + np.minimum(edges, extra) - 1]

    cumulative = np.cumsum(cnt)
    cuts = np.empty(max_bins - 1, dtype=np.float64)
    target = len(x) / float(max_bins)
    for used in range(max_bins - 1):
        # Reserve one distinct value for each remaining bin, including the last.
        last = len(x) - max_bins + used
        i = cumulative.searchsorted(ceil(target))
        if i >= last:
            cuts[used:] = x[last:-1]
            break
        cuts[used] = x[i]
        seen = int(cumulative[i])
        target = seen + (sum_cnt - seen) / (max_bins - used - 1)
    return cuts


def construct_bin_column(x: np.array, max_bins: int) -> np.array:
    """Build native histogram cuts for a finite numeric feature column.

    Args:
        x: One-dimensional input values.
        max_bins: Maximum bin count, from 1 through 65,535.

    Returns:
        Sorted float64 thresholds suitable for native tree comparisons.

    Raises:
        ValueError: If the column or the requested bin count is invalid.
    """
    if not isinstance(max_bins, Integral) or not 1 <= max_bins <= np.iinfo(np.uint16).max:
        raise ValueError("max_bins must be an integer between 1 and 65535.")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("A feature column must be one-dimensional, nonempty and finite.")

    return _construct_bin_column_impl(x, int(max_bins))


def map_bin_column(x, bins):
    """Map feature values to the first bin with an inclusive upper bound.

    Args:
        x: Feature values to assign.
        bins: Sorted cuts returned by ``construct_bin_column``.

    Returns:
        Array of uint16 bin indices with the same shape as ``x``.
    """
    return np.searchsorted(bins, x, side="left").astype(np.uint16)


def _get_bins_maps(x_column: np.array, max_bins: int) -> tuple:
    """Construct native histogram cuts and row assignments for one input column.

    Args:
        x_column: Nonempty one-dimensional array of feature values.
        max_bins: Maximum number of bins.

    Returns:
        Increasing cuts and the uint16 index for each row.
    """
    bins = construct_bin_column(x_column, max_bins)
    maps = map_bin_column(x_column, bins)

    return (bins, maps)


def get_bins_maps(x: np.array, max_bins: int, threads: int = 1) -> (list, np.array):
    """Construct histogram cuts and bin indices for each feature independently.

    Args:
        x: Numeric matrix of shape (n_samples, n_features).
        max_bins: Maximum number of bins per feature, from 1 through 65,535.
        threads: Number of workers used to construct the histograms.

    Returns:
        Pair containing the per-feature cuts and a uint16 map shaped like ``x``.

    Raises:
        ValueError: If the matrix, bin budget, or worker count is invalid.
        TypeError: If the parallel worker count is not integral.
    """
    x = np.asarray(x)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("Features must be a nonempty two-dimensional matrix.")
    build_column = partial(_get_bins_maps, max_bins=max_bins)
    if threads == 1:
        columns = map(build_column, x.T)
        bins, maps = zip(*columns)
    else:
        workers = None if threads is None else index(threads)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            bins, maps = zip(*pool.map(build_column, x.T))
    return list(bins), np.stack(maps, axis=1)


if __name__ == '__main__':
    x = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(x, 8, 2)
    bin = bins[0]
    print(bin)
