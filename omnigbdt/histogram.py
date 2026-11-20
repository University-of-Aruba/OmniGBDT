import numpy as np
from multiprocessing import Pool
from functools import partial
from numbers import Integral

try:
    from numba import jit
except ImportError:
    jit = None


def _construct_bin_column_impl(x: np.array, max_bins: int) -> np.array:
    """Choose increasing cuts between distinct finite feature values.

    Args:
        x: Nonempty, finite float64 feature column.
        max_bins: Maximum number of bins in the native histogram.

    Returns:
        Array of at most ``max_bins - 1`` inclusive upper bounds.
    """
    x, cnt = np.unique(x, return_counts=True)
    sum_cnt = np.sum(cnt)
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

    cumulative = np.cumsum(cnt)
    cuts = np.empty(max_bins - 1, dtype=np.float64)
    target = len(x) / float(max_bins)
    used = 0
    for i in range(len(x) - 1):
        # Reserve one distinct value for each remaining bin, including the last.
        if cumulative[i] >= target or len(x) - i == max_bins - used:
            cuts[used] = x[i]
            used += 1
            if used == len(cuts):
                break
            target = cumulative[i] + (sum_cnt - cumulative[i]) / (max_bins - used)
    return cuts


if jit is not None:
    _construct_bin_column_jitted = jit(forceobj=True)(_construct_bin_column_impl)
else:
    _construct_bin_column_jitted = None


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
    global _construct_bin_column_jitted

    if not isinstance(max_bins, Integral) or not 1 <= max_bins <= np.iinfo(np.uint16).max:
        raise ValueError("max_bins must be an integer between 1 and 65535.")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("A feature column must be one-dimensional, nonempty and finite.")

    if _construct_bin_column_jitted is None:
        return _construct_bin_column_impl(x, max_bins)

    try:
        return _construct_bin_column_jitted(x, max_bins)
    except Exception:
        # Keep the Python path available when Numba is unavailable or unstable.
        _construct_bin_column_jitted = None
        return _construct_bin_column_impl(x, max_bins)


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
    """
    x = np.asarray(x)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError("Features must be a nonempty two-dimensional matrix.")
    out = []
    if threads == 1:
        for i in range(x.shape[-1]):
            out.append(_get_bins_maps(x[:, i], max_bins))
    else:
        x = list(np.transpose(x))
        f = partial(_get_bins_maps, max_bins=max_bins)
        with Pool(threads) as pool:
            out = pool.map(f, x)

    bins, maps = [], []
    while out:
        _bin, _map = out.pop(0)
        bins.append(_bin)
        maps.append(_map)
    return bins, np.stack(maps, axis=1)


if __name__ == '__main__':
    x = np.random.rand(10000, 10)
    bins, maps = get_bins_maps(x, 8, 2)
    bin = bins[0]
    print(bin)
