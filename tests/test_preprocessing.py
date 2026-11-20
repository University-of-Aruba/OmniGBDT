import numpy as np
import pytest

from omnigbdt import SingleOutputGBDT, get_bins_maps


def test_skewed_histogram_agrees_with_native_prediction():
    """Keep ordered, usable cuts and replay training rows into the same leaves."""
    x = np.repeat(np.arange(1.0, 6.0), [1, 1, 1, 1, 100]).reshape(-1, 1)
    y = (x[:, 0] <= 2.0).astype(np.float64)
    bins, maps = get_bins_maps(x, max_bins=4)
    assert np.all(np.diff(bins[0]) > 0)
    for index, cut in enumerate(bins[0]):
        np.testing.assert_array_equal(maps[:, 0] <= index, x[:, 0] <= cut)

    booster = SingleOutputGBDT(
        params={"max_bins": 4, "max_depth": 1, "num_threads": 1, "min_samples": 1, "verbosity": 0}
    )
    try:
        booster.set_data((x, y))
        booster.train(1)
        preds = booster.predict(x)
        assert preds[0] > preds[-1]
        np.testing.assert_allclose(preds, booster.preds_train)
    finally:
        booster.close()


@pytest.mark.parametrize(("counts", "max_bins", "expected_cuts"), [
    ([1] * 10, 4, [2, 5, 7]),
    ([1, 4, 1, 1, 2, 1, 3, 1, 1, 1], 4, [1, 4, 6]),
    ([10] * 64, 4, [15, 31, 47]),
    ([1, 1, 1, 1, 1, 1, 1, 1, 40, 1], 6, [4, 5, 6, 7, 8]),
], ids=["distinct", "repeated", "equal-frequency", "reserve-remaining-values"])
def test_histogram_cut_placement_across_workers(counts, max_bins, expected_cuts):
    """Check row-count quantiles and inclusive assignments across workers.

    Args:
        counts: Occurrences of successive distinct input values.
        max_bins: Histogram bin budget below the distinct value count.
        expected_cuts: Inclusive upper bounds after reserving distinct values for later bins.
    """
    values = np.repeat(np.arange(len(counts), dtype=np.float64), counts)
    x = np.column_stack([values[::-1], values + 100])
    expected_cuts = np.asarray(expected_cuts, dtype=np.float64)
    expected_maps = np.column_stack([
        np.searchsorted(expected_cuts, values[::-1]),
        np.searchsorted(expected_cuts, values),
    ])
    for workers in (1, 2):
        cuts, maps = get_bins_maps(x, max_bins, threads=workers)
        np.testing.assert_array_equal(cuts[0], expected_cuts)
        np.testing.assert_array_equal(cuts[1], expected_cuts + 100)
        np.testing.assert_array_equal(maps, expected_maps)
        assert maps.dtype == np.uint16


@pytest.mark.parametrize(("workers", "error"), [(0, ValueError), (1.5, TypeError)])
def test_invalid_histogram_worker_count(workers, error):
    """Reject zero and fractional worker counts before running histogram tasks.

    Args:
        workers: Unsupported worker limit.
        error: Expected exception type.
    """
    with pytest.raises(error):
        get_bins_maps(np.ones((2, 2)), 2, threads=workers)


@pytest.mark.parametrize("values", [
    [1e308, 1.1e308],
    [-1.1e308, -1e308, 1e308],
    [np.nextafter(1.0, 2.0), np.nextafter(np.nextafter(1.0, 2.0), 2.0)],
])
def test_distinct_float_values_stay_separable(values):
    """Keep a representable cut between each distinct pair of feature values.

    Args:
        values: Ordered finite inputs that test overflow or midpoint rounding.
    """
    x = np.array(values)[:, None]
    bins, maps = get_bins_maps(x, max_bins=4)
    np.testing.assert_array_equal(maps[:, 0], np.arange(len(x)))
    assert np.isfinite(bins[0]).all()


def test_bin_budget_fits_native_count():
    """Accept one and the largest native bin count and reject out-of-range budgets."""
    x = np.arange(5.0)[:, None]
    bins, maps = get_bins_maps(x, max_bins=1)
    assert bins[0].size == 0
    np.testing.assert_array_equal(maps, np.zeros_like(x))
    _, maps = get_bins_maps(x, max_bins=65535)
    np.testing.assert_array_equal(maps[:, 0], np.arange(len(x)))
    for unsupported in (0, 65536):
        with pytest.raises(ValueError, match="max_bins"):
            get_bins_maps(x, max_bins=unsupported)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-np.inf], []])
def test_invalid_feature_column_is_rejected(values):
    """Reject feature columns that cannot define a finite native histogram.

    Args:
        values: Empty column or a column containing a nonfinite observation.
    """
    with pytest.raises(ValueError, match="finite|nonempty"):
        get_bins_maps(np.array(values).reshape(-1, 1), max_bins=4)


@pytest.mark.parametrize("estimator_name", ["SingleOutputGBDTRegressor", "MultiOutputGBDTRegressor"])
def test_sklearn_failed_fit_preserves_fitted_state(estimator_name):
    """Publish trained boosters atomically and clear learned attributes on close.

    Args:
        estimator_name: Public estimator class whose fit lifecycle is exercised.
    """
    pytest.importorskip("sklearn")
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    from omnigbdt import sklearn as wrappers

    def reject_round(preds, targets):
        """Interrupt a fit from the user objective.

        Args:
            preds: Current training predictions.
            targets: Training labels.

        Raises:
            ValueError: Always raised to simulate an unsuccessful training round.
        """
        raise ValueError("training interrupted")

    x = np.arange(16.0).reshape(-1, 1)
    y = x[:, 0]
    estimator = getattr(wrappers, estimator_name)(num_threads=1, num_rounds=1, min_samples=1)
    estimator.set_params(objective=reject_round)
    with pytest.raises(ValueError, match="training interrupted"):
        estimator.fit(x, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)

    estimator.set_params(objective=None).fit(x, y)
    expected = estimator.predict(x)
    estimator.set_params(objective=reject_round)
    with pytest.raises(ValueError, match="training interrupted"):
        estimator.fit(np.column_stack((x, x)), y)
    np.testing.assert_array_equal(estimator.predict(x), expected)
    estimator.close()
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)
