"""Public API and numerical contracts for Gaussian split projections."""

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


def _sample():
    """Make four distinct regression outputs with competing split directions.

    Returns:
        Float64 feature and target matrices with 180 rows.
    """
    rng = np.random.default_rng(340)
    x = rng.normal(size=(180, 5))
    y = np.column_stack((x[:, 0], x[:, 1], x[:, 2] * x[:, 3], x[:, 4] ** 2))
    return x, y


def _model(**options):
    """Construct a dense four-target booster with overridable training settings.

    Args:
        **options: Native model parameters for this fit.

    Returns:
        An untrained multi-output model.
    """
    settings = dict(max_depth=3, min_samples=3, num_threads=1, verbosity=0, early_stop=0)
    return MultiOutputGBDT(out_dim=4, params={**settings, **options})


@pytest.mark.parametrize("metric", [None, (np.eye(4) + np.ones((4, 4)) / 4) / 8])
def test_projected_tree_fits_all_original_outputs(metric):
    """Recover the four analytic regularized leaf values from one split channel.

    Args:
        metric: Ordinary or coupled loss defining the split geometry.
    """
    x = np.repeat([0., 1.], 48)[:, None]
    y = np.repeat([[-2., 1., 4., 6.], [5., 9., 3., -1.]], 48, axis=0)
    model = _model(projection_dim=1, output_metric=metric, max_depth=1, gamma=0.,
                   reg_l2=3., base_score=0., lr=1., seed=17)
    model.set_data((x, y))
    model.train(1)
    np.testing.assert_allclose(model.predict(x), y * 48 / 51, rtol=1e-14)
    model.close()


def test_projection_respects_the_metric_before_scoring_splits():
    """Exclude a large zero-weight direction from split selection while fitting its leaf."""
    x = np.repeat([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], 12, axis=0)
    a, b = x.T
    y = np.column_stack((1e6 * b + 3 * a, a, 2 * a, 4e5 * b))
    model = _model(projection_dim=1, output_metric=[0., 1., 2., 0.], seed=17,
                   max_depth=1, gamma=0., reg_l2=0., base_score=0., lr=1.)
    model.set_data((x, y))
    model.train(1)
    np.testing.assert_allclose(model.predict(x), a[:, None] * [3., 1., 2., 0.], atol=1e-12)
    model.close()


@pytest.mark.parametrize("metric,width", [
    (None, 4), (np.eye(4) / 4, 4), ([0., 2., 0., 1.], 2),
    (np.ones((4, 4)), 1),
])
def test_full_width_preserves_exact_tree_builder(metric, width, tmp_path):
    """Require the exact default tree ensemble when the metric fits in the requested width.

    Args:
        metric: Output metric, possibly singular.
        width: Requested projection width covering all active directions.
        tmp_path: Directory for comparing native tree files.
    """
    x, y = _sample()
    dumps = []
    for requested in (None, width):
        model = _model(projection_dim=requested, output_metric=metric, subsample=.75, seed=17)
        model.set_data((x, y))
        model.train(7)
        path = tmp_path / f"trees-{requested}.txt"
        model.dump(path)
        dumps.append(path.read_bytes())
        model.close()
    assert dumps[0] == dumps[1]


def test_equal_output_weights_preserve_projected_mse():
    """Keep equal-weight and ordinary split scores identical with projection active."""
    x, y = _sample()
    predictions = []
    for metric in (None, np.ones(4) / 4, np.eye(4) / 4):
        model = _model(projection_dim=2, seed=17, output_metric=metric)
        model.set_data((x, y))
        model.train(7)
        predictions.append(model.predict(x))
        model.close()
    for weighted in predictions[1:]:
        np.testing.assert_array_equal(weighted, predictions[0])


def test_projection_seed_and_continuation(tmp_path):
    """Replay the retained tree index across train, dump/load, and OpenMP team sizes.

    Args:
        tmp_path: Directory for checking the inference and continued-training format.
    """
    x, y = _sample()
    settings = dict(projection_dim=2, seed=0, subsample=.7)
    whole = _model(**settings)
    whole.set_data((x, y))
    whole.train(9)
    split = _model(**settings, num_threads=2)
    split.set_data((x, y))
    split.train(4)
    path = tmp_path / "projected.txt"
    split.dump(path)
    resumed = _model(**settings)
    resumed.set_data((x, y))
    resumed.load(path)
    plain = _model()
    plain.set_booster(x.shape[1], y.shape[1])
    plain.load(path)
    np.testing.assert_allclose(plain.predict(x), split.predict(x), rtol=1e-12, atol=1e-12)
    split.train(5)
    resumed.train(5)
    np.testing.assert_array_equal(whole.predict(x), split.predict(x))
    np.testing.assert_allclose(whole.predict(x), resumed.predict(x), rtol=1e-12, atol=1e-12)
    for model in (whole, split, resumed, plain):
        model.close()


def test_projection_varies_with_seed_on_full_rows():
    """Detect a disabled seed-zero sketch or an RNG that ignores the specified seed."""
    x, y = _sample()
    predictions = []
    for dim, seed in [(None, 0), (1, 0), (1, 17)]:
        model = _model(projection_dim=dim, seed=seed)
        model.set_data((x, y))
        model.train(7)
        predictions.append(model.predict(x))
        model.close()
    assert not np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[1], predictions[2])


@pytest.mark.parametrize("metric", [None, (np.eye(4) + np.ones((4, 4)) / 4) / 8])
def test_early_stopping_monitors_original_loss(metric, capfd):
    """Find the best full-objective prefix independently of the split channels.

    Args:
        metric: Ordinary or coupled quadratic loss used for early stopping.
        capfd: Captured validation metrics from the native booster.
    """
    x, y = _sample()
    params = dict(projection_dim=1, seed=29, output_metric=metric, base_score=0., lr=.2)
    reference = _model(**params)
    reference.set_data((x, y))
    reference.train(24)
    # The validation labels equal the third ensemble prediction on deliberately
    # different rows, making the best retained round unambiguous.
    xv = x[::3] + .01
    target = reference.predict(xv, 3)
    stopped = _model(**params, early_stop=3, verbosity=2)
    stopped.set_data((x, y), (xv, target))
    stopped.train(24)
    residual = reference.predict(xv, 1) - target
    squared_errors = residual ** 2 if metric is None else np.einsum("ij,jk,ik->i", residual, metric, residual)
    first_score = np.sqrt(squared_errors.mean())
    assert f"eval->{first_score:.5f}" in capfd.readouterr().out.splitlines()[0]
    assert stopped.lib.GetTreeCount(stopped._boostnode) == 3
    np.testing.assert_array_equal(stopped.predict(xv), target)
    np.testing.assert_array_equal(stopped.preds_eval, target)
    stopped.train(4)
    np.testing.assert_array_equal(stopped.predict(xv, 4), reference.predict(xv, 4))
    reference.close()
    stopped.close()


@pytest.mark.parametrize("invalid", [0, -1, 5, True, np.bool_(True), 1.5])
def test_invalid_projection_preserves_fitted_booster(invalid):
    """Reject invalid widths without replacing a working booster.

    Args:
        invalid: Width outside the public integer contract.
    """
    x, y = _sample()
    model = _model()
    model.set_data((x, y))
    model.train(2)
    expected = model.predict(x)
    model.params["projection_dim"] = invalid
    with pytest.raises((TypeError, ValueError), match="projection_dim"):
        model.set_data((x, y))
    np.testing.assert_array_equal(model.predict(x), expected)
    model.close()


@pytest.mark.parametrize("cls,width,options", [
    (SingleOutputGBDT, 1, {}), (MultiOutputGBDT, 1, {}),
    (MultiOutputGBDT, 4, {"topk": 1}), (MultiOutputGBDT, 4, {"loss": b"ce"}),
    (MultiOutputGBDT, 4, {"reg_l1": .1}),
])
def test_unsupported_projection_configuration(cls, width, options):
    """Reject incompatible boosting methods before fitting.

    Args:
        cls: Low-level estimator type.
        width: Configured output width.
        options: Additional incompatible model options.
    """
    with pytest.raises(ValueError, match="projection_dim"):
        model = cls(out_dim=width, params={"projection_dim": 1, **options})
        model.set_booster(2, width)


def test_manual_projection_training_rejected():
    """Reject external gradients and retain a usable built-in training path."""
    x, y = _sample()
    model = _model(projection_dim=2)
    model.set_data((x, y))
    with pytest.raises(ValueError, match="projection_dim"):
        model.train(1, objective=lambda pred, target: (pred - target, np.ones_like(pred)))
    with pytest.raises(ValueError, match="projection_dim"):
        model._set_gh(y, np.ones_like(y))
    model.train(2)
    assert model.lib.GetTreeCount(model._boostnode) == 2
    model.close()


def test_sklearn_projection_matches_native_and_clones():
    """Fit the public estimator, compare native predictions and preserve the width on clone."""
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from omnigbdt.sklearn import MultiOutputGBDTRegressor

    x, y = _sample()
    options = dict(projection_dim=np.int64(2), seed=17, max_depth=3, min_samples=3,
                   early_stop=0, num_threads=1, verbosity=0)
    native = _model(**options)
    native.set_data((x, y))
    native.train(6)
    wrapper = MultiOutputGBDTRegressor(num_rounds=6, **options)
    assert clone(wrapper).get_params()["projection_dim"] == 2
    wrapper.fit(x, y)
    np.testing.assert_array_equal(wrapper.predict(x), native.predict(x))
    native.close()
    wrapper.close()
