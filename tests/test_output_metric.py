"""Quadratic output loss tests in the original coefficient units."""

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


def _two_modes():
    """Build a common level change and a larger alternating coefficient change.

    Returns:
        Features and two coefficient targets for four equally frequent cases.
    """
    x = np.repeat([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], 4, axis=0)
    y = 0.75 * x[:, [0]] + x[:, [1]] * np.array([1., -1.])
    return x, y


@pytest.mark.parametrize("metric,weight", [
    ([0.5, 0.5], 0.5), (np.eye(2) / 2, 0.5), (2. * np.eye(2), 2.),
])
def test_equal_weights_reproduce_ordinary_splits(metric, weight, capfd):
    """Compare fitted routes on feature combinations absent from training.

    Args:
        metric: Vector or matrix of equal diagonal weights.
        weight: Weight per target in the configured metric.
        capfd: Captures the validation error reported by native training.
    """
    group = np.repeat(np.arange(4, dtype=float), 8)
    x = np.column_stack((group, group >= 2))
    y = 100. * x[:, [1]] + np.random.default_rng(4).normal(size=(len(x), 2))
    validation = y.mean(axis=0) + 0.13 * (y - y.mean(axis=0))
    # The two candidate rules agree on training rows but disagree on these probes.
    probes = np.array([[0., 1.], [3., 0.]])
    opts = dict(max_depth=1, min_samples=2, lr=0.05, early_stop=2, num_threads=1, verbosity=2)
    ordinary = MultiOutputGBDT(out_dim=2, params=dict(opts, gamma=0.2))
    weighted = MultiOutputGBDT(out_dim=2, params=dict(opts, gamma=0.2 * 2 * weight, output_metric=metric))
    ordinary.set_data((x, y), (x, validation))
    ordinary.train(15)
    capfd.readouterr()
    weighted.set_data((x, y), (x, validation))
    weighted.train(15)
    rounds = ordinary.lib.GetTreeCount(ordinary._boostnode)
    assert 1 < rounds < 15
    assert weighted.lib.GetTreeCount(weighted._boostnode) == rounds
    for prefix in range(1, rounds + 1):
        np.testing.assert_array_equal(weighted.predict(probes, prefix), ordinary.predict(probes, prefix))
    score = np.sqrt(weight * np.mean(np.sum((weighted.predict(x) - validation)**2, axis=1)))
    assert f"eval->{score:.5f}" in capfd.readouterr().out
    ordinary.close()
    weighted.close()


@pytest.mark.parametrize("weight,gamma,should_split", [(2., 18., True), (0.125, 3., False)])
def test_uniform_metric_keeps_gamma_in_weighted_units(weight, gamma, should_split):
    """Check split acceptance above and below the gain of an ordinary stump.

    Args:
        weight: Equal per-target quadratic weight.
        gamma: Minimum gain in the configured loss units.
        should_split: Whether the weighted gain exceeds gamma.
    """
    x, y = _two_modes()
    model = MultiOutputGBDT(out_dim=2, params=dict(
        output_metric=[weight, weight], gamma=gamma, base_score=0., reg_l2=0.,
        min_samples=1, lr=1., max_depth=1, num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    # Splitting on x[:, 1] has ordinary gain 8 and weighted gain 16 * weight.
    expected = x[:, [1]] * [1., -1.] if should_split else np.zeros_like(y)
    np.testing.assert_array_equal(model.predict(x), expected)
    model.close()


def test_nearly_equal_weights_keep_target_preference():
    """Respect a small specified weight difference between equally strong targets."""
    x, _ = _two_modes()
    model = MultiOutputGBDT(out_dim=2, params=dict(
        output_metric=[0.5, 0.500001], gamma=0., base_score=0., reg_l2=0.,
        min_samples=1, lr=1., max_depth=1, num_threads=1, verbosity=0,
    ))
    model.set_data((x, x))
    model.train(1)
    np.testing.assert_array_equal(model.predict(x), x * [0., 1.])
    model.close()


@pytest.mark.parametrize("metric,base,target,expected", [
    (None, 0., 1e155, 1e155),
    ([0.5, 0.5], 0., 1e155, 1e155),
    ([0.125, 0.125], 1.4e308, -1.4e308, 1.4e308),
])
def test_lsq_reports_large_finite_rmse(metric, base, target, expected, capfd):
    """Keep a representable validation score finite at extreme residual sizes.

    Args:
        metric: Implicit or explicit equal target weights.
        base: Training target and starting prediction.
        target: Constant validation target.
        expected: Weighted RMSE in the original loss units.
        capfd: Captures the native validation and early-stopping output.
    """
    x = np.ones((4, 1))
    y = np.full((4, 2), base)
    model = MultiOutputGBDT(out_dim=2, params=dict(base_score=base, output_metric=metric, verbosity=2, num_threads=1))
    model.set_data((x, y), (x, np.full_like(y, target)))
    model.train(1)
    output = capfd.readouterr().out
    validation = float(output.split("eval->")[1].split()[0])
    best = float(output.split("Best score ")[1].split()[0])
    assert validation == pytest.approx(expected)
    assert best == validation
    model.close()


@pytest.mark.parametrize("metric,gamma,split", [
    (np.ones((2, 2)), 14.3, "level"),
    (np.ones((2, 2)), 14.5, "none"),
    ([1., 1.], 0., "alternating"),
    (np.eye(2), 0., "alternating"),
])
def test_quadratic_gain_and_original_leaf_values(metric, gamma, split):
    """Check a rank-one reconstruction loss, gain threshold and L2 shrinkage.

    Args:
        metric: Output Gram matrix or vector of diagonal weights.
        gamma: Minimum required split gain.
        split: Analytically optimal permitted split.
    """
    x, y = _two_modes()
    model = MultiOutputGBDT(out_dim=2, params=dict(
        output_metric=metric, base_score=0., lr=1., reg_l2=2., max_depth=1,
        min_samples=1, gamma=gamma, num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    # Each child has eight observations. The sum of its two coefficient
    # residual sums has magnitude 12, giving a coupled gain of 12**2 / 10.
    if split == "level":
        expected = np.repeat(0.75 * x[:, [0]], 2, axis=1)
    elif split == "alternating":
        expected = x[:, [1]] * [1., -1.]
    else:
        expected = np.zeros_like(y)
    np.testing.assert_allclose(model.predict(x), expected * (8. / 10.), atol=1e-14)
    model.close()


def test_zero_weight_channel_retains_original_predictions():
    """A high-variance ignored target still gets its conditional leaf mean."""
    x, _ = _two_modes()
    shared = x[:, 1]
    y = np.column_stack((1.e6 * x[:, 0] + 3. * shared, 2. * shared))
    model = MultiOutputGBDT(out_dim=2, params=dict(
        output_metric=[0., 1.], max_depth=1, min_samples=1, reg_l2=0.,
        gamma=0., lr=1., num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    np.testing.assert_allclose(model.predict(x), np.column_stack((3. * shared, 2. * shared)))
    model.close()


@pytest.mark.parametrize("axis,null_axis,rho,weight,level_step", [
    ([1., 1.], [1., -1.], np.nextafter(1., 0.), 1., .02),
    ([1., 1.], [1., -1.], 1., 1., .02),
    ([1., -2., 4.], [2., 1., 0.], 1., 1., .02),
    ([1., 15.], [15., -1.], 1., 5., 2.**-10),
])
def test_weak_and_null_reconstruction_directions(axis, null_axis, rho, weight, level_step):
    """Compare nearly singular and truly singular metrics to direct signal error.

    Args:
        axis: Coefficients of a linear reconstruction.
        null_axis: A direction invisible to the rank-one reconstruction.
        rho: Off-diagonal correlation, set below one for the full-rank case.
        weight: Common multiplier on the reconstruction loss.
        level_step: Small change along the observed axis.
    """
    x, _ = _two_modes()
    axis = np.array(axis)
    metric = weight * np.outer(axis, axis)
    if rho < 1.:
        metric[0, 1] = metric[1, 0] = rho
        basis = np.linalg.cholesky(metric)
    else:
        basis = np.sqrt(weight) * axis[:, None]
    level = level_step * x[:, [0]] * axis
    alternating = 1.e7 * x[:, [1]] * null_axis
    y = level + alternating
    # Each binary feature explains exactly one component. Compare the residual
    # curve energies after fitting each possible split, using the known basis.
    error_after_level_split = np.mean((alternating @ basis) ** 2)
    error_after_alternating_split = np.mean((level @ basis) ** 2)
    assert (error_after_alternating_split < error_after_level_split) == (rho < 1.)
    model = MultiOutputGBDT(out_dim=len(axis), params=dict(
        output_metric=metric, max_depth=1, min_samples=1, reg_l2=0.,
        base_score=0., gamma=0., lr=1., num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    expected = alternating if rho < 1. else level
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-8)
    model.close()


def test_singular_block_preserves_separate_weak_signal():
    """Select a weakly weighted coefficient direction beside an exact nullspace."""
    x, _ = _two_modes()
    rho = np.nextafter(1., 0.)
    metric = np.zeros((4, 4))
    metric[:2, :2] = [[1., rho], [rho, 1.]]
    metric[2:, 2:] = 1.
    # The first pair reconstructs two signal coordinates; only the sum of
    # the last pair contributes to a third signal coordinate.
    basis = np.zeros((4, 3))
    basis[:2, :2] = np.linalg.cholesky(metric[:2, :2])
    basis[2:, 2] = 1.
    level = x[:, [0]] * [.02, .02, .01, .01]
    alternating = 1.e7 * x[:, [1]] * [1., -1., 1., -1.]
    assert np.mean((alternating @ basis) ** 2) > np.mean((level @ basis) ** 2)
    model = MultiOutputGBDT(out_dim=4, params=dict(
        output_metric=metric, max_depth=1, min_samples=1, reg_l2=0.,
        base_score=0., gamma=0., lr=1., num_threads=1, verbosity=0,
    ))
    model.set_data((x, level + alternating))
    model.train(1)
    np.testing.assert_allclose(model.predict(x), alternating, atol=1e-8)
    model.close()


@pytest.mark.parametrize("scale,amplitude", [(1., 1.), (1.e-15, 1.), (1.e305, 100.)])
def test_metric_scale_preserves_best_split_when_gamma_is_zero(scale, amplitude):
    """Compare split gains below absolute tolerance and above float64 range.

    Args:
        scale: Common multiplier on the quadratic loss.
        amplitude: Coefficient scale giving two distinct positive gains.
    """
    x, _ = _two_modes()
    y = amplitude * x * [1., 2.]
    model = MultiOutputGBDT(out_dim=2, params=dict(
        output_metric=[scale, scale], max_depth=1, min_samples=1, reg_l2=0.,
        gamma=0., lr=1., num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    np.testing.assert_allclose(model.predict(x), amplitude * x * [0., 2.])
    model.close()


@pytest.mark.parametrize("fraction", [1., 0.7])
def test_quadratic_trees_match_transformed_mse(fraction):
    """Compare a deep, regularized ensemble to a change-of-coordinates oracle.

    Args:
        fraction: Fraction of rows used by both native ensembles for each tree.
    """
    rng = np.random.default_rng(142)
    # Two bins per feature give each split a unique threshold, including
    # when row sampling leaves some combinations absent from a child node.
    levels = (np.arange(32)[:, None] >> np.arange(5)) % 2
    x = np.repeat(2. * levels - 1., 8, axis=0)
    y = np.column_stack((x[:, 0] + .8*x[:, 1]*x[:, 2] + .4*x[:, 3],
                         1.3*x[:, 2] - .9*x[:, 0]*x[:, 4] + .3*x[:, 1]))
    y += rng.normal(scale=.1, size=y.shape)
    xt = rng.uniform(-1., 1., size=(31, x.shape[1]))
    metric = np.array([[2., -0.6], [-0.6, 0.7]])
    lower = np.linalg.cholesky(metric)
    base = np.array([0.3, -0.1])
    opts = dict(num_threads=2, verbosity=0, min_samples=3, lr=.4, max_depth=3,
                subsample=fraction, seed=76, reg_l2=2.3)
    model = MultiOutputGBDT(out_dim=2, params=dict(opts, output_metric=metric, base_score=base, gamma=.24))
    reference = MultiOutputGBDT(out_dim=2, params=dict(opts, base_score=base @ lower, gamma=.12))
    model.set_data((x, y), (xt, None))
    reference.set_data((x, y @ lower))
    for rounds in (1, 7):
        model.train(rounds)
        reference.train(rounds)
        np.testing.assert_allclose(model.predict(xt) @ lower, reference.predict(xt), atol=1e-11)
        np.testing.assert_allclose(model.preds_train, model.predict(x), atol=1e-11)
        np.testing.assert_allclose(model.preds_eval, model.predict(xt), atol=1e-11)
    model.close()
    reference.close()


def test_validation_uses_reconstruction_error(capfd):
    """Retain different optimal rounds and report the error in signal units.

    Args:
        capfd: Native stdout capture fixture.
    """
    x = np.ones((8, 1))
    y = np.ones((8, 2))
    yval = np.tile([0., 1.], (8, 1))
    opts = dict(num_threads=1, early_stop=1, verbosity=2, base_score=0., lr=.5, reg_l2=0.)
    metric = [[1., .25], [.25, 4.]]
    ordinary = MultiOutputGBDT(out_dim=2, params=opts)
    model = MultiOutputGBDT(out_dim=2, params=dict(opts, output_metric=metric))
    ordinary.set_data((x, y), (x, yval))
    ordinary.train(6)
    capfd.readouterr()
    model.set_data((x, y), (x, yval))
    model.train(6)
    first_error = np.array([.5, -.5])
    eval_score = np.sqrt(first_error @ metric @ first_error)
    assert f"eval->{eval_score:.5f}" in capfd.readouterr().out
    np.testing.assert_array_equal(ordinary.predict(x), .5)
    np.testing.assert_array_equal(model.predict(x), .75)
    np.testing.assert_array_equal(model.preds_eval, model.predict(x))
    ordinary.close()
    model.close()


@pytest.mark.parametrize("invalid", [
    1., [1.], [1., -1.], [0., 0.], [np.nan, 1.], [1., np.inf],
    [[1., 0.]], [[1., .5], [0., 1.]], [[1., 2.], [2., 1.]],
    np.array([[1., 2.], [2., 1.]]) * 1.e-40, [1.j, 1.j],
])
def test_invalid_metric_preserves_existing_booster(invalid):
    """Reject invalid metrics before replacing a fitted native model.

    Args:
        invalid: Malformed metric, including indefinite or nonfinite inputs.
    """
    x, y = _two_modes()
    model = MultiOutputGBDT(out_dim=2, params=dict(min_samples=1, verbosity=0, num_threads=1))
    model.set_data((x, y))
    model.train(1)
    saved = model.predict(x)
    model.params["output_metric"] = invalid
    with pytest.raises((TypeError, ValueError)):
        model.set_data((x, y))
    np.testing.assert_array_equal(model.predict(x), saved)
    model.close()


@pytest.mark.parametrize("cls,width,opts", [
    (SingleOutputGBDT, 1, {}),
    (MultiOutputGBDT, 1, {}),
    (MultiOutputGBDT, 2, {"loss": b"ce"}),
    (MultiOutputGBDT, 2, {"loss": b"bce"}),
    (MultiOutputGBDT, 2, {"topk": 1}),
    (MultiOutputGBDT, 2, {"reg_l1": .1}),
])
def test_incompatible_metric_configuration(cls, width, opts):
    """Reject combinations outside dense multi-output squared-error training.

    Args:
        cls: Low-level booster type.
        width: Configured target width.
        opts: Additional native options.
    """
    with pytest.raises(ValueError):
        model = cls(out_dim=width, params=dict(opts, output_metric=np.ones(width)))
        model.set_data((np.ones((8, 1)), np.ones((8, width))))


def test_custom_objective_cannot_override_quadratic_geometry():
    """Reject custom derivatives and keep the configured built-in path usable."""
    x, y = _two_modes()
    model = MultiOutputGBDT(out_dim=2, params=dict(output_metric=[0., 1.], num_threads=1, verbosity=0))
    model.set_data((x, y))
    with pytest.raises(ValueError):
        model.train(1, objective=lambda pred, target: (pred - target, np.ones_like(pred)))
    with pytest.raises(ValueError):
        model._set_gh(y, np.ones_like(y))
    with pytest.raises((ValueError, RuntimeError)):
        model.boost()
    model.train(1)
    assert model.lib.GetTreeCount(model._boostnode) == 1
    model.close()


def test_dump_and_continue_in_original_coordinates(tmp_path):
    """Load for plain inference and continue fitting with the original metric.

    Args:
        tmp_path: Directory containing the tree file.
    """
    x, y = _two_modes()
    opts = dict(output_metric=np.ones((2, 2)), min_samples=1, gamma=0.,
                base_score=[.5, -.25], num_threads=1, verbosity=0)
    model = MultiOutputGBDT(out_dim=2, params=opts)
    model.set_data((x, y))
    model.train(3)
    path = tmp_path / "coefficients.txt"
    model.dump(path)
    inference = MultiOutputGBDT(out_dim=2)
    inference.set_booster(x.shape[1], 2)
    inference.load(path)
    expected = model.predict(x)
    np.testing.assert_allclose(inference.predict(x), expected, atol=1e-12)
    resumed = MultiOutputGBDT(out_dim=2, params=opts)
    resumed.set_data((x, y))
    resumed.load(path)
    np.testing.assert_allclose(resumed.preds_train, expected, atol=1e-12)
    model.train(2)
    resumed.train(2)
    np.testing.assert_allclose(resumed.predict(x), model.predict(x), atol=1e-12)
    for booster in (model, inference, resumed):
        booster.close()


def test_sklearn_refit_and_clone():
    """Preserve a valid fitted regressor if a later fit rejects the metric."""
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from omnigbdt.sklearn import MultiOutputGBDTRegressor

    x, y = _two_modes()
    reg = MultiOutputGBDTRegressor(
        output_metric=np.ones((2, 2)), min_samples=1, max_depth=1, num_threads=1,
        num_rounds=1, lr=1., reg_l2=0., gamma=0., early_stop=0,
    )
    reg.fit(x, y)
    expected = np.tile(.75 * x[:, [0]], (1, 2))
    np.testing.assert_allclose(reg.predict(x), expected)
    other = clone(reg)
    np.testing.assert_array_equal(other.output_metric, reg.output_metric)
    reg.set_params(output_metric=[-1., 1.])
    with pytest.raises(ValueError):
        reg.fit(x, y)
    np.testing.assert_allclose(reg.predict(x), expected)
    reg.close()
