"""Regression tests for row sampling and built-in classifier stopping metrics."""

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


@pytest.mark.parametrize("fraction,count", [(0.35, 3), (1e-9, 1)])
def test_subsample_root_uses_distinct_training_rows(fraction, count):
    """Decode the rows contributing to a fitted root leaf from its mean.

    Args:
        fraction: Fraction of rows requested for each tree.
        count: Expected number after flooring and applying the one-row minimum.
    """
    x = np.zeros((10, 1))
    y = 2. ** np.arange(len(x))
    model = SingleOutputGBDT(params=dict(
        subsample=fraction, base_score=0., lr=1., reg_l2=0., max_leaves=1,
        num_threads=1, verbosity=0,
    ))
    model.set_data((x, y))
    model.train(1)
    value = model.predict(x)[0]
    subset_sum = round(value * count)
    assert value == pytest.approx(subset_sum / count)
    assert subset_sum.bit_length() <= len(y) and subset_sum.bit_count() == count
    np.testing.assert_array_equal(model.preds_train, np.full(len(x), value))


@pytest.mark.parametrize("multi", [False, True])
def test_subsample_reproducibility_and_continuation(multi):
    """Check sample progression, the full-data default, and all prediction caches.

    Args:
        multi: Whether to train two regression outputs jointly.
    """
    rng = np.random.default_rng(122)
    x = rng.normal(size=(96, 5))
    y = x[:, 0] - x[:, 2] ** 2
    cls = MultiOutputGBDT if multi else SingleOutputGBDT
    if multi:
        y = np.column_stack((y, x[:, 1] * x[:, 4]))
    preds = []
    for options, rounds in [
        ({"subsample": .7, "seed": 8}, [6]),
        ({"subsample": .7, "seed": 8}, [2, 4]),
        ({"subsample": .7, "seed": 9}, [6]),
        ({"subsample": 1., "seed": 8}, [6]),
        ({}, [6]),
    ]:
        model = cls(out_dim=2 if multi else 1, params=dict(
            min_samples=2, num_threads=1, verbosity=0, **options,
        ))
        model.set_data((x, y), (x[::3], None))
        for count in rounds:
            model.train(count)
        prediction = model.predict(x)
        np.testing.assert_array_equal(model.preds_train, prediction)
        np.testing.assert_array_equal(model.preds_eval, prediction[::3])
        preds.append(prediction)
        model.close()
    np.testing.assert_array_equal(preds[0], preds[1])
    np.testing.assert_array_equal(preds[3], preds[4])
    assert not np.array_equal(preds[0], preds[2])
    assert not np.array_equal(preds[0], preds[3])


@pytest.mark.parametrize("fraction", [0., -.1, 1.1, np.nan, np.inf, "0.8"])
def test_invalid_subsample_is_rejected(fraction):
    """Reject invalid sampling configurations before fitting.

    Args:
        fraction: Out-of-range, nonfinite, or nonnumeric sampling setting.
    """
    with pytest.raises((ValueError, TypeError), match="subsample"):
        model = MultiOutputGBDT(out_dim=2, params={"subsample": fraction})
        model.set_booster(1, 2)


@pytest.mark.parametrize("loss", [b"bce", b"ce", b"ce_column"])
@pytest.mark.parametrize("next_metric", [None, "accuracy"])
def test_classifier_logloss_selects_validation_prefix(loss, next_metric):
    """Compare the retained rounds to cross-entropy from an independent reference fit.

    Args:
        loss: Binary, joint multiclass, or legacy per-class objective.
        next_metric: Default or explicit accuracy selection on continued training.
    """
    binary = loss == b"bce"
    n_class = 2 if binary else 3
    cls = MultiOutputGBDT if loss == b"ce" else SingleOutputGBDT
    width = 1 if binary else n_class
    labels = np.tile(np.arange(n_class, dtype=np.int32), 35)
    x = labels[:, None].astype(np.float64)
    heldout_labels = labels.copy()
    heldout_labels[::5] = (heldout_labels[::5] + 1) % n_class
    params = dict(loss=loss, lr=.3, min_samples=1, early_stop=2, num_threads=1, verbosity=0)
    model = cls(out_dim=width, params=params)
    reference = cls(out_dim=width, params=params)
    reference.set_data((x, labels))
    model.set_data((x, labels), (x, heldout_labels))
    train = model.train_multi if loss == b"ce_column" else model.train
    train_reference = reference.train_multi if loss == b"ce_column" else reference.train
    train_reference(30)
    losses = []
    for r in range(1, 31):
        margin = reference.predict(x, num_trees=r)
        if binary:
            cross_entropy = np.logaddexp(0., margin) - heldout_labels * margin
        else:
            cross_entropy = np.logaddexp.reduce(margin, axis=1) - margin[np.arange(len(x)), heldout_labels]
        losses.append(cross_entropy.mean())
    best = int(np.argmin(losses)) + 1
    assert 1 < best < 29
    train(30, eval_metric="logloss")
    np.testing.assert_array_equal(model.predict(x), reference.predict(x, num_trees=best))

    # All classes are now correct. Accuracy plateaus, while log loss keeps falling.
    model.set_data(eval_set=(x, labels))
    train(6, eval_metric=next_metric)
    np.testing.assert_array_equal(model.predict(x), reference.predict(x, num_trees=best + 1))


def test_regression_rejects_classification_metric():
    """Keep a regression fit intact when a classification-only metric is requested."""
    x = np.arange(64., dtype=np.float64)[:, None]
    model = SingleOutputGBDT(params={"num_threads": 1, "verbosity": 0})
    model.set_data((x, x[:, 0]))
    model.train(1)
    before = model.predict(x)
    with pytest.raises(ValueError):
        model.train(2, eval_metric="logloss")
    np.testing.assert_array_equal(model.predict(x), before)


@pytest.mark.parametrize("loss", [b"ce", b"ce_column"])
def test_logloss_on_large_common_multiclass_margin(loss, capfd):
    """Preserve the loss of three equiprobable classes under a common logit offset.

    Args:
        loss: Joint or legacy multiclass objective.
        capfd: Native metric output capture.
    """
    cls = MultiOutputGBDT if loss == b"ce" else SingleOutputGBDT
    model = cls(out_dim=3, params=dict(
        loss=loss, base_score=1e30, lr=0., num_threads=2, verbosity=2,
    ))
    x = np.ones((513, 1))
    y = np.arange(len(x), dtype=np.int32) % 3
    model.set_data((x, y), (x, y))
    train = model.train if loss == b"ce" else model.train_multi
    train(1, eval_metric="logloss")
    assert f"eval->{np.log(3.):.5f}" in capfd.readouterr().out


@pytest.mark.parametrize("multi", [False, True])
def test_sklearn_forwards_row_sampling(multi):
    """Check that cloning and fitting a wrapper retains native row sampling.

    Args:
        multi: Whether to use the multi-output regressor.
    """
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from omnigbdt import MultiOutputGBDTRegressor, SingleOutputGBDTRegressor

    rng = np.random.default_rng(732)
    x = rng.normal(size=(72, 3))
    y = x[:, 0] if not multi else x[:, :2]
    cls = MultiOutputGBDTRegressor if multi else SingleOutputGBDTRegressor
    sampled = clone(cls(subsample=.6, min_samples=2, num_threads=1, num_rounds=3))
    full = clone(sampled).set_params(subsample=1.)
    assert sampled.get_params()["subsample"] == .6
    assert not np.array_equal(sampled.fit(x, y).predict(x), full.fit(x, y).predict(x))


@pytest.mark.parametrize("loss", [b"bce", b"ce"])
def test_logloss_mean_at_float_limit(loss, capfd):
    """Keep the mean of three maximal finite classification losses representable.

    Args:
        loss: Binary or joint multiclass objective.
        capfd: Native metric output capture.
    """
    limit = np.finfo(np.float64).max
    params = dict(loss=loss, lr=0., num_threads=1, verbosity=2)
    if loss == b"bce":
        model = SingleOutputGBDT(params={**params, "base_score": limit})
    else:
        model = MultiOutputGBDT(out_dim=2, params={**params, "base_score": [0., limit]})
    x = np.zeros((3, 1))
    y = np.zeros(3, dtype=np.int32)
    model.set_data((x, y), (x, y))
    model.train(1, eval_metric="logloss")
    printed = capfd.readouterr().out.split("eval->")[1].splitlines()[0]
    assert float(printed) == limit


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("margin,step", [(40., 1.), (-720., -1.), (900., 0.)])
def test_saturated_bce_leaf(multi, margin, step):
    """Retain a finite Newton update in a correctly predicted, saturated leaf.

    Args:
        multi: Whether to fit two output columns in the same leaf.
        margin: Initial log-odds for the first output.
        step: Expected update at unit learning rate.
    """
    x = np.ones((8, 3))
    base = np.array([margin, -margin]) if multi else margin
    shape = (8, 2) if multi else (8,)
    y = np.broadcast_to(base > 0, shape).astype(np.int32)
    expected = base + (np.array([step, -step]) if multi else step)
    cls = MultiOutputGBDT if multi else SingleOutputGBDT
    model = cls(out_dim=2 if multi else 1, params=dict(
        loss=b"bce", base_score=base, lr=1., reg_l2=0., colsample_bytree=.5,
        num_threads=1, verbosity=0,
    ))
    try:
        model.set_data((x, y))
        model.train(1)
        np.testing.assert_array_equal(model.predict(x), np.broadcast_to(expected, shape))
    finally:
        model.close()


def test_flat_binary_output_preserves_other_split_gain():
    """Split on an informative binary target when another target is saturated."""
    bit = np.repeat([0., 1.], 8)
    x = np.column_stack((bit, bit))
    y = np.column_stack((np.ones_like(bit), bit)).astype(np.int32)
    model = MultiOutputGBDT(out_dim=2, params=dict(
        loss=b"bce", base_score=[900., 0.], lr=1., reg_l2=0., max_depth=1,
        min_samples=1, colsample_bytree=.5, num_threads=1, verbosity=0,
    ))
    try:
        model.set_data((x, y))
        model.train(1)
        np.testing.assert_array_equal(model.predict(x), np.column_stack((np.full(16, 900.), 4 * bit - 2)))
    finally:
        model.close()


@pytest.mark.parametrize("margin", [-900., -40., 40., 900.])
def test_wrong_saturated_binary_predictions_take_finite_steps(margin):
    """Correct confident mistakes with pure and mixed labels in the same tree.

    Args:
        margin: Initial log-odds on either side of the decision boundary.
    """
    bit = np.repeat([0., 1.], 8)
    x = np.column_stack((bit, bit))
    correct = int(margin > 0)
    labels = np.where(bit == 0, correct, 1 - correct)
    y = np.column_stack((labels, np.full(16, 1 - correct))).astype(np.int32)
    model = MultiOutputGBDT(out_dim=2, params=dict(
        loss=b"bce", base_score=margin, lr=1., reg_l2=0., max_depth=1,
        min_samples=1, colsample_bytree=.5, num_threads=1, verbosity=0,
    ))
    try:
        model.set_data((x, y))
        model.train(1)
        pred = model.predict(x)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(pred[:, 1], np.full(len(x), margin - 4 * np.sign(margin)))
        assert np.all(np.sign(margin) * (pred[bit == 1, 0] - margin) < 0)
    finally:
        model.close()


@pytest.mark.parametrize("sparse", [False, True])
def test_binary_histogram_subtraction_near_saturation(sparse):
    """Keep child updates finite across mixed curvatures and sampled trees.

    Args:
        sparse: Whether to select the two best outputs for each leaf.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(size=(513, 8))
    held = rng.normal(size=(347, 8))
    y = np.column_stack((x[:, 2] > .2, x[:, 1] * x[:, 5] > .2,
                         x[:, 3] < 0, rng.random(len(x)) < .03)).astype(np.int32)
    model = MultiOutputGBDT(out_dim=4, params=dict(
        loss=b"bce", base_score=[37., -37., 37., -37.], lr=.1, reg_l2=0.,
        max_depth=4, min_samples=5, colsample_bytree=.7, subsample=.7,
        seed=0, topk=2 if sparse else 0, num_threads=1, verbosity=0,
    ))
    try:
        model.set_data((x, y), (held, None))
        model.train(15)
        assert np.isfinite(model.preds_train).all()
        assert np.isfinite(model.predict(held)).all()
        np.testing.assert_array_equal(model.preds_eval, model.predict(held))
    finally:
        model.close()
