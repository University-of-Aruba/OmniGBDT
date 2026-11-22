import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


def _single_or_multi(multi):
    """Build a small deterministic training case.

    Args:
        multi: Whether to use two joint targets.

    Returns:
        tuple: Booster, features, and float64 labels.
    """
    x = np.arange(64, dtype=np.float64).reshape(32, 2)
    params = {"num_threads": 1, "verbosity": 0, "min_samples": 2, "early_stop": 1}
    if multi:
        return MultiOutputGBDT(out_dim=2, params=params), x, x.copy()
    return SingleOutputGBDT(params=params), x, x[:, 0].copy()


def _objective(preds, labels):
    """Compute the per-target squared-error derivatives.

    Args:
        preds: Current predictions.
        labels: Training targets.

    Returns:
        tuple: Gradient and Hessian of the same shape as predictions.
    """
    return preds - labels, np.ones_like(preds)


@pytest.mark.parametrize("multi", [False, True])
def test_large_constant_labels_have_a_finite_mean_prediction(multi):
    """Fit a constant target at the finite float64 limit without overflowing.

    Args:
        multi: Whether to fit both signs of the limiting value as two targets.
    """
    model, x, _ = _single_or_multi(multi)
    y = np.full(len(x), np.finfo(np.float64).max)
    if multi:
        y = np.column_stack((y, -y))
    model.set_data((x, y))
    np.testing.assert_array_equal(model.predict(x), y)
    model.train(1)
    np.testing.assert_array_equal(model.predict(x), y)
    model.close()


def test_overflowing_target_sum_preserves_cancellation_and_other_targets():
    """Recover a small residual mean and retain ordinary rounding on another target."""
    model, x, _ = _single_or_multi(True)
    limit = np.finfo(np.float64).max
    tiny = 1e-300
    y = np.array([[limit, .3], [limit, .01], [-limit, .05], [-limit, .7], [tiny, .2]])
    model.set_data((x[:5], y))
    expected = np.array([tiny / 5, np.mean(y[:, 1])])
    np.testing.assert_array_equal(model.predict(x[:1])[0], expected)
    model.close()


@pytest.mark.parametrize("multi", [False, True])
def test_replace_dataset_labels_and_attach_eval(multi):
    """Check retained labels, delayed eval registration and replacing a dataset.

    Args:
        multi: Whether to test a joint model.
    """
    model, x, y = _single_or_multi(multi)
    model.set_data((x, y))
    changed_y = y + 8
    model._set_label(changed_y, True)
    model.train(1, objective=_objective)
    assert model.label is changed_y or np.shares_memory(model.label, changed_y)
    model.set_data(eval_set=(x[:7], y[:7]))
    np.testing.assert_allclose(model.preds_eval, model.predict(x[:7]))
    model.set_data((x, y))
    model.train(1, objective=_objective)
    assert not hasattr(model, "preds_eval")
    model.close()


def test_interleaved_model_registration():
    """Register eval splits with different output ranks through a shared library."""
    single, x, y = _single_or_multi(False)
    multi, _, y_multi = _single_or_multi(True)
    single.set_data((x, y))
    multi.set_data((x, y_multi))
    single.set_data(eval_set=(x, y))
    multi.set_data(eval_set=(x, y_multi))
    single.train(1)
    multi.train(1)
    np.testing.assert_allclose(single.preds_eval, single.predict(x))
    np.testing.assert_allclose(multi.preds_eval, multi.predict(x))
    single.close()
    multi.close()


@pytest.mark.parametrize("multi", [False, True])
def test_custom_early_stop_preserves_previous_rounds(multi):
    """Continue twice with one improving round per call and compare to a control.

    Args:
        multi: Whether to train two outputs.
    """
    model, x, y = _single_or_multi(multi)
    control, _, _ = _single_or_multi(multi)
    model.set_data((x, y), (x, y))
    control.set_data((x, y))
    for _ in range(2):
        model.train(3, objective=_objective, eval_metric=lambda p, t: 0.0, maximize=False)
    control.train(2, objective=_objective)
    np.testing.assert_allclose(model.predict(x), control.predict(x))
    np.testing.assert_allclose(model.preds_train, control.predict(x))
    model.close()
    control.close()


@pytest.mark.parametrize("multi", [False, True])
def test_builtin_integer_targets_and_unlabeled_eval(multi):
    """Verify supported label dtypes and that unlabeled eval rows still update.

    Args:
        multi: Whether to test the joint predictor.
    """
    model, x, y = _single_or_multi(multi)
    control, _, _ = _single_or_multi(multi)
    control.set_data((x, y))
    control.train(3)
    model.set_data((x, y.astype(np.int32)), (x, y))
    model.set_data(eval_set=(x, None))
    model.train(3)
    np.testing.assert_allclose(model.preds_eval, control.predict(x))
    model.close()
    control.close()


@pytest.mark.parametrize("multi", [False, True])
def test_load_refreshes_registered_train_and_eval(multi, tmp_path):
    """Restore a model into a configured dataset and continue its training.

    Args:
        multi: Whether to train jointly.
        tmp_path: Temporary directory for the model dump.
    """
    model, x, y = _single_or_multi(multi)
    restored, _, _ = _single_or_multi(multi)
    model.set_data((x, y))
    model.train(2)
    path = tmp_path / "model.txt"
    model.dump(path)
    restored.set_data((x, y), (x[:6], None))
    restored.load(path)
    np.testing.assert_allclose(restored.preds_train, model.predict(x))
    np.testing.assert_allclose(restored.preds_eval, model.predict(x[:6]))
    model.train(1)
    restored.train(1)
    np.testing.assert_allclose(restored.predict(x), model.predict(x))
    restored.close()
    model.close()


def test_invalid_data_preserves_configured_dataset():
    """Reject incompatible row counts and widths before replacing the old split."""
    model, x, y = _single_or_multi(False)
    model.set_data((x, y))
    model.train(1)
    before = model.predict(x)
    with pytest.raises(ValueError, match="rows"):
        model.set_data((x, y[:-1]))
    with pytest.raises(ValueError, match="columns"):
        model.set_data((x, y), (x[:, :1], y))
    with pytest.raises(ValueError, match="columns"):
        model.predict(x[:, :1])
    np.testing.assert_allclose(model.predict(x), before)
    model.set_data((x, None))
    with pytest.raises(ValueError, match="labels"):
        model.train(1)
    with pytest.raises(RuntimeError, match="_set_gh"):
        model.boost()
    model.close()
    with pytest.raises(RuntimeError, match="set_booster"):
        model.predict(x)


def test_multiclass_labels_and_shape_validation():
    """Require a valid class per row and retain the example's 1D label layout."""
    _, x, _ = _single_or_multi(False)
    y = (np.arange(len(x)) % 3).astype(np.int32)
    model = MultiOutputGBDT(out_dim=3, params={"loss": b"ce", "num_threads": 1, "verbosity": 0})
    model.set_data((x, y))
    model.train(1)
    assert model.predict(x).shape == (len(x), 3)
    for invalid in (y + 1, y.astype(np.float64) + 0.5):
        model._set_label(invalid, True)
        with pytest.raises(ValueError, match="Class indices"):
            model.train(1)
    model.close()


def test_configured_model_keeps_native_loss_and_dimensions():
    """Keep training-compatible buffers after Python parameter edits."""
    model, x, y = _single_or_multi(False)
    model.set_data((x, y))
    model.params["loss"] = b"bce"
    model.train(1)
    assert np.isfinite(model.predict(x)).all()
    model.out_dim = 2
    with pytest.raises(ValueError, match="out_dim"):
        model.predict(x)
    model.close()

    model = MultiOutputGBDT(out_dim=2, params={"out_dim": 1, "num_threads": 1, "verbosity": 0})
    model.set_data((x, x.copy()))
    model.train(1)
    assert model.predict(x).shape == x.shape
    model.close()


@pytest.mark.parametrize("name,value", [("num_threads", 0), ("min_samples", 0), ("topk", -1), ("hist_cache", -1)])
def test_invalid_native_parameter_is_rejected(name, value):
    """Validate unsafe native dimensions before binning or booster construction.

    Args:
        name: Native integer parameter.
        value: Value outside its supported range.
    """
    model, x, y = _single_or_multi(False)
    model.params[name] = value
    with pytest.raises(ValueError, match=name):
        model.set_data((x, y))
    model.close()


def test_custom_objective_receives_matching_label_shape():
    """Reject a single target column for a two-output regression callback."""
    model, x, y = _single_or_multi(True)
    model.set_data((x, y[:, :1]))
    with pytest.raises(ValueError, match="shape"):
        model.train(1, objective=_objective)
    model.close()
