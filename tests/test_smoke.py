from pathlib import Path
import importlib.util

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT, Verbosity, load_lib
from omnigbdt.lib_utils import _resolve_packaged_library_path


def mse_objective(preds, target):
    """Return MSE gradients and Hessians for callback-based training tests."""
    return preds - target, np.ones_like(preds)


def rmse_metric(preds, target):
    """Return the root-mean-squared error for callback-based training tests."""
    return float(np.sqrt(np.mean((preds - target) ** 2)))


def invalid_single_shape_objective(preds, target):
    """Return a deliberately invalid single-output gradient shape for tests."""
    del target
    return (
        np.ones((len(preds), 2), dtype=np.float64),
        np.ones((len(preds), 2), dtype=np.float64),
    )


def invalid_integer_objective(preds, target):
    """Return deliberately invalid integer gradients and Hessians for tests."""
    del target
    return (
        np.ones_like(preds, dtype=np.int32),
        np.ones_like(preds, dtype=np.int32),
    )


class EvalMetricSequence:
    """Emit a fixed evaluation metric sequence while keeping train metrics numeric."""

    def __init__(self, values, eval_size):
        """Store the evaluation-only metric sequence.

        Args:
            values: Metric values to emit for the evaluation split.
            eval_size: Number of rows in the evaluation split.
        """
        self.values = list(values)
        self.eval_size = eval_size
        self.index = 0

    def __call__(self, preds, target):
        """Return the next configured eval metric or a real train RMSE value.

        Args:
            preds: Prediction array provided by the booster.
            target: Label array provided by the booster.

        Returns:
            float: Metric value for the current split.
        """
        if len(preds) == self.eval_size:
            value = self.values[min(self.index, len(self.values) - 1)]
            self.index += 1
            return float(value)
        return rmse_metric(preds, target)


def test_multioutputgbdt_smoke(tmp_path):
    rng = np.random.default_rng(0)
    inp_dim = 10
    out_dim = 5
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "num_threads": 1,
        "verbose": False,
    }

    x_train = rng.random((512, inp_dim)).astype("float64")
    y_train = rng.random((512, out_dim)).astype("float64")
    x_valid = rng.random((128, inp_dim)).astype("float64")
    y_valid = rng.random((128, out_dim)).astype("float64")

    booster = MultiOutputGBDT(out_dim=out_dim, params=params)
    booster.set_data((x_train, y_train), (x_valid, y_valid))
    booster.train(1)
    preds = booster.predict(x_valid)

    assert preds.shape == (len(x_valid), out_dim)

    model_path = tmp_path / "model.txt"
    booster.dump(model_path)
    assert model_path.is_file()

    loaded = MultiOutputGBDT(out_dim=out_dim, params=params)
    loaded.set_booster(inp_dim, out_dim)
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_valid)

    np.testing.assert_allclose(preds, loaded_preds)

    explicit_lib = load_lib(str(_resolve_packaged_library_path()))
    assert explicit_lib is booster.lib

    booster.close()
    loaded.close()


def test_multioutputgbdt_verbosity_levels(capfd):
    rng = np.random.default_rng(1)
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
    }

    x_train = rng.random((64, 4)).astype("float64")
    y_train = rng.random((64, 2)).astype("float64")
    x_valid = rng.random((32, 4)).astype("float64")
    y_valid = rng.random((32, 2)).astype("float64")

    silent = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.SILENT})
    silent.set_data((x_train, y_train), (x_valid, y_valid))
    silent.train(2)
    silent.close()

    silent_out = capfd.readouterr().out
    assert "Best score" not in silent_out
    assert "[0] train->" not in silent_out

    summary = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.SUMMARY})
    summary.set_data((x_train, y_train), (x_valid, y_valid))
    summary.train(2)
    summary.close()

    summary_out = capfd.readouterr().out
    assert "Best score" in summary_out
    assert "[0] train->" not in summary_out

    full = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.FULL})
    full.set_data((x_train, y_train), (x_valid, y_valid))
    full.train(2)
    full.close()

    full_out = capfd.readouterr().out
    assert "Best score" in full_out
    assert "[0] train->" in full_out


def test_legacy_verbose_flag_maps_to_full_and_silent(capfd):
    rng = np.random.default_rng(2)
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
    }

    x_train = rng.random((64, 4)).astype("float64")
    y_train = rng.random((64, 2)).astype("float64")
    x_valid = rng.random((32, 4)).astype("float64")
    y_valid = rng.random((32, 2)).astype("float64")

    silent = MultiOutputGBDT(out_dim=2, params={**params, "verbose": False})
    silent.set_data((x_train, y_train), (x_valid, y_valid))
    silent.train(2)
    silent.close()
    assert capfd.readouterr().out == ""

    noisy = MultiOutputGBDT(out_dim=2, params={**params, "verbose": True})
    noisy.set_data((x_train, y_train), (x_valid, y_valid))
    noisy.train(2)
    noisy.close()

    noisy_out = capfd.readouterr().out
    assert "Best score" in noisy_out
    assert "[0] train->" in noisy_out


def test_multioutputgbdt_falls_back_to_root_leaf_when_no_split_meets_min_samples(tmp_path):
    rng = np.random.default_rng(3)
    x_train = rng.random((6, 2)).astype("float64")
    y_train = rng.random((6, 2)).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "max_leaves": 8,
        "min_samples": 4,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = MultiOutputGBDT(out_dim=2, params=params)
    booster.set_data((x_train, y_train), (x_train, y_train))
    booster.train(1)
    preds = booster.predict(x_train)

    expected = np.repeat(preds[:1], len(x_train), axis=0)
    np.testing.assert_allclose(preds, expected)

    model_path = tmp_path / "multi_root_leaf.txt"
    booster.dump(model_path)
    dump_text = model_path.read_text()
    assert "\t-" not in dump_text

    loaded = MultiOutputGBDT(out_dim=2, params=params)
    loaded.set_booster(x_train.shape[1], 2)
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_train)

    np.testing.assert_allclose(preds, loaded_preds)

    booster.close()
    loaded.close()


def test_singleoutputgbdt_falls_back_to_root_leaf_when_no_split_meets_min_samples(tmp_path):
    rng = np.random.default_rng(4)
    x_train = rng.random((6, 2)).astype("float64")
    y_train = rng.random(6).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "max_leaves": 8,
        "min_samples": 4,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = SingleOutputGBDT(params=params)
    booster.set_data((x_train, y_train), (x_train, y_train))
    booster.train(1)
    preds = booster.predict(x_train)

    np.testing.assert_allclose(preds, np.repeat(preds[:1], len(x_train)))

    model_path = tmp_path / "single_root_leaf.txt"
    booster.dump(model_path)
    dump_text = model_path.read_text()
    assert "\t-" not in dump_text

    loaded = SingleOutputGBDT(params=params)
    loaded.set_booster(x_train.shape[1])
    loaded.load(model_path)
    loaded_preds = loaded.predict(x_train)

    np.testing.assert_allclose(preds, loaded_preds)

    booster.close()
    loaded.close()


def test_singleoutput_auto_base_score_matches_training_mean_and_persists_dump(tmp_path):
    """Ensure automatic single-output base-score initialization survives reload."""
    rng = np.random.default_rng(5)
    x_train = rng.random((32, 3)).astype("float64")
    y_train = (2.0 * x_train[:, 0] - 0.5 * x_train[:, 1] + 2.5).astype("float64")
    params = {
        "loss": b"mse",
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = SingleOutputGBDT(params=params)
    booster.set_data((x_train, y_train))
    expected = np.full(6, float(np.mean(y_train)), dtype=np.float64)
    np.testing.assert_allclose(booster.predict(x_train[:6]), expected)

    model_path = tmp_path / "single_auto_base_score.txt"
    booster.dump(model_path)

    loaded = SingleOutputGBDT(params=params)
    loaded.set_booster(x_train.shape[1])
    loaded.load(model_path)
    np.testing.assert_allclose(loaded.predict(x_train[:6]), expected)

    booster.close()
    loaded.close()


def test_multioutput_auto_and_explicit_base_scores_work_and_persist_dump(tmp_path):
    """Ensure multi-output base-score vectors initialize and reload correctly."""
    rng = np.random.default_rng(6)
    x_train = rng.random((32, 3)).astype("float64")
    y_train = np.column_stack(
        [
            2.0 * x_train[:, 0] + 0.25,
            -1.5 * x_train[:, 1] + 0.75,
        ]
    ).astype("float64")
    params = {
        "loss": b"mse",
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = MultiOutputGBDT(out_dim=2, params=params)
    booster.set_data((x_train, y_train))
    expected_auto = np.repeat(np.mean(y_train, axis=0, keepdims=True), 4, axis=0)
    np.testing.assert_allclose(booster.predict(x_train[:4]), expected_auto)

    model_path = tmp_path / "multi_auto_base_score.txt"
    booster.dump(model_path)

    loaded = MultiOutputGBDT(out_dim=2, params=params)
    loaded.set_booster(x_train.shape[1], 2)
    loaded.load(model_path)
    np.testing.assert_allclose(loaded.predict(x_train[:4]), expected_auto)

    explicit = MultiOutputGBDT(
        out_dim=2,
        params={**params, "base_score": np.array([0.2, 0.8], dtype=np.float64)},
    )
    explicit.set_data((x_train, y_train))
    expected_explicit = np.repeat(np.array([[0.2, 0.8]], dtype=np.float64), 4, axis=0)
    np.testing.assert_allclose(explicit.predict(x_train[:4]), expected_explicit)

    booster.close()
    loaded.close()
    explicit.close()


def test_singleoutput_custom_objective_matches_builtin_mse():
    rng = np.random.default_rng(7)
    x_train = rng.random((96, 4)).astype("float64")
    y_train = (1.7 * x_train[:, 0] - 0.4 * x_train[:, 1] + 0.2 * x_train[:, 2]).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "num_threads": 1,
        "seed": 11,
        "verbosity": Verbosity.SILENT,
    }

    built_in = SingleOutputGBDT(params=params)
    built_in.set_data((x_train, y_train))
    built_in.train(4)

    custom = SingleOutputGBDT(params=params)
    custom.set_data((x_train, y_train))
    custom.train(4, objective=mse_objective)

    np.testing.assert_allclose(custom.predict(x_train), built_in.predict(x_train))

    built_in.close()
    custom.close()


def test_multioutput_custom_objective_matches_builtin_mse():
    rng = np.random.default_rng(8)
    x_train = rng.random((96, 4)).astype("float64")
    y_train = np.column_stack(
        [
            1.2 * x_train[:, 0] - 0.5 * x_train[:, 1],
            -0.7 * x_train[:, 2] + 0.9 * x_train[:, 3],
        ]
    ).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "num_threads": 1,
        "seed": 13,
        "verbosity": Verbosity.SILENT,
    }

    built_in = MultiOutputGBDT(out_dim=2, params=params)
    built_in.set_data((x_train, y_train))
    built_in.train(4)

    custom = MultiOutputGBDT(out_dim=2, params=params)
    custom.set_data((x_train, y_train))
    custom.train(4, objective=mse_objective)

    np.testing.assert_allclose(custom.predict(x_train), built_in.predict(x_train))

    built_in.close()
    custom.close()


def test_singleoutput_manual_set_gh_accepts_1d_and_column_vector():
    rng = np.random.default_rng(9)
    x_train = rng.random((64, 3)).astype("float64")
    y_train = rng.random(64).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    booster = SingleOutputGBDT(params=params)
    booster.set_data((x_train, y_train))

    grad_1d = booster.preds_train.copy() - booster.label.copy()
    hess_1d = np.ones_like(grad_1d)
    booster._set_gh(grad_1d, hess_1d)
    booster.boost()
    preds_1d = booster.predict(x_train[:4])

    booster.reset()

    grad_2d = grad_1d.reshape(-1, 1)
    hess_2d = hess_1d.reshape(-1, 1)
    booster._set_gh(grad_2d, hess_2d)
    booster.boost()
    preds_2d = booster.predict(x_train[:4])

    np.testing.assert_allclose(preds_1d, preds_2d)
    booster.close()


def test_custom_objective_rejects_invalid_shapes_and_dtypes():
    rng = np.random.default_rng(10)
    x_train = rng.random((48, 3)).astype("float64")
    y_single = rng.random(48).astype("float64")
    y_multi = rng.random((48, 2)).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 2,
        "num_threads": 1,
        "verbosity": Verbosity.SILENT,
    }

    single = SingleOutputGBDT(params=params)
    single.set_data((x_train, y_single))
    with pytest.raises(ValueError, match="shape"):
        single.train(1, objective=invalid_single_shape_objective)

    multi = MultiOutputGBDT(out_dim=2, params=params)
    multi.set_data((x_train, y_multi))
    with pytest.raises(ValueError, match="floating-point"):
        multi.train(1, objective=invalid_integer_objective)

    single.close()
    multi.close()


def test_custom_eval_metric_controls_verbosity_and_early_stopping(tmp_path, capfd):
    rng = np.random.default_rng(11)
    x_train = rng.random((96, 4)).astype("float64")
    y_train = np.column_stack(
        [
            1.4 * x_train[:, 0] - 0.3 * x_train[:, 1],
            0.8 * x_train[:, 2] + 0.2 * x_train[:, 3],
        ]
    ).astype("float64")
    x_valid = rng.random((24, 4)).astype("float64")
    y_valid = np.column_stack(
        [
            1.4 * x_valid[:, 0] - 0.3 * x_valid[:, 1],
            0.8 * x_valid[:, 2] + 0.2 * x_valid[:, 3],
        ]
    ).astype("float64")
    params = {
        "loss": b"mse",
        "lr": 0.1,
        "max_depth": 3,
        "num_threads": 1,
        "seed": 17,
        "early_stop": 1,
        "verbosity": Verbosity.FULL,
    }

    booster = MultiOutputGBDT(out_dim=2, params=params)
    booster.set_data((x_train, y_train), (x_valid, y_valid))
    metric = EvalMetricSequence([3.0, 2.0, 2.5], eval_size=len(x_valid))
    booster.train(5, objective=mse_objective, eval_metric=metric, maximize=False)

    output = capfd.readouterr().out
    assert "[0] train->" in output
    assert "Best score 2.0 at round 1" in output

    control = MultiOutputGBDT(
        out_dim=2,
        params={**params, "early_stop": 0, "verbosity": Verbosity.SILENT},
    )
    control.set_data((x_train, y_train), (x_valid, y_valid))
    control.train(2, objective=mse_objective)

    np.testing.assert_allclose(booster.predict(x_valid), control.predict(x_valid))

    model_path = tmp_path / "custom_early_stop.txt"
    booster.dump(model_path)
    assert model_path.read_text().count("Booster[") == 2

    loaded = MultiOutputGBDT(out_dim=2, params={**params, "verbosity": Verbosity.SILENT})
    loaded.set_booster(x_train.shape[1], 2)
    loaded.load(model_path)
    np.testing.assert_allclose(loaded.predict(x_valid), control.predict(x_valid))

    booster.close()
    control.close()
    loaded.close()


def test_public_deterministic_flag_is_normalized_on_python_wrappers():
    """Ensure deterministic CPU mode is exposed on the direct Python wrappers."""
    single = SingleOutputGBDT(params={"loss": b"mse", "deterministic": True})
    assert single.params["deterministic"] is True
    assert single.deterministic is True

    multi = MultiOutputGBDT(out_dim=2, params={"loss": b"mse", "deterministic": False})
    assert multi.params["deterministic"] is False
    assert multi.deterministic is False

    with pytest.raises(ValueError, match="deterministic"):
        SingleOutputGBDT(params={"loss": b"mse", "deterministic": "yes"})

    single.close()
    multi.close()


def test_optional_sklearn_wrappers_are_lazy():
    import omnigbdt

    if importlib.util.find_spec("sklearn") is None:
        with pytest.raises(ImportError, match="scikit-learn"):
            getattr(omnigbdt, "SingleOutputGBDTRegressor")
        with pytest.raises(ImportError, match="scikit-learn"):
            getattr(omnigbdt, "MultiOutputGBDTRegressor")
    else:
        from omnigbdt import MultiOutputGBDTRegressor, SingleOutputGBDTRegressor

        assert MultiOutputGBDTRegressor is not None
        assert SingleOutputGBDTRegressor is not None


def test_singleoutput_sklearn_wrapper_supports_permutation_importance():
    pytest.importorskip("sklearn")
    from sklearn.inspection import permutation_importance

    from omnigbdt import SingleOutputGBDTRegressor

    rng = np.random.default_rng(5)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = (x_train[:, 0] * 2.0 - x_train[:, 1]).astype("float64")

    model = SingleOutputGBDTRegressor(
        num_rounds=5,
        max_depth=3,
        deterministic=True,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8,)
    assert model.booster_.params["deterministic"] is True

    result = permutation_importance(
        model,
        x_train,
        y_train,
        scoring="r2",
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )

    assert result.importances_mean.shape == (x_train.shape[1],)
    model.close()


def test_multioutput_sklearn_wrapper_smoke():
    pytest.importorskip("sklearn")

    from omnigbdt import MultiOutputGBDTRegressor

    rng = np.random.default_rng(6)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = np.column_stack(
        [
            x_train[:, 0] + x_train[:, 1],
            x_train[:, 2] - x_train[:, 3],
        ]
    ).astype("float64")

    model = MultiOutputGBDTRegressor(
        num_rounds=5,
        max_depth=3,
        deterministic=True,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8, y_train.shape[1])
    assert model.booster_.params["deterministic"] is True
    assert np.isfinite(model.score(x_train, y_train))
    model.close()


def test_singleoutput_sklearn_wrapper_supports_custom_objective():
    pytest.importorskip("sklearn")

    from omnigbdt import SingleOutputGBDTRegressor

    rng = np.random.default_rng(12)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = (1.5 * x_train[:, 0] - 0.8 * x_train[:, 2]).astype("float64")

    model = SingleOutputGBDTRegressor(
        num_rounds=4,
        objective=mse_objective,
        eval_metric=rmse_metric,
        maximize=False,
        max_depth=3,
        deterministic=True,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8,)
    assert np.isfinite(model.score(x_train, y_train))
    model.close()


def test_multioutput_sklearn_wrapper_supports_custom_objective():
    pytest.importorskip("sklearn")

    from omnigbdt import MultiOutputGBDTRegressor

    rng = np.random.default_rng(13)
    x_train = rng.random((128, 4)).astype("float64")
    y_train = np.column_stack(
        [
            x_train[:, 0] + x_train[:, 1],
            x_train[:, 2] - x_train[:, 3],
        ]
    ).astype("float64")

    model = MultiOutputGBDTRegressor(
        num_rounds=4,
        objective=mse_objective,
        eval_metric=rmse_metric,
        maximize=False,
        max_depth=3,
        deterministic=True,
        num_threads=1,
        verbosity=Verbosity.SILENT,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_train[:8])

    assert preds.shape == (8, y_train.shape[1])
    assert np.isfinite(model.score(x_train, y_train))
    model.close()
