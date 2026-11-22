"""Per-tree input feature sampling contracts for the native and sklearn APIs."""

from itertools import product

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


@pytest.mark.parametrize("options", [
    None, {}, {"topk": 1}, {"topk": 1, "one_side": False},
    {"output_metric": [[.4, .1], [.1, .6]]},
    {"projection_dim": 1, "output_metric": [[.4, .1], [.1, .6]]},
])
@pytest.mark.parametrize("fraction, expected", [(1e-8, 1), (.59, 2)])
def test_tree_uses_one_subset_of_original_columns(options, fraction, expected, tmp_path):
    """Identify each tree's eligible input bits from a complete factorial design.

    Args:
        options: Vector split configuration, or None for a scalar booster.
        fraction: Input sampling fraction.
        expected: Number of independently predictive features available per tree.
        tmp_path: Directory for inspecting and reloading the model.
    """
    bits = np.array(list(product([0., 1.], repeat=5)))
    x = bits + 10 * np.arange(5)
    signal = bits @ np.arange(1, 6)
    y = signal if options is None else np.column_stack((signal, -2 * signal))
    cls = SingleOutputGBDT if options is None else MultiOutputGBDT
    params = dict(colsample_bytree=fraction, lr=.05, max_depth=4, min_samples=1,
                  reg_l2=0., gamma=0., seed=21, num_threads=1, verbosity=0)
    width = 1 if options is None else 2
    model = cls(out_dim=width, params={**params, **(options or {})})
    model.set_data((x, y), (x[::-1], None))
    model.train(10)
    path = tmp_path / "sampled.txt"
    model.dump(path)
    all_columns = set()
    for tree in path.read_text().split("Booster[")[1:]:
        selected = set()
        for line in tree.splitlines():
            if line.startswith("\t-"):
                fields = line.split(",")
                column, threshold = int(fields[4]), float(fields[5])
                assert threshold == 10 * column + .5
                selected.add(column)
        assert len(selected) == expected
        all_columns.update(selected)
    assert len(all_columns) > expected
    preds = model.predict(x)
    np.testing.assert_array_equal(model.preds_train, preds)
    np.testing.assert_array_equal(model.preds_eval, preds[::-1])
    reader = cls(out_dim=width, params={"num_threads": 1, "verbosity": 0})
    reader.set_data((x, y))
    reader.load(path)
    np.testing.assert_allclose(reader.predict(x), preds, rtol=1e-14, atol=1e-14)
    model.close()
    reader.close()


def test_column_sampling_replays_the_retained_tree_index():
    """Replay row, input and output sampling after an early-stopping rollback."""
    rng = np.random.default_rng(57)
    x = rng.normal(size=(180, 9))
    y = x[:, :5] + x[:, 4:9] ** 2
    settings = dict(colsample_bytree=.55, subsample=.8, projection_dim=2, seed=113,
                    min_samples=2, num_threads=1, early_stop=3, verbosity=0)
    reference = MultiOutputGBDT(out_dim=5, params=settings)
    reference.set_data((x, y))
    reference.train(12)
    stopped = MultiOutputGBDT(out_dim=5, params={**settings, "num_threads": 2})
    x_val = x[::3] + .1
    stopped.set_data((x, y), (x_val, reference.predict(x_val, 4)))
    stopped.train(12)
    assert stopped.lib.GetTreeCount(stopped._boostnode) == 4
    stopped.set_data(eval_set=(x_val, None))
    stopped.train(8)
    np.testing.assert_array_equal(stopped.predict(x), reference.predict(x))
    reference.close()
    stopped.close()


@pytest.mark.parametrize("fraction", [0., -1., 1.01, np.nan, np.inf, "0.8", True])
def test_invalid_column_fraction(fraction):
    """Reject invalid fractions before replacing the native handle.

    Args:
        fraction: Non-numeric, non-finite or out-of-range parameter.
    """
    model = MultiOutputGBDT(out_dim=2, params={"colsample_bytree": fraction})
    with pytest.raises((ValueError, TypeError), match="colsample_bytree"):
        model.set_booster(4, 2)


@pytest.mark.parametrize("multi", [False, True])
def test_sklearn_forwards_column_sampling(multi):
    """Compare the cloned wrapper to the corresponding direct booster.

    Args:
        multi: Whether to train a shared vector-leaf ensemble.
    """
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from omnigbdt import MultiOutputGBDTRegressor, SingleOutputGBDTRegressor

    rng = np.random.default_rng(42)
    x = rng.normal(size=(90, 6))
    y = x[:, :2] if multi else x[:, 0]
    cls = MultiOutputGBDTRegressor if multi else SingleOutputGBDTRegressor
    sampled = clone(cls(colsample_bytree=.3, num_rounds=5, num_threads=1, min_samples=2))
    assert sampled.get_params()["colsample_bytree"] == .3
    full = clone(sampled).set_params(colsample_bytree=1.)
    actual = sampled.fit(x, y).predict(x)
    direct_cls = MultiOutputGBDT if multi else SingleOutputGBDT
    direct = direct_cls(out_dim=2 if multi else 1, params=dict(
        colsample_bytree=.3, num_threads=1, min_samples=2, verbosity=0))
    direct.set_data((x, y))
    direct.train(5)
    np.testing.assert_array_equal(actual, direct.predict(x))
    assert not np.array_equal(actual, full.fit(x, y).predict(x))
    sampled.close()
    full.close()
    direct.close()
