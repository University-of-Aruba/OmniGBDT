"""Regression tests for retained rounds, native metrics, and saved trees."""

import errno
import os
from pathlib import Path

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT, create_graph


@pytest.mark.parametrize("multi", [False, True])
def test_builtin_early_stopping_continues_from_retained_rounds(multi):
    """Compare two early-stopped fits with two uninterrupted boosting rounds.

    Args:
        multi: Whether to fit two outputs jointly.
    """
    x = np.ones((20, 1))
    y = np.ones(20) if not multi else np.tile([1., -2.], (20, 1))
    params = dict(base_score=0., lr=0.5, reg_l2=0., early_stop=1, num_threads=1, verbosity=0)
    cls = MultiOutputGBDT if multi else SingleOutputGBDT
    model = cls(out_dim=2 if multi else 1, params=params)
    reference = cls(out_dim=2 if multi else 1, params=params)
    model.set_data((x, y), (x[:6], -y[:6]))
    reference.set_data((x, y))
    for _ in range(2):
        model.train(5)
        reference.train(1)
        expected = reference.predict(x)
        np.testing.assert_array_equal(model.predict(x), expected)
        np.testing.assert_array_equal(model.preds_train, expected)
        np.testing.assert_array_equal(model.preds_eval, expected[:6])


@pytest.mark.parametrize("n", [12, 96])
def test_legacy_class_rounds_and_column_loss(n, capfd, tmp_path):
    """Check complete class groups, tied accuracy, and mean cross-entropy.

    Args:
        n: Training size on each side of the native parallel scoring threshold.
        capfd: Native output capture fixture.
        tmp_path: Temporary directory for the saved ensemble.
    """
    labels = np.arange(n, dtype=np.int32) % 3
    x = labels.astype(np.float64)[:, None]
    model = SingleOutputGBDT(out_dim=3, params=dict(
        loss=b"ce_column", base_score=0.25, num_threads=2, min_samples=1,
        early_stop=1, verbosity=2, gamma=0.,
    ))
    model.set_data((x, labels), (x, labels))
    with pytest.raises(ValueError, match="out_dim"):
        model._set_gh(np.ones(3 * n), np.ones(3 * n))
    model.train_multi(4)
    output = capfd.readouterr().out
    pred = model.predict(x)
    np.testing.assert_array_equal(pred.argmax(axis=1), labels)
    expected_loss = np.mean(np.logaddexp.reduce(pred, axis=1) - pred[np.arange(n), labels])
    assert f"train->{expected_loss:.5f}" in output
    assert "at round 0" in output
    for fit_number in range(2):
        np.testing.assert_array_equal(model.preds_train.reshape(3, n).T, model.predict(x))
        np.testing.assert_array_equal(model.preds_eval.reshape(3, n).T, model.predict(x))
        np.testing.assert_array_equal(model.predict(x, num_trees=1), pred)
        np.testing.assert_array_equal(model.predict(x, num_trees=999), model.predict(x))
        if fit_number == 0:
            model.train_multi(4)
    path = tmp_path / "legacy.txt"
    model.dump(path)
    assert path.read_text().count("Booster[") == 6
    expected = model.predict(x)
    model.reset()
    np.testing.assert_array_equal(model.preds_train, 0.25)
    model.load(path)
    np.testing.assert_array_equal(model.predict(x), expected)


def test_extreme_binary_log_loss(capfd):
    """Check that a large finite margin produces the expected finite log loss.

    Args:
        capfd: Native output capture fixture.
    """
    x = np.ones((4, 1))
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    model = SingleOutputGBDT(params=dict(loss=b"bce", base_score=-1000., lr=0., num_threads=1))
    model.set_data((x, y))
    model.train(1)
    assert "score->500.00000" in capfd.readouterr().out


def test_model_file_failures_preserve_predictions(tmp_path):
    """Check I/O errors, incompatible widths and headerless legacy loading.

    Args:
        tmp_path: Temporary directory for model files.
    """
    model = MultiOutputGBDT(out_dim=2, params=dict(base_score=[1., 2.], num_threads=1))
    model.set_booster(1, 2)
    x = np.ones((6, 1))
    path = tmp_path / "tree.txt"
    with pytest.raises(FileNotFoundError):
        model.load(path)
    with pytest.raises(OSError):
        model.dump(tmp_path)
    path.write_text("BaseScore:6,8\n")
    model.load(path)
    np.testing.assert_array_equal(model.predict(x), np.tile([6., 8.], (6, 1)))
    path.write_bytes(b"")
    model.load(path)
    np.testing.assert_array_equal(model.predict(x), np.tile([1., 2.], (6, 1)))
    path.write_text("Booster[0]:\n\t\t1,2.5,3.5\n")
    model.load(path)
    expected = model.predict(x)
    np.testing.assert_array_equal(expected, np.tile([3.5, 5.5], (6, 1)))
    for content in ("BaseScore:4\n", "Booster[0]:\n\t\t1,0.3\n", "Booster[0]:\n"):
        path.write_text(content)
        with pytest.raises(ValueError):
            model.load(path)
        np.testing.assert_array_equal(model.predict(x), expected)


def test_plot_root_leaf_and_split_tree(tmp_path):
    """Plot original dumps with a leaf root and with a split root.

    Args:
        tmp_path: Temporary directory containing a two-tree model file.
    """
    pytest.importorskip("graphviz")
    path = tmp_path / "plot.txt"
    path.write_text(
        "BaseScore:0,0\nBooster[0]:\n\t\t1,2.5,3.5\n"
        "Booster[1]:\n\t-1,-1,1,2,0,0.5\n\t\t1,0,1\n\t\t2,2,3\n"
    )
    leaf = create_graph(path, value_list=[1]).source
    assert "3.5000" in leaf and "->" not in leaf
    split = create_graph(path, tree_index=1).source
    assert split.count("->") == 2
    assert "X[0] 0.5000" in split


def test_partial_dump_preserves_existing_model(tmp_path, monkeypatch):
    """Retain a readable model after the native writer fails partway through.

    Args:
        tmp_path: Directory containing the original model.
        monkeypatch: Fixture that simulates a failed native write.
    """
    model = SingleOutputGBDT(params=dict(num_threads=1, base_score=7.0))
    model.set_booster(1)
    path = tmp_path / "existing.txt"
    model.dump(path)
    original = path.read_bytes()

    def fail_during_write(handle, encoded_path):
        """Write a partial file and report a disk-full condition.

        Args:
            handle: Native booster handle.
            encoded_path: Native byte-string output path.

        Returns:
            The disk-full errno.
        """
        Path(os.fsdecode(encoded_path)).write_bytes(b"partial")
        return errno.ENOSPC

    monkeypatch.setattr(model.lib, "DumpChecked", fail_during_write)
    with pytest.raises(OSError) as failure:
        model.dump(path)
    assert failure.value.errno == errno.ENOSPC
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    model.load(path)
    np.testing.assert_array_equal(model.predict(np.ones((2, 1))), [7.0, 7.0])


def test_identical_rmse_never_extends_patience(tmp_path):
    """Check strict RMSE ties on a parallel evaluation split.

    Args:
        tmp_path: Directory containing the resulting model.
    """
    rng = np.random.default_rng(730)
    x = np.zeros((16, 1), dtype=np.float64)
    x_eval = np.zeros((4097, 1), dtype=np.float64)
    y_eval = rng.normal(size=len(x_eval))
    model = SingleOutputGBDT(params=dict(
        lr=0., early_stop=4, num_threads=4, base_score=0., verbosity=0
    ))
    model.set_data((x, np.ones(len(x))), (x_eval, y_eval))
    model.train(20)
    path = tmp_path / "plateau.txt"
    model.dump(path)
    assert path.read_text().count("Booster[") == 1
