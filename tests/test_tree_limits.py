import os
import subprocess
import sys

import numpy as np
import pytest

from omnigbdt import MultiOutputGBDT, SingleOutputGBDT


@pytest.mark.parametrize("model_kind", ["single", "dense", "sparse"])
@pytest.mark.parametrize("max_leaves,hist_cache", [(1, 16), (3, 16), (16, 1)])
def test_tree_limits_preserve_fitted_leaves(model_kind, max_leaves, hist_cache):
    """Check fitted leaf weights even when a budget stops expansion.

    Args:
        model_kind: Single-output, dense multi-output, or sparse multi-output mode.
        max_leaves: Upper bound on the number of terminal leaves.
        hist_cache: Maximum number of pending split candidates.
    """
    x = np.arange(32.0)[:, None]
    target = x[:, 0] + 1
    params = dict(
        base_score=0.0, lr=1.0, reg_l2=0.0, min_samples=1, gamma=0.0,
        num_threads=1, verbosity=0, max_depth=4, max_leaves=max_leaves,
        hist_cache=hist_cache,
    )
    if model_kind == "single":
        y = target
        model = SingleOutputGBDT(params=params)
    else:
        y = np.column_stack((target, np.zeros_like(target)))
        model = MultiOutputGBDT(out_dim=2, params={**params, "topk": int(model_kind == "sparse")})
    try:
        model.set_data((x, y))
        model.train(1)
        predictions = model.predict(x)
        if model_kind != "single":
            np.testing.assert_array_equal(predictions[:, 1], 0.0)
            predictions = predictions[:, 0]
        unique = np.unique(predictions)
        assert 1 <= len(unique) <= max_leaves
        for value in unique:
            assert value == pytest.approx(target[predictions == value].mean())
        if max_leaves == 3:
            # Equal-gain candidates are considered in left-to-right order.
            np.testing.assert_allclose(predictions, np.repeat([4.5, 12.5, 24.5], [8, 8, 16]))
        elif hist_cache == 1:
            assert len(unique) > 1
    finally:
        model.close()


@pytest.mark.parametrize("out_dim", [1, 2])
def test_prediction_with_smaller_openmp_team(tmp_path, out_dim):
    """Compare batch prediction to serial slices with one available worker.

    Args:
        tmp_path: Temporary directory for the trained model.
        out_dim: Number of outputs used in the model.
    """
    x = np.arange(32.0)[:, None]
    y = x[:, 0] + 1
    params = {"num_threads": 1, "verbosity": 0, "base_score": 0.0, "min_samples": 2}
    if out_dim == 1:
        model = SingleOutputGBDT(params=params)
    else:
        model = MultiOutputGBDT(out_dim=out_dim, params=params)
        y = np.column_stack((y, -y))
    path = tmp_path / "model.txt"
    try:
        model.set_data((x, y))
        model.train(2)
        model.dump(path)
        library = model.lib._name
    finally:
        model.close()

    # OpenMP reads its team limit at process startup.
    code = """
import sys
import numpy as np
from omnigbdt import MultiOutputGBDT, SingleOutputGBDT, load_lib

library, path, width = sys.argv[1:]
params = {"num_threads": 2}
lib = load_lib(library)
if int(width) == 1:
    model = SingleOutputGBDT(lib=lib, params=params)
    model.set_booster(1)
else:
    model = MultiOutputGBDT(lib=lib, out_dim=int(width), params=params)
    model.set_booster(1, int(width))
model.load(path)
x = (np.arange(301.0) % 32)[:, None]
expected = np.concatenate([model.predict(x[:150]), model.predict(x[150:])])
np.testing.assert_array_equal(model.predict(x), expected)
model.close()
"""
    subprocess.run(
        [sys.executable, "-c", code, library, str(path), str(out_dim)],
        env={**os.environ, "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "1"},
        check=True,
    )
