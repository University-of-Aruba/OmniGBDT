"""Compare a vector-leaf regression profile using nested group cross-validation.

Run each method and loss as a separate process. Each repeat scores every row
once. All three methods use the same outer and inner group assignments.
"""

import argparse
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import median
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

if sys.flags.isolated:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_benchmark_data import load_energy_csv
from regression_benchmarks import coupled_metric, error_scores
from scm_benchmark_data import _read_archive
from stock_portfolio_comparison import load_csv


CAP = 200
PATIENCE = 15
DEPTHS = (2, 4)
REPEATS = (20261121, 20261122, 20261123)
DATASET_ID = {"stock": 1, "energy": 2, "scm": 3}


def fit_omni(x, y, features, validation, depth, rounds, *, metric, projected, native):
    """Fit OmniGBDT and return predictions at the selected ensemble prefix.

    Args:
        x: Fitting features.
        y: Fitting targets in original units.
        features: Inner validation or outer scoring features.
        validation: Inner validation targets, or None for fixed-round outer fitting.
        depth: Maximum tree depth.
        rounds: Inner cap or fixed outer round count.
        metric: Coupled metric matrix, or None for ordinary MSE.
        projected: Whether to use Gaussian projection for split selection.
        native: Loaded OmniGBDT native library.

    Returns:
        Original-coordinate predictions and selected round count.
    """
    from omnigbdt import MultiOutputGBDT

    width = y.shape[1]
    params = dict(loss=b"mse", max_bins=128, max_depth=depth, lr=0.05, min_samples=20,
                  reg_l1=0., reg_l2=1., gamma=0.001, num_threads=1, subsample=1.,
                  early_stop=PATIENCE if validation is not None else 0, verbosity=0)
    if metric is not None:
        params["output_metric"] = metric
    if projected:
        params.update(seed=17, projection_dim=1 if width == 2 else 2)
    model = MultiOutputGBDT(lib=native, out_dim=width, params=params)
    try:
        model.set_data((x, y), None if validation is None else (features, validation))
        model.train(rounds)
        retained = model.lib.GetTreeCount(model._boostnode)
        if validation is None:
            assert retained == rounds
        elif retained == CAP:
            # At the cap the native booster keeps all trees when patience has
            # not expired. Its earliest best prefix is inside this final window.
            metric_index = int(metric is not None)
            window = range(CAP - PATIENCE + 1, CAP + 1)
            retained = min(window, key=lambda count:
                           error_scores(model.predict(features, count) - validation)[metric_index])
        return model.predict(features, retained), retained
    finally:
        model.close()


def fit_xgb(x, y, features, validation, depth, rounds, *, metric):
    """Fit XGBoost's shared vector-leaf model and return its best prefix.

    Args:
        x: Fitting features.
        y: Fitting targets in original units.
        features: Inner validation or outer scoring features.
        validation: Inner validation targets, or None for fixed-round outer fitting.
        depth: Maximum tree depth.
        rounds: Inner cap or fixed outer round count.
        metric: Coupled metric matrix, or None for ordinary MSE.

    Returns:
        Original-coordinate predictions and selected round count.
    """
    import xgboost as xgb

    width = y.shape[1]
    factor = None if metric is None else np.linalg.cholesky(width * metric)
    train = xgb.DMatrix(x, label=y if factor is None else y @ factor, nthread=1)
    params = dict(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                  multi_strategy="multi_output_tree", device="cpu", max_depth=depth,
                  max_bin=128, eta=0.05, reg_lambda=1., reg_alpha=0., min_child_weight=20,
                  gamma=0.002 * width, nthread=1, seed=0, subsample=1.)
    options = {}
    if validation is not None:
        targets = validation if factor is None else validation @ factor
        valid = xgb.DMatrix(features, label=targets, nthread=1)
        options = dict(evals=[(valid, "validation")], early_stopping_rounds=PATIENCE)
    model = xgb.train(params, train, rounds, verbose_eval=False, **options)
    used = rounds if validation is None else model.best_iteration + 1
    prediction = model.inplace_predict(features, iteration_range=(0, used))
    if validation is None:
        trees = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
        assert len(trees) == rounds and all(int(tree["tree_param"]["size_leaf_vector"]) == width for tree in trees)
    if factor is None:
        return prediction.astype(np.float64), used
    original = np.linalg.solve(factor.T, prediction.T).T
    np.testing.assert_allclose(original @ factor, prediction, rtol=1e-12, atol=1e-12)
    return original, used


def evaluate_depth(x, y, groups, inner, depth, fit, index):
    """Evaluate one depth on every inner validation row using grouped stopping.

    Args:
        x: Outer training features.
        y: Outer training targets.
        groups: Group ID for each outer training row.
        inner: Three disjoint validation group blocks covering the training groups.
        depth: Candidate depth shared across libraries.
        fit: Model-specific fit and prediction function.
        index: Objective index in the paired ordinary/coupled error scores.

    Returns:
        Inner out-of-fold scores, selected prefix per fold and row counts.
    """
    predictions = np.full_like(y, np.nan)
    rounds, counts = [], []
    for held in inner:
        valid = np.isin(groups, held)
        train = ~valid
        assert train.any() and valid.any() and not np.isfinite(predictions[valid]).any()
        predictions[valid], chosen = fit(x[train], y[train], x[valid], y[valid], depth, CAP)
        rounds.append(chosen)
        counts.append([int(train.sum()), int(valid.sum())])
    assert np.isfinite(predictions).all()
    metrics = error_scores(predictions - y)
    return dict(depth=depth, score=metrics[index], metrics=metrics, best_rounds=rounds, row_counts=counts)


def run_repeat(name, data, repeat, fit, index, output):
    """Select parameters inside each outer training partition and score every row once.

    Args:
        name: Dataset key defining group-fold count and RNG namespace.
        data: Features, targets, and numeric group IDs.
        repeat: Seed defining the shuffled outer group assignment.
        fit: Model-specific fit and predict function.
        index: Objective index in the two error metrics.
        output: Path for the out-of-fold prediction archive for this repeat.

    Returns:
        Fold manifests, selections, original-unit scores and prediction digest.
    """
    x, y, groups = data
    unique_groups = np.unique(groups)
    permutation = np.random.default_rng(repeat).permutation(unique_groups)
    outer = np.array_split(permutation, 4 if name == "energy" else 5)
    np.testing.assert_array_equal(np.sort(np.concatenate(outer)), unique_groups)
    predictions = np.full_like(y, np.nan)
    records = []
    for fold, held in enumerate(outer):
        test = np.isin(groups, held)
        train = ~test
        training_groups = np.unique(groups[train])
        assert not set(training_groups).intersection(held)
        rng = np.random.default_rng(np.random.SeedSequence([20261122, DATASET_ID[name], repeat, fold]))
        inner = np.array_split(rng.permutation(training_groups), 3)
        np.testing.assert_array_equal(np.sort(np.concatenate(inner)), training_groups)

        candidates = [evaluate_depth(x[train], y[train], groups[train], inner, depth, fit, index)
                      for depth in DEPTHS]
        best = min(candidates, key=lambda candidate: candidate["score"])
        rounds = int(median(best["best_rounds"]))
        pred, used = fit(x[train], y[train], x[test], None, best["depth"], rounds)
        assert used == rounds and pred.shape == y[test].shape and np.isfinite(pred).all()
        assert not np.isfinite(predictions[test]).any()
        predictions[test] = pred
        records.append(dict(fold=fold, test_groups=held.tolist(), inner_val_groups=[ids.tolist() for ids in inner],
                            candidates=candidates, depth=best["depth"], rounds=rounds, test_rows=int(test.sum()),
                            test_scores=error_scores(pred - y[test])))
    assert np.isfinite(predictions).all()
    np.savez_compressed(output, predictions=predictions)
    errors = predictions - y
    by_group = {str(group): error_scores(errors[groups == group]) for group in unique_groups}
    return dict(repeat=repeat, folds=records, group_scores=by_group, oof_scores=error_scores(errors),
                rows=len(y), targets=y.shape[1], predictions=output.name,
                prediction_sha256=sha256(predictions.tobytes()).hexdigest())


def main():
    """Run one model and loss in an isolated process and write its audit records."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stock", "energy", "scm", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--method", choices=["full", "projected", "xgboost"], required=True)
    parser.add_argument("--loss", choices=["mse", "coupled"], required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The official eight-game test archive is excluded from this group CV.
    raw = _read_archive(args.scm / "scm20d-train.zip")
    game = np.cumsum(np.r_[False, np.diff(raw[:, 0]) < 0])
    assert (len(raw), len(np.unique(game))) == (7463, 40)
    datasets = {"stock": load_csv(args.stock), "energy": load_energy_csv(args.energy),
                "scm": (raw[:, :61], raw[:, 61:], game)}
    if args.method == "xgboost":
        import xgboost as xgb
        if xgb.__version__ != "3.4.1":
            raise RuntimeError(f"Expected XGBoost 3.4.1, got {xgb.__version__}.")
        library = Path(xgb.core._LIB._name)
        package = "xgboost"
        fit = fit_xgb
    else:
        from omnigbdt import load_lib
        native = load_lib()
        if args.method == "projected" and not hasattr(native, "SetProjectionDim"):
            raise RuntimeError("Install a build with native random projection support.")
        library = Path(native._name)
        package = "omnigbdt"
        fit = partial(fit_omni, projected=args.method == "projected", native=native)

    report = dict(method=args.method, loss=args.loss, numpy=version("numpy"), version=version(package),
                  library_sha256=sha256(library.read_bytes()).hexdigest(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), cases={})
    label = f"{args.method}-{args.loss}"
    for name, data in datasets.items():
        metric = None if args.loss == "mse" else coupled_metric(data[1].shape[1])
        predict = partial(fit, metric=metric)
        repeats = REPEATS[:1] if name == "scm" else REPEATS
        report["cases"][name] = [
            run_repeat(name, data, seed, predict, int(metric is not None), args.out / f"{label}-{name}-{seed}.npz")
            for seed in repeats
        ]
    (args.out / f"{label}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
