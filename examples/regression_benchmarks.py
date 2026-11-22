"""Benchmark the repository regression examples against XGBoost 3.4.1 vector leaves."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

# Isolated workers can import the adjacent benchmark helper and installed wheels.
if sys.flags.isolated:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_benchmark_data import ENERGY_CSV_SHA256, energy_split, load_energy_csv
from scm_benchmark_data import SCM_ARCHIVE_SHA256, load_scm_split
from stock_portfolio_comparison import CSV_DIGEST, SETTINGS, fit_model, load_csv, partitions


PROFILES = {
    "OmniGBDT": ("OmniGBDT", False),
    "XGBoost matched": ("XGBoost matched", False),
    "OmniGBDT quadratic": ("OmniGBDT", True),
    "XGBoost quadratic": ("XGBoost matched", True),
}
STOCK_CASES = {"stock-rows": "rows", "stock-portfolios": "portfolios"}
CASE_SEEDS = {"stock-rows": 10, "stock-portfolios": 10, "readme": 3, "independent": 3}
ENERGY_CASES = ("energy-rows", "energy-geometries")
REPEATS = 11
PREDICTIONS_PER_REPEAT = 10


def example_split(case, seed, stock):
    """Return training, validation and held-out feature/target pairs.

    Args:
        case: Dataset key from CASE_SEEDS.
        seed: Split seed for stock or generator seed for a synthetic dataset.
        stock: Feature, target and portfolio ID arrays from the checked UCI CSV.

    Returns:
        Six contiguous arrays in train, validation and test pair order.
    """
    if case in STOCK_CASES:
        x, y, ids = stock
        train, valid, test = partitions(ids, STOCK_CASES[case], seed)
        return x[train], y[train], x[valid], y[valid], x[test], y[test]
    if case == "readme":
        x = np.random.default_rng(seed).normal(size=(400, 6))
        shared = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
        y = np.column_stack((shared + 0.3 * x[:, 4], shared - 0.2 * x[:, 5]))
        return x[:240], y[:240], x[240:320], y[240:320], x[320:], y[320:]
    # examples/test.py draws training and validation arrays with np.random.rand.
    # Seed its original generator and append an independent test batch.
    rng = np.random.RandomState(seed)
    return (rng.rand(10000, 10), rng.rand(10000, 5), rng.rand(10000, 10),
            rng.rand(10000, 5), rng.rand(10000, 10), rng.rand(10000, 5))


def coupled_metric(width):
    """Weight individual squared errors and squared error of their mean equally.

    Args:
        width: Number of regression targets.

    Returns:
        Positive-definite (width, width) matrix in the original target units.
    """
    return (np.eye(width) + np.ones((width, width)) / width) / (2 * width)


def error_scores(errors):
    """Score original target units with ordinary and coupled losses.

    Args:
        errors: (rows, targets) matrix of prediction errors.

    Returns:
        Pooled test RMSE and square root of the mean coupled error.
    """
    mean_square = np.mean(errors ** 2)
    coupled = 0.5 * (mean_square + np.mean(errors.mean(axis=1) ** 2))
    return float(np.sqrt(mean_square)), float(np.sqrt(coupled))


def fit_profile(profile, train_valid, settings):
    """Fit one model and include model-specific target setup in its fit time.

    Args:
        profile: Profile name from PROFILES.
        train_valid: Four feature/target arrays for training and validation.
        settings: Shared boosting parameters for this dataset.

    Returns:
        Model, prediction rounds, total fit time in milliseconds and optional
        inverse transform mapping XGBoost predictions to the original targets.
    """
    library, coupled = PROFILES[profile]
    setup_ms = 0.0
    metric = inverse = None
    if coupled:
        start = perf_counter_ns()
        x_train, y_train, x_valid, y_valid = train_valid
        width = y_train.shape[1]
        metric = coupled_metric(width)
        if library == "XGBoost matched":
            # XGBoost averages squares over target columns when monitoring RMSE.
            # Y @ L gives the desired scale because L @ L.T = width * metric.
            factor = np.linalg.cholesky(width * metric)
            inverse = np.linalg.inv(factor)
            train_valid = x_train, y_train @ factor, x_valid, y_valid @ factor
            metric = None
        setup_ms = (perf_counter_ns() - start) / 1e6
    model, rounds, fit_ms = fit_model(library, *train_valid, subsample=1.0,
                                      settings=settings, output_metric=metric)
    return model, rounds, setup_ms + fit_ms, inverse


def predict_original(x, *, model, iteration_range, inverse):
    """Return XGBoost's transformed-label predictions in original target units.

    Args:
        x: Feature matrix for the entire prediction batch.
        model: Fitted vector-leaf XGBoost model.
        iteration_range: Start and stop boosting rounds for prediction.
        inverse: Inverse of the transform used to train the model.

    Returns:
        Original-unit prediction matrix with one column per target.
    """
    return model.inplace_predict(x, iteration_range=iteration_range) @ inverse


def measure_case(profile, case, data_splits):
    """Score each split and time warm fits and full-batch predictions on seed zero.

    Args:
        profile: OmniGBDT or matched XGBoost vector-leaf model.
        case: Benchmark case name, used to select its boosting settings.
        data_splits: Iterable of train, validation and test feature/target pairs
            in split-seed order. SCM20D has one fixed split.

    Returns:
        Split scores, retained rounds, row counts and individual timings in ms.
    """
    settings = dict(SETTINGS)
    library, coupled = PROFILES[profile]
    if case == "independent":
        settings.update(max_depth=5, learning_rate=0.2, round_cap=20)
    splits, fit_ms, predict_ms = [], [], []
    for seed, arrays in enumerate(data_splits):
        x_train, y_train, x_valid, y_valid, x_test, y_test = arrays
        train_valid = (x_train, y_train, x_valid, y_valid)
        model, rounds, _, inverse = fit_profile(profile, train_valid, settings)
        if library == "OmniGBDT":
            predict, kwargs = model.predict, {}
        else:
            predict, kwargs = model.inplace_predict, {"iteration_range": (0, rounds)}
            if inverse is not None:
                predict, kwargs = predict_original, {**kwargs, "inverse": inverse, "model": model}
            if seed == 0:
                config = json.loads(model.save_config())["learner"]
                assert config["learner_train_param"]["multi_strategy"] == "multi_output_tree"
                trees = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
                assert len(trees) == model.num_boosted_rounds()
                assert all(int(t["tree_param"]["size_leaf_vector"]) == y_test.shape[1] for t in trees)

        predictions = predict(x_test, **kwargs)
        assert predictions.shape == y_test.shape and np.isfinite(predictions).all()
        test_rmse, test_quadratic_rmse = error_scores(predictions - y_test)
        baseline_rmse, baseline_quadratic_rmse = error_scores(y_train.mean(axis=0) - y_test)
        if coupled:
            width = y_test.shape[1]
            factor = np.linalg.cholesky(width * coupled_metric(width))
            projected = predictions @ factor
            if inverse is not None:
                raw = model.inplace_predict(x_test, iteration_range=(0, rounds))
                np.testing.assert_allclose(projected, raw, rtol=1e-12, atol=1e-12)
            projected_rmse = np.sqrt(np.mean((projected - y_test @ factor) ** 2))
            np.testing.assert_allclose(test_quadratic_rmse, projected_rmse, rtol=1e-12, atol=1e-12)
        splits.append({
            "seed": seed, "rounds": rounds, "rows": [len(x_train), len(x_valid), len(x_test)],
            "outputs": y_test.shape[1], "test_rmse": test_rmse, "test_quadratic_rmse": test_quadratic_rmse,
            "baseline_rmse": baseline_rmse, "baseline_quadratic_rmse": baseline_quadratic_rmse,
        })
        if seed == 0:
            # The score above warmed the fitted model on this same contiguous input batch.
            for _ in range(REPEATS):
                start = perf_counter_ns()
                for _ in range(PREDICTIONS_PER_REPEAT):
                    batch = predict(x_test, **kwargs)
                predict_ms.append((perf_counter_ns() - start) / (1e6 * PREDICTIONS_PER_REPEAT))
                np.testing.assert_array_equal(batch, predictions)
            for _ in range(REPEATS):
                repeat, repeat_rounds, elapsed, _ = fit_profile(profile, train_valid, settings)
                assert repeat_rounds == rounds
                fit_ms.append(elapsed)
                del repeat
    return {
        "settings": settings, "splits": splits,
        "mean_test_rmse": statistics.mean(split["test_rmse"] for split in splits),
        "mean_test_quadratic_rmse": statistics.mean(split["test_quadratic_rmse"] for split in splits),
        "mean_baseline_rmse": statistics.mean(split["baseline_rmse"] for split in splits),
        "mean_baseline_quadratic_rmse": statistics.mean(split["baseline_quadratic_rmse"] for split in splits),
        "seed0_fit_ms": fit_ms, "seed0_predict_ms": predict_ms,
        "median_fit_ms": statistics.median(fit_ms), "median_predict_ms": statistics.median(predict_ms),
    }


def run_worker(profile, csv_path, energy_csv=None, scm_dir=None):
    """Run one model profile across the requested datasets in its native runtime.

    Args:
        profile: Regression model to benchmark in this process.
        csv_path: Original UCI Stock Portfolio Performance CSV.
        energy_csv: Optional path to the UCI Energy Efficiency CSV.
        scm_dir: Optional directory containing the SCM20D publisher ZIPs.

    Returns:
        Loaded package version, native library name and results for each dataset.

    Raises:
        RuntimeError: If a package version, native feature or library location differs.
    """
    library, _ = PROFILES[profile]
    if library == "OmniGBDT":
        from omnigbdt import load_lib

        lib = load_lib()
        native = Path(lib._name).resolve()
        package_version = version("omnigbdt")
        if package_version != "0.7.0" or not hasattr(lib, "SetOutputMetric"):
            raise RuntimeError(f"Expected OmniGBDT 0.7.0 with native output_metric, found {package_version}.")
    else:
        import xgboost as xgb

        package_version = xgb.__version__
        if package_version != "3.4.1":
            raise RuntimeError(f"Expected XGBoost 3.4.1, found {package_version}.")
        native = Path(xgb.core._LIB._name).resolve()
    if not {"site-packages", "dist-packages"}.intersection(native.parts):
        raise RuntimeError(f"Benchmark requires a native library from an installed wheel, found {native}.")
    stock = load_csv(csv_path)
    cases = {}
    for case, seed_count in CASE_SEEDS.items():
        splits = (example_split(case, seed, stock) for seed in range(seed_count))
        cases[case] = measure_case(profile, case, splits)
    if energy_csv is not None:
        energy = load_energy_csv(energy_csv)
        for case in ENERGY_CASES:
            splits = (energy_split(case, seed, energy) for seed in range(10))
            cases[case] = measure_case(profile, case, splits)
    if scm_dir is not None:
        cases["scm20d"] = measure_case(profile, "scm20d", [load_scm_split(scm_dir)])
    return {"version": package_version, "native_library": native.name, "cases": cases}


def main():
    """Run each model profile in isolation and write the scores and timings as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="UCI Stock Portfolio Performance CSV")
    parser.add_argument("output", type=Path, help="Destination JSON for scores and timings")
    parser.add_argument("--energy-csv", type=Path, help="UCI Energy Efficiency CSV")
    parser.add_argument("--scm-dir", type=Path, help="Directory with scm20d-train.zip and scm20d-test.zip")
    parser.add_argument("--worker", choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args.worker, args.csv_path, args.energy_csv, args.scm_dir), allow_nan=False))
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")
    results = {
        "stock_sha256": CSV_DIGEST, "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "fit_repeats": REPEATS, "predict_repeats": REPEATS,
        "predictions_per_repeat": PREDICTIONS_PER_REPEAT, "profiles": {},
        "quadratic_metric": "M = (I + ones(d, d) / d) / (2 * d)",
    }
    dataset_args = []
    if args.energy_csv is not None:
        results["energy_sha256"] = ENERGY_CSV_SHA256
        dataset_args.extend(["--energy-csv", str(args.energy_csv)])
    if args.scm_dir is not None:
        results["scm_archives_sha256"] = SCM_ARCHIVE_SHA256
        dataset_args.extend(["--scm-dir", str(args.scm_dir)])
    for profile in PROFILES:
        command = [sys.executable, "-I", str(Path(__file__).resolve()), str(args.csv_path),
                   str(args.output), "--worker", profile, *dataset_args]
        child = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True, env=env)
        results["profiles"][profile] = json.loads(child.stdout)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
