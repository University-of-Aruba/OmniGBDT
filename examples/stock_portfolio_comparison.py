"""Compare OmniGBDT and XGBoost 3.4.1 vector-leaf regressors on UCI 390.

Download https://archive.ics.uci.edu/static/public/390/data.csv as stock.csv.
Run with installed wheels for OmniGBDT and XGBoost. The published comparison
used Python 3.13.9 and NumPy 2.5.5. Row sampling requires OmniGBDT 0.6.0 or
a build with native row sampling support.
"""

import argparse
import csv
import hashlib
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


FEATURES = ["Large B/P", "Large ROE", "Large S/P", "Large Return Rate in the last quarter",
            "Large Market Value", "Small systematic Risk"]
TARGETS = ["Annual Return", "Excess Return", "Systematic Risk", "Total Risk", "Abs. Win Rate",
           "Rel. Win Rate"]
CSV_DIGEST = "24a9c22e57c5a12398b5a6c0c639f0a403ad4d7255dceace2ed6031b7c8a8a19"
PROFILES = ["OmniGBDT", "XGBoost", "XGBoost matched"]
SETTINGS = {"max_depth": 4, "bins": 128, "learning_rate": 0.05, "round_cap": 200,
            "patience": 15, "threads": 1, "model_seed": 0, "L1": 0, "L2": 1}


def load_csv(path):
    """Read portfolio weights and normalized responses from UCI's original CSV.

    Args:
        path: Downloaded UCI 390 data.csv.

    Returns:
        Feature and target arrays, plus the portfolio ID for each observation.
    """
    if hashlib.sha256(path.read_bytes()).hexdigest() != CSV_DIGEST:
        raise ValueError("CSV does not match the UCI snapshot used by this benchmark.")
    with path.open(newline="", encoding="utf-8-sig") as source:
        headers, *rows = csv.reader(source)
    # The normalization block repeats six header names, appearing last in the CSV.
    assert headers == ["period", "ID", *FEATURES, *TARGETS, *TARGETS]
    x = np.array([r[2:8] for r in rows], dtype=np.float64)
    y = np.array([r[-6:] for r in rows], dtype=np.float64)
    ids = np.array([int(r[1]) for r in rows])
    assert x.shape == y.shape == (315, 6) and np.isfinite(x).all() and np.isfinite(y).all()
    groups, counts = np.unique(ids, return_counts=True)
    assert len(groups) == 63 and (counts == 5).all() and len(np.unique(x, axis=0)) == 63
    assert all((x[ids == g] == x[ids == g][0]).all() for g in groups)
    return x, y, ids


def partitions(ids, scheme, seed):
    """Make the documentation partition or partition unique portfolios.

    Args:
        ids: Portfolio group identifier for every row.
        scheme: Either rows or portfolios.
        seed: NumPy split seed.

    Returns:
        Three disjoint row-index arrays covering the data.
    """
    units = np.arange(len(ids)) if scheme == "rows" else np.unique(ids)
    units = np.random.default_rng(seed).permutation(units)
    parts = np.split(units, [int(0.6 * len(units)), int(0.8 * len(units))])
    if scheme == "portfolios":
        parts = [np.flatnonzero(np.isin(ids, p)) for p in parts]
        assert sum(len(np.unique(ids[p])) for p in parts) == 63
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids)))
    return parts


def fit_model(profile, x_train, y_train, x_valid, y_valid, subsample, *, settings=None, output_metric=None):
    """Fit one full-gradient joint model including native data preparation.

    Args:
        profile: Model and regularization setting.
        x_train: Training features.
        y_train: Training responses.
        x_valid: Validation features.
        y_valid: Validation responses for early stopping.
        subsample: Fraction of training rows used to fit each tree.
        settings: Benchmark settings. Defaults to the documented stock example.
        output_metric: Optional dense quadratic loss matrix for OmniGBDT.

    Returns:
        Booster, inference round count and elapsed fitting time in milliseconds.
    """
    s = SETTINGS if settings is None else settings
    if profile == "OmniGBDT":
        from omnigbdt import MultiOutputGBDT, Verbosity

        start = perf_counter_ns()
        params = {
            "loss": b"mse", "max_depth": s["max_depth"], "max_bins": s["bins"],
            "num_threads": s["threads"], "lr": s["learning_rate"], "early_stop": s["patience"],
            "verbosity": Verbosity.SILENT, "subsample": subsample,
        }
        if output_metric is not None:
            params["output_metric"] = output_metric
        model = MultiOutputGBDT(out_dim=y_train.shape[1], params=params)
        model.set_data((x_train, y_train), (x_valid, y_valid))
        model.train(s["round_cap"])
        elapsed = perf_counter_ns() - start
        rounds = model.lib.GetTreeCount(model._boostnode)
    else:
        import xgboost as xgb

        if output_metric is not None:
            raise ValueError("XGBoost fits this quadratic objective in transformed coordinates.")
        params = {"objective": "reg:squarederror", "tree_method": "hist",
                  "multi_strategy": "multi_output_tree", "device": "cpu", "eval_metric": "rmse",
                  "max_depth": s["max_depth"], "max_bin": s["bins"], "eta": s["learning_rate"],
                  "nthread": s["threads"], "seed": s["model_seed"], "subsample": subsample}
        if profile == "XGBoost matched":
            # Omni averages half the gain over targets. XGBoost sums full gains.
            params.update(min_child_weight=20, gamma=2 * y_train.shape[1] * 0.001)
        start = perf_counter_ns()
        train = xgb.DMatrix(x_train, label=y_train, nthread=1)
        valid = xgb.DMatrix(x_valid, label=y_valid, nthread=1)
        model = xgb.train(params, train, s["round_cap"], evals=[(valid, "valid")],
                          early_stopping_rounds=s["patience"], verbose_eval=False)
        elapsed = perf_counter_ns() - start
        built = model.num_boosted_rounds()
        best = model.best_iteration + 1
        # Match Omni's public API: trim when patience fires, keep all rounds at the cap otherwise.
        rounds = best if built - best >= s["patience"] else built
    assert 0 < rounds <= s["round_cap"]
    return model, rounds, elapsed / 1e6


def run_profile(profile, csv_path, subsample):
    """Evaluate one library in a dedicated process and time warm fits on the example split.

    Args:
        profile: Model and regularization setting.
        csv_path: Path to the UCI CSV.
        subsample: Fraction of training rows used to fit each tree.

    Returns:
        Runtime identity, split scores, baseline scores and individual fit timings.
    """
    x, y, ids = load_csv(csv_path)
    omni = profile == "OmniGBDT"
    package = "omnigbdt" if omni else "xgboost"
    package_version = version(package)
    if not omni and package_version != "3.4.1":
        raise ValueError(f"This benchmark requires XGBoost 3.4.1, found {package_version}.")
    if omni and subsample < 1.0:
        from omnigbdt import load_lib

        if not hasattr(load_lib(), "SetSubsample"):
            raise RuntimeError("The installed OmniGBDT library lacks native row sampling.")
    results = []
    for scheme in ("rows", "portfolios"):
        for seed in range(10):
            train, valid, test = partitions(ids, scheme, seed)
            data = (x[train], y[train], x[valid], y[valid])
            model, rounds, _ = fit_model(profile, *data, subsample=subsample)
            if omni:
                pred = model.predict(x[test])
                native_path = Path(model.lib._name)
            else:
                import xgboost as xgb

                pred = model.inplace_predict(x[test], iteration_range=(0, rounds))
                native_path = Path(xgb.core._LIB._name)
                raw = json.loads(model.save_raw(raw_format="json"))["learner"]
                trees = raw["gradient_booster"]["model"]["trees"]
                assert all(int(t["tree_param"]["size_leaf_vector"]) == 6 for t in trees)
                assert len(trees) == model.num_boosted_rounds()
                assert json.loads(model.save_config())["learner"]["learner_train_param"][
                    "multi_strategy"] == "multi_output_tree"
            assert {"site-packages", "dist-packages"}.intersection(native_path.parts)
            assert pred.shape == y[test].shape and np.isfinite(pred).all()
            squared_errors = (y[test] - pred) ** 2
            baseline = y[train].mean(axis=0)
            results.append({
                "split": scheme, "seed": seed, "sizes": [len(train), len(valid), len(test)],
                "test_rows_with_portfolio_in_train": int(np.isin(ids[test], ids[train]).sum()),
                "rounds_used": rounds, "test_rmse": float(np.sqrt(squared_errors.mean())),
                "target_rmse": np.sqrt(squared_errors.mean(axis=0)).tolist(),
                "baseline_rmse": float(np.sqrt(np.mean((baseline - y[test]) ** 2))),
            })
            if scheme == "rows" and seed == 0:
                timings = []
                for _ in range(11):
                    timed_model, timed_rounds, elapsed = fit_model(profile, *data, subsample=subsample)
                    assert timed_rounds == rounds
                    timings.append(elapsed)
                    del timed_model
    return {"profile": profile, "version": package_version, "native_library": native_path.name,
            "split_results": results, "seed0_fit_ms": timings,
            "median_seed0_fit_ms": statistics.median(timings)}


def main():
    """Write all benchmark results, isolating each native runtime in a separate process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Original UCI Stock Portfolio Performance CSV")
    parser.add_argument("output", type=Path, help="Destination JSON file for split scores and timings")
    parser.add_argument("--subsample", type=float, choices=(1.0, 0.8), default=1.0,
                        help="Training row fraction for both libraries (default: 1.0)")
    parser.add_argument("--profile", choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.profile:
        print(json.dumps(run_profile(args.profile, args.csv_path, args.subsample)))
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")
    output = {"source": "https://archive.ics.uci.edu/static/public/390/data.csv",
              "sha256": CSV_DIGEST, "targets": TARGETS,
              "settings": {**SETTINGS, "subsample": args.subsample},
              "python": platform.python_version(), "platform": platform.platform(),
              "versions": {p: version(p) for p in ("omnigbdt", "xgboost", "numpy", "numba",
                                                    "llvmlite", "scipy")}, "profiles": []}
    for profile in PROFILES:
        child = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), str(args.csv_path),
                                str(args.output), "--subsample", str(args.subsample),
                                "--profile", profile], check=True, env=env,
                               stdout=subprocess.PIPE, text=True)
        output["profiles"].append(json.loads(child.stdout))
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
