from __future__ import annotations

import ctypes
import os
from pathlib import Path
import stat
import tempfile
from operator import index
from typing import Callable

import numpy as np

from .histogram import get_bins_maps
from .lib_utils import (
    Verbosity,
    _normalize_deterministic,
    _normalize_verbosity,
    default_params,
    load_lib,
)

ObjectiveCallback = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
MetricCallback = Callable[[np.ndarray, np.ndarray], float]


class BoostUtils:
    """Shared Python helpers around the native OmniGBDT booster objects."""

    _multi_output = False

    def __init__(self, lib=None, free_fn_name=None):
        """Initialize the shared native booster wrapper state.

        Args:
            lib: Optional pre-loaded native library handle.
            free_fn_name: Name of the native destructor function for the booster.
        """
        self._boostnode = None
        self._free_fn_name = free_fn_name
        self._gh_buffers = None
        self._input_width = None
        self._active_out_dim = None
        self._active_loss = None
        self._resolved_base_scores = np.ones(1, dtype=np.float64) * 0.0
        self.lib = lib if lib is not None else load_lib()

    def _base_score_count(self):
        """Return the number of base-score entries used by the model.

        Returns:
            int: Number of resolved base-score values.
        """
        return 1

    def _is_regression_loss(self):
        """Return whether the configured built-in loss is regression MSE.

        Returns:
            bool: ``True`` when the configured loss is ``mse``.
        """
        loss = self.params.get("loss", b"mse")
        if isinstance(loss, bytes):
            return loss == b"mse"
        return str(loss).lower() == "mse"

    def _normalize_base_score_param(self, value):
        """Normalize an explicit base-score parameter.

        Args:
            value: User-supplied base-score value.

        Returns:
            numpy.ndarray | None: Contiguous 1D ``float64`` base-score vector, or
            ``None`` when automatic initialization should be used.

        Raises:
            ValueError: If the supplied base-score shape is incompatible.
        """
        if value is None:
            return None

        expected = self._base_score_count()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(expected, float(array), dtype=np.float64)

        array = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
        if len(array) == 1:
            return np.full(expected, float(array[0]), dtype=np.float64)
        if len(array) != expected:
            raise ValueError(
                f"base_score must be a scalar or contain {expected} value(s), "
                f"but received shape {tuple(np.asarray(value).shape)}."
            )
        return array

    def _auto_base_scores(self, labels=None):
        """Build automatically inferred base scores.

        Args:
            labels: Optional training labels used to infer the initial prediction.

        Returns:
            numpy.ndarray: Contiguous 1D ``float64`` base-score vector.
        """
        expected = self._base_score_count()
        if not self._is_regression_loss() or labels is None:
            return np.zeros(expected, dtype=np.float64)

        label_array = np.asarray(labels, dtype=np.float64)
        if expected == 1:
            return np.array([float(np.mean(label_array))], dtype=np.float64)

        return np.ascontiguousarray(np.mean(label_array, axis=0), dtype=np.float64)

    def _resolve_base_scores(self, labels=None):
        """Resolve the effective base-score vector for the current model state.

        Args:
            labels: Optional training labels for automatic initialization.

        Returns:
            numpy.ndarray: Contiguous 1D ``float64`` base-score vector.
        """
        explicit = self._normalize_base_score_param(self.params.get("base_score"))
        if explicit is not None:
            return explicit
        return self._auto_base_scores(labels)

    def _apply_resolved_base_scores(self, base_scores):
        """Store resolved base scores on the Python wrapper.

        Args:
            base_scores: Base-score vector to cache on the wrapper.
        """
        self._resolved_base_scores = np.ascontiguousarray(base_scores, dtype=np.float64)

    def _set_native_base_scores(self, base_scores):
        """Send resolved base scores to the native booster.

        Args:
            base_scores: Base-score vector to register natively.
        """
        array = np.ascontiguousarray(base_scores, dtype=np.float64)
        self.lib.SetBaseScore(
            self._boostnode,
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(array),
        )

    def _sync_base_scores_from_native(self):
        """Refresh cached base scores from the native booster state."""
        count = self.lib.GetBaseScoreSize(self._boostnode)
        base_scores = np.empty(count, dtype=np.float64)
        self.lib.GetBaseScore(
            self._boostnode,
            base_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        self._apply_resolved_base_scores(base_scores)

    def _fill_prediction_buffer(self, array):
        """Fill a prediction buffer with the resolved base scores.

        Args:
            array: Prediction buffer to initialize in place.
        """
        if array.ndim == 1:
            array.fill(float(self._resolved_base_scores[0]))
            return
        array[...] = self._resolved_base_scores

    def _custom_output_shape(self):
        """Return the expected gradient and Hessian shape for custom objectives.

        Returns:
            tuple[int, ...]: Expected callback output shape.

        Raises:
            RuntimeError: If training data has not been registered yet.
        """
        if not hasattr(self, "preds_train"):
            raise RuntimeError("Call set_data(...) before using custom objectives.")
        return tuple(self.preds_train.shape)

    def _normalize_gh_array(self, value, *, name):
        """Normalize one gradient or Hessian callback output array.

        Args:
            value: Array-like gradient or Hessian output.
            name: Human-readable label for error messages.

        Returns:
            numpy.ndarray: A contiguous ``float64`` array that matches the model.

        Raises:
            ValueError: If the array shape or dtype is incompatible with the model.
        """
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must be a floating-point array.")

        array = np.ascontiguousarray(array, dtype=np.float64)
        expected_shape = self._custom_output_shape()

        if len(expected_shape) == 1:
            valid_shapes = {expected_shape, (expected_shape[0], 1)}
            if tuple(array.shape) not in valid_shapes:
                raise ValueError(
                    f"{name} must have shape {expected_shape} or {(expected_shape[0], 1)}, "
                    f"but received {tuple(array.shape)}."
                )
            return array

        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, but received {tuple(array.shape)}."
            )
        return array

    def _set_gh(self, g, h):
        """Register gradients and Hessians for the next manual ``boost()`` call.

        Args:
            g: Gradient array matching the active training layout.
            h: Hessian array matching the active training layout.

        Raises:
            RuntimeError: If training data has not been registered yet.
            ValueError: If the arrays do not match the expected shape or dtype.
        """
        if self._boostnode is None:
            raise RuntimeError("Call set_data(...) before setting gradients and Hessians.")

        self._validate_custom_objective_support()
        g_array = self._normalize_gh_array(g, name="g")
        h_array = self._normalize_gh_array(h, name="h")
        self._check_booster()
        self._gh_buffers = (g_array, h_array)
        self.lib.SetGH(
            self._boostnode,
            g_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            h_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def _set_bin(self, bins):
        """Send the feature histogram bin definitions to the native booster.

        Args:
            bins: Sequence of per-feature bin edge arrays.
        """
        num = np.array([len(column_bins) for column_bins in bins], dtype=np.uint16)
        value = np.concatenate(bins, axis=0)
        self.lib.SetBin(self._boostnode, num, value)

    def _check_features(self, features, width=None):
        """Validate and normalize features before handing a pointer to native code.

        Args:
            features: Two-dimensional float64 feature array.
            width: Expected number of columns, if already known.

        Returns:
            numpy.ndarray: Contiguous feature matrix.
        """
        x = np.asarray(features)
        if x.ndim != 2 or x.dtype != np.float64 or x.shape[1] == 0:
            raise ValueError("Features must be a 2D float64 array with at least one column.")
        if width is not None and x.shape[1] != width:
            raise ValueError(f"Expected {width} feature columns, received {x.shape[1]}.")
        if not np.isfinite(x).all():
            raise ValueError("Features must be finite.")
        return np.ascontiguousarray(x)

    def _check_label(self, label, n):
        """Check label storage shared by built-in and custom objectives.

        Args:
            label: Array of training or evaluation labels.
            n: Expected number of rows.

        Returns:
            numpy.ndarray: Contiguous label array.
        """
        y = np.asarray(label)
        if y.ndim not in (1, 2) or y.shape[0] != n:
            raise ValueError(f"Labels must be a 1D or 2D array with {n} rows.")
        if y.ndim == 2 and y.shape[1] not in (1, self.out_dim):
            raise ValueError(f"Expected 1 or {self.out_dim} label columns.")
        if self.out_dim == 1:
            y = y.reshape(n, 1) if self._multi_output else y.reshape(n)
        if y.dtype not in (np.float64, np.int32):
            raise ValueError("Label dtype must be float64 or int32.")
        return np.ascontiguousarray(y)

    def _register_label(self, y, is_train):
        """Retain the array whose address is registered with the native dataset.

        Args:
            y: Native-compatible, contiguous labels.
            is_train: Whether to register training labels.
        """
        setter = self.lib.SetLabelDouble if y.dtype == np.float64 else self.lib.SetLabelInt
        setter(self._boostnode, y, is_train)
        if is_train:
            self._native_train_label = y
        else:
            self._native_eval_label = y

    def _set_label(self, x, is_train):
        """Replace labels used by both native training and Python callbacks.

        Args:
            x: Float64 or int32 labels with one row per registered sample.
            is_train: Whether to replace training labels.
        """
        self._check_booster()
        data = getattr(self, "data" if is_train else "data_eval", None)
        if data is None:
            raise RuntimeError("Call set_data(...) for this split before setting labels.")
        y = self._check_label(x, len(data))
        self._register_label(y, is_train)
        if is_train:
            self.label = y
        else:
            self.label_eval = y

    def _prepare_builtin_labels(self):
        """Register correctly typed buffers for the configured native loss."""
        if getattr(self, "label", None) is None:
            raise ValueError("Built-in training requires training labels.")
        loss = self._active_loss
        for is_train in (True, False):
            y = self.label if is_train else getattr(self, "label_eval", None)
            if y is None:
                continue
            if loss in (b"ce", b"ce_column"):
                if y.size != len(y):
                    raise ValueError("Cross-entropy requires one class index per row.")
                if ((y < 0) | (y >= self.out_dim) | (y != np.floor(y))).any():
                    raise ValueError(f"Class indices must be integers in [0, {self.out_dim}).")
            elif y.size != len(y) * self.out_dim:
                raise ValueError(f"This loss requires {self.out_dim} target value(s) per row.")
            elif loss == b"bce" and ((y != 0) & (y != 1)).any():
                raise ValueError("Binary labels must be 0 or 1.")
            if loss == b"mse" and not np.isfinite(y).all():
                raise ValueError("Regression labels must be finite.")
            if not self._multi_output and self.out_dim > 1 and y.ndim == 2:
                y = y.T
            dtype = np.float64 if loss == b"mse" else np.int32
            self._register_label(np.ascontiguousarray(y, dtype=dtype), is_train)

    def _check_booster(self):
        """Require a live native model before invoking the C interface."""
        if self._boostnode is None:
            raise RuntimeError("Call set_data(...) or set_booster(...) before using the model.")
        if self.out_dim != self._active_out_dim:
            raise ValueError("out_dim has changed; configure a new booster before using this width.")

    def _check_configuration(self, inp_dim):
        """Validate dimensions and loss before constructing a native model.

        Args:
            inp_dim: Number of input features.
        """
        if index(inp_dim) <= 0 or index(self.out_dim) <= 0:
            raise ValueError("Input and output dimensions must be positive.")
        for name, minimum in (("min_samples", 1), ("num_threads", 1), ("topk", 0), ("hist_cache", 0)):
            if index(self.params[name]) < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
        if self.params["loss"] not in (b"mse", b"bce", b"ce", b"ce_column"):
            raise ValueError("loss must be one of b'mse', b'bce', b'ce', b'ce_column'.")

    def _callback_input(self, array):
        """Create a read-only callback view of a model buffer.

        Args:
            array: Array to expose to a callback.

        Returns:
            numpy.ndarray: Snapshot copy marked as read-only.
        """
        snapshot = np.array(array, copy=True)
        snapshot.setflags(write=False)
        return snapshot

    def _call_eval_metric(self, metric, preds, labels):
        """Evaluate a custom metric callback and coerce it to a scalar float.

        Args:
            metric: Metric callback.
            preds: Prediction array for the current split.
            labels: Label array for the current split.

        Returns:
            float: Scalar metric value.

        Raises:
            ValueError: If the metric does not return a scalar float.
        """
        value = metric(self._callback_input(preds), self._callback_input(labels))
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("eval_metric must return a scalar float.") from exc

    def _has_eval_labels(self):
        """Return whether evaluation labels are available for custom metrics.

        Returns:
            bool: ``True`` when evaluation labels were registered.
        """
        return getattr(self, "label_eval", None) is not None and hasattr(self, "preds_eval")

    def _validate_custom_objective_support(self):
        """Validate model-specific preconditions for the custom objective path."""
        return None

    def _validate_custom_training_configuration(self, objective, eval_metric, maximize):
        """Validate public custom-objective training arguments.

        Args:
            objective: Custom objective callback.
            eval_metric: Optional custom metric callback.
            maximize: Optional flag indicating whether larger metric values are better.

        Raises:
            RuntimeError: If training data has not been configured.
            TypeError: If callback arguments are not callable.
            ValueError: If labels or early-stopping requirements are missing.
        """
        if self._boostnode is None or not hasattr(self, "preds_train"):
            raise RuntimeError("Call set_data(...) before train(...).")
        if getattr(self, "label", None) is None:
            raise ValueError("Custom objectives require training labels.")
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if eval_metric is not None and not callable(eval_metric):
            raise TypeError("eval_metric must be callable.")
        if self.early_stop > 0 and self._has_eval_labels():
            if eval_metric is None:
                raise ValueError("Custom early stopping requires eval_metric.")
            if maximize is None:
                raise ValueError("Custom early stopping requires maximize to be set explicitly.")
        self._validate_custom_objective_support()
        splits = [(self.label, self.preds_train)]
        if self._has_eval_labels():
            splits.append((self.label_eval, self.preds_eval))
        for labels, preds in splits:
            class_indices = self._active_loss in (b"ce", b"ce_column") and labels.size == len(labels)
            if labels.shape != preds.shape and not class_indices:
                raise ValueError("Custom objective labels must match the prediction shape.")

    def _show_custom_metrics(self, round_index, train_metric, eval_metric):
        """Print Python-side custom metric logging using native-style formatting.

        Args:
            round_index: Zero-based boosting round.
            train_metric: Optional training metric value.
            eval_metric: Optional evaluation metric value.
        """
        if self.verbosity < int(Verbosity.FULL) or train_metric is None:
            return
        if eval_metric is None:
            print(f"[{round_index}] score->{train_metric:.5f}")
            return
        print(f"[{round_index}] train->{train_metric:.5f}\teval->{eval_metric:.5f}")

    def _show_custom_summary(self, best_metric, best_round):
        """Print the final best-score summary for Python-managed training.

        Args:
            best_metric: Best observed evaluation metric.
            best_round: Zero-based boosting round of the best metric.
        """
        if self.verbosity >= int(Verbosity.SUMMARY) and best_metric is not None and best_round is not None:
            print(f"Best score {best_metric} at round {best_round}")

    def _trim_trees(self, num_trees):
        """Trim the native tree ensemble to a fixed number of trees.

        Args:
            num_trees: Number of trees to keep.
        """
        self.lib.TrimTrees(self._boostnode, num_trees)

    def _new_predictions(self, n):
        """Allocate a native prediction buffer initialized with the base scores.

        Args:
            n: Number of rows in the split.

        Returns:
            numpy.ndarray: Buffer in the native booster's layout.
        """
        shape = (n, self.out_dim) if self._multi_output else n * self.out_dim
        preds = np.empty(shape, dtype=np.float64)
        self._fill_prediction_buffer(preds)
        return preds

    def _predict_into(self, x, preds, num_trees=0):
        """Add tree predictions to an initialized native-layout buffer.

        Args:
            x: Contiguous feature matrix of the configured width.
            preds: Initialized native prediction buffer.
            num_trees: Tree limit, or zero for the full ensemble.
        """
        if not self._multi_output and self.out_dim > 1:
            self.lib.PredictMulti(self._boostnode, x, preds, len(x), self.out_dim, num_trees)
        else:
            self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)

    def _refresh_prediction_cache(self):
        """Replay the current ensemble into registered prediction buffers."""
        if hasattr(self, "preds_train"):
            self._fill_prediction_buffer(self.preds_train)
            self._predict_into(self.data, self.preds_train)
        if hasattr(self, "preds_eval"):
            self._fill_prediction_buffer(self.preds_eval)
            self._predict_into(self.data_eval, self.preds_eval)

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):
        """Register training and optional evaluation data.

        Args:
            train_set: Optional (features, labels) pair that starts a fresh model.
            eval_set: Optional (features, labels) pair for evaluation.
        """
        width = self._input_width
        if train_set is not None:
            x, y = train_set
            x = self._check_features(x)
            if not len(x):
                raise ValueError("Training requires at least one sample.")
            y = None if y is None else self._check_label(y, len(x))
            width = x.shape[1]
        if eval_set is not None:
            if train_set is None:
                self._check_booster()
            x_eval, y_eval = eval_set
            x_eval = self._check_features(x_eval, width)
            y_eval = None if y_eval is None else self._check_label(y_eval, len(x_eval))

        if train_set is not None:
            self._check_configuration(width)
            base_scores = self._resolve_base_scores(y)
            bins, maps = get_bins_maps(x, self.params["max_bins"], self.params["num_threads"])
            if self._multi_output:
                self.set_booster(width, self.out_dim, base_scores=base_scores)
            else:
                self.set_booster(width, base_scores=base_scores)
            self.bins, self.maps = bins, np.ascontiguousarray(maps.T)
            self._set_bin(bins)
            self.data, self.label = x, y
            self.preds_train = self._new_predictions(len(x))
            self.lib.SetData(self._boostnode, self.maps, x, self.preds_train, len(x), True)
            if y is not None:
                self._register_label(y, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = x_eval, y_eval
            self.preds_eval = self._new_predictions(len(x_eval))
            self._predict_into(x_eval, self.preds_eval)
            # Evaluation traverses raw features and does not read histogram maps.
            eval_maps = np.empty((0, 0), dtype=np.uint16)
            self.lib.SetData(self._boostnode, eval_maps, x_eval, self.preds_eval, len(x_eval), False)
            self._native_eval_label = None
            if y_eval is not None:
                self._register_label(y_eval, False)

    def predict(self, x, num_trees=0):
        """Predict on a feature matrix using all or a prefix of the ensemble.

        Args:
            x: Two-dimensional float64 feature matrix.
            num_trees: Nonnegative tree limit, or zero for the full ensemble.

        Returns:
            numpy.ndarray: Predictions in row-major sample order.
        """
        self._check_booster()
        x = self._check_features(x, self._input_width)
        if index(num_trees) < 0:
            raise ValueError("num_trees must be nonnegative.")
        preds = self._new_predictions(len(x))
        self._predict_into(x, preds, num_trees)
        if not self._multi_output and self.out_dim > 1:
            return preds.reshape((self.out_dim, len(x))).T
        return preds

    def _train_custom(self, num, objective, eval_metric=None, maximize=None):
        """Run the Python-managed custom objective training loop.

        Args:
            num: Number of boosting rounds to run.
            objective: Callback returning ``(grad, hess)``.
            eval_metric: Optional callback returning a scalar metric.
            maximize: Whether larger evaluation metric values are better.
        """
        self._validate_custom_training_configuration(objective, eval_metric, maximize)

        initial_trees = self.lib.GetTreeCount(self._boostnode)
        best_metric = None
        best_round = None
        should_stop_early = False

        for round_index in range(num):
            result = objective(self._callback_input(self.preds_train), self._callback_input(self.label))
            try:
                grad, hess = result
            except (TypeError, ValueError) as exc:
                raise ValueError("objective must return a tuple of (grad, hess).") from exc

            self._set_gh(grad, hess)
            self.boost()

            train_metric_value = None
            eval_metric_value = None
            if eval_metric is not None:
                train_metric_value = self._call_eval_metric(eval_metric, self.preds_train, self.label)
                if self._has_eval_labels():
                    eval_metric_value = self._call_eval_metric(
                        eval_metric,
                        self.preds_eval,
                        self.label_eval,
                    )

            self._show_custom_metrics(round_index, train_metric_value, eval_metric_value)

            if eval_metric_value is None or maximize is None:
                continue

            is_better = (
                best_metric is None
                or (eval_metric_value > best_metric if maximize else eval_metric_value < best_metric)
            )
            if is_better:
                best_metric = eval_metric_value
                best_round = round_index

            if self.early_stop > 0 and round_index >= best_round + self.early_stop:
                should_stop_early = True
                break

        if should_stop_early and best_round is not None:
            self._trim_trees(initial_trees + best_round + 1)
            self._refresh_prediction_cache()

        self._show_custom_summary(best_metric, best_round)

    def boost(self):
        """Grow a single tree using the currently registered gradients and Hessians."""
        self._check_booster()
        if self._gh_buffers is None:
            raise RuntimeError("Call _set_gh(...) before boost().")
        self.lib.Boost(self._boostnode)

    def dump(self, path):
        """Serialize the current model to a text file.

        Args:
            path: Output path for the dumped model.
        """
        self._check_booster()
        destination = Path(os.fsdecode(path)).resolve()
        try:
            mode = destination.stat().st_mode
        except FileNotFoundError:
            mode = None

        if mode is not None and not stat.S_ISREG(mode):
            error = self.lib.DumpChecked(self._boostnode, os.fsencode(destination))
            if error:
                raise OSError(error, os.strerror(error), path)
            return
        if mode is not None:
            # Respect the existing file's write permissions before replacing it.
            os.close(os.open(destination, os.O_WRONLY))

        with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
            staged = Path(staging) / "model"
            error = self.lib.DumpChecked(self._boostnode, os.fsencode(staged))
            if error:
                raise OSError(error, os.strerror(error), path)
            if mode is not None:
                staged.chmod(stat.S_IMODE(mode))
            os.replace(staged, destination)

    def load(self, path):
        """Load a text-dumped model from disk.

        Args:
            path: Path to a text model generated by ``dump``.
        """
        self._check_booster()
        round_width = 1 if self._multi_output else self.out_dim
        error = self.lib.LoadChecked(self._boostnode, os.fsencode(path), round_width)
        if error < 0:
            raise ValueError(f"Invalid or incompatible OmniGBDT model: {path!r}")
        if error:
            raise OSError(error, os.strerror(error), path)
        self._sync_base_scores_from_native()
        self._refresh_prediction_cache()

    def train(self, num, objective=None, eval_metric=None, maximize=None):
        """Train the booster with either a built-in loss or a custom objective.

        Args:
            num: Number of boosting rounds to run.
            objective: Optional callback implementing ``(grad, hess)`` generation.
            eval_metric: Optional scalar metric callback for Python-managed training.
            maximize: Whether larger evaluation metric values are better.
        """
        self._check_booster()
        if not hasattr(self, "preds_train"):
            raise RuntimeError("Call set_data(...) with training data before train(...).")
        if index(num) < 0:
            raise ValueError("num must be nonnegative.")
        if objective is None:
            if not self._multi_output and self.out_dim != 1:
                raise ValueError("Use train_multi(...) for legacy multi-class training.")
            if self._multi_output and self._active_loss == b"ce_column":
                raise ValueError("Use b'ce' for row-major multi-output training.")
            self._prepare_builtin_labels()
            self._gh_buffers = None
            if num == 0:
                return
            self.lib.Train(self._boostnode, num)
            return
        self._train_custom(num, objective, eval_metric=eval_metric, maximize=maximize)

    def close(self):
        """Release the underlying native booster handle."""
        self._gh_buffers = None
        if self._boostnode is None or self._free_fn_name is None:
            return
        try:
            getattr(self.lib, self._free_fn_name)(self._boostnode)
        except Exception:
            pass
        finally:
            self._boostnode = None
            self._input_width = None
            self._active_out_dim = None
            self._active_loss = None
            for name in ("data", "label", "preds_train", "data_eval", "label_eval",
                         "preds_eval", "maps", "bins", "_native_train_label", "_native_eval_label"):
                self.__dict__.pop(name, None)

    def __del__(self):
        """Release the native booster during garbage collection."""
        self.close()


class SingleOutputGBDT(BoostUtils):
    """Python wrapper around the native single-output OmniGBDT booster."""

    def __init__(self, lib=None, out_dim=1, params=None):
        """Initialize the single-output booster wrapper.

        Args:
            lib: Optional pre-loaded native library handle.
            out_dim: Output dimension used by legacy helper APIs.
            params: Optional parameter overrides.
        """
        super().__init__(lib=lib, free_fn_name="SingleFree")
        self.out_dim = index(out_dim)
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        user_params = {} if params is None else dict(params)
        defaults = default_params()
        self.params = {**defaults, **user_params}
        self.params["deterministic"] = _normalize_deterministic(user_params, self.params)
        self.params["verbosity"] = _normalize_verbosity(user_params, self.params)
        self.params["verbose"] = self.params["verbosity"] >= int(Verbosity.FULL)
        self.__dict__.update((name, self.params[name]) for name in defaults)
        self._apply_resolved_base_scores(self._resolve_base_scores())

    def _base_score_count(self):
        """Return the number of base-score entries for single-output models.

        Returns:
            int: Always ``1`` for ``SingleOutputGBDT``.
        """
        return 1

    def _validate_custom_objective_support(self):
        """Validate the custom-objective contract for single-output training.

        Raises:
            ValueError: If the legacy multi-output helper mode is active.
        """
        if self.out_dim != 1:
            raise ValueError(
                "Custom objectives on SingleOutputGBDT require out_dim == 1. "
                "Use MultiOutputGBDT for multi-output custom training."
            )

    def set_booster(self, inp_dim, base_scores=None):
        """Create the native single-output booster for a fixed input width.

        Args:
            inp_dim: Number of input features.
            base_scores: Optional resolved base-score vector.
        """
        self._check_configuration(inp_dim)
        resolved_base_scores = (self._resolve_base_scores() if base_scores is None
                                else self._normalize_base_score_param(base_scores))
        new_booster = self.lib.SingleNew(
            inp_dim,
            self.params["loss"],
            self.params["max_depth"],
            self.params["max_leaves"],
            self.params["seed"],
            self.params["min_samples"],
            self.params["num_threads"],
            self.params["lr"],
            self.params["reg_l1"],
            self.params["reg_l2"],
            self.params["gamma"],
            float(resolved_base_scores[0]),
            self.params["early_stop"],
            self.params["verbosity"],
            self.params["hist_cache"],
        )
        self.close()
        self._boostnode = new_booster
        self._input_width = inp_dim
        self._active_out_dim = self.out_dim
        self._active_loss = self.params["loss"]
        self._apply_resolved_base_scores(resolved_base_scores)
        self._set_native_base_scores(resolved_base_scores)

    def train_multi(self, num):
        """Train the legacy multi-class helper path.

        Args:
            num: Number of boosting rounds.
        """
        self._check_booster()
        if self.out_dim <= 1:
            raise ValueError("train_multi requires out_dim > 1.")
        if self._active_loss == b"ce":
            raise ValueError("Use b'ce_column' for column-major train_multi(...).")
        if index(num) < 0:
            raise ValueError("num must be nonnegative.")
        self._prepare_builtin_labels()
        self._gh_buffers = None
        if num == 0:
            return
        self.lib.TrainMulti(self._boostnode, num, self.out_dim)

    def reset(self):
        """Clear learned trees and reset cached predictions to ``base_score``."""
        self._check_booster()
        self.lib.Reset(self._boostnode)
        if self.out_dim > 1:
            self._refresh_prediction_cache()


class MultiOutputGBDT(BoostUtils):
    """Python wrapper around the native multi-output OmniGBDT booster."""

    _multi_output = True

    def __init__(self, lib=None, out_dim=1, params=None):
        """Initialize the multi-output booster wrapper.

        Args:
            lib: Optional pre-loaded native library handle.
            out_dim: Number of output columns.
            params: Optional parameter overrides.
        """
        super().__init__(lib=lib, free_fn_name="MultiFree")
        self.out_dim = index(out_dim)
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        user_params = {} if params is None else dict(params)
        defaults = default_params()
        self.params = {**defaults, **user_params}
        self.params["deterministic"] = _normalize_deterministic(user_params, self.params)
        self.params["verbosity"] = _normalize_verbosity(user_params, self.params)
        self.params["verbose"] = self.params["verbosity"] >= int(Verbosity.FULL)
        self.__dict__.update((name, self.params[name]) for name in defaults)
        self._apply_resolved_base_scores(self._resolve_base_scores())

    def _base_score_count(self):
        """Return the number of base-score entries for multi-output models.

        Returns:
            int: One base-score entry per output column.
        """
        return self.out_dim

    def set_booster(self, inp_dim, out_dim, base_scores=None):
        """Create the native multi-output booster for a fixed input width.

        Args:
            inp_dim: Number of input features.
            out_dim: Number of output columns.
            base_scores: Optional resolved base-score vector.
        """
        if out_dim != self.out_dim:
            raise ValueError("set_booster output width must match the model's out_dim.")
        self._check_configuration(inp_dim)
        resolved_base_scores = (self._resolve_base_scores() if base_scores is None
                                else self._normalize_base_score_param(base_scores))
        new_booster = self.lib.MultiNew(
            inp_dim,
            out_dim,
            self.params["topk"],
            self.params["loss"],
            self.params["max_depth"],
            self.params["max_leaves"],
            self.params["seed"],
            self.params["min_samples"],
            self.params["num_threads"],
            self.params["lr"],
            self.params["reg_l1"],
            self.params["reg_l2"],
            self.params["gamma"],
            float(resolved_base_scores[0]),
            self.params["early_stop"],
            self.params["one_side"],
            self.params["verbosity"],
            self.params["hist_cache"],
        )
        self.close()
        self._boostnode = new_booster
        self._input_width = inp_dim
        self._active_out_dim = self.out_dim
        self._active_loss = self.params["loss"]
        self._apply_resolved_base_scores(resolved_base_scores)
        self._set_native_base_scores(resolved_base_scores)
