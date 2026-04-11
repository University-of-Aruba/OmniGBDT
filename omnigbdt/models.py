from __future__ import annotations

import ctypes
from typing import Callable

import numpy as np

from .histogram import get_bins_maps
from .lib_utils import (
    Verbosity,
    _normalize_deterministic,
    _normalize_verbosity,
    array_1d_double,
    array_1d_int,
    array_2d_double,
    array_2d_int,
    array_2d_uint16,
    default_params,
    load_lib,
)

ObjectiveCallback = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
MetricCallback = Callable[[np.ndarray, np.ndarray], float]


def _as_bytes(path):
    """Convert a filesystem path into a byte string for the native API.

    Args:
        path: Path-like value accepted by the public Python wrapper.

    Returns:
        bytes: Encoded path ready for the native library.
    """
    if isinstance(path, bytes):
        return path
    return str(path).encode()


class BoostUtils:
    """Shared Python helpers around the native OmniGBDT booster objects."""

    def __init__(self, lib=None, free_fn_name=None):
        """Initialize the shared native booster wrapper state.

        Args:
            lib: Optional pre-loaded native library handle.
            free_fn_name: Name of the native destructor function for the booster.
        """
        self._boostnode = None
        self._free_fn_name = free_fn_name
        self._gh_buffers = None
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

        g_array = self._normalize_gh_array(g, name="g")
        h_array = self._normalize_gh_array(h, name="h")
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

    def _set_label(self, x: np.array, is_train: bool):
        """Register training or evaluation labels with the native booster.

        Args:
            x: Label array.
            is_train: Whether the labels belong to the training split.

        Raises:
            AssertionError: If the label array has an unsupported dtype or rank.
        """
        if x.dtype == np.float64:
            if x.ndim == 1:
                self.lib.SetLabelDouble.argtypes = [ctypes.c_void_p, array_1d_double, ctypes.c_bool]
            elif x.ndim == 2:
                self.lib.SetLabelDouble.argtypes = [ctypes.c_void_p, array_2d_double, ctypes.c_bool]
            else:
                assert False, "label must be 1D or 2D array"
            self.lib.SetLabelDouble(self._boostnode, x, is_train)
        elif x.dtype == np.int32:
            if x.ndim == 1:
                self.lib.SetLabelInt.argtypes = [ctypes.c_void_p, array_1d_int, ctypes.c_bool]
            elif x.ndim == 2:
                self.lib.SetLabelInt.argtypes = [ctypes.c_void_p, array_2d_int, ctypes.c_bool]
            else:
                assert False, "label must be 1D or 2D array"
            self.lib.SetLabelInt(self._boostnode, x, is_train)
        else:
            assert False, "dtype of label must be float64 or int32"

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

    def _validate_custom_training_configuration(self, num, objective, eval_metric, maximize):
        """Validate public custom-objective training arguments.

        Args:
            num: Number of boosting rounds.
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
        if num < 0:
            raise ValueError("num must be greater than or equal to 0.")

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

    def _refresh_prediction_cache(self):
        """Rebuild cached train and eval predictions after structural changes.

        Raises:
            NotImplementedError: Always. Subclasses provide the actual replay logic.
        """
        raise NotImplementedError

    def _train_custom(self, num, objective, eval_metric=None, maximize=None):
        """Run the Python-managed custom objective training loop.

        Args:
            num: Number of boosting rounds to run.
            objective: Callback returning ``(grad, hess)``.
            eval_metric: Optional callback returning a scalar metric.
            maximize: Whether larger evaluation metric values are better.
        """
        self._validate_custom_training_configuration(num, objective, eval_metric, maximize)

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
            self._trim_trees(best_round + 1)
            self._refresh_prediction_cache()

        self._show_custom_summary(best_metric, best_round)

    def boost(self):
        """Grow a single tree using the currently registered gradients and Hessians."""
        self.lib.Boost(self._boostnode)

    def dump(self, path):
        """Serialize the current model to a text file.

        Args:
            path: Output path for the dumped model.
        """
        self.lib.Dump(self._boostnode, _as_bytes(path))

    def load(self, path):
        """Load a text-dumped model from disk.

        Args:
            path: Path to a text model generated by ``dump``.
        """
        self.lib.Load(self._boostnode, _as_bytes(path))
        self._sync_base_scores_from_native()

    def train(self, num, objective=None, eval_metric=None, maximize=None):
        """Train the booster with either a built-in loss or a custom objective.

        Args:
            num: Number of boosting rounds to run.
            objective: Optional callback implementing ``(grad, hess)`` generation.
            eval_metric: Optional scalar metric callback for Python-managed training.
            maximize: Whether larger evaluation metric values are better.
        """
        if objective is None:
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
        self.out_dim = out_dim
        user_params = {} if params is None else dict(params)
        self.params = default_params()
        self.params.update(user_params)
        self.params["deterministic"] = _normalize_deterministic(user_params, self.params)
        self.params["verbosity"] = _normalize_verbosity(user_params, self.params)
        self.params["verbose"] = self.params["verbosity"] >= int(Verbosity.FULL)
        self.__dict__.update(self.params)
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

    def _refresh_prediction_cache(self):
        """Replay the current ensemble into cached train and eval prediction buffers."""
        self._fill_prediction_buffer(self.preds_train)
        self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_1d_double, ctypes.c_int, ctypes.c_int]
        self.lib.Predict(self._boostnode, self.data, self.preds_train, len(self.data), 0)

        if hasattr(self, "preds_eval"):
            self._fill_prediction_buffer(self.preds_eval)
            self.lib.Predict(self._boostnode, self.data_eval, self.preds_eval, len(self.data_eval), 0)

    def set_booster(self, inp_dim, base_scores=None):
        """Create the native single-output booster for a fixed input width.

        Args:
            inp_dim: Number of input features.
            base_scores: Optional resolved base-score vector.
        """
        resolved_base_scores = self._resolve_base_scores() if base_scores is None else base_scores
        resolved_base_scores = np.ascontiguousarray(resolved_base_scores, dtype=np.float64)
        self._apply_resolved_base_scores(resolved_base_scores)
        self.close()
        self._boostnode = self.lib.SingleNew(
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
        self._set_native_base_scores(resolved_base_scores)

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):
        """Register training and optional evaluation data.

        Args:
            train_set: Optional ``(X, y)`` training tuple.
            eval_set: Optional ``(X, y)`` evaluation tuple.
        """
        if train_set is not None:
            self.data, self.label = train_set
            resolved_base_scores = self._resolve_base_scores(self.label)
            self.set_booster(self.data.shape[-1], base_scores=resolved_base_scores)
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins, self.num_threads)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.empty(len(self.data) * self.out_dim, dtype="float64")
            self._fill_prediction_buffer(self.preds_train)

            self.lib.SetData.argtypes = [
                ctypes.c_void_p,
                array_2d_uint16,
                array_2d_double,
                array_1d_double,
                ctypes.c_int,
                ctypes.c_bool,
            ]
            self.lib.SetData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data), True)
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.empty(len(self.data_eval) * self.out_dim, dtype="float64")
            self._fill_prediction_buffer(self.preds_eval)
            maps = np.zeros((1, 1), "uint16")
            self.lib.SetData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval), False)
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def train_multi(self, num):
        """Train the legacy multi-class helper path.

        Args:
            num: Number of boosting rounds.
        """
        assert self.out_dim > 1, "out_dim must bigger than 1"
        self.lib.TrainMulti(self._boostnode, num, self.out_dim)

    def predict(self, x, num_trees=0):
        """Predict from the current single-output ensemble.

        Args:
            x: Feature matrix.
            num_trees: Optional tree limit. ``0`` means all trees.

        Returns:
            numpy.ndarray: Prediction vector or matrix depending on ``out_dim``.
        """
        preds = np.empty(len(x) * self.out_dim, dtype="float64")
        self._fill_prediction_buffer(preds)

        if self.out_dim == 1:
            self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_1d_double, ctypes.c_int, ctypes.c_int]
            self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)
            return preds

        self.lib.PredictMulti(self._boostnode, x, preds, len(x), self.out_dim, num_trees)
        preds = np.reshape(preds, (self.out_dim, len(x)))
        return np.transpose(preds)

    def reset(self):
        """Clear learned trees and reset cached predictions to ``base_score``."""
        self.lib.Reset(self._boostnode)


class MultiOutputGBDT(BoostUtils):
    """Python wrapper around the native multi-output OmniGBDT booster."""

    def __init__(self, lib=None, out_dim=1, params=None):
        """Initialize the multi-output booster wrapper.

        Args:
            lib: Optional pre-loaded native library handle.
            out_dim: Number of output columns.
            params: Optional parameter overrides.
        """
        super().__init__(lib=lib, free_fn_name="MultiFree")
        self.out_dim = out_dim
        user_params = {} if params is None else dict(params)
        self.params = default_params()
        self.params.update(user_params)
        self.params["deterministic"] = _normalize_deterministic(user_params, self.params)
        self.params["verbosity"] = _normalize_verbosity(user_params, self.params)
        self.params["verbose"] = self.params["verbosity"] >= int(Verbosity.FULL)
        self.__dict__.update(self.params)
        self._apply_resolved_base_scores(self._resolve_base_scores())

    def _base_score_count(self):
        """Return the number of base-score entries for multi-output models.

        Returns:
            int: One base-score entry per output column.
        """
        return self.out_dim

    def _refresh_prediction_cache(self):
        """Replay the current ensemble into cached train and eval prediction buffers."""
        self._fill_prediction_buffer(self.preds_train)
        self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_2d_double, ctypes.c_int, ctypes.c_int]
        self.lib.Predict(self._boostnode, self.data, self.preds_train, len(self.data), 0)

        if hasattr(self, "preds_eval"):
            self._fill_prediction_buffer(self.preds_eval)
            self.lib.Predict(self._boostnode, self.data_eval, self.preds_eval, len(self.data_eval), 0)

    def set_booster(self, inp_dim, out_dim, base_scores=None):
        """Create the native multi-output booster for a fixed input width.

        Args:
            inp_dim: Number of input features.
            out_dim: Number of output columns.
            base_scores: Optional resolved base-score vector.
        """
        resolved_base_scores = self._resolve_base_scores() if base_scores is None else base_scores
        resolved_base_scores = np.ascontiguousarray(resolved_base_scores, dtype=np.float64)
        self._apply_resolved_base_scores(resolved_base_scores)
        self.close()
        self._boostnode = self.lib.MultiNew(
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
        self._set_native_base_scores(resolved_base_scores)

    def set_data(self, train_set: tuple = None, eval_set: tuple = None):
        """Register training and optional evaluation data.

        Args:
            train_set: Optional ``(X, y)`` training tuple.
            eval_set: Optional ``(X, y)`` evaluation tuple.
        """
        if train_set is not None:
            self.data, self.label = train_set
            resolved_base_scores = self._resolve_base_scores(self.label)
            self.set_booster(self.data.shape[-1], self.out_dim, base_scores=resolved_base_scores)
            self.bins, self.maps = get_bins_maps(self.data, self.max_bins, self.num_threads)
            self._set_bin(self.bins)
            self.maps = np.ascontiguousarray(self.maps.transpose())
            self.preds_train = np.empty((len(self.data), self.out_dim), dtype="float64")
            self._fill_prediction_buffer(self.preds_train)
            self.lib.SetData.argtypes = [
                ctypes.c_void_p,
                array_2d_uint16,
                array_2d_double,
                array_2d_double,
                ctypes.c_int,
                ctypes.c_bool,
            ]
            self.lib.SetData(self._boostnode, self.maps, self.data, self.preds_train, len(self.data), True)
            if self.label is not None:
                self._set_label(self.label, True)

        if eval_set is not None:
            self.data_eval, self.label_eval = eval_set
            self.preds_eval = np.empty((len(self.data_eval), self.out_dim), dtype="float64")
            self._fill_prediction_buffer(self.preds_eval)
            maps = np.zeros((1, 1), "uint16")
            self.lib.SetData(self._boostnode, maps, self.data_eval, self.preds_eval, len(self.data_eval), False)
            if self.label_eval is not None:
                self._set_label(self.label_eval, False)

    def predict(self, x, num_trees=0):
        """Predict from the current multi-output ensemble.

        Args:
            x: Feature matrix.
            num_trees: Optional tree limit. ``0`` means all trees.

        Returns:
            numpy.ndarray: Prediction matrix with shape ``(n_samples, out_dim)``.
        """
        preds = np.empty((len(x), self.out_dim), dtype="float64")
        self._fill_prediction_buffer(preds)
        self.lib.Predict.argtypes = [ctypes.c_void_p, array_2d_double, array_2d_double, ctypes.c_int, ctypes.c_int]
        self.lib.Predict(self._boostnode, x, preds, len(x), num_trees)
        return preds
