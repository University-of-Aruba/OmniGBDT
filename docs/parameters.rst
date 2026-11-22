Parameters
==========

This page describes the Python parameter dictionary used by ``SingleOutputGBDT`` and ``MultiOutputGBDT`` in this fork. Unless noted otherwise, defaults come from ``omnigbdt.lib_utils.default_params()``. Several defaults intentionally differ from the original package; see :doc:`differences`.

General
-------

- ``loss``: default = ``b"mse"``, type = bytes
  - Supported values are ``b"mse"``, ``b"bce"``, ``b"ce"``, and ``b"ce_column"``
  - ``b"ce_column"`` is only relevant to legacy ``SingleOutputGBDT`` classification-style workflows
  - The Python API expects a byte string, for example ``b"mse"``
  - For binary cross-entropy, a confidently wrong prediction beyond about 36.74 absolute log-odds uses the logistic curvature bound of 0.25. This keeps the corrective step finite even with zero L2 regularization.
  - when training with a custom ``objective=...``, the native booster still requires ``loss`` to be a supported built-in value at construction time, but custom rounds will use the custom callback instead of the built-in objective

- ``verbosity``: default = ``Verbosity.FULL`` (``2``), type = ``Verbosity`` or int
  - ``Verbosity.SILENT`` / ``0`` prints nothing from the native trainer
  - ``Verbosity.SUMMARY`` / ``1`` prints only the final best score when evaluation data is present
  - ``Verbosity.FULL`` / ``2`` prints per-round metrics and the final best score

- ``verbose``: default = ``True``, type = bool
  - Backward-compatible alias for the old two-level behavior
  - ``False`` maps to ``Verbosity.SILENT``
  - ``True`` maps to ``Verbosity.FULL``

- ``num_threads``: default = ``2``, type = int
  - Number of training threads

- ``deterministic``: default = ``True``, type = bool
  - Enables the documented fixed-thread CPU repeatability mode on the same platform
  - The current packaged CPU implementation already uses deterministic split selection

- ``seed``: default = ``0``, type = int
  - Controls the training row samples when ``subsample < 1.0`` and input feature samples when ``colsample_bytree < 1.0``
  - Controls Gaussian directions when ``projection_dim`` is set. Each tree uses the seed and its position in the retained ensemble.

- ``hist_cache``: default = ``16``, type = int
  - Maximum number of histogram caches

- ``max_bins``: default = ``128``, type = int
  - Maximum number of bins for each input feature, from 1 through 65,535
  - When distinct values exceed this limit, the cut targets are based on row counts. Repeated values stay together.
  - Retraining a 0.5.1 model on repeated-valued features can therefore produce different splits. Loading a saved model uses its stored thresholds.

- ``topk``: default = ``0``, type = int
  - Sparse split-finding parameter
  - If ``0``, the dense split-search path is used

- ``one_side``: default = ``True``, type = bool
  - Selects the sparse split-search variant
  - Only used when ``topk != 0``

Tree
----

- ``max_depth``: default = ``4``, type = int
  - Maximum tree depth
  - Must be at least ``1``

- ``max_leaves``: default = ``32``, type = int
  - Maximum number of leaves per tree

- ``min_samples``: default = ``20``, type = int
  - Minimum number of samples allowed in a leaf

- ``early_stop``: default = ``15``, type = int
  - Early-stopping patience in rounds
  - If no evaluation labels are registered, early stopping stays inactive

Learning
--------

- ``output_metric``: default = ``None``, type = array-like
  - A weight vector sets how much each target's squared error counts. For example, ``[0.8, 0.2]`` gives the first target four times the weight of the second. A matrix also accounts for errors that reinforce or offset each other across targets.
  - Use nonnegative vector weights or a symmetric positive-semidefinite matrix with one row and column per target. At least one direction must have positive weight.
  - For an error vector ``e = prediction - target`` and matrix ``M``, the loss per row is ``0.5 * e @ M @ e``. A weight vector supplies the diagonal of ``M``. Validation reports the square root of the mean of ``e @ M @ e`` over rows and minimizes it for early stopping.
  - With equal weights of ``1 / out_dim``, training produces the same splits, leaf values and early-stopping rounds as ordinary least squares. For any common positive weight ``w``, the equivalent ordinary split threshold is ``gamma / (out_dim * w)``, and the reported validation score is ordinary RMSE times ``sqrt(out_dim * w)``.
  - Supported for regression with two or more targets in ``MultiOutputGBDT`` and ``MultiOutputGBDTRegressor``. Use the built-in ``loss=b"mse"``, ``topk=0``, and ``reg_l1=0``. The same metric applies to every row. The L2 leaf penalty uses the same metric as the loss.
  - A direction with zero weight contributes nothing to split selection. Its prediction follows the usual regularized mean residual in each leaf. A Gram matrix can acquire very small negative eigenvalues through rounding. In that case, eigenvalues within roundoff of zero in the scaled matrix are treated as zero.

- ``projection_dim``: default = ``None``, type = int or ``None``
  - Use a positive integer ``k`` up to the number of targets to score candidate splits with Gaussian projections. For example, ``projection_dim=2`` uses two random mixtures of the residuals to grow each tree in a six-target regression. Every leaf still fits all six outputs.
  - ``None`` uses the exact split score. A width covering all active output directions also uses exact scoring. With ``output_metric``, the projection acts on the metric-weighted residuals.
  - A fresh matrix is drawn per tree with independent entries of variance ``1 / k``. Validation and early stopping use the full objective. Prediction requires only the fitted trees.
  - Requires two or more targets, built-in ``loss=b"mse"``, ``topk=0``, ``reg_l1=0``, and finite, nonnegative ``reg_l2``. These settings apply to ``MultiOutputGBDT`` and ``MultiOutputGBDTRegressor``.

- ``lr``: default = ``0.05``, type = float
  - Learning rate

- ``base_score``: default = ``None``, type = ``None`` | float | sequence of floats
  - ``None`` enables automatic regression mean initialization
  - ``SingleOutputGBDT`` resolves one scalar base score
  - ``MultiOutputGBDT`` accepts either one scalar or one value per output column

- ``reg_l1``: default = ``0.0``, type = float
  - L1 regularization term
  - The upstream code notes that this is not currently used for sparse split finding

- ``reg_l2``: default = ``1.0``, type = float
  - L2 regularization term

- ``gamma``: default = ``1e-3``, type = float
  - Minimum objective gain required for a split
  - Applies to the root split as well as deeper nodes

- ``subsample``: default = ``1.0``, type = float
  - Fraction of training rows used to fit each tree, in the range ``(0, 1]``
  - A value below one draws a fresh sample without replacement for each tree. The sample size is the fraction times the training row count, rounded down, with a minimum of one row.
  - Split statistics and leaf values use the sampled rows. Each tree updates predictions for every training and validation row.

- ``colsample_bytree``: default = ``1.0``, type = float
  - Fraction of input features available for splitting, in the range ``(0, 1]``. Each tree draws a fresh set without replacement and uses it at every node. With 61 features, ``colsample_bytree=0.8`` draws 48 per tree. The count is rounded down, with a minimum of one.
  - Available on both boosters and both sklearn regressors. It can be combined with row sampling and, in dense multi-output regression, random output projection. All input columns are still supplied to ``predict()`` in their original order.
  - To continue the same sampling sequence after loading a model, supply the original fraction and seed. Prediction uses the feature indices stored in the trees.

Training call hooks
-------------------

The public callback hooks live on ``train(...)`` rather than inside the ``params`` dictionary:

- ``train(num, objective=None, eval_metric=None, maximize=None)``
  - available on ``SingleOutputGBDT`` and ``MultiOutputGBDT``
  - Built-in classifiers maximize validation accuracy by default. Pass ``eval_metric="logloss"`` to minimize validation log loss for this training call. The legacy ``train_multi`` method accepts the same metric.
  - ``objective(preds, y_true)`` must return ``(grad, hess)``
  - For custom objectives, ``eval_metric(preds, y_true)`` must return a scalar float and ``maximize`` controls whether larger values are better.

Shape rules:

- ``SingleOutputGBDT.train(..., objective=...)`` uses 1D prediction and label arrays shaped ``(n_samples,)``
- ``MultiOutputGBDT.train(..., objective=...)`` uses 2D prediction and label arrays shaped ``(n_samples, out_dim)``

Custom early stopping:

- if ``early_stop > 0`` and evaluation labels are registered on the custom-objective path, then ``eval_metric`` and ``maximize`` must also be provided
- the protected ``_set_gh(...)`` plus ``boost()`` workflow remains available for advanced manual control

Model-specific notes
--------------------

- ``MultiOutputGBDT`` expects multi-output labels shaped like ``(n_samples, out_dim)``
- ``SingleOutputGBDT`` is best used with one target column at a time
- for a comparison with a multi-output baseline using ``SingleOutputGBDT``, train one model per target column and stack their predictions manually
- ``SingleOutputGBDT.train_multi(...)`` is a legacy helper for multi-class classification style workflows, not the common baseline path used in this fork's examples
