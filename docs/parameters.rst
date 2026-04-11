Parameters
==========

This page describes the Python parameter dictionary used by ``SingleOutputGBDT`` and ``MultiOutputGBDT`` in this fork. Unless noted otherwise, defaults come from ``omnigbdt.lib_utils.default_params()``. Several defaults intentionally differ from the original package; see :doc:`differences`.

General
-------

- ``loss``: default = ``b"mse"``, type = bytes
  - Supported values are ``b"mse"``, ``b"bce"``, ``b"ce"``, and ``b"ce_column"``
  - ``b"ce_column"`` is only relevant to legacy ``SingleOutputGBDT`` classification-style workflows
  - The Python API expects a byte string, for example ``b"mse"``
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
  - Retained for API compatibility
  - The current deterministic CPU training path does not actively use randomness during tree growth

- ``hist_cache``: default = ``16``, type = int
  - Maximum number of histogram caches

- ``max_bins``: default = ``128``, type = int
  - Maximum number of bins for each input feature

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

- ``subsample``: default = ``1.0``, type = float
  - Present in the Python defaults for compatibility
  - The current native implementation does not actively use it

Training call hooks
-------------------

The public callback hooks live on ``train(...)`` rather than inside the ``params`` dictionary:

- ``train(num, objective=None, eval_metric=None, maximize=None)``
  - available on ``SingleOutputGBDT`` and ``MultiOutputGBDT``
  - ``objective(preds, y_true)`` must return ``(grad, hess)``
  - ``eval_metric(preds, y_true)`` must return a scalar float
  - ``maximize`` controls whether larger evaluation metric values are better

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
