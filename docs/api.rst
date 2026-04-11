Python API
==========

This page summarizes the main Python entry points in OmniGBDT. For installation and runnable examples, see :doc:`install` and :doc:`example`.

Common data requirements
------------------------

- feature arrays should be ``float64`` and two-dimensional with shape ``(n_samples, n_features)``
- multi-output labels should be ``float64`` or ``int32`` and two-dimensional with shape ``(n_samples, out_dim)``
- single-output labels should be contiguous ``float64`` or ``int32`` one-dimensional arrays
- For slicing one column out of a 2D label matrix, use ``np.ascontiguousarray(...)`` before passing it to ``SingleOutputGBDT``

Core models
-----------

MultiOutputGBDT
^^^^^^^^^^^^^^^

.. class:: MultiOutputGBDT(lib=None, out_dim=1, params=None)

   Multi-output boosted tree model.

   :param lib: optional handle returned by ``load_lib()``
   :param int out_dim: number of output columns
   :param dict params: training parameters; missing values fall back to defaults

   ``MultiOutputGBDT`` is the main entry point to learn multiple outputs jointly.

   When ``params["base_score"]`` is left as ``None`` with ``loss=b"mse"``, the initial prediction is inferred from the training-label mean for each output column.

   When ``params["deterministic"]`` is ``True``, repeated CPU runs on the same platform are intended to be repeatable for a fixed ``num_threads`` setting.

   .. method:: set_data(train_set=None, eval_set=None)

      Register training and optional evaluation data.

      ``train_set`` and ``eval_set`` are tuples of ``(X, y)``.

      - ``X`` must be a 2D ``float64`` array
      - ``y`` may be ``None`` or a 2D ``float64`` / ``int32`` array with one column per output

   .. method:: train(num, objective=None, eval_metric=None, maximize=None)

      Train the model for ``num`` boosting rounds.

      - when ``objective`` is omitted, OmniGBDT uses the built-in native loss from ``params["loss"]``
      - when ``objective`` is provided, it must return ``(grad, hess)`` from the current prediction matrix and label matrix
      - for ``MultiOutputGBDT``, those callback arrays are 2D with shape ``(n_samples, out_dim)``
      - ``eval_metric`` may be used to report a scalar metric for the train and eval splits during custom-objective training
      - if ``early_stop > 0`` and evaluation labels are registered on the custom-objective path, then ``eval_metric`` and ``maximize`` must also be provided

   .. method:: predict(x, num_trees=0)

      Predict on a 2D ``float64`` feature matrix.

      - when ``num_trees == 0``, all learned trees are used
      - returns a 2D array with shape ``(n_samples, out_dim)``

   .. method:: dump(path)

      Write the learned model to a text file.

      ``path`` accepts ``str``, ``bytes``, and ``pathlib.Path``.

   .. method:: load(path)

      Load a text-dumped model from disk.

      ``path`` accepts ``str``, ``bytes``, and ``pathlib.Path``.

   .. method:: _set_gh(g, h)

      Set gradient and hessian arrays for the next call to ``boost()``. This is an advanced escape hatch for manual custom-loss workflows.

   .. method:: _set_label(x, is_train)

      Replace labels for the training or evaluation dataset without rebuilding the feature binning.

   .. method:: boost()

      Grow a single tree after calling ``_set_gh(...)``.

   .. method:: close()

      Release the underlying native model explicitly. This is optional, but useful in longer-running scripts.

SingleOutputGBDT
^^^^^^^^^^^^^^^^

.. class:: SingleOutputGBDT(lib=None, out_dim=1, params=None)

   Single-output boosted tree model.

   :param lib: optional handle returned by ``load_lib()``
   :param int out_dim: output dimension used by prediction helpers; for the common single-target case, leave this at ``1``
   :param dict params: training parameters; missing values fall back to defaults

   ``SingleOutputGBDT`` can be used to train one model per target column as a simple baseline.

   When ``params["base_score"]`` is left as ``None`` with ``loss=b"mse"``, the initial prediction is inferred from the training-label mean.

   When ``params["deterministic"]`` is ``True``, repeated CPU runs on the same platform are intended to be repeatable for a fixed ``num_threads`` setting.

   .. method:: set_data(train_set=None, eval_set=None)

      Register training and optional evaluation data.

      ``train_set`` and ``eval_set`` are tuples of ``(X, y)`` where:

      - ``X`` is a 2D ``float64`` array
      - ``y`` is typically a contiguous 1D ``float64`` or ``int32`` array

   .. method:: train(num, objective=None, eval_metric=None, maximize=None)

      Train a single-output model for ``num`` boosting rounds.

      - when ``objective`` is omitted, OmniGBDT uses the built-in native loss from ``params["loss"]``
      - when ``objective`` is provided, it must return ``(grad, hess)`` from the current prediction vector and label vector
      - for ``SingleOutputGBDT``, callback arrays are 1D with shape ``(n_samples,)``
      - the custom-objective path is only supported for the normal ``out_dim == 1`` workflow
      - if ``early_stop > 0`` and evaluation labels are registered on the custom-objective path, then ``eval_metric`` and ``maximize`` must also be provided

   .. method:: predict(x, num_trees=0)

      Predict on a 2D ``float64`` feature matrix.

      - with ``out_dim == 1``, the return value is a 1D array
      - with ``out_dim > 1``, the return value is shaped as ``(n_samples, out_dim)``

   .. method:: train_multi(num)

      Legacy helper used by the original code for multi-classification style workflows.

   .. method:: reset()

      Clear learned trees and reset predictions back to the resolved base score.

   .. method:: close()

      Release the underlying native model explicitly.

Optional sklearn wrappers
-------------------------

The sklearn-compatible wrappers are optional and require the ``sklearn`` extra:

.. code-block:: bash

   pip install "omnigbdt[sklearn]"

This is a fork-specific addition intended to make OmniGBDT work with sklearn tooling such as ``sklearn.inspection.permutation_importance``.

SingleOutputGBDTRegressor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: SingleOutputGBDTRegressor(...)

   sklearn-compatible single-target regressor wrapper around ``SingleOutputGBDT``.

   It exposes ``fit(...)``, ``predict(...)``, and ``score(...)`` for use with tools such as ``sklearn.inspection.permutation_importance``.

   Its constructor also accepts ``objective=None``, ``eval_metric=None``, ``maximize=None``, and ``deterministic=True`` and forwards them to ``SingleOutputGBDT``.

MultiOutputGBDTRegressor
^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MultiOutputGBDTRegressor(...)

   sklearn-compatible multi-output regressor wrapper around ``MultiOutputGBDT``.

   It exposes ``fit(...)``, ``predict(...)``, and ``score(...)`` for sklearn-style multi-output workflows.

   Its constructor also accepts ``objective=None``, ``eval_metric=None``, ``maximize=None``, and ``deterministic=True`` and forwards them to ``MultiOutputGBDT``.

Utilities
---------

load_lib
^^^^^^^^

.. function:: load_lib(path=None)

   Load the compiled native library and return a configured ``ctypes`` handle.

   ``path`` may be:

   - omitted, in which case the packaged native library is loaded automatically
   - a direct path to the compiled library file
   - a directory that contains the compiled library

   Most users do not need to call this directly.

Verbosity
^^^^^^^^^

.. class:: Verbosity

   Small enum-like helper for training output levels.

   - ``Verbosity.SILENT``: no native training output
   - ``Verbosity.SUMMARY``: only the final best score when evaluation data is present
   - ``Verbosity.FULL``: per-round metrics plus the final best score

create_graph
^^^^^^^^^^^^

.. function:: create_graph(file_name, tree_index=0, value_list=None)

   Build a ``graphviz.Digraph`` from a dumped text model.

   This helper is optional and requires the plotting dependency:

   .. code-block:: bash

      pip install "omnigbdt[plot]"

   :param file_name: path to a text model dump
   :param int tree_index: zero-based tree index
   :param value_list: optional list of output indices to display in leaf nodes
