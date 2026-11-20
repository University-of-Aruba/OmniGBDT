Examples
========

This page contains short, self-contained examples for the packaged OmniGBDT fork. The UCI stock portfolio benchmark is the main real-world tabular example used throughout the documentation. Installation details are available on :doc:`install`.

For benchmark scripts and extended evaluation workflows, please see the upstream repositories:

- `GBDTMO <https://github.com/zzd1992/GBDTMO>`_
- `GBDTMO-EX <https://github.com/zzd1992/GBDTMO-EX>`_

UCI stock portfolio benchmark
-----------------------------

Install the UCI dataset helper first:

.. code-block:: bash

   pip install ucimlrepo

The example below uses the `UCI Machine Learning Repository Stock Portfolio Performance dataset <https://archive.ics.uci.edu/dataset/390/stock+portfolio+performance>`_. It loads one real-world financial tabular benchmark and splits it into train, validation, and test partitions.

.. code-block:: python

   import numpy as np
   from ucimlrepo import fetch_ucirepo
   from omnigbdt import MultiOutputGBDT, Verbosity

   stock_portfolio = fetch_ucirepo(id=390)
   frame = stock_portfolio.data.original

   feature_columns = [
       "Large B/P",
       "Large ROE",
       "Large S/P",
       "Large Return Rate in the last quarter",
       "Large Market Value",
       "Small systematic Risk",
   ]
   target_columns = [
       "Annual Return.1",
       "Excess Return.1",
       "Systematic Risk.1",
       "Total Risk.1",
       "Abs. Win Rate.1",
       "Rel. Win Rate.1",
   ]

   X = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
   Y = frame.loc[:, target_columns].to_numpy(dtype=np.float64)

   rng = np.random.default_rng(0)
   indices = rng.permutation(len(X))
   train_end = int(len(X) * 0.6)
   valid_end = int(len(X) * 0.8)
   train_idx = indices[:train_end]
   valid_idx = indices[train_end:valid_end]
   test_idx = indices[valid_end:]

   X_train, Y_train = X[train_idx], Y[train_idx]
   X_valid, Y_valid = X[valid_idx], Y[valid_idx]
   X_test, Y_test = X[test_idx], Y[test_idx]

   params = {
       "loss": b"mse",
       "max_depth": 4,
       "max_bins": 128,
       "lr": 0.05,
       "early_stop": 15,
       "num_threads": 1,
       "verbosity": Verbosity.SILENT,
   }

   booster = MultiOutputGBDT(out_dim=Y.shape[1], params=params)
   booster.set_data((X_train, Y_train), (X_valid, Y_valid))
   booster.train(200)

   preds = booster.predict(X_test[:5])
   print(preds.shape)

The UCI export includes both formatted percentage columns and normalized numeric target columns. The example uses the normalized target columns from ``data.original``, which carry the ``.1`` suffix in the spreadsheet-derived column names. ``base_score`` remains unset, so regression training starts from the training-label mean automatically.

Comparison with XGBoost vector leaves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OmniGBDT and XGBoost 3.4.1 both retained 84 boosting rounds on the example split
when their leaf settings were matched. The table reports test RMSE across the six
normalized targets. The last two columns average ten splits each (seeds 0 through 9).

.. list-table::
   :header-rows: 1

   * - Model
     - Example split
     - Random rows
     - Unseen portfolios
   * - OmniGBDT 0.6.0
     - 0.12013
     - 0.11974
     - 0.12408
   * - XGBoost 3.4.1, matched leaf settings
     - 0.12013
     - 0.11974
     - 0.12765
   * - XGBoost 3.4.1, default leaf settings
     - 0.12599
     - 0.12363
     - 0.13105
   * - Training-target mean
     - 0.14087
     - 0.13749
     - 0.14670

All 63 test rows in the example represent portfolios also seen in training.
Grouping by portfolio keeps all five observations for each portfolio in one partition.
Two of the ten grouped splits accounted for about 90% of the mean difference
between OmniGBDT and matched XGBoost. With ``subsample=0.8`` in both libraries,
their grouped mean RMSE was 0.12279 and 0.12658, respectively.

The matched XGBoost run used ``multi_strategy="multi_output_tree"``,
``min_child_weight=20``, and ``gamma=0.012``. Each model used one CPU thread.
On an Apple M2, the median of 11 warm fits on the example split was 3.48 ms
for OmniGBDT at commit ``8274e88`` and 12.07 ms for matched XGBoost,
including data preparation.

To repeat the comparison from the repository root:

.. code-block:: bash

   python -m pip install "omnigbdt==0.6.0" "xgboost==3.4.1"
   curl -fL https://archive.ics.uci.edu/static/public/390/data.csv -o stock.csv
   python examples/stock_portfolio_comparison.py stock.csv results.json
   python examples/stock_portfolio_comparison.py stock.csv results-80.json --subsample 0.8

Comparing ``SingleOutputGBDT`` and ``MultiOutputGBDT``
------------------------------------------------------

Continuing from the UCI stock portfolio example above, one simple baseline is to train:

- one ``MultiOutputGBDT`` model on the full target matrix
- one ``SingleOutputGBDT`` model per target column

.. code-block:: python

   import numpy as np
   from omnigbdt import SingleOutputGBDT

   multi_preds = booster.predict(X_test)

   single_models = []
   for col in range(Y.shape[1]):
       model = SingleOutputGBDT(params=params)
       target = np.ascontiguousarray(Y_train[:, col])
       eval_target = np.ascontiguousarray(Y_valid[:, col])
       model.set_data((X_train, target), (X_valid, eval_target))
       model.train(200)
       single_models.append(model)

   single_preds = np.column_stack([model.predict(X_test) for model in single_models])

   multi_rmse = np.sqrt(np.mean((multi_preds - Y_test) ** 2))
   single_rmse = np.sqrt(np.mean((single_preds - Y_test) ** 2))

   print("Held-out RMSE from MultiOutputGBDT:", round(float(multi_rmse), 6))
   print("Held-out RMSE from stacked SingleOutputGBDT models:", round(float(single_rmse), 6))

Dumping and loading a model
---------------------------

Continuing from the UCI stock portfolio example above:

.. code-block:: python

   from pathlib import Path

   model_path = Path("omnigbdt_model.txt")
   booster.dump(model_path)

   reloaded = MultiOutputGBDT(out_dim=Y.shape[1], params=params)
   reloaded.set_booster(X.shape[1], Y.shape[1])
   reloaded.load(model_path)

Advanced manual loading
-----------------------

Normal usage does not require manual shared-library handling, but the compatibility helper is still available:

.. code-block:: python

   from omnigbdt import MultiOutputGBDT, load_lib

   lib = load_lib("/path/to/native/library/or/folder")
   booster = MultiOutputGBDT(lib=lib, out_dim=3, params={"loss": b"mse"})

Optional plotting
-----------------

Install the optional plotting dependency first:

.. code-block:: bash

   pip install "omnigbdt[plot]"

Then render a dumped tree:

.. code-block:: python

   from omnigbdt import create_graph

   graph = create_graph("omnigbdt_model.txt", tree_index=0, value_list=[0, 1])
   graph.render("tree_0", format="pdf")

Custom objective
----------------

``MultiOutputGBDT`` supports public callback-based custom objectives through ``train(..., objective=...)``:

Continuing from the UCI stock portfolio example above:

.. code-block:: python

   import numpy as np
   from omnigbdt import MultiOutputGBDT, Verbosity

   def mse_objective(preds, target):
       return preds - target, np.ones_like(preds)

   def rmse_metric(preds, target):
       return float(np.sqrt(np.mean((preds - target) ** 2)))

   booster = MultiOutputGBDT(
       out_dim=Y_train.shape[1],
       params={
           "loss": b"mse",
           "max_depth": 4,
           "max_bins": 128,
           "lr": 0.05,
           "early_stop": 15,
           "num_threads": 1,
           "verbosity": Verbosity.FULL,
       },
   )
   booster.set_data((X_train, Y_train), (X_valid, Y_valid))
   booster.train(
       200,
       objective=mse_objective,
       eval_metric=rmse_metric,
       maximize=False,
   )

This uses a Python callback to supply gradients and Hessians round by round.

The protected ``_set_gh(...)`` plus ``boost()`` workflow still exists as an advanced escape hatch:

.. code-block:: python

   g, h = mse_objective(booster.preds_train.copy(), booster.label.copy())
   booster._set_gh(g, h)
   booster.boost()

For ``SingleOutputGBDT``, the custom-objective callback receives 1D arrays. For ``MultiOutputGBDT``, it receives 2D arrays shaped ``(n_samples, out_dim)``.

The sklearn-compatible wrappers forward the same callback arguments:

.. code-block:: python

   from omnigbdt import MultiOutputGBDTRegressor

   model = MultiOutputGBDTRegressor(
       num_rounds=200,
       objective=mse_objective,
       eval_metric=rmse_metric,
       maximize=False,
       max_depth=4,
       max_bins=128,
       lr=0.05,
       early_stop=15,
       num_threads=1,
   )
   model.fit(X_train, Y_train)

Permutation importance with sklearn
-----------------------------------

Install the optional sklearn extra and the UCI dataset helper first:

.. code-block:: bash

   pip install "omnigbdt[sklearn]"
   pip install ucimlrepo

Then use the sklearn-compatible wrapper with ``permutation_importance``:

.. code-block:: python

   import numpy as np
   from sklearn.inspection import permutation_importance
   from ucimlrepo import fetch_ucirepo

   from omnigbdt import MultiOutputGBDTRegressor

   stock_portfolio = fetch_ucirepo(id=390)
   frame = stock_portfolio.data.original
   feature_columns = [
       "Large B/P",
       "Large ROE",
       "Large S/P",
       "Large Return Rate in the last quarter",
       "Large Market Value",
       "Small systematic Risk",
   ]
   target_columns = [
       "Annual Return.1",
       "Excess Return.1",
       "Systematic Risk.1",
       "Total Risk.1",
       "Abs. Win Rate.1",
       "Rel. Win Rate.1",
   ]
   X = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
   Y = frame.loc[:, target_columns].to_numpy(dtype=np.float64)

   rng = np.random.default_rng(0)
   indices = rng.permutation(len(X))
   train_end = int(len(X) * 0.8)
   train_idx = indices[:train_end]
   test_idx = indices[train_end:]

   X_train, Y_train = X[train_idx], Y[train_idx]
   X_test, Y_test = X[test_idx], Y[test_idx]

   model = MultiOutputGBDTRegressor(
       num_rounds=200,
       max_depth=4,
       max_bins=128,
       lr=0.05,
       early_stop=15,
       num_threads=1,
   )
   model.fit(X_train, Y_train)

   result = permutation_importance(
       model,
       X_test,
       Y_test,
       scoring="r2",
       n_repeats=5,
       random_state=42,
       n_jobs=1,
   )

   print(result.importances_mean)
