OmniGBDT Documentation
======================

OmniGBDT packages the original `GBDT-MO <https://github.com/zzd1992/GBDTMO>`__ algorithm as a regular Python library. The native C++ training core remains in place, while the Python layer adds wheel-based installation, public custom-objective hooks, optional sklearn-compatible wrappers, and accuracy-oriented regression defaults.

At the mathematical level, GBDT-MO replaces the usual one-tree-per-target strategy with a single tree whose split score is the summed second-order objective gain across all outputs. This allows one tree to reuse shared structure across correlated targets, supports optional sparse leaves that update only the most relevant outputs, and extends histogram-based split search to the multi-output case so the costs of training remain practical at larger output dimensions.

The main user-facing entry points are ``MultiOutputGBDT`` and ``SingleOutputGBDT``.

Why OmniGBDT
------------

- Joint multi-output gradient boosting from the original GBDT-MO research codebase
- Standard ``pip`` and ``uv`` installation with the native library bundled inside the package
- Public Python callbacks for custom gradients, Hessians, metrics, and early stopping
- Fixed-thread deterministic CPU training through the public ``deterministic`` parameter
- Optional sklearn-compatible wrappers for tools such as permutation importance
- Accuracy-oriented regression defaults in the current fork: ``num_rounds=200``, ``lr=0.05``, ``max_bins=128``, ``early_stop=15``, and automatic mean initialization when ``base_score`` is unset

For the original project, benchmark figures, experiment scripts, and upstream research context, please see:

- `Original repository <https://github.com/zzd1992/GBDTMO>`_
- `Experiment repository <https://github.com/zzd1992/GBDTMO-EX>`_

Installation
------------

Install the released package:

.. code-block:: bash

   pip install omnigbdt

or with uv:

.. code-block:: bash

   uv add omnigbdt

For optional extras, source installs, local-path installs, and Windows build notes, see :doc:`install`.

Quick start
-----------

This source example uses the 0.7.0 coupled quadratic loss and random projection.
Install from the repository root with ``python -m pip install .``.
The two targets share a signal. In this example, the model needs to predict
both targets accurately and get their average right.

.. code-block:: python

   import numpy as np

   from omnigbdt import MultiOutputGBDT, Verbosity

   rng = np.random.default_rng(0)
   X = rng.normal(size=(400, 6))
   shared_signal = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3]
   Y = np.column_stack(
       [
           shared_signal + 0.3 * X[:, 4],
           shared_signal - 0.2 * X[:, 5],
       ]
   )

   X_train, Y_train = X[:240], Y[:240]
   X_valid, Y_valid = X[240:320], Y[240:320]
   X_test = X[320:]

   model = MultiOutputGBDT(
       out_dim=Y.shape[1],
       params={
           "output_metric": np.array([[3, 1], [1, 3]]) / 8,
           "projection_dim": 1,
           "seed": 17,
           "num_threads": 1,
           "verbosity": Verbosity.SILENT,
       },
   )
   model.set_data((X_train, Y_train), (X_valid, Y_valid))
   model.train(200)

   preds = model.predict(X_test)
   print(preds.shape)  # (80, 2)

The ``output_metric`` matrix, ``np.array([[3, 1], [1, 3]]) / 8``, defines
the coupled quadratic loss. For each row, half the weight goes to the mean
of the two targets’ squared prediction errors. The other half goes to the
squared difference between the predicted average and the true average.

If both predictions are high by one, their average is also high by one.
The loss is ``0.5 × 1 + 0.5 × 1 = 1``. If one prediction is high by one
and the other is low by one, each still has a squared error of 1, but
their average is correct. The loss is then ``0.5 × 1 + 0.5 × 0 = 0.5``.

With ``projection_dim=1``, each tree
uses one fresh random combination of the weighted target errors to choose
its splits. Each leaf still predicts both targets. See :doc:`parameters` for
other weights and projection dimensions.

For a real-world financial benchmark based on the UCI Stock Portfolio Performance dataset, a comparison with ``SingleOutputGBDT``, and an sklearn ``permutation_importance`` example, see :doc:`example`.

Documentation guide
-------------------

- :doc:`install` for released-package installs, source installs, and platform notes
- :doc:`example` for runnable examples, including the UCI stock portfolio dataset
- :doc:`benchmarks` compares test error, training time and prediction time with XGBoost 3.4.1 on stock portfolios, building energy loads, simulated product prices and two synthetic examples
- :doc:`api` for the main Python entry points
- :doc:`parameters` for the parameter dictionary and callback hook signatures
- :doc:`differences` for fork-specific behavior and deviations from the original package
- :doc:`development` for local contributor workflows

Differences from upstream
-------------------------

Compared with the upstream repository, OmniGBDT currently adds:

- standard Python packaging and bundled native-library loading
- public Python callback hooks for custom gradients, Hessians, metrics, and early stopping
- public ``deterministic`` parameter for fixed-thread CPU repeatability on the same platform
- optional sklearn-compatible wrappers
- automatic regression mean initialization when ``base_score`` is omitted
- scalar or per-output ``base_score`` values for ``MultiOutputGBDT``
- accuracy-oriented wrapper defaults for regression workflows

See :doc:`differences` for a fuller summary, including native-code adjustments that can change same-seed trees relative to older buggy runs.

Project provenance
------------------

This fork builds on the shared-tree, vector-leaf algorithm of Zhang and
Jung. The 50/50 loss in the quick start is a choice for this example.
Random projection for split selection draws on work by Joly, Wehenkel and
Geurts and on SketchBoost by Iosipoi and Vakhrushev.

- `GBDT-MO paper <https://arxiv.org/abs/1909.04373>`_
- `Gradient tree boosting with random output projections <https://arxiv.org/abs/1905.07558>`_
- `SketchBoost paper <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html>`_

If this project is used in research, please credit the original paper:

.. code-block:: bibtex

   @article{zhang2020gbdt,
     title={GBDT-MO: Gradient-boosted decision trees for multiple outputs},
     author={Zhang, Zhendong and Jung, Cheolkon},
     journal={IEEE transactions on neural networks and learning systems},
     volume={32},
     number={7},
     pages={3156--3167},
     year={2021},
     doi={10.1109/TNNLS.2020.3009776},
     publisher={IEEE}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Installation <install>
   Examples <example>
   Benchmarks <benchmarks>
   Python API <api>
   Parameters <parameters>
   Differences from upstream <differences>
   Development <development>
