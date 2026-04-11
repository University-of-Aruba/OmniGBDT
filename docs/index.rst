OmniGBDT Documentation
======================

OmniGBDT packages the original `GBDT-MO <https://github.com/zzd1992/GBDTMO>`__ algorithm as a regular Python library. The native C++ training core remains in place, while the Python layer adds wheel-based installation, public custom-objective hooks, optional sklearn-compatible wrappers, and accuracy-oriented regression defaults.

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

The example below trains one ``MultiOutputGBDT`` model on two correlated targets using only NumPy:

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
           "loss": b"mse",
           "max_depth": 4,
           "max_bins": 128,
           "lr": 0.05,
           "early_stop": 15,
           "num_threads": 1,
           "verbosity": Verbosity.SILENT,
       },
   )
   model.set_data((X_train, Y_train), (X_valid, Y_valid))
   model.train(200)
   preds = model.predict(X_test[:5])
   print(preds.shape)

For a real-world financial benchmark based on the UCI Stock Portfolio Performance dataset, a comparison with ``SingleOutputGBDT``, and an sklearn ``permutation_importance`` example, see :doc:`example`.

Documentation guide
-------------------

- :doc:`install` for released-package installs, source installs, and platform notes
- :doc:`example` for runnable examples, including the financial benchmark
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

This fork builds directly on the original GBDT-MO implementation by Zhendong Zhang and Cheolkon Jung.

OmniGBDT is intended to make the package easier to build, install, and distribute. It is not the canonical source for the paper, benchmark tables, figures, or research documentation.

If this project is used in research, please credit the original paper:

.. code-block:: bibtex

   @article{zhang2020gbdt,
     title={GBDT-MO: Gradient-boosted decision trees for multiple outputs},
     author={Zhang, Zhendong and Jung, Cheolkon},
     journal={IEEE transactions on neural networks and learning systems},
     volume={32},
     number={7},
     pages={3156--3167},
     year={2020},
     publisher={Ieee}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Installation <install>
   Examples <example>
   Python API <api>
   Parameters <parameters>
   Differences from upstream <differences>
   Development <development>
