OmniGBDT Documentation
======================

OmniGBDT packages the original `GBDT-MO <https://github.com/zzd1992/GBDTMO>`__ algorithm as a regular Python library. It keeps the native C++ core and adds modern packaging, cross-platform wheels, and optional sklearn-compatible wrappers.

The main user-facing entry points are ``MultiOutputGBDT`` and ``SingleOutputGBDT``.

For the original project, benchmark figures, experiment scripts, and upstream research context, please see:

- `Original repository <https://github.com/zzd1992/GBDTMO>`_
- `Experiment repository <https://github.com/zzd1992/GBDTMO-EX>`_

Installation
------------

Released package
^^^^^^^^^^^^^^^^

Install the released package:

.. code-block:: bash

   pip install omnigbdt

or with uv:

.. code-block:: bash

   uv add omnigbdt

Optional extras
^^^^^^^^^^^^^^^

Optional extras are available for plotting and sklearn-compatible wrappers:

.. code-block:: bash

   pip install "omnigbdt[plot]"
   pip install "omnigbdt[sklearn]"

or with uv:

.. code-block:: bash

   uv add "omnigbdt[plot]"
   uv add "omnigbdt[sklearn]"

The optional sklearn wrappers are a fork-specific addition. They make it possible to use sklearn inspection utilities such as permutation-based feature importance:

- `Feature importance based on feature permutation <https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation>`_
- `Permutation feature importance <https://scikit-learn.org/stable/modules/permutation_importance.html>`_

Quick start
-----------

The example below shows the normal multi-output training flow:

.. code-block:: python

   import numpy as np
   from omnigbdt import MultiOutputGBDT, Verbosity

   rng = np.random.default_rng(0)

   X = rng.random((512, 4)).astype("float64")
   shared_signal = (
       1.5 * X[:, 0]
       - 0.8 * X[:, 1]
       + 0.4 * np.sin(np.pi * X[:, 2])
   )
   Y = np.column_stack([
       1.2 * shared_signal + 0.3 * X[:, 2] * X[:, 3],
       0.9 * shared_signal - 0.4 * X[:, 0] + 0.2 * X[:, 3],
       1.1 * shared_signal + 0.5 * X[:, 1] * X[:, 3],
   ]).astype("float64")
   Y += 0.05 * rng.standard_normal(512)[:, None]
   Y += 0.02 * rng.standard_normal((512, 3))

   model = MultiOutputGBDT(
       out_dim=Y.shape[1],
       params={"loss": b"mse", "max_depth": 3, "lr": 0.1, "verbosity": Verbosity.SILENT},
   )
   model.set_data((X, Y))
   model.train(1)
   preds = model.predict(X[:5])
   print(preds.shape)

For a fuller example, a comparison with ``SingleOutputGBDT``, and an sklearn ``permutation_importance`` example, see :doc:`example`.

Source and development installs
-------------------------------

From source
^^^^^^^^^^^

Install from source:

.. code-block:: bash

   pip install .

or with uv from a parent directory:

.. code-block:: bash

   uv add ./OmniGBDT

On Windows, use either ``uv add .\\OmniGBDT`` or ``uv add ./OmniGBDT``.
Do not use ``uv add OmniGBDT`` without ``./`` or ``.\\`` because that asks the package registry for a published package named ``omnigbdt`` instead of using the local folder.

The wheel-first release target for this repository is:

- Linux x86_64
- Windows x86_64
- macOS arm64 (Apple Silicon, 14+)

Use OmniGBDT inside an existing uv project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add OmniGBDT as a normal released dependency:

.. code-block:: bash

   uv add omnigbdt

Add a sibling checkout as an editable dependency:

.. code-block:: bash

   uv add --editable ../OmniGBDT

If you copy the ``OmniGBDT`` folder inside an existing ``uv`` workspace and run:

.. code-block:: bash

   uv add ./OmniGBDT

then ``uv`` may treat it as a workspace member. If you want it to remain a plain path dependency instead, use:

.. code-block:: bash

   uv add --no-workspace ./OmniGBDT

The equivalent manual configuration in ``pyproject.toml`` is:

.. code-block:: toml

   [project]
   dependencies = ["omnigbdt"]

   [tool.uv.sources]
   omnigbdt = { path = "../OmniGBDT", editable = true }

Windows source builds
^^^^^^^^^^^^^^^^^^^^^

Local path installs such as ``uv add ./OmniGBDT`` and source installs such as ``pip install .`` compile the native C++ library during installation.

On Windows, install Visual Studio Build Tools 2022 (or Visual Studio 2022) with the ``Desktop development with C++`` workload before building from source.

If CMake reports that ``nmake`` is missing and ``CMAKE_CXX_COMPILER`` is not set, the current shell does not have a usable MSVC toolchain configured. Reopen the terminal from ``x64 Native Tools Command Prompt for VS 2022`` and try the install again. If the toolchain is already installed, check that ``CMAKE_GENERATOR`` is not forcing ``NMake Makefiles`` in a shell where ``nmake.exe`` is unavailable.

What OmniGBDT adds
------------------

Compared with the upstream repository, OmniGBDT:

- replaces the old ``make.sh`` and manual shared-library workflow with standard Python packaging
- bundles the native library inside the Python package
- keeps ``load_lib(path=None)`` for advanced or compatibility workflows
- adds wheel automation for Linux, macOS, and Windows
- adds optional sklearn-compatible wrappers so users can apply sklearn inspection tools such as permutation-based feature importance

Project provenance
------------------

This fork builds directly on the original GBDT-MO implementation by Zhendong Zhang and Cheolkon Jung.

OmniGBDT is intended to make the package easier to build, install, and distribute. It is not the canonical source for the paper, benchmark tables, figures, or research documentation.

If you use this project in research, please credit the original paper:

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

   Python API <api>
   Parameters <parameters>
   Examples <example>
