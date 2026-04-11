Differences From Upstream
=========================

This page summarizes the main ways in which OmniGBDT differs from the original GBDT-MO package and repository workflow.

Packaging and distribution
--------------------------

Compared with the upstream repository, OmniGBDT:

- replaces the old ``make.sh`` and manual shared-library workflow with standard Python packaging
- bundles the native library inside the Python package
- keeps ``load_lib(path=None)`` for advanced or compatibility workflows
- adds wheel automation for Linux, macOS, and Windows

Python API additions
--------------------

OmniGBDT adds several public Python-facing features on top of the native core:

- public callback hooks for custom gradients, Hessians, metrics, and Python-side early stopping through ``train(..., objective=..., eval_metric=..., maximize=...)``
- public ``deterministic`` parameter for fixed-thread CPU repeatability on the same platform
- optional sklearn-compatible wrappers for ``SingleOutputGBDT`` and ``MultiOutputGBDT``
- direct interoperability with sklearn tooling such as permutation importance

Modeling behavior and defaults
------------------------------

The current OmniGBDT fork also differs from the original package in a few modeling-oriented areas:

- regression wrappers now default to ``num_rounds=200``, ``lr=0.05``, ``max_bins=128``, and ``early_stop=15``
- leaving ``base_score`` unset now enables automatic regression mean initialization
- ``MultiOutputGBDT`` accepts either one scalar ``base_score`` or one value per output column

These changes are intended to provide stronger out-of-the-box regression accuracy on the examples bundled with the fork, including the financial benchmark in :doc:`example`.

Native behavior adjustments
---------------------------

Most changes in the fork remain packaging and Python-API changes, but a small number of native-code adjustments are also present:

- stricter ``min_samples`` enforcement during split scoring
- safe child-node materialization after a split
- proper root-leaf fallback when no valid split exists
- deterministic split selection for fixed-thread CPU repeatability

As a result, same-seed runs are not guaranteed to match older buggy runs exactly. Trees may differ because invalid small-child splits are filtered earlier and the resulting control flow changes accordingly.

For the original project, benchmark figures, experiment scripts, and research context, please refer to:

- `Original repository <https://github.com/zzd1992/GBDTMO>`_
- `Experiment repository <https://github.com/zzd1992/GBDTMO-EX>`_
