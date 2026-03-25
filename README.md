# OmniGBDT

OmniGBDT packages the original [GBDT-MO](https://github.com/zzd1992/GBDTMO) algorithm as a regular Python library. It keeps the native C++ training core and adds modern Python packaging, cross-platform wheels, and optional sklearn-compatible wrappers.

The main public classes are `MultiOutputGBDT` and `SingleOutputGBDT`.

For the original project, benchmark figures, experiment scripts, and upstream research context, please see:

- Original repository: <https://github.com/zzd1992/GBDTMO>
- Experiment and evaluation repository: <https://github.com/zzd1992/GBDTMO-EX>

## Installation

### Install the released package

```bash
pip install omnigbdt
```

or with `uv`:

```bash
uv add omnigbdt
```

OmniGBDT targets wheel-based installs on:

- Linux x86_64
- Windows x86_64
- macOS arm64 (Apple Silicon, 14+)

The GitHub Actions workflow builds these wheels in CI and publishes them on version tags matching `v*`.

### Optional extras

Install plotting support if you want to render dumped trees with `create_graph()`:

```bash
pip install "omnigbdt[plot]"
```

Install sklearn-compatible wrappers if you want to use tools such as `permutation_importance`:

```bash
pip install "omnigbdt[sklearn]"
```

The same extras can be installed with `uv`:

```bash
uv add "omnigbdt[plot]"
uv add "omnigbdt[sklearn]"
```

The optional sklearn wrappers are a fork-specific addition. They make it possible to use sklearn inspection utilities such as permutation-based feature importance:

- [Feature importances with a forest of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-feature-permutation)
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

## Quick Start

### Minimum workable example

The example below creates a multi-output regression problem with intentionally correlated targets. It compares:

- one `MultiOutputGBDT` model trained on the full target matrix
- one `SingleOutputGBDT` model per target column

```python
import numpy as np
from omnigbdt import SingleOutputGBDT, MultiOutputGBDT, Verbosity

rng = np.random.default_rng(0)

n_samples = 512
n_features = 4
n_outputs = 3

X = rng.random((n_samples, n_features)).astype("float64")
shared_signal = (
    1.5 * X[:, 0]
    - 0.8 * X[:, 1]
    + 0.4 * np.sin(np.pi * X[:, 2])
)
target_specific = np.column_stack([
    0.3 * X[:, 2] * X[:, 3],
    -0.4 * X[:, 0] + 0.2 * X[:, 3],
    0.5 * X[:, 1] * X[:, 3],
])
shared_noise = 0.05 * rng.standard_normal(n_samples)[:, None]
independent_noise = 0.02 * rng.standard_normal((n_samples, n_outputs))
Y = np.column_stack([
    1.2 * shared_signal,
    0.9 * shared_signal,
    1.1 * shared_signal,
]).astype("float64")
Y += target_specific + shared_noise + independent_noise

params = {
    "loss": b"mse",
    "max_depth": 3,
    "lr": 0.1,
    "num_threads": 1,
    "verbosity": Verbosity.SILENT,
}

multi = MultiOutputGBDT(out_dim=n_outputs, params=params)
multi.set_data((X, Y))
multi.train(1)
multi_preds = multi.predict(X)

single_models = []
for col in range(n_outputs):
    model = SingleOutputGBDT(params=params)
    target = np.ascontiguousarray(Y[:, col])
    model.set_data((X, target))
    model.train(1)
    single_models.append(model)

single_preds = np.column_stack([model.predict(X) for model in single_models])

multi_rmse = np.sqrt(np.mean((multi_preds - Y) ** 2))
single_rmse = np.sqrt(np.mean((single_preds - Y) ** 2))

print("MultiOutputGBDT RMSE:", round(float(multi_rmse), 6))
print("SingleOutputGBDT-per-target RMSE:", round(float(single_rmse), 6))
print("Prediction shape from MultiOutputGBDT:", multi.predict(X[:3]).shape)
print("Prediction shape from stacked SingleOutputGBDT models:", single_preds[:3].shape)
```

### Permutation importance with sklearn

Install the optional sklearn extra first:

```bash
pip install "omnigbdt[sklearn]"
```

or with `uv`:

```bash
uv add "omnigbdt[sklearn]"
```

Then use the sklearn-compatible multi-output wrapper with permutation importance:

```python
import time

import numpy as np
from sklearn.inspection import permutation_importance

from omnigbdt import MultiOutputGBDTRegressor

rng = np.random.default_rng(0)
X = rng.random((256, 4)).astype("float64")
shared_signal = (
    1.2 * X[:, 0]
    - 0.7 * X[:, 1]
    + 0.3 * np.sin(np.pi * X[:, 2])
)
shared_noise = 0.05 * rng.standard_normal(256)[:, None]
Y = np.column_stack([
    1.1 * shared_signal + 0.2 * X[:, 3],
    0.9 * shared_signal - 0.3 * X[:, 0],
    1.0 * shared_signal + 0.4 * X[:, 1] * X[:, 3],
]).astype("float64")
Y += shared_noise + 0.02 * rng.standard_normal((256, 3))

model = MultiOutputGBDTRegressor(
    num_rounds=10,
    max_depth=3,
    num_threads=1,
)
model.fit(X, Y)

start_time = time.time()
result = permutation_importance(
    model,
    X,
    Y,
    scoring="r2",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)
elapsed_time = time.time() - start_time

print(f"Elapsed time: {elapsed_time:.3f} seconds")
print(result.importances_mean)
```

## Source and Development Installs

### Install from source

```bash
pip install .
```

or with `uv`:

```bash
uv add ./OmniGBDT
```

That `uv add ./OmniGBDT` form is a local path dependency, so it builds from source on the current machine.

On Windows, use either `uv add .\\OmniGBDT` or `uv add ./OmniGBDT`.
Do not use `uv add OmniGBDT` without `./` or `.\\`, because that asks the package registry for a published package named `omnigbdt` instead of using the local folder.

### Use OmniGBDT inside an existing uv project

Add OmniGBDT as a normal released dependency:

```bash
uv add omnigbdt
```

Add a sibling checkout as an editable dependency while developing two projects side by side:

```bash
uv add --editable ../OmniGBDT
```

If you copy the `OmniGBDT` folder inside an existing `uv` workspace and run:

```bash
uv add ./OmniGBDT
```

then `uv` may treat it as a workspace member. If you want it to remain a plain path dependency, use:

```bash
uv add --no-workspace ./OmniGBDT
```

On Windows, the same commands are:

```bash
uv add .\\OmniGBDT
uv add --no-workspace .\\OmniGBDT
```

The equivalent manual configuration in `pyproject.toml` is:

```toml
[project]
dependencies = ["omnigbdt"]

[tool.uv.sources]
omnigbdt = { path = "../OmniGBDT", editable = true }
```

### Windows source builds

Local path installs such as `uv add ./OmniGBDT` and source installs such as `pip install .` compile the native C++ library during installation.

On Windows, install:

- Visual Studio Build Tools 2022 (or Visual Studio 2022)
- the `Desktop development with C++` workload
- MSVC build tools and a working OpenMP-capable compiler

If CMake fails with an error such as:

```text
Running 'nmake' '-?' failed with: no such file or directory
CMAKE_CXX_COMPILER not set, after EnableLanguage
```

then the package is being built from source but the MSVC toolchain is not available in the current shell.

The most reliable fix is:

1. Install Visual Studio Build Tools 2022 with the C++ workload.
2. Reopen the terminal from `x64 Native Tools Command Prompt for VS 2022`.
3. Rerun `uv add ./OmniGBDT` or `pip install .`.

If the toolchain is already installed, also check that `CMAKE_GENERATOR` is not forcing `NMake Makefiles` in a shell where `nmake.exe` is unavailable.

## What OmniGBDT Adds

Compared with the upstream repository, OmniGBDT:

- replaces the old `make.sh` and manual shared-library workflow with standard Python packaging
- bundles the native library inside the Python package
- keeps `load_lib(path=None)` for advanced or compatibility workflows
- adds wheel automation for Linux, macOS, and Windows
- adds optional sklearn-compatible wrappers so users can apply sklearn inspection tools such as permutation-based feature importance

### Core native-code deviations from upstream GBDT-MO

Most changes in this fork are packaging and distribution changes. The native C/C++ training code has only been changed in a few targeted ways so far:

- stricter `min_samples` enforcement during split scoring: candidate split points are rejected unless both child branches satisfy `min_samples`
- safe child-node materialization after a split: if a branch cannot be split further, it is emitted as an explicit leaf instead of being left implicit or partially unassigned
- proper root-leaf fallback: if no valid split exists at the root, the model stores a true single-leaf tree and prediction, dump, and load work cleanly for that case

As a consequence, same-seed runs do not necessarily match older buggy runs exactly. Trees can be smaller because invalid small-child splits are filtered earlier, and the control flow through the native code changes accordingly.

Outside of those fixes, the core objective functions, histogram-based split search, and overall training structure are still inherited from the original repository.

## Project Provenance

This fork builds directly on the original GBDT-MO implementation by Zhendong Zhang and Cheolkon Jung.

OmniGBDT is intended to make the package easier to build, install, and distribute. It is not the canonical source for the paper, benchmark tables, figures, or research documentation.

For evaluation metrics, dataset-specific experiments, and extended project context, please refer to:

- <https://github.com/zzd1992/GBDTMO>
- <https://github.com/zzd1992/GBDTMO-EX>

## Development

### Run tests

For local development with `uv`, sync the project together with the optional test dependencies:

```bash
uv sync --extra test
```

Then run the test suite with:

```bash
uv run pytest
```

If you only want the smoke coverage in this repository, you can run:

```bash
uv run pytest tests/test_smoke.py
```

### Build the documentation locally

The hosted documentation is configured through the repository-level `.readthedocs.yaml` file and the pinned dependencies in `docs/requirements.txt`.

To preview the docs locally with the same Sphinx dependency set used on Read the Docs, run:

```bash
uv run --no-project --with-requirements docs/requirements.txt sphinx-build -W -n -b html docs _build/html
```

### Build the native library directly

If you only want to build the native library:

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Versioning

This fork follows Semantic Versioning independently from the upstream GBDT-MO repository.

## License

This fork is distributed under the Apache License 2.0. The main license text for this fork is in [LICENSE](LICENSE).

Because this repository incorporates and modifies the original GBDT-MO codebase, the original upstream MIT license notice from Zhendong Zhang is preserved in [LICENSE.upstream](LICENSE.upstream). Additional attribution and fork-specific notice text is provided in [NOTICE](NOTICE).

## Citation

If you use this project in research, please credit the original paper by Zhang and Jung:

```bibtex
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
```
