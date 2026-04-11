# OmniGBDT

OmniGBDT packages the original [GBDT-MO](https://github.com/zzd1992/GBDTMO) algorithm as a regular Python library. The native C++ training core remains in place, while the Python layer adds wheel-based installation, public custom-objective hooks, optional sklearn-compatible wrappers, and accuracy-oriented regression defaults.

The main public classes are `MultiOutputGBDT` and `SingleOutputGBDT`.

## Why OmniGBDT

- Joint multi-output gradient boosting from the original GBDT-MO research codebase
- Standard `pip` and `uv` installation with the native library bundled inside the package
- Public Python callbacks for custom gradients, Hessians, metrics, and early stopping
- Fixed-thread deterministic CPU training through the public `deterministic` parameter
- Optional sklearn-compatible wrappers for tools such as permutation importance
- Accuracy-oriented regression defaults in the current fork: `num_rounds=200`, `lr=0.05`, `max_bins=128`, `early_stop=15`, and automatic mean initialization when `base_score` is unset

For the original project, benchmark figures, experiment scripts, and upstream research context, please see:

- Original repository: <https://github.com/zzd1992/GBDTMO>
- Experiment and evaluation repository: <https://github.com/zzd1992/GBDTMO-EX>

## Installation

Install the released package:

```bash
pip install omnigbdt
```

or with `uv`:

```bash
uv add omnigbdt
```

Optional extras:

```bash
pip install "omnigbdt[plot]"
pip install "omnigbdt[sklearn]"
```

The current wheel targets are:

- Linux x86_64
- Windows x86_64
- macOS arm64 (Apple Silicon, 14+)

## First Model

The example below trains one `MultiOutputGBDT` model on two correlated targets using only NumPy:

```python
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

preds = model.predict(X_test)
print(preds.shape)
```

`SingleOutputGBDT` can be used to train one model per target column as a simple baseline. A real-world financial benchmark based on the UCI Stock Portfolio Performance dataset, together with custom-objective and sklearn examples, is available in the [hosted examples page](https://omnigbdt.readthedocs.io/en/latest/example.html).

## Differences From The Original Package

Compared with the upstream GBDT-MO repository, OmniGBDT currently adds:

- standard Python packaging and bundled native-library loading
- wheel automation for Linux, macOS, and Windows
- public Python callback hooks for custom gradients, Hessians, metrics, and early stopping
- public `deterministic` parameter for fixed-thread CPU repeatability on the same platform
- optional sklearn-compatible wrappers
- automatic regression mean initialization when `base_score` is omitted
- scalar or per-output `base_score` values for `MultiOutputGBDT`
- accuracy-oriented wrapper defaults for regression workflows

Several targeted native-code fixes are also part of the fork, so same-seed runs are not guaranteed to match older buggy runs exactly. A fuller summary is available in the [Differences From Upstream page](https://omnigbdt.readthedocs.io/en/latest/differences.html).

## Documentation Guide

- Hosted documentation: <https://omnigbdt.readthedocs.io>
- Installation and source-build notes: [install page](https://omnigbdt.readthedocs.io/en/latest/install.html)
- Worked examples, including the financial benchmark: [examples page](https://omnigbdt.readthedocs.io/en/latest/example.html)
- Python API reference: [API page](https://omnigbdt.readthedocs.io/en/latest/api.html)
- Parameter reference: [parameters page](https://omnigbdt.readthedocs.io/en/latest/parameters.html)
- Fork-specific differences: [differences page](https://omnigbdt.readthedocs.io/en/latest/differences.html)
- Development workflow: [development page](https://omnigbdt.readthedocs.io/en/latest/development.html)

## Project Provenance

This fork builds directly on the original GBDT-MO implementation by Zhendong Zhang and Cheolkon Jung.

OmniGBDT is intended to make the package easier to build, install, and distribute. It is not the canonical source for the paper, benchmark tables, figures, or research documentation.

## Versioning

This fork follows Semantic Versioning independently from the upstream GBDT-MO repository.

## License

This fork is distributed under the Apache License 2.0. The main license text for this fork is in [LICENSE](LICENSE).

Because this repository incorporates and modifies the original GBDT-MO codebase, the original upstream MIT license notice from Zhendong Zhang is preserved in [LICENSE.upstream](LICENSE.upstream). Additional attribution and fork-specific notice text is provided in [NOTICE](NOTICE).

## Citation

If this project is used in research, please credit the original paper by Zhang and Jung:

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
