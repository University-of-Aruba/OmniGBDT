# OmniGBDT

Use OmniGBDT to predict several targets with one gradient boosting model in Python. All targets share the same splits in each tree. Each leaf stores a vector of updates, one per target. The C++ implementation builds on [GBDT-MO](https://github.com/zzd1992/GBDTMO).

[Installation](#installation) · [First model](#first-model) · [Benchmarks](#benchmarks) · [API reference](https://omnigbdt.readthedocs.io/en/latest/api.html)

Train a joint model with `MultiOutputGBDT`, or use `SingleOutputGBDT` for a single target. In 0.7.0, `output_metric` sets how target errors count, and `projection_dim` sets how many random combinations of those errors each tree uses to choose splits. Both are optional. For custom objectives, use the Python callbacks. For pipelines and model selection, use the scikit-learn estimator.

## Installation

Install the released package with pip.

```bash
pip install omnigbdt
```

If you use uv, add OmniGBDT to your project.

```bash
uv add omnigbdt
```

To use plotting or scikit-learn integration, install the corresponding extra.

```bash
pip install "omnigbdt[plot]"
pip install "omnigbdt[sklearn]"
```

Wheels are available for Linux x86_64, Windows x86_64, and macOS 14 or later on Apple silicon. To try the 0.7.0 quadratic loss and random projection from source, run `python -m pip install .` from the repository root. [Source installation](docs/install.rst)

## First model

Train a model on two targets that share a signal. In this example, the model needs to predict both targets accurately and get their average right.

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
```

The `output_metric` matrix, `np.array([[3, 1], [1, 3]]) / 8`, defines the coupled quadratic loss. For each row, half the weight goes to the mean of the two targets’ squared prediction errors. The other half goes to the squared difference between the predicted average and the true average.

If both predictions are high by one, their average is also high by one. The loss is `0.5 × 1 + 0.5 × 1 = 1`. If one prediction is high by one and the other is low by one, each still has a squared error of 1, but their average is correct. The loss is then `0.5 × 1 + 0.5 × 0 = 0.5`.

With `projection_dim=1`, each tree uses one fresh random combination of the weighted target errors to choose its splits. Each leaf still predicts both targets. Joly, Wehenkel and Geurts studied projecting residuals when growing trees. Iosipoi and Vakhrushev used projections to score candidate splits in SketchBoost. [Random output projections](https://arxiv.org/abs/1905.07558) · [SketchBoost](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html)

## Benchmarks

OmniGBDT 0.7.0 and XGBoost 3.4.1 used the same grouped cross-validation folds. Each test fold held out whole portfolios, building geometries or simulation games. Tuning and early stopping used only the training groups. The table reports **RMSE in the original target units** for both training losses. Lower is better within each dataset.

| Dataset | Training loss | OmniGBDT full | OmniGBDT projected | XGBoost vector-leaf |
| --- | --- | ---: | ---: | ---: |
| Stock | MSE | 0.11839 | **0.11835** | 0.12145 |
| Stock | Coupled quadratic | 0.11871 | **0.11851** | 0.12139 |
| Energy | MSE | 8.209 | 8.462 | **7.681** |
| Energy | Coupled quadratic | 10.787 | 10.671 | **7.667** |
| SCM20D | MSE | 202.789 | **198.765** | 203.086 |
| SCM20D | Coupled quadratic | 203.130 | **200.684** | 203.250 |

Projection made little difference on Stock. Its largest gain was on SCM20D, where it lowered MSE-trained RMSE by 2% versus full OmniGBDT. On the first SCM20D fold, projection cut fit time from 573 to 203 ms and increased batch prediction time from 1.66 to 1.93 ms. XGBoost had the lowest RMSE on Energy under both losses. [Quadratic scores, timings and fold details](docs/benchmarks.rst#nested-group-cross-validation) · [Per-repeat scores](docs/data/grouped-projection-benchmarks.json)

## Documentation

- [Installation and source builds](https://omnigbdt.readthedocs.io/en/latest/install.html)
- [Worked examples](https://omnigbdt.readthedocs.io/en/latest/example.html)
- [Python API](https://omnigbdt.readthedocs.io/en/latest/api.html) and [training parameters](https://omnigbdt.readthedocs.io/en/latest/parameters.html)
- [Development guide](https://omnigbdt.readthedocs.io/en/latest/development.html)

## Project origins

Zhendong Zhang and Cheolkon Jung developed the shared-tree, vector-leaf algorithm underlying OmniGBDT. [GBDT-MO paper](https://arxiv.org/abs/1909.04373) · [GBDT-MO source](https://github.com/zzd1992/GBDTMO) · [Upstream experiments](https://github.com/zzd1992/GBDTMO-EX) · [Differences in this fork](docs/differences.rst)

## Versioning

This fork follows Semantic Versioning independently from the upstream GBDT-MO repository.

## License

OmniGBDT uses the [Apache License 2.0](LICENSE). The repository preserves the [upstream MIT license](LICENSE.upstream) for the original GBDT-MO code. See [NOTICE](NOTICE) for attribution.

## Citation

If you use OmniGBDT in your research, cite the original paper by Zhang and Jung.

```bibtex
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
```
