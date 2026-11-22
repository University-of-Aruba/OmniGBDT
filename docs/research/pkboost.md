# What PKBoost claims and what its source implements

Audited Pushp Kharat’s public repository at commit `f09100d680bfcd52895904047185cfe5d0532d4c`. The numbers below are the project’s own reports. The review traced the training and prediction paths and inspected the available experiment code. The feature numbers match the upstream catalogue. [Pinned source tree][repo] · [Feature catalogue][features]

## What transfers to OmniGBDT

PKBoost’s regressor makes 80% of the input columns available to each tree. It draws without replacement, sorts the original feature indices, and reuses them throughout that tree. This can reduce repeated splits on correlated predictors and cuts the number of histograms required during growth. OmniGBDT already has row sampling, output projection and sibling histogram subtraction. Feature sampling adds a separate choice that works for ordinary, coupled and projected regression. OmniGBDT implements this directly in its C++ tree builder. [PKBoost regressor][sampling]

Other candidates had narrower payoffs. A Huber loss changes how large target errors count. The drift wrappers need newly observed labels and a time-ordered evaluation. PKBoost’s Shannon term measures binary label entropy. Its regressor calls the same tree scorer, where the term interprets a sum of labels as a count of positives. That interpretation does not extend to arbitrary continuous target vectors. [Regression training][sampling] · [Split scoring][gain] · [Classifier adaptation][adapt]

## Learning and prediction

| Catalogue items | Claimed advantage | What the implementation does |
| --- | --- | --- |
| 1, 2, 16 | Find informative splits on rare classes | Binary classification combines the regularized Newton split gain with a term for reduction in binary label entropy. The entropy weight decays with depth and a gain screen skips some candidates. The derivatives are ordinary class-weighted logistic derivatives. No term measures feature stability across time. [Split scoring][gain] · [Derivatives][deriv] |
| 3, 14, 15 | Fast histograms and resistance to distribution shifts | Fits medians and cut points on training data and reuses them at prediction. It sorts **distinct** values, puts extra cuts in their outer deciles, and imputes missing values with a training median. For a heavily repeated value, its frequency drops out of rank calculation. Auto mode sets a bin budget of 32. [Binning][bins] · [Tail allocation][tails] |
| 4–8, 33, 34 | Automatic settings, rare-class emphasis and lower variance | The auto builder derives a positive-class weight from class counts. Size, feature count and prevalence govern depth, shrinkage, regularization and sampling via hand-written rules. The classifier and regressor sample rows and columns per tree. This is a preset based on training statistics, with no validation search for optimal parameters. [Builder][builder] · [Tuner][tuner] · [Regression sampling][sampling] |
| 11, 20, 21 | Stop on minority-class performance and optimize thresholds | The binary loop monitors average precision at the first iteration and every 20th iteration, smoothing across up to three evaluations. Ranking metrics process score ties one row at a time, which can make results depend on their order. The estimator exposes a user-set decision threshold. A built-in F1-maximizing threshold search was not found in the pinned sources. [Stopping][stop] · [Metrics][metrics] · [Threshold][threshold] |
| 12, 13 | Parallel multiclass classification with separate tuning by class | One binary model is trained for each class. The final softmax receives already-sigmoided probabilities from those models. It normalizes scores, but the repository supplies no probability-calibration study. OmniGBDT’s native shared vector-leaf trees are a different model. [Multiclass implementation][ovr] |
| 22–24, 45 | Robust regression, automatic outlier handling and prediction uncertainty | Scalar regression exposes MSE, Huber and Poisson. Its auto rule counts targets outside the 1.5-IQR fences and switches to Huber above 5%, then sets delta from the training target MAD. MAE is an evaluation metric. The uncertainty output is the spread of cumulative boosting predictions for a row across rounds. Its interval coverage has not been established. [Loss selection][reg] · [Uncertainty][uncert] |
| 17–19 | Fast adaptation by retraining affected specialists | HAB clusters rows, trains one binary specialist per partition and tracks its classification error once new labels arrive. Retraining requires an error moving average above 1.3 times baseline and at least 50 buffered rows for a selected partition. This is applicable to labeled streams. [Specialists and buffer update][hab] |
| 25, 28–30, 43, 44 | Detect local or systemic drift, prune stale trees and recover | Separate “living” wrappers monitor labeled errors, maintain feature-usage and vulnerability histories, and update the ensemble from buffered observations. Regression diagnostics include residual autocorrelation and changing variance. These features need a sequential evaluation with label arrival and update timing specified. [Classifier adaptation][adapt] · [Regression diagnostics][regdrift] · [Feature tracking][metab] |

## Execution and tooling

| Catalogue items | Claimed advantage | What the implementation does |
| --- | --- | --- |
| 9, 10, 27 | Adapt to hardware and reduce memory and fit time | Rayon parallelism, feature-major byte bins and cached sibling histogram subtraction are in source. CPU counts are queried. Available RAM is estimated from an environment setting or thread count. Batch prediction first bins the complete input and then processes smaller prediction blocks, so the feature matrix still scales with total rows. [Hardware rules][hw] · [Histograms][hist] · [Batch prediction][batch] |
| 26, 31, 32, 35–37, 42 | Configuration, data loading, validation, persistence and integration | The repo includes constants, ndarray inputs, optional validation and logging, PyO3/NumPy bindings, sklearn-style wrappers and serialization tests. The CSV loading example needs caller-provided code. The Cargo manifest declares external dependencies. The estimator takes a supplied validation set, and the documented stratified splits belong to benchmark preparation. [Manifest][cargo] · [Bindings][bindings] · [Serialization tests][sertest] |
| 38–41 | Documented benchmarks, drift tests and runnable examples | The project publishes benchmark reports and guides. Its checked-in Rust benchmark generates synthetic binary data and measures throughput. The guide’s CSVs, `prepare_data.py`, `run_single_benchmark.py`, `drift_comparison_all.py` and `src/bin/benchmark.rs` are absent at this revision. [Reproduction guide][repro] · [Actual benchmark][bench] |

## How to interpret the reported gains

* **Classification and drift.** On Credit Card, the README reports clean PR-AUC of 0.878 for PKBoost, 0.745 for XGBoost and 0.793 for LightGBM. The headline 1.8%, 31.8% and 42.5% declines are relative drops after simulated feature corruption. The README says adaptive retraining never triggered in that run. A separate 16-scenario report gives a 2.82% average decline for PKBoost. Dry Bean reports 92.54% clean and 92.14% corrupted accuracy for PKBoost against 92.25% and 91.41% for XGBoost. The advertised 2–17× factor compares relative degradation, not a multiplication of accuracy. Comparisons used package defaults and do not isolate any individual feature. [README benchmarks][headline] · [Scenario report][scenarios] · [Dry Bean][beans]
* **Entropy and specialists.** A synthetic multiclass ablation reports accuracy of 46.3% without the entropy term and 47.3% at its best listed weight, against 48.6% for XGBoost. The separate 165× adaptation figure compares 92.7 seconds for full retraining with 0.56 seconds for selective updates. Another upstream report measures about 74% PR-AUC for HAB and 82% for one model. These answer different questions. [Entropy ablation][ablation] · [Adaptation timings][habtime] · [HAB accuracy][habresult]
* **Training and inference speed.** The v2.2.1 notes report speedups over the previous PKBoost implementation on four Windows CPU threads. Reported fit speedups are 5.27× at 20k rows, 3.23× at 200k and 1.98× at 1M. Reported inference speedup is 1.86×. The changes include level-parallel tree growth, byte bins written directly in column-major order, dual accumulators, interleaved sibling subtraction, reusable classifier derivative buffers, ordered Bernoulli row sampling, node totals calculated once and a static entropy lookup. The regressor still obtains a fresh gradient/Hessian pair in its loop. These are combined measurements and give no per-change speedup for OmniGBDT. [Release measurements][release] · [Tree construction][growth] · [Regression loop][sampling]

The upstream results also show losses. On Breast Cancer, PKBoost reports PR-AUC of 0.979, compared with 0.992 for XGBoost. Its feature catalogue reports a Credit Card fit taking 92.7 seconds, compared with 12.0 for XGBoost and 11.2 for LightGBM. Those are separate comparisons from the version-to-version timings. [Standard datasets][headline] · [Speed comparison][habtime]

The Cargo and Python manifests identify version 2.2.0, while the release notes say 2.2.1. The README offers a choice of Apache 2.0 or GPL-3.0-or-later and the Cargo manifest lists GPL-3.0. Both license texts are in the pinned repository. This inventory cites the ideas and source paths without reproducing implementation code. [README licensing][license] · [Cargo manifest][cargo] · [Python manifest][pyproject]

[repo]: https://github.com/Pushp-Kharat1/PkBoost/tree/f09100d680bfcd52895904047185cfe5d0532d4c
[features]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/FEATURES.md
[gain]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/tree.rs#L859-L935
[deriv]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/loss.rs#L24-L58
[bins]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/histogram_builder.rs#L24-L99
[tails]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/histogram_builder.rs#L150-L175
[builder]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/model.rs#L133-L188
[tuner]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/auto_tuner.rs#L54-L169
[sampling]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/regression.rs#L227-L286
[stop]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/model.rs#L525-L592
[metrics]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/metrics.rs#L35-L96
[threshold]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/pkboost_sklearn/sklearn_interface.py#L240-L244
[ovr]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/multiclass.rs#L39-L119
[reg]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/regression.rs#L14-L180
[uncert]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/regression.rs#L392-L435
[hab]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/partitioned_classifier.rs#L403-L527
[adapt]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/living_booster.rs#L267-L530
[regdrift]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/living_regressor.rs#L338-L589
[metab]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/metabolism.rs
[hw]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/adaptive_parallel.rs#L15-L58
[hist]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/optimized_data.rs#L63-L101
[batch]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/model.rs#L659-L693
[cargo]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/Cargo.toml#L14-L116
[bindings]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/python_bindings.rs
[sertest]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/tests/test_sklearn_serialization.py
[repro]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/BENCHMARK_REPRODUCTION.md#L46-L202
[bench]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/benches/benchmark.rs#L17-L90
[headline]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/README.md#L196-L272
[scenarios]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/DRIFT_BENCHMARK_REPORT.md#L12-L65
[beans]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/DRYBEAN_DRIFT_RESULTS.md#L10-L45
[ablation]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/SHANNON_ANALYSIS.md#L7-L59
[habtime]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/FEATURES.md#L423-L449
[habresult]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/docs/HAB_CONCLUSION.md#L3-L19
[release]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/RELEASE_NOTES.md#L9-L63
[growth]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/src/tree.rs#L288-L333
[license]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/README.md#L418-L425
[pyproject]: https://github.com/Pushp-Kharat1/PkBoost/blob/f09100d680bfcd52895904047185cfe5d0532d4c/pyproject.toml#L5-L12
