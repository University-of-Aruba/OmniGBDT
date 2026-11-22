Regression benchmarks
=====================

OmniGBDT 0.7.0 and XGBoost 3.4.1 each fit one vector-leaf tree per boosting
round. The comparison covers three public datasets and two synthetic examples.
The first two tables report **OmniGBDT / XGBoost** in every cell. Lower values are
better. Compare error within a row because the targets have different units
across datasets.

Mean squared error (MSE)
------------------------

RMSE measures test error across all rows and targets in their original units.

.. csv-table::
   :header: "Dataset and holdout", "Test RMSE", "Fit (ms)", "Predict (µs)"

   "Stock, random rows", 0.11974 / 0.11974, 3.22 / 11.22, 94.7 / 82.1
   "Stock, held-out portfolios", 0.12408 / 0.12765, 2.58 / 10.16, 68.0 / 75.4
   "Energy, random rows", 1.09565 / 1.09559, 9.13 / 20.16, 755.9 / 243.0
   "Energy, held-out geometries", 7.66306 / 7.09766, 1.97 / 2.08, 54.7 / 38.6
   "SCM20D, held-out games", 198.14 / 199.31, 742.64 / 1525.06, 2256.8 / 1131.2
   "Synthetic, correlated targets", 0.52355 / 0.51426, 8.86 / 32.12, 323.4 / 149.9
   "Synthetic, independent targets", 0.28855 / 0.28855, 32.89 / 41.50, 383.9 / 421.1

Coupled quadratic loss
----------------------

The quadratic score Q gives equal weight to the average squared error of the
targets and the squared error of their average. For example, an
overprediction on one target and an underprediction on another can cancel
when predicting their average. For a row of target errors ``e``, the score is

.. code-block:: text

   Q = sqrt(mean_over_rows(0.5 * mean(e**2) + 0.5 * mean(e)**2))

The quadratic models train and validate on this loss. For ``d`` targets,
OmniGBDT sets ``output_metric = (eye(d) + ones((d, d)) / d) / (2 * d)``.
XGBoost trains on ``Y @ L``, where ``L = cholesky(d * output_metric)``.
RMSE in those coordinates equals Q. XGBoost predictions are transformed
back to the original units for scoring. The RMSE column uses the same test
metric as the MSE table, so the two training losses can be compared within
each dataset. The Q column measures the error under the quadratic loss.

.. csv-table::
   :header: "Dataset and holdout", "Test RMSE", "Test Q", "Fit (ms)", "Predict (µs)"

   "Stock, random rows", 0.11989 / 0.11989, 0.10019 / 0.10019, 4.86 / 16.17, 109.5 / 117.9
   "Stock, held-out portfolios", 0.12447 / 0.12793, 0.10398 / 0.10619, 3.22 / 9.62, 67.5 / 78.4
   "Energy, random rows", 1.10479 / 1.10479, 0.94359 / 0.94357, 10.44 / 20.14, 706.6 / 244.4
   "Energy, held-out geometries", 10.06504 / 7.09678, 10.04177 / 7.06082, 1.16 / 2.50, 8.2 / 50.7
   "SCM20D, held-out games", 197.11 / 197.87, 160.82 / 161.73, 1385.82 / 1267.50, 1893.4 / 945.0
   "Synthetic, correlated targets", 0.50711 / 0.51263, 0.50464 / 0.51028, 11.75 / 32.21, 307.5 / 152.0
   "Synthetic, independent targets", 0.28853 / 0.28853, 0.22336 / 0.22336, 46.95 / 42.04, 350.1 / 511.4

OmniGBDT has lower error on held-out portfolios and SCM20D under both
objectives. Two of the ten portfolio splits account for 90% of its ordinary
RMSE advantage. XGBoost has lower error on held-out building geometries,
where OmniGBDT's quadratic fit also exceeds the training-mean baseline
(Q = 9.50). On the correlated synthetic targets, XGBoost has the lower RMSE
with MSE training and OmniGBDT has the lower Q with the coupled loss.
The training mean beats both libraries on the independent random targets.

Datasets and holdouts
---------------------

- **UCI Stock Portfolio Performance.** Six portfolio weights predict six
  normalized performance measures. Each of the 63 portfolios appears in five
  period summaries. Splitting rows gives 189/63/63 training, validation and
  test observations. Holding out entire portfolios gives 37/13/13 portfolios
  (185/65/65 rows) and measures performance on new portfolios.
  `Stock Portfolio Performance dataset <https://archive.ics.uci.edu/dataset/390/stock+portfolio+performance>`_.
- **UCI Energy Efficiency.** Eight building inputs predict heating and cooling
  loads. The 768 simulations cover 12 building geometries with 64 configurations
  each. Splitting rows gives 460/154/154 observations. Holding out entire
  geometries gives 7/2/3 geometries (448/128/192 rows) and tests generalization
  to new building shapes.
  `Energy Efficiency dataset <https://archive.ics.uci.edu/dataset/242/energy+efficiency>`_.
- **Mulan SCM20D.** The task forecasts 16 simulated product prices 20 days
  ahead from 61 inputs. Game boundaries are inferred from resets in the day
  counter. The first 32 games of the published training file form the training
  set and the next eight form the validation set. The published test file
  contains eight held-out games, giving 5,971/1,492/1,503 rows.
  `Mulan dataset directory <https://mulan.sourceforge.net/datasets-mtr.html>`_.
- **Correlated targets.** The data generator from the README example produces six features and two
  targets that share a signal. Each dataset has 240/80/80 rows.
- **Independent targets.** The example in ``examples/test.py`` generates ten
  features and five independent random targets. Each dataset has 10,000 rows
  in each of the three partitions. The training-mean baseline provides a
  reference for this noise-only example.

For each UCI split rule, the tables average test errors over ten random
holdouts (seeds 0 through 9). Test sets can overlap between repetitions.
The synthetic rows average three independently generated datasets (seeds
0 through 2). SCM20D uses one fixed split.

Training and timing
-------------------

Both libraries use one CPU thread, depth 4, 128 histogram bins, a learning
rate of 0.05, all training rows, at least 20 rows per leaf and unit L2
regularization. The cap is 200 rounds, with 15 rounds of validation patience.
The independent-target example uses depth 5, a learning rate of 0.2 and a
20-round cap. XGBoost uses ``tree_method="hist"``,
``multi_strategy="multi_output_tree"``, ``min_child_weight=20`` and
``gamma=2 * n_targets * 0.001`` to match OmniGBDT's split threshold.
`XGBoost vector-leaf documentation <https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html#training-with-vector-leaf>`_.

Each library and objective runs in a separate process. Times are warm medians
of 11 measurements on an Apple M2 (macOS 26.6.2, Python 3.13.9, NumPy 2.5.5)
using the seed-0 split or the fixed SCM20D split. Fit time includes data
preparation for the model, training, validation
and any target transformation. Each prediction measurement averages ten calls
on the full test batch already in memory and includes returning predictions
in the original units. With MSE training, OmniGBDT and XGBoost retained
16 and 1 rounds on the timed geometry split and 79 and 93 rounds on SCM20D.
These differences also affect prediction time. The
:download:`full results <data/regression-benchmarks.json>` contain per-split
scores, baseline errors, retained rounds and individual timings.

Reproduce the comparison
------------------------

From the OmniGBDT 0.7.0 source root, build the package with an OpenMP-capable
C++ toolchain (see :doc:`install`). The benchmark checks the UCI CSVs and Mulan
archives against their recorded SHA-256 digests and generates the synthetic
datasets locally.

.. code-block:: bash

   python3 -m venv .benchmark-venv
   .benchmark-venv/bin/python3 -m pip install . "xgboost==3.4.1" "numpy==2.5.5"
   curl -fL https://archive.ics.uci.edu/static/public/390/data.csv -o stock.csv
   curl -fL https://archive.ics.uci.edu/static/public/242/data.csv -o energy.csv
   scm_url='https://downloads.sourceforge.net/project/mulan/datasets/multi-target%20regression%20datasets'
   curl -fL "$scm_url/scm20d-train.zip" -o scm20d-train.zip
   curl -fL "$scm_url/scm20d-test.zip" -o scm20d-test.zip
   .benchmark-venv/bin/python3 examples/regression_benchmarks.py stock.csv results.json \
       --energy-csv energy.csv --scm-dir .

The optional :download:`quadratic timing chart <_static/regression-benchmarks.png>`
covers the two stock splits and the two synthetic examples. To regenerate
that chart from a run, use the plotting script.

.. code-block:: bash

   .benchmark-venv/bin/python3 -m pip install matplotlib
   .benchmark-venv/bin/python3 examples/plot_regression_benchmarks.py results.json regression-benchmarks.png

Nested group cross-validation
-----------------------------

This comparison holds out whole portfolios, building geometries and simulation
games. Each row is predicted once per outer repeat. The table averages the
out-of-fold scores over repeats. It reports RMSE for MSE fits and Q for
coupled quadratic fits. Lower is better within each row.

.. csv-table::
   :header: "Dataset", "Loss and score", "OmniGBDT full", "OmniGBDT projected", "XGBoost 3.4.1"

   Stock, MSE (RMSE), 0.11839, 0.11835, 0.12145
   Stock, Coupled (Q), 0.09883, 0.09855, 0.10060
   Energy, MSE (RMSE), 8.209, 8.462, 7.681
   Energy, Coupled (Q), 10.757, 10.641, 7.627
   SCM20D, MSE (RMSE), 202.789, 198.765, 203.086
   SCM20D, Coupled (Q), 175.242, 173.281, 175.240

Stock uses three five-fold repeats across 63 portfolios. Energy uses three
four-fold repeats across 12 building geometries. SCM20D uses one five-fold
repeat across the 40 games in the publisher's training file. All methods use
identical group assignments. Within each outer training set, three grouped
inner folds evaluate depths 2 and 4 with up to 200 rounds and 15 rounds of
early-stopping patience. The lowest pooled inner validation error selects
the depth. Each model then refits on all outer training groups for the median
of that depth's three best round counts.

The projected fits use seed 17, one direction for Energy and two for Stock
and SCM20D. On SCM20D, projection improved on full OmniGBDT in 30 of 40 games
under MSE and 27 under the coupled loss. On Stock, the MSE difference was
0.036%, with lower error on 30 of 63 portfolios. XGBoost had the lowest
error on Energy under both losses. Repeated folds reuse the same groups.
The README reports ordinary RMSE for both losses so the fits can be compared
in the same units. The :download:`grouped results <data/grouped-projection-benchmarks.json>`
include both scores for each repeat.

The Stock folds measure performance on new portfolio allocations over the
same historical periods. Forecasting a future period requires a chronological
holdout. On Energy, full OmniGBDT's MSE-trained RMSE ranges from 6.88 to 9.07
across the three partitions of the same 12 geometries. On SCM20D, only the
40 training games enter cross-validation. The eight publisher test games
were scored in the earlier fixed-split comparison. All projected results
use one projection seed, so they do not measure variability across seeds.

Grouped benchmark timings
~~~~~~~~~~~~~~~~~~~~~~~~~

Each cell gives fit time in **milliseconds** and full test-batch prediction
time in **microseconds**. These are warm medians of five single-threaded runs
on an Apple M2 for the first outer fold. Each model used the depth and number
of trees selected within its training fold.

.. csv-table::
   :header: "Dataset (test rows)", "Training loss", "OmniGBDT full", "OmniGBDT projected", "XGBoost 3.4.1"

   Stock (65), MSE, 2.48 / 92.0, 4.81 / 137.2, 6.81 / 92.7
   Stock (65), Coupled, 4.58 / 94.7, 5.22 / 124.8, 6.18 / 89.1
   Energy (192), MSE, 0.96 / 41.4, 0.95 / 37.9, 4.93 / 142.4
   Energy (192), Coupled, 0.62 / 21.2, 0.85 / 29.5, 5.19 / 133.8
   "SCM20D (1,498)", MSE, 572.56 / 1664.6, 203.01 / 1931.1, 781.39 / 1180.7
   "SCM20D (1,498)", Coupled, 680.13 / 1245.6, 335.98 / 2869.8, 705.31 / 1263.6

Reproduce the grouped comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same downloaded data and benchmark environment as above. Each command
writes its group assignments, selected depths and rounds, scores, and
out-of-fold predictions to the output directory.

.. code-block:: bash

   for loss in mse coupled; do
       for method in full projected xgboost; do
           .benchmark-venv/bin/python3 -I examples/grouped_projection_benchmark.py \
               --stock stock.csv --energy energy.csv --scm . --out grouped-results \
               --method "$method" --loss "$loss"
       done
   done

Per-tree input feature sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OmniGBDT 0.8.0 can repeat this comparison with
``--colsample-bytree 0.8``. The fraction was fixed before evaluation.
Each tree considers four of the six Stock inputs, six of the eight Energy
inputs, or 48 of the 61 SCM20D inputs. All training rows are used.

.. csv-table::
   :header: "Dataset", "MSE, all inputs", "MSE, 80%", "Coupled, all inputs", "Coupled, 80%"

   Stock, 0.11839, 0.11898, 0.11871, 0.11893
   Energy, 8.209, 7.531, 10.787, 8.635
   SCM20D, 202.789, 201.947, 203.130, 202.286

These are held-out RMSEs for OmniGBDT with exact split scoring across outputs,
on the same grouped folds described above. Sampling reduced RMSE on nine of
the twelve Energy geometries under MSE and eleven under the coupled loss.
The Stock average rose slightly under both losses. On SCM20D the gains were
small, and adding feature sampling to output projection slightly increased
RMSE under both losses. Each configuration used a single model seed.

For a fixed budget of 100 depth-four MSE trees on the first SCM20D fold,
sampling reduced median fit time from 792 to 621 ms. Fit includes binning.
Prediction for all 1,498 held-out rows took 2.67 and 2.71 ms respectively.
These are seven warm, alternating pairs on one Apple M2 thread.

The :download:`PKBoost source inventory <research/pkboost.md>` explains
the choice of feature sampling and the scope of PKBoost's published results.
