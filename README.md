# GSI-MLC-PA: Experimental Code

This repository contains the experimental code for research on multi-label
classification with partial abstention. It provides the GSI-MLC-PA method,
reference baselines, dataset loaders, cross-validation procedures, evaluation
metrics and saved predictions used in the experiments.

GSI-MLC-PA identifies independent labels through out-of-fold evaluation,
models dependent labels using residual correlations, and applies calibrated
probability thresholds to accept or abstain from individual label decisions.
Each prediction is positive, negative or abstained.

The repository supports reproducing the benchmark and checking reported
metrics directly against saved out-of-sample predictions.

## Repository structure

| Path | Contents |
|---|---|
| [src/models/](src/models) | GSI-MLC-PA and reference estimators |
| [src/selection/](src/selection) | Label partitioning and selection |
| [src/calibration/](src/calibration) | Probability calibration |
| [src/data/](src/data) | Dataset loading and normalization |
| [src/evaluation/](src/evaluation) | Prediction metrics and evaluation contracts |
| [configs/](configs) | Experimental settings and dataset manifests |
| [scripts/](scripts) | Experiment, validation and verification commands |
| [tests/](tests) | Regression tests |
| [data/](data) | Dataset metadata and local raw inputs |
| [results/](results) | Saved benchmark outputs |

## Environment

The experiments were run on Windows x64 with Python 3.13.14 and PyTorch
2.6.0+cu124. The MLP benchmark requires CUDA; the reference GPU is an NVIDIA
GeForce RTX 4060 Laptop GPU. Logistic Regression and calibrated SVM can run
on the CPU. Git is required to record source provenance.

From the repository root in PowerShell, create the pinned environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_final_env.ps1
```

If needed, select the Python executable explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_final_env.ps1 -Python "C:\path\to\Python313\python.exe"
```

The installer creates `.venv-final` and checks dependency consistency with
`pip check`. Runtime and test dependencies are pinned in
[requirements-final-env.lock.txt](requirements-final-env.lock.txt), which
includes [requirements-final.lock.txt](requirements-final.lock.txt).
The commands below use the environment's interpreter directly.

## Datasets

The benchmark uses ten multi-label datasets:

| Dataset | Samples | Features | Labels |
|---|---:|---:|---:|
| Emotions | 593 | 72 | 6 |
| Scene | 2,407 | 294 | 6 |
| Yeast | 2,417 | 103 | 14 |
| CHD49 | 555 | 49 | 6 |
| Music | 592 | 71 | 6 |
| GpositivePseAAC | 519 | 440 | 4 |
| Genbase | 662 | 1,185 | 27 |
| VirusPseAAC | 207 | 440 | 6 |
| PlantPseAAC | 978 | 440 | 12 |
| HumanPseAAC | 3,106 | 440 | 14 |

Raw inputs are stored locally under `data/` and excluded from Git. Dataset
sources, file layout, citation requirements and normalization rules are
documented in [data/README.md](data/README.md). The compact collection summary
is [data/dataset_manifest.csv](data/dataset_manifest.csv).

Validate the inputs before running experiments:

```powershell
.\.venv-final\Scripts\python.exe scripts/validate_datasets.py
```

Validation checks SHA-256 hashes, file sizes, matrix dimensions, finite
features and binary labels. A subset can be checked with `--datasets`:

```powershell
.\.venv-final\Scripts\python.exe scripts/validate_datasets.py --datasets emotions scene
```

Genbase's unique `protein` identifier is excluded. PseAAC train/test files are
joined before generating cross-validation folds. Exact duplicate rows are
retained: Scene has 7, Genbase has 455 after identifier removal, and
VirusPseAAC has 1. The folds do not group duplicates. Music and Emotions are
related audio benchmark variants. These properties should be considered
when interpreting generalization and statistical comparisons.

## Experimental protocol

The final benchmark evaluates four models across ten datasets, three base
learners and five outer folds, for a total of **600 fitted models**: Binary
Relevance (BR), Classifier Chains (CC), MLC-PA and GSI-MLC-PA v6.3.3. Exact
model settings are captured with each experiment.

| Setting | Protocol |
|---|---|
| Base learners | Logistic Regression, calibrated SVM, PyTorch MLP |
| Outer evaluation | Five shuffled multilabel-stratified folds, split seed 42 |
| Model seeds | `42 + fold - 1`, shared across methods |
| Preprocessing | MaxAbsScaler fitted on the outer training fold |
| Internal evaluation | Five folds for GSI label selection and calibration |
| Abstention cost | `0.30`, linear penalty |
| Abstention encoding | `-1` |
| Training coverage target | `0.70` on out-of-fold training predictions |
| Aggregation | Fold means within each dataset, then equal dataset weights |

Calibration, label selection and threshold estimation use training data only.
The training coverage target is not a lower bound on test coverage.

All methods share the same binary estimator settings. MLP uses a per-label
64-unit network; SVM uses nested sigmoid calibration. Base estimator
hyperparameters are specified in [configs/experiment.json](configs/experiment.json).
Execution uses one Torch/BLAS thread, deterministic Torch operations and
disabled TF32.

## Run experiments

Run the four final models on Emotions with Logistic Regression across five
folds (**20 fits**):

```powershell
.\.venv-final\Scripts\python.exe scripts/run_experiments.py --datasets emotions --base-learners logistic
```

Run the complete benchmark:

```powershell
.\.venv-final\Scripts\python.exe scripts/run_experiments.py
```

New runs are written to `results/reproductions/run_<hash>/`. The command prints
the actual run directory. Dataset, learner and fold subsets can be selected:

```powershell
.\.venv-final\Scripts\python.exe scripts/run_experiments.py --datasets viruspseaac --base-learners logistic --folds 1
```

Use `--output-root` to choose another output directory, or `--config` to supply
an experimental configuration. Commit source and configuration changes before
running experiments so the recorded Git commit identifies the tested code.

The run identity includes source, configuration, dataset manifest, environment
and selected datasets, learners and folds. Repeating the same command resumes
completed fits after checking their prediction hashes. Changes to the run
identity create a separate directory. A corrupted checkpoint stops resume.

## Tests and smoke runs

Run the regression suite:

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py --stage tests
```

Run all four models on one VirusPseAAC fold with Logistic Regression, check
checkpoint resume and compare predictions against the saved reference:

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py --stage smoke
```

Add `--gpu-smoke` to exercise all three learners, including CUDA MLP
(**12 fits**). Smoke outputs are stored under `scratch/final_checks/`.
[pytest-final.ini](pytest-final.ini) selects the experiment regression suite,
covering data loading, metrics, calibration, label selection, checkpoints
and packaging.

## Saved results and verification

The saved benchmark is available under [results/](results). Each run includes
predictions, fold assignments, fitted thresholds, label partitions and a
manifest recording the source commit, configuration and environment.

Verify the saved benchmark without retraining:

```powershell
.\.venv-final\Scripts\python.exe scripts/verify_results.py
```

To verify a reproduced run, provide its actual directory:

```powershell
.\.venv-final\Scripts\python.exe scripts/verify_results.py "results/reproductions/run_<hash>"
```

Verification checks source consistency, artifact checksums, fold indices,
prediction validity and metrics recomputed from the saved predictions.

| Artifact | Contents |
|---|---|
| `detailed_folds.csv` | One metric row per dataset, learner, model and fold |
| `dataset_summary.csv` | Fold means and sample standard deviations (`ddof=1`) |
| `overall_summary.csv` | Equally weighted dataset means for each learner/model pair |
| `per_label_folds.csv` | Label support, confusion counts, coverage and F1 |
| `thresholds_audit.csv` | Per-label acceptance and rejection thresholds |
| `partitions_audit.csv` | Independent/dependent label membership |
| `predictions/*.npz` | Test indices, truth, probabilities, predictions and scaler state |
| `cells/*.json` | Per-fit checkpoints, metrics and recorded warnings |
| `fold_indices.json` | Original training/test row indices |
| `run_manifest.json`, `*_snapshot.json` | Source provenance and experiment settings |
| `checksums.sha256` | Artifact integrity inventory |

`Full_Macro_F1` evaluates predictions at every sample-label position.
`Selective_Macro_F1` computes F1 on decided positions for each label and then
averages across labels; a label with no decided positions receives zero.
`Coverage` is the fraction of decided sample-label positions. Interpret
selective F1 together with coverage, and use dataset-level means for
comparisons across datasets.
