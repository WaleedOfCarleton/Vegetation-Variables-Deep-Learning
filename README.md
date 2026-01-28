# HemiPy fork — Simulation + ML workflows

This repository is based on the original **HemiPy** project by **Luke A. Brown**.
The core `hemipy` package (canopy variable estimation from hemispherical photographs) is the original library; this fork focuses on running HemiPy on a simulated image dataset, validating outputs against provided “truth” values, and building ML baselines (including a CNN).

For the original upstream project documentation, see [OldREADME.md](OldREADME.md).

## Project owner / timeline

- **Author:** Waleed Abu-Osbeh
- **Work started:** Jan 16, 2026
- **Status:** Ongoing (actively in progress)

## Summary of what I added/changed

### 1) Added a simulation workflow (Sylvain Leblanc dataset)

I adapted/added scripts so the code can be run in batch on the simulation image dataset located under:

- [Simulations/](Simulations/)

The simulation runner scripts are:

- [Simulations/run_simulations.py](Simulations/run_simulations.py): batch-processes the simulation plots and writes outputs
- [Simulations/run_example.py](Simulations/run_example.py): smaller example-style runner

Outputs produced:

- [Simulations/simulations_output.csv](Simulations/simulations_output.csv): predicted biophysical variables (e.g., PAI-related outputs)
- [Simulations/simulations_errors.csv](Simulations/simulations_errors.csv): errors encountered while processing cases/plots

### 2) Joined HemiPy predictions to “truth” values (Leblanc Excel)

To validate the HemiPy predictions against the provided reference (“true”) values, I added a join step:

- [truth_join/join_truth_to_hemipy.py](truth_join/join_truth_to_hemipy.py)

Inputs:

- [Simulations/simulations_output.csv](Simulations/simulations_output.csv)
- [Inputs Cases LAI.xlsx](Inputs%20Cases%20LAI.xlsx) (Excel sheet containing the provided truth values)

Output:

- [truth_join/truth_joined_to_hemipy.csv](truth_join/truth_joined_to_hemipy.csv)

This produces a single table containing both:

- HemiPy-predicted values from the simulation images
- Truth/reference values from the Excel file

### 3) Evaluated prediction quality (metrics + residuals)

To quantify agreement between predictions and truth, I added an evaluation step:

- [estimations_eval/evaluate_estimations.py](estimations_eval/evaluate_estimations.py)

Outputs:

- [estimations_eval/estimation_metrics_summary.csv](estimations_eval/estimation_metrics_summary.csv)
- [estimations_eval/estimation_residuals.csv](estimations_eval/estimation_residuals.csv)

### 4) Prepared an image-level ML dataset index

To support ML experiments, I added a script that converts the simulation folder structure into an “image dataset index” CSV:

- [dataset_index/build_image_dataset_index.py](dataset_index/build_image_dataset_index.py)

Output:

- [dataset_index/image_dataset_index.csv](dataset_index/image_dataset_index.csv)

This index is used by the baseline models (feature baseline and CNN baseline) to locate images and their associated labels.

Notes:

- Hinge-region truth columns in the joined truth table may appear as strings like `1.66+/-0.06`; the dataset index builder parses these into numeric values so hinge CNN training has valid labels.
- Hinge-region truth is expected to be present for **ERECT** and **PLANO**; **RND** typically has missing hinge truth in the provided dataset.

### 5) Started ML baselines (including a CNN)

I am currently working toward training models to predict biophysical variables from the images.

- Simple baseline features + metrics: [image_baseline/image_baseline_features.py](image_baseline/image_baseline_features.py)
  - Output: [image_baseline/image_baseline_metrics.csv](image_baseline/image_baseline_metrics.csv)
- CNN baseline training: [cnn_baseline/train_cnn_pai.py](cnn_baseline/train_cnn_pai.py)
  - Outputs are written as CSVs and plots; see **Where to find CNN results** below.

Current status: the CNN baseline is actively developed and produces strong correlation with PAI (with optional post-hoc calibration to remove systematic bias).

Note: the CNN baseline requires PyTorch/torchvision and Pillow.

### 6) Added a pipeline runner

To make the workflow easier to reproduce, I added:

- [run_pipeline.py](run_pipeline.py)

It runs the steps in order (and optionally includes the CNN step).

## Quick start (this fork)

From the repo root, run (using your Python executable):

- `python run_pipeline.py` (runs truth join → dataset index → image baseline → evaluation)
- `python run_pipeline.py --include-cnn` (also attempts CNN training)

## Note on generated CSV outputs

Some of the CSV files in this repository (e.g., simulation outputs, joined truth tables, and baseline prediction outputs) are generated artifacts that can take a while to reproduce.
They are intentionally kept under version control during active development so results can be inspected and shared without rerunning long jobs.

## Where to find CNN results

The CNN baseline produces multiple output files (progress, metrics, per-image predictions, per-site/case predictions, and plots). To keep the root [cnn_baseline/](cnn_baseline/) folder clean, runs are archived under:

- [cnn_baseline/archive/](cnn_baseline/archive/)

The most recent run paths are tracked in:

- [cnn_baseline/latest_run.txt](cnn_baseline/latest_run.txt)

Training progress is written to a CSV (intended for live monitoring). Runs are disambiguated by a `run_id` column so multiple runs can share a single progress file.

### Calibration outputs

We support post-hoc linear calibration to reduce bias:

- Script: [cnn_baseline/calibrate_cnn_outputs.py](cnn_baseline/calibrate_cnn_outputs.py)

Calibration options:

- **Leakage-safe CV reporting:** use the `cal_oof_foldwise` outputs (fits calibration on other folds, then applies to the held-out fold).
- **Best corrected predictions for deployment:** use the `cal_global_casefit` outputs (fits calibration on all OOF points; best RMSE/MAE but optimistic as a CV score).

Plots are generated via:

- [cnn_baseline/visualize_cnn_results.py](cnn_baseline/visualize_cnn_results.py)

## Hinge-region PAIe + clumping (cnn_hinge)

The hinge region is the zenith angle neighborhood around ~57.3° (often referenced as 57.5° in the literature). In this repo we treat hinge-region **PAIe** as **orientation-dependent**, and compute hinge-region clumping as:

$$\Omega_{hinge} = \frac{PAIe_{hinge}}{PAI}$$

### 1) Ensure the dataset index contains hinge labels

The image dataset index is built from the simulation folder structure and joined to truth values. For hinge training you need the numeric column:

- `truth_PAIe_hinge`

Build (or rebuild) the index:

```bash
python dataset_index/build_image_dataset_index.py --out dataset_index/image_dataset_index.csv
```

### 2) Train a hinge PAIe CNN

Training script:

- [cnn_hinge/train_cnn_paie_hinge.py](cnn_hinge/train_cnn_paie_hinge.py)

It produces metrics/per-image/per-site CSVs and writes a live-updated progress CSV (by default):

- `cnn_hinge/cnn_hinge_training_progress.csv`

Progress CSV notes:

- Each row includes `run_id` so you can append multiple runs into a single file.
- Batch-level rows are optional (see `--log-batches`) and are off by default.

Archived runs:

- Hinge CNN runs are archived under [cnn_hinge/archive/](cnn_hinge/archive/) (raw outputs, derived clumping outputs, plots, and a snapshot of logs/progress).
- The most recent archived run id is stored in [cnn_hinge/latest_run.txt](cnn_hinge/latest_run.txt).

Example (ERECT + PLANO only):

```bash
python cnn_hinge/train_cnn_paie_hinge.py \
  --index dataset_index/image_dataset_index.csv \
  --orientations ERECT PLANO \
  --epochs 25 --early-stop 6 --aug --aug-hflip --aug-color-jitter 0.1 \
  --nonnegative-head \
  --run-id 20260127_paie_hinge_ep25_aug_es
```

Batch-level progress logging is optional and **off by default** (to keep the CSV readable). If you want it:

```bash
python cnn_hinge/train_cnn_paie_hinge.py --log-batches ...
```

### 3) Monitor training progress

To summarize the progress CSV (latest + per-fold latest/best), use:

- [cnn_hinge/summarize_progress.py](cnn_hinge/summarize_progress.py)

```bash
python cnn_hinge/summarize_progress.py
```

By default, the summarizer will try to use the run id from [cnn_hinge/latest_run.txt](cnn_hinge/latest_run.txt).

If multiple runs share the same progress CSV, pass `--run-id` explicitly:

```bash
python cnn_hinge/summarize_progress.py --run-id 20260127_paie_hinge_ep25_aug_es
```

### 4) Compute clumping-hinge from PAI (baseline) + PAIe-hinge (hinge CNN)

Clumping-hinge is computed by combining per-site predictions:

- baseline PAI per-site predictions (from `cnn_baseline/`)
- hinge PAIe per-site predictions (from `cnn_hinge/`)

Script:

- [cnn_hinge/compute_clumping_hinge.py](cnn_hinge/compute_clumping_hinge.py)

### 5) Visualize hinge/clumping results

Plots are generated via:

- [cnn_hinge/visualize_cnn_hinge_results.py](cnn_hinge/visualize_cnn_hinge_results.py)

## Credit / citation

Please credit and cite the original HemiPy authors as described in [OldREADME.md](OldREADME.md) and [CITATION.cff](CITATION.cff).

## License

See [LICENSE.md](LICENSE.md).
