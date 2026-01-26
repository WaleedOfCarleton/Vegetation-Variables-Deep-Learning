# NewREADME — Changes from the original HemiPy project

This repository is based on the original **HemiPy** project by **Luke A. Brown**.
The core `hemipy` package (canopy variable estimation from hemispherical photographs) is the original library; this fork focuses on running HemiPy on a simulated image dataset, validating outputs against provided “truth” values, and building ML baselines (including a CNN).

For the original project documentation, see [README.md](README.md).

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

### 5) Started ML baselines (including a CNN)

I am currently working toward training models to predict biophysical variables from the images.

- Simple baseline features + metrics: [image_baseline/image_baseline_features.py](image_baseline/image_baseline_features.py)
  - Output: [image_baseline/image_baseline_metrics.csv](image_baseline/image_baseline_metrics.csv)
- CNN baseline training: [cnn_baseline/train_cnn_pai.py](cnn_baseline/train_cnn_pai.py)
  - Outputs in [cnn_baseline/](cnn_baseline/):
    - `cnn_baseline_metrics.csv`
    - `cnn_per_image_predictions.csv`
    - `cnn_per_site_predictions.csv`

Current status: the CNN baseline is **preliminary and not performing that well yet**; it is included as a starting point and is still being improved.

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

## Credit / citation

Please credit and cite the original HemiPy authors as described in [README.md](README.md) and [CITATION.cff](CITATION.cff).

## License

See [LICENSE.md](LICENSE.md).
