# HemiPy simulation + ML workflows

This repo is based on the original **HemiPy** project by **Luke A. Brown** and adds a reproducible workflow for:

- Running HemiPy on a hemispherical-photo simulation dataset
- Joining HemiPy estimates to the provided Excel “truth” values
- Computing evaluation metrics/residuals
- Building an image-level dataset index for deep learning
- Training CNN models to predict **PAI** and **clumping** from images, plus a small GUI for inference

For upstream HemiPy documentation, see [hemipy_core/OldREADME.md](hemipy_core/OldREADME.md).

## Repository layout

- [hemipy_core/](hemipy_core/) — the upstream HemiPy package (vendored)
- [Simulations/](Simulations/) — simulation dataset + runners
- [shared/](shared/) — truth join, dataset index, evaluation
- [ml/](ml/) — CNN training/inference (PAI + clumping)

## Setup

Requirements depend on what you run:

- **HemiPy pipeline**: numpy/scipy/scikit-image/imageio/rawpy/uncertainties + pandas
- **ML/CNN**: PyTorch + torchvision (+ tqdm recommended)

Suggested setup (from the repo root):

1) Create/activate a virtual environment.
2) Install core dependencies:

- `python -m pip install -e hemipy_core`
- `python -m pip install pandas`

3) If you plan to train CNNs:

- Install PyTorch/torchvision appropriate to your OS + CUDA: https://pytorch.org/get-started/locally/
- `python -m pip install tqdm pillow`

4) If you plan to use the web predictor:

- `python -m pip install fastapi uvicorn`

## How to use (end-to-end)

### 1) Run HemiPy on the simulation images

This processes the images and writes per-case/per-plot outputs:

- `python Simulations/run_simulations.py`

Outputs:

- [Simulations/simulations_output.csv](Simulations/simulations_output.csv)
- [Simulations/simulations_errors.csv](Simulations/simulations_errors.csv)

If you already have [Simulations/simulations_output.csv](Simulations/simulations_output.csv), you can skip this step.

### 2) Join to truth, build dataset index, and evaluate

Run the pipeline wrapper:

- `python run_pipeline.py`

This runs, in order:

- [shared/truth_join/join_truth_to_hemipy.py](shared/truth_join/join_truth_to_hemipy.py)
- [shared/dataset_index/build_image_dataset_index.py](shared/dataset_index/build_image_dataset_index.py)
- [shared/estimations_eval/evaluate_estimations.py](shared/estimations_eval/evaluate_estimations.py)

Expected inputs:

- [Inputs Cases LAI.xlsx](Inputs%20Cases%20LAI.xlsx) (truth values)
- [Simulations/simulations_output.csv](Simulations/simulations_output.csv) (HemiPy outputs)

Key outputs:

- [shared/truth_join/truth_joined_to_hemipy.csv](shared/truth_join/truth_joined_to_hemipy.csv)
- [shared/dataset_index/image_dataset_index.csv](shared/dataset_index/image_dataset_index.csv)
- [shared/estimations_eval/estimation_metrics_summary.csv](shared/estimations_eval/estimation_metrics_summary.csv)
- [shared/estimations_eval/estimation_residuals.csv](shared/estimations_eval/estimation_residuals.csv)

Notes:

- The truth join currently excludes the **RND** orientation for some truth columns (depending on what the Excel provides).
- The dataset index builder parses hinge-style strings like `1.66+/-0.06` into numeric values.

## ML/CNN (PAI + clumping)

ML tooling lives in [ml/](ml/).

- [ml/pai/](ml/pai/) — train + predict **PAI** + Tkinter GUI
- [ml/clumping_cnn/](ml/clumping_cnn/) — train + predict **clumping** using the Excel truth clumping
- [ml/runs/](ml/runs/) — training outputs (checkpoints, metrics, splits)

Common commands (from the repo root):

- Train PAI: `python ml/pai/train_cnn_pai.py --amp --num-workers 0`
- Train clumping (Excel truth): `python ml/clumping_cnn/train_cnn_clumping_truth.py --amp --num-workers 0`
- Run GUI (loads either a PAI or clumping checkpoint): `python ml/pai/gui_predict_pai.py`
- Sweep training cases vs. val MAE (PAI): `python ml/pai/run_sweep_train_cases.py --train-cases-list 15,25,35,45,55,65,70 --epochs 10 --patience 3 --best-metric val_mae_case --batch-size 16 --img-size 224 --lr 1e-4 --weight-decay 1e-4 --num-workers 0 --amp --pretrained`

### ML prerequisites (dataset index)

Most ML scripts expect an image-level index CSV at:

- `shared/dataset_index/image_dataset_index.csv`

You can build it via the end-to-end pipeline:

- `python run_pipeline.py`

Or rebuild just the index (after changing truth labels or simulation imagery):

- `python shared/dataset_index/build_image_dataset_index.py`

Notes:

- The large simulation imagery under `Simulations/` is intentionally not versioned (see `.gitignore`). You’ll need the dataset present locally for training.
- Some labels are orientation-specific (hinge-region labels), but `truth_PAI` is joined per-case so that RND/Sunny images can still be labeled.

### Train models

PAI training (writes checkpoints under `ml/runs/pai_cnn/<timestamp>/` by default):

- `python ml/pai/train_cnn_pai.py --pretrained --amp --num-workers 0`

Common useful options:

- Filter the data: `--orientation ERECT` and/or `--simulation-set "DHP - ERECT - 4000x4000"`
- Cross-validation: `--kfold 5 --fold 0`
- Reproducibility: `--seed 42`
- Output location: `--run-dir ml/runs/pai_cnn/my_run_name`

Clumping training (Excel truth clumping, writes under `ml/runs/clumping_cnn_truth/<timestamp>/` by default):

- `python ml/clumping_cnn/train_cnn_clumping_truth.py --pretrained --amp --num-workers 0`

Hinge clumping (optional):

- `python ml/clumping_cnn/train_cnn_clumping_hinge.py --pretrained --amp --num-workers 0`

### Run inference (CLI, GUI, or Web)

All inference methods require a trained checkpoint (`.pt`) such as `model_best.pt` produced by the training scripts.

#### 1) GUI (local desktop)

- `python ml/pai/gui_predict_pai.py`

Workflow:

1) Load a checkpoint (`.pt`) produced by training (PAI or clumping).
2) Choose a single image, or choose a folder to average predictions.

Tip: the folder option supports either a single case folder or a parent folder containing multiple `Case */` subfolders.

#### 2) Web predictor (browser)

Start the server from the repo root:

- `python -m uvicorn ml.pai.web_predict_pai:app --host 127.0.0.1 --port 8000 --log-level warning`

Then open:

- http://127.0.0.1:8000/

You can paste checkpoint paths or pick from the auto-suggest list (it scans under `ml/runs/`). Upload one or more images to get a table + downloadable CSV.

#### 3) CLI predictors (batch)

Predict PAI for one case using the dataset index:

- `python ml/pai/predict_case_pai.py --checkpoint <path_to_model_best.pt> --case "Case 001"`

Predict PAI across the entire dataset index (writes `pred_all_images.csv` and `pred_all_cases.csv` next to the checkpoint by default):

- `python ml/pai/predict_all_cases_pai.py --checkpoint <path_to_model_best.pt>`

Predict PAI (and optionally clumping) for *any folder of images* (recursively), without requiring the dataset index:

- `python ml/tools/predict_folder_pai_clumping.py --images-dir <folder> --pai-checkpoint <pai_model.pt> --clumping-checkpoint <clumping_model.pt>`

If you don’t have a clumping model, omit `--clumping-checkpoint` and the output CSV will leave clumping empty.

More details:

- [ml/README.md](ml/README.md)
- [ml/pai/README.md](ml/pai/README.md)
- [ml/clumping_cnn/README.md](ml/clumping_cnn/README.md)

## Credit / citation

Please credit and cite the original HemiPy authors as described in [hemipy_core/OldREADME.md](hemipy_core/OldREADME.md) and [CITATION.cff](CITATION.cff).

## License

See [LICENSE.md](LICENSE.md).
