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

More details:

- [ml/README.md](ml/README.md)
- [ml/pai/README.md](ml/pai/README.md)
- [ml/clumping_cnn/README.md](ml/clumping_cnn/README.md)

## Credit / citation

Please credit and cite the original HemiPy authors as described in [OldREADME.md](OldREADME.md) and [CITATION.cff](CITATION.cff).

## License

See [LICENSE.md](LICENSE.md).
