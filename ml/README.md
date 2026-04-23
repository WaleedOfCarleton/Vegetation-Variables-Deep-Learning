# ML

This folder contains the deep learning / ML tooling for predicting **PAI** and **clumping** from hemispherical images.

## Layout

- `ml/pai/`: PAI CNN training + inference (ResNet18 regression) + GUI + web app
- `ml/clumping_cnn/`: clumping CNN training + inference (Excel truth clumping and optional hinge clumping)
- `ml/tools/`: utility scripts (plots, folder predictors, baseline comparisons)
- `ml/runs/`: training outputs (checkpoints, metrics, splits)

## Prerequisites

Install PyTorch/torchvision appropriate to your OS + CUDA:

- https://pytorch.org/get-started/locally/

Recommended extras:

- `python -m pip install tqdm pillow`

For the web predictor:

- `python -m pip install fastapi uvicorn`

### Dataset index

Most training and “predict from index” scripts use:

- `shared/dataset_index/image_dataset_index.csv`

Build it via the end-to-end pipeline:

- `python run_pipeline.py`

Or rebuild just the index:

- `python shared/dataset_index/build_image_dataset_index.py`

## Train

PAI model (outputs under `ml/runs/pai_cnn/<timestamp>/` by default):

- `python ml/pai/train_cnn_pai.py --pretrained --amp --num-workers 0`

Useful flags:

- `--orientation ERECT` and/or `--simulation-set "DHP - ERECT - 4000x4000"`
- `--kfold 5 --fold 0`
- `--best-metric val_mae_case` (saves `model_best.pt` based on that metric)
- `--run-dir ml/runs/pai_cnn/my_run_name`

Clumping model (Excel truth clumping `truth_Clumping`, outputs under `ml/runs/clumping_cnn_truth/<timestamp>/`):

- `python ml/clumping_cnn/train_cnn_clumping_truth.py --pretrained --amp --num-workers 0`

Optional hinge clumping:

- `python ml/clumping_cnn/train_cnn_clumping_hinge.py --pretrained --amp --num-workers 0`

## Inference

All inference methods require a trained checkpoint (`.pt`) produced by the trainers (e.g. `model_best.pt`).

### GUI

- `python ml/pai/gui_predict_pai.py`

Load a checkpoint (PAI or clumping), then predict a single image or average a folder.

### Web (browser)

- `python -m uvicorn ml.pai.web_predict_pai:app --host 127.0.0.1 --port 8000 --log-level warning`

Then open: http://127.0.0.1:8000/

### CLI (using the dataset index)

One case:

- `python ml/pai/predict_case_pai.py --checkpoint <model_best.pt> --case "Case 001"`

All cases:

- `python ml/pai/predict_all_cases_pai.py --checkpoint <model_best.pt>`

### CLI (predict any folder of images)

This does not require the dataset index. It recursively scans a folder and writes a CSV:

- `python ml/tools/predict_folder_pai_clumping.py --images-dir <folder> --pai-checkpoint <pai_model.pt> --clumping-checkpoint <clumping_model.pt>`

Omit `--clumping-checkpoint` if you only have a PAI model.

## More details

- `ml/pai/README.md`
- `ml/clumping_cnn/README.md`
