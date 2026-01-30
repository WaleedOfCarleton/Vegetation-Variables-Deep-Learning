# ML — PAI CNN

This folder contains a small PyTorch CNN pipeline that predicts **PAI** directly from the simulated hemisphere images under `Simulations/`.

At a high level:
- Training reads a prebuilt image/label index CSV: `shared/dataset_index/image_dataset_index.csv`
- Labels come from `shared/truth_join/truth_joined_to_hemipy.csv` (the index includes `truth_PAI`)
- The model is a ResNet18 regressor (single scalar output)

## Setup

From the repo root, use your preferred environment (venv/conda). You’ll need at least:
- Python + pip
- `torch`, `torchvision`
- `Pillow`

If you want GPU, install a CUDA-enabled PyTorch build. The scripts automatically use CUDA when available (unless you pass `--cpu`).

## Data / index CSV

Most scripts default to:
- `shared/dataset_index/image_dataset_index.csv`

If you modify simulations or truth labels, rebuild the index:

`python shared/dataset_index/build_image_dataset_index.py`

That script scans `Simulations/` recursively and writes/overwrites:
- `shared/dataset_index/image_dataset_index.csv`

## Train (single train/val split)

Train with a random case-wise split (default `--val-fraction 0.2`):

`python ml/train_cnn_pai.py`

Common options:
- Train only one orientation: `--orientation ERECT` (or `PLANO`, `RND`)
- Train only one simulation set: `--simulation-set "DHP - ERECT - 4000x4000"` (example)
- Mixed precision on GPU: `--amp`
- Use ImageNet init: `--pretrained`

Outputs go to:
- `ml/runs/pai_cnn/<timestamp>/`
  - `model_best.pt`, `model_last.pt`
  - `metrics.csv` (loss + MAE/RMSE metrics)
  - `config.json` (what was trained)
  - `splits/train_cases.txt`, `splits/val_cases.txt`

## K-fold validation (case-wise)

The training script also supports case-wise k-fold splits (prevents leakage across a case):

`python ml/train_cnn_pai.py --kfold 5 --fold 0 --amp`

To run all folds automatically and write an aggregate summary:

`python ml/run_kfold_train_cnn_pai.py --k 5 --epochs 30 --amp`

Outputs:
- `ml/runs/pai_cnn_kfold/<timestamp>/fold_<i>/...`
- `ml/runs/pai_cnn_kfold/<timestamp>/kfold_summary.csv`
- `ml/runs/pai_cnn_kfold/<timestamp>/kfold_aggregate.json`

## Predict / evaluate

### Predict a single case

Use a saved checkpoint and a case id from the index (e.g. `Case 001`):

`python ml/predict_case_pai.py --checkpoint ml/runs/pai_cnn/<timestamp>/model_best.pt --case "Case 001"`

This prints the case mean prediction and writes a per-image CSV next to the checkpoint.

### Predict all images + per-case mean (full evaluation)

This runs a checkpoint over all indexed images and writes:
- `pred_all_images.csv` (one row per image)
- `pred_all_cases.csv` (case mean prediction)

`python ml/predict_all_cases_pai.py --checkpoint ml/runs/pai_cnn/<timestamp>/model_best.pt`

## GUI

Run a local Tkinter GUI for interactive prediction:

`python ml/gui_predict_pai.py`

Workflow:
1) Load checkpoint (`.pt`)
2) Choose an image or a folder
3) Click **Predict**

Notes:
- Folder prediction searches recursively (works with nested `Case/Plot/...` layouts) and reports mean/std.

## Troubleshooting

- If the Predict button is disabled: you haven’t loaded a checkpoint yet, or you haven’t selected an image/folder.
- If inference is slow on CPU: install a CUDA-enabled PyTorch build and confirm `torch.cuda.is_available()` is true.
- If you changed/added images: rebuild the index CSV first.

## Design notes

- Splitting is **by case** (not by image) to avoid leakage.
- Default model is ResNet18. `--pretrained` uses ImageNet initialization.
