# PAI — CNN

This folder contains the end-to-end pipeline to train and use a CNN (ResNet18 regression) to predict **PAI** from simulation images.

## Train

From the repo root:

`python ml/pai/train_cnn_pai.py`

Common options:
- `--orientation ERECT` (or `PLANO`, `RND`)
- `--simulation-set "DHP - ERECT - 4000x4000"` (example)
- `--amp` (mixed precision on GPU)
- `--pretrained` (ImageNet init)

Outputs default to:
- `ml/runs/pai_cnn/<timestamp>/`

## K-fold (case-wise)

Single fold:

`python ml/pai/train_cnn_pai.py --kfold 5 --fold 0 --amp`

Run all folds:

`python ml/pai/run_kfold_train_cnn_pai.py --k 5 --epochs 30 --amp`

Outputs default to:
- `ml/runs/pai_cnn_kfold/<timestamp>/...`

## Predict / evaluate

Predict one case:

`python ml/pai/predict_case_pai.py --checkpoint ml/runs/pai_cnn/<timestamp>/model_best.pt --case "Case 001"`

Predict all images + per-case mean:

`python ml/pai/predict_all_cases_pai.py --checkpoint ml/runs/pai_cnn/<timestamp>/model_best.pt`

## GUI

`python ml/pai/gui_predict_pai.py`

Workflow:
1) Load checkpoint (`.pt`)
2) Choose an image or folder
3) Click **Predict**
