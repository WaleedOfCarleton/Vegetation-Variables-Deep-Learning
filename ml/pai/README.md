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
2) Choose an image, a single case folder, **or a parent folder containing multiple Case */ subfolders**
3) Click **Predict**

Folder mode averages predictions:
- Single case folder → mean/std over all images in that case.
- Parent folder with multiple cases → per-case means plus an overall mean/std across cases.

## Case-count sweep (train size vs. val MAE)

Quickly assess how many training cases you need before gains plateau:

```
python ml/pai/run_sweep_train_cases.py \
	--train-cases-list 15,25,35,45,55,65,70 \
	--epochs 10 \
	--patience 3 \
	--best-metric val_mae_case \
	--batch-size 16 --img-size 224 --lr 1e-4 --weight-decay 1e-4 --num-workers 0 --amp --pretrained
```

Outputs (timestamped under ml/runs/pai_cnn_case_sweep/):
- One subfolder per train count (metrics.csv, val_cases.txt, checkpoints)
- Aggregates: case_sweep_summary.csv and case_sweep_aggregate.json (use summary CSV to plot val_mae_case vs n_train_cases)
