import torch
import numpy as np
from pathlib import Path
from ml.pai.pai_cnn.common import (
    PaiIndexDataset, build_transforms, build_model,
    read_index_csv, split_cases_kfold
)

device = torch.device("cpu")

# Best all-data model (fold_3)
model_path = Path("ml/runs/pai_cnn_kfold/20260206-130133/fold_3/model_best.pt")
index_csv = Path("shared/dataset_index/image_dataset_index.csv")

if not model_path.exists():
    print(f"Model not found: {model_path}")
else:
    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = build_model(pretrained=True)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    rows = read_index_csv(index_csv)
    train_rows, val_rows = split_cases_kfold(rows, k=5, fold=3, seed=42)
    
    transform = build_transforms(224, train=False)
    val_ds = PaiIndexDataset(val_rows, Path.cwd(), transform)
    
    preds = []
    truths = []
    
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            # item is (x, truth, meta) from collate_keep_meta
            x = item[0]
            truth = item[1]
            x_batch = x.unsqueeze(0).to(device)
            pred = model(x_batch).squeeze().item()
            preds.append(pred)
            truths.append(truth)
    
    preds = np.array(preds)
    truths = np.array(truths)
    errors = preds - truths
    
    print(f"\nAll-data best CNN (fold_3, epoch 14):")
    print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}")
    print(f"  Ground truth: mean={truths.mean():.4f}, std={truths.std():.4f}")
    print(f"  Errors (pred - truth):")
    print(f"    Mean error (bias): {errors.mean():.4f}")
    print(f"    MAE: {np.abs(errors).mean():.4f}")
    print(f"    % positive errors (over-predict): {(errors > 0).sum() / len(errors) * 100:.1f}%")
    print(f"    % negative errors (under-predict): {(errors < 0).sum() / len(errors) * 100:.1f}%")
