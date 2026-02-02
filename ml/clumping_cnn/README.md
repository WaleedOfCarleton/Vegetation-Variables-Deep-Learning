# Clumping — CNN

This folder contains CNN workflows to predict **clumping** from simulation images.

## Default target: Excel truth clumping

By default we train against the simulation/Excel truth that is already in the dataset index:
- `truth_Clumping`

This is **not** hinge-method clumping; it’s the truth clumping value joined into the image index.

### Train

`python ml/clumping_cnn/train_cnn_clumping_truth.py --amp`

Common options:
- `--orientation ERECT` (or `PLANO`, `RND`)
- `--simulation-set "DHP - ERECT - 4000x4000"`

### Predict

`python ml/clumping_cnn/predict_case_clumping_truth.py --checkpoint <model_best.pt> --case "Case 001"`

## Optional target: hinge clumping

If you specifically want hinge-based clumping:

$$\Omega_{hinge} = \frac{PAIe_{hinge}}{PAI}$$

Scripts:
- `python ml/clumping_cnn/train_cnn_clumping_hinge.py --amp`
- `python ml/clumping_cnn/predict_case_clumping_hinge.py --checkpoint <model_best.pt> --case "Case 001"`

Notes:
- Hinge truth is orientation-specific and often missing for `RND`.

## Dataset index

All scripts use:
- `shared/dataset_index/image_dataset_index.csv`

If you modify simulations or truth labels, rebuild the index:

`python shared/dataset_index/build_image_dataset_index.py`
