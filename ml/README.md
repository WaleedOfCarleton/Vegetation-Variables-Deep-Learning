# ML

This folder contains the deep learning / ML tooling.

Current layout:
- `ml/pai/`: train + predict **PAI** from simulation images (ResNet18 regression)
- `ml/clumping_cnn/`: compute **clumping** using a CNN model (scaffold)
- `ml/runs/`: training outputs (checkpoints, metrics, splits)

Most scripts use the dataset index:
- `shared/dataset_index/image_dataset_index.csv`

If you modify simulations or truth labels, rebuild the index:

`python shared/dataset_index/build_image_dataset_index.py`

See:
- `ml/pai/README.md`
- `ml/clumping_cnn/README.md`
