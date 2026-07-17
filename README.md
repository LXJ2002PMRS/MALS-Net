# MALS-Net

Official PyTorch implementation of **MALS-Net: A multimodal attention-enhanced segmentation network for detecting lunar lobate scarps**.

MALS-Net accepts two aligned single-band 512 x 512 pixel inputs: a lunar digital orthophoto map (DOM) tile and its DAM V2-derived relative-depth tile. It produces a binary lobate-scarp mask.

## 1. Repository layout

```text
MALS-Net/
├── data/
│   ├── train/{image,dem,label}/
│   ├── val/{image,dem,label}/
│   └── test/{image,dem,label}/
├── datasets/datasets.py
├── network/
├── utils/environment.yaml
├── train.py
├── predict.py
├── estimate_model.py
└── eval_performance.py
```

The folder name `dem` is retained for compatibility with the released code; in the DOM + Depth experiment it contains the DAM V2 relative-depth tiles, not a metric DEM.

## 2. System requirements

The paper experiments used Python 3.10, PyTorch 2.5.1, CUDA 12.4, and an NVIDIA RTX 3090 GPU. A CUDA-capable GPU is recommended for training. Prediction can fall back to CPU, but `eval_performance.py` requires CUDA because it synchronizes GPU timing.

## 3. Create the environment

Install Miniconda or Anaconda, clone the repository, and create the provided environment:

```bash
git clone https://github.com/LXJ2002PMRS/MALS-Net.git
cd MALS-Net
conda env create -f utils/environment.yaml
conda activate torch
```

Verify the main dependencies and GPU:

```bash
python -c "import torch, osgeo, cv2; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 4. Prepare the data

Each split must contain three subfolders with matching TIFF filenames:

```text
data/test/image/00007.tif   # DOM
data/test/dem/00007.tif     # aligned DAM V2 relative depth
data/test/label/00007.tif   # binary reference mask (0/1 or 0/255)
```

Requirements:

- Every DOM, depth, and label triplet must have the same filename and dimensions.
- Inputs used in the paper are single-band 512 x 512 pixel TIFFs.
- Labels are binary. The loader converts 0/255 labels to 0/1.
- The repository includes a small sample subset for checking the pipeline. Reproducing the paper's reported values requires the complete released train/validation/test split and the same split assignment used in the manuscript.

Check that all file triplets are present:

```bash
python -c "from pathlib import Path; root=Path('data'); [(lambda a,b,c: print(s, len(a), 'triplets', 'OK' if a==b==c else 'MISMATCH'))({p.name for p in (root/s/'image').glob('*.tif')},{p.name for p in (root/s/'dem').glob('*.tif')},{p.name for p in (root/s/'label').glob('*.tif')}) for s in ('train','val','test')]"
```
##### Dataset
```text
https://pan.baidu.com/s/17ER19Lti0JibkG3-oZ0sWw?pwd=uany
```
## 5. Train MALS-Net

From the repository root, run:

```bash
python train.py \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --seed 42 \
  --checkpoint utils/check.pth
```

Outputs:

- best validation checkpoint: `utils/check.pth`
- timestamped training log: `training_logs/`
- TensorBoard events: `log/`
- loss and accuracy curves: `utils/loss.jpg` and `utils/acc.jpg`

If GPU memory is insufficient, reduce `--batch-size`, but keep it at 2 or greater because the ASPP global-pooling branch uses BatchNorm on a 1 x 1 feature map. On Windows, keep `--num-workers 0`; Linux users may increase it after confirming stable data loading.

To inspect learning curves:

```bash
tensorboard --logdir log
```

## 6. Run inference on the test set

```bash
python predict.py \
  --test-dir data/test \
  --checkpoint utils/check.pth \
  --output-dir utils/predict_test \
  --threshold 0.5
```

The predicted binary TIFF masks are written to `utils/predict_test/` with the same filenames as the inputs.

## 7. Segmentation metrics

After inference, calculate pixel-level, object-level, and geometric metrics:

```bash
python estimate_model.py \
  --prediction-dir utils/predict_test \
  --label-dir data/test/label \
  --output evaluation_results.txt
```

The report includes pixel precision, recall, F1 and IoU; object precision, recall, F1 and matched-object IoU; normalized centroid distance (NCD); and compactness similarity (CS). The object matching threshold and minimum component size follow the released evaluation implementation.

## 8. Model cost and inference speed

```bash
python eval_performance.py
```

This script profiles a batch of one 2 x 512 x 512 multimodal sample, performs 50 warm-up iterations and times 1,000 CUDA iterations. On the RTX 3090 configuration used in the manuscript, MALS-Net contained 40.77 M parameters, required 149.62 GFLOPs, and achieved 12.30 ms per tile (81.33 FPS).

## Citation

```text
Under review
```
