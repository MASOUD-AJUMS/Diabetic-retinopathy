# DR Multi-Task Learning Framework

A hierarchical multi-task deep learning framework for simultaneous diabetic retinopathy grading, lesion segmentation, and lesion detection from color fundus photographs.

---

## Overview

This framework jointly learns three clinically complementary tasks in a single forward pass:

- **Lesion Segmentation** — pixel-level delineation of microaneurysms (MA), hemorrhages (HE), hard exudates (EX), and soft exudates (SE)
- **Lesion Detection** — bounding-box localization of the same four lesion types
- **DR Grading** — six-class severity classification (No DR / Mild / Moderate / Severe / PDR / Ungradable)

The grading head is conditioned on predicted lesion features through a cross-task attention mechanism, so each severity decision is accompanied by lesion-level spatial evidence.

---

## Architecture

```
Input Image (512×512)
        │
   ResNet-50 Backbone
        │
 Feature Pyramid Network
   ┌────┴────────────┐
   │                 │
Seg Head         Det Head
(4 lesion maps)  (FCOS-style)
   │                 │
   └────┐   ┌────────┘
        ▼   ▼
   Cross-Task Attention
   (seg context → grade feat)
   (det context → grade feat)
        │
  Classification Head
   (6-class grade logits)
```

**Key design choices:**

- Shared ResNet-50 encoder with FPN (256-channel pyramid)
- Segmentation head: lightweight decoder with bilinear upsampling
- Detection head: anchor-free FCOS-style with center sampling
- Classification head: two serial cross-task attention blocks (gated residuals, γ initialized to 0)
- Homoscedastic uncertainty weighting across the three task losses
- Progressive three-stage training schedule (Seg → Seg+Det → Full)
- Partial-label handling via gradient masking (no pseudo-labels)

---

## Results

### DDR Test Split (Internal)

| Task | Metric | Value |
|---|---|---|
| Segmentation | Mean Dice | 0.535 |
| Segmentation | Mean IoU | — |
| Detection | mAP@0.5 | 0.289 |
| Detection | Sens@1FPI | 0.436 |
| Grading | Accuracy | 0.823 |
| Grading | QWK | 0.858 |
| Grading | Macro F1 | 0.612 |
| Referable DR | Sensitivity | 0.928 |
| Referable DR | Specificity | 0.901 |
| VT-DR | Sensitivity | 0.871 |

### IDRiD External Test (No Tuning)

| Task | Metric | Value |
|---|---|---|
| Segmentation | Mean Dice | 0.489 |
| Grading | Accuracy | 0.781 |
| Grading | QWK | 0.821 |
| Referable DR | Sensitivity | 0.889 |

### Per-Lesion Detection (DDR Test, AP@0.5)

| Lesion | AP |
|---|---|
| EX | 0.392 |
| HE | 0.331 |
| SE | 0.275 |
| MA | 0.158 |

---

## Project Structure

```
dr_multitask/
├── configs/
│   └── config.py               # All hyperparameters and settings
├── data/
│   ├── dataset.py              # Dataset class, transforms, collate
│   └── data_utils.py           # DDR/IDRiD loaders, fold splitting, deduplication
├── models/
│   ├── model.py                # Full architecture (backbone, FPN, heads, attention)
│   ├── losses.py               # Seg/Det/Cls losses, uncertainty weighting
│   └── trainer.py              # Progressive training engine, optimizer, scheduler
├── utils/
│   ├── metrics.py              # Dice, IoU, AUPR, QWK, ECE, bootstrap CI
│   └── visualization.py        # Grad-CAM, confusion matrix, ROC, t-SNE, reliability
├── scripts/
│   ├── train_crossval.py       # Stratified 5-fold cross-validation training
│   ├── evaluate_external.py    # External evaluation on IDRiD
│   ├── run_ablation.py         # Full ablation study with significance tests
│   ├── interpretability.py     # Grad-CAM, SHAP, t-SNE analysis
│   └── predict.py              # Single image or folder inference
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/[organization]/dr-multitask.git
cd dr-multitask
pip install -r requirements.txt
```

PyTorch with CUDA support must be installed separately following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Data Preparation

### DDR Dataset

Download from the [DDR dataset page](https://github.com/nkicsl/DDR-dataset). Expected structure:

```
ddr_root/
├── grading/
│   ├── 0/          ← No DR images
│   ├── 1/          ← Mild NPDR
│   ├── 2/          ← Moderate NPDR
│   ├── 3/          ← Severe NPDR
│   ├── 4/          ← PDR
│   └── 5/          ← Ungradable
└── lesion_annotation/
    ├── MA/         ← Binary masks (.png)
    ├── HE/
    ├── EX/
    ├── SE/
    └── bboxes.json ← [{"image_id": "...", "x1":..., "y1":..., "x2":..., "y2":..., "label": 0}, ...]
```

### IDRiD Dataset

Download from the [IDRiD challenge page](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid). Expected structure:

```
idrid_root/
├── A. Segmentation/
│   └── 2. All Segmentation Groundtruths/
│       └── a. Training Set/
│           ├── 1. Microaneurysms/
│           ├── 2. Haemorrhages/
│           ├── 3. Hard Exudates/
│           └── 4. Soft Exudates/
└── B. Disease Grading/
    ├── 1. Original Images/
    │   └── a. Training Set/
    └── 2. Groundtruths/
        └── a. IDRiD_Disease Grading_Training Labels.csv
```

---

## Usage

### Cross-Validation Training

```bash
python scripts/train_crossval.py \
    --ddr_root /path/to/ddr \
    --lesion_annotation_dir /path/to/ddr/lesion_annotation \
    --output_dir outputs/crossval \
    --device cuda \
    --n_folds 5 \
    --n_seeds 3
```

Results per fold are saved to `outputs/crossval/fold_X_seed_Y/`. A summary JSON is written to `outputs/crossval/crossval_summary.json`.

### External Evaluation on IDRiD

```bash
python scripts/evaluate_external.py \
    --idrid_root /path/to/idrid \
    --checkpoint outputs/crossval/fold_0_seed_0/final_model.pth \
    --thresholds outputs/crossval/fold_0_seed_0/val_results.json \
    --output_dir outputs/external_eval \
    --device cuda
```

Outputs: per-task metrics JSON, confusion matrix, ROC curves, reliability diagram.

### Ablation Study

```bash
python scripts/run_ablation.py \
    --ddr_root /path/to/ddr \
    --lesion_annotation_dir /path/to/ddr/lesion_annotation \
    --output_dir outputs/ablation \
    --device cuda \
    --fold 0 \
    --seed 42
```

Runs all ablation variants and produces a comparison table with Benjamini-Hochberg adjusted p-values.

### Interpretability Analysis

```bash
python scripts/interpretability.py \
    --data_root /path/to/ddr \
    --dataset ddr \
    --lesion_annotation_dir /path/to/ddr/lesion_annotation \
    --checkpoint outputs/crossval/fold_0_seed_0/final_model.pth \
    --output_dir outputs/interpretability \
    --device cuda \
    --n_per_grade 3
```

Produces Grad-CAM panels, segmentation overlays, t-SNE embedding projection, and SHAP attribution maps.

### Single Image or Folder Prediction

```bash
python scripts/predict.py \
    --input /path/to/image_or_folder \
    --checkpoint outputs/crossval/fold_0_seed_0/final_model.pth \
    --thresholds outputs/crossval/fold_0_seed_0/val_results.json \
    --output_dir outputs/predictions \
    --device cuda \
    --visualize
```

Predictions are saved to `outputs/predictions/predictions.json`. With `--visualize`, overlay figures are also saved.

---

## Configuration

All hyperparameters are centralized in `configs/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `image_size` | 512 | Input resolution (px) |
| `base_lr` | 1e-4 | Base learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `layer_decay` | 0.9 | Backbone layer-wise LR decay |
| `seg_epochs` | 100 | Max epochs for stage 1 |
| `det_epochs` | 100 | Max epochs for stage 2 |
| `full_epochs` | 100 | Max epochs for stage 3 |
| `patience` | 10 | Early stopping patience |
| `batch_size_lesion` | 4 | Batch size for annotated subset |
| `batch_size_full` | 8 | Batch size for full multi-task stage |
| `n_folds` | 5 | Number of CV folds |
| `n_seeds` | 3 | Number of random seeds |
| `bootstrap_n_resamples` | 1000 | Bootstrap CI resamples |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |

---

## Training Stages

The progressive schedule stabilizes multi-task optimization under the partial-label structure of DDR (only 757 of 13,673 images have dense lesion annotation):

| Stage | What trains | Loss terms | Batch source |
|---|---|---|---|
| 1 – Segmentation | Backbone, FPN, Seg Head | Dice + BCE | 757 annotated images |
| 2 – Detection | + Det Head | Seg + Focal | 757 annotated images |
| 3 – Full | All (with layer-LR decay) | Seg + Det + Cls (uncertainty-weighted) | All 13,673 images |

During stages 1 and 2, the classification loss is masked for mini-batches with no grade-only images. During stage 3, segmentation and detection losses are masked for mini-batches containing no annotated images.

---

## Evaluation Protocol

- **Internal**: stratified 5-fold cross-validation, 3 seeds; near-duplicate control via perceptual hashing (Hamming ≤ 5)
- **External**: IDRiD evaluated once with thresholds fixed from DDR validation folds (no leakage)
- **Statistics**: 95% bootstrap CIs (1,000 resamples), McNemar test for accuracy, Wilcoxon for Dice, DeLong for AUC; Benjamini-Hochberg FDR correction across ablation comparisons

---

## Ablation Variants

| Configuration | Purpose |
|---|---|
| Single-task classification only | Benefit of lesion supervision for grading |
| Single-task segmentation only | Effect of joint training on dense prediction |
| Without cross-task attention | Value of lesion-conditioned grading |
| Two-task (Seg + Cls, no Det) | Contribution of the detection head |
| Fixed equal task weights | Value of uncertainty weighting |
| No progressive schedule | Value of the staged curriculum |
| Full model | Reference |

---

## Hardware

Trained on a single NVIDIA RTX 3090 (24 GB). Inference throughput: ~96 images/second at batch size 32.

For higher input resolutions (768 or 1024) beyond 512, gradient accumulation or larger-memory hardware is required to maintain stable batch sizes for the small annotated subset.
