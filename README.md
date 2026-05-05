# PocketED-K: Single-Lead ECG Hypokalemia Risk Tool

PocketED-K is a lightweight research toolkit for single-lead ECG based hypokalemia screening. It is distilled from the internal `mc-med` workflow into a GitHub-friendly tool repository, with a simpler structure for training, temporal splitting, and batch inference.

## Features

- Single-lead ECG training pipeline based on WFDB records.
- Strict temporal split utility with patient-overlap removal.
- Balanced mini-batch sampling for imbalanced hypokalemia labels.
- Batch inference with ROC export and prediction CSV generation.
- Example baseline summary and representative result figures included.

## Repository Structure

```text
PocketED-K/
├── checkpoint/
│   └── README.md
├── data/
│   ├── README.md
│   └── examples/
│       ├── matched_cohort_schema.csv
│       └── temporal_split_schema.csv
├── figures/
│   ├── ROC_Curve_Publication.png
│   └── ex_predictions_boxplot_custom_bins_beautified.png
├── outputs/
│   ├── README.md
│   └── examples/
│       └── baseline_summary_train_test_mimic.csv
├── utils/
│   ├── config.py
│   ├── dataset.py
│   ├── focal_loss.py
│   └── net1d.py
├── prepare_split.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## Expected Data Format

### Matched cohort CSV

The main cohort CSV should include at least:

- `MRN`
- `Order_time`
- `Component_value`
- `hypo_class`
- `num_matched_ecgs`
- `dat_path_1`, `index_interval_1`

The current dataset loader reads WFDB records from `dat_path_*` columns and slices 10-second segments using `index_interval_*`.

### Temporal split CSV

The split file should include:

- `MRN`
- `Order_time`
- `split`

Optional:

- `CSN`

## Installation

```bash
git clone https://github.com/yourusername/PocketED-K.git
cd PocketED-K
pip install -r requirements.txt
```

## Quick Start

### 1. Create a strict temporal split

```bash
python prepare_split.py \
    --data-csv /path/to/matched_cohort.csv \
    --output-csv data/temporal_split.csv
```

### 2. Train

```bash
python train.py \
    --data-csv /path/to/matched_cohort.csv \
    --split-csv data/temporal_split.csv \
    --output-dir outputs/train_run \
    --pretrained-ckpt /path/to/pretrained_backbone.pth
```

### 3. Run inference

```bash
python inference.py \
    --data-csv /path/to/matched_cohort.csv \
    --split-csv data/temporal_split.csv \
    --ckpt-path outputs/train_run/best_checkpoint.pth \
    --output-dir outputs/inference_run
```

## Notes

- This repository is a tool-style extraction of the internal workflow, not a full archival dump.
- Large checkpoints, TensorBoard logs, and raw sensitive cohort files are intentionally excluded.
- Clinical use requires separate validation, signal quality control, and governance review.
