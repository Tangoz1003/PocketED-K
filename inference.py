import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.config import DEFAULT_MODEL_KWARGS, resolve_device
from utils.dataset import ECGDataset
from utils.net1d import Net1D


def build_dataset(df, args):
    pos_df = df[df[args.label_col] == 1]
    neg_df = df[df[args.label_col] == 0]
    pos_ids = pos_df[args.patient_col].unique().tolist()
    neg_ids = neg_df[args.patient_col].unique().tolist()
    pos_dataset = ECGDataset(
        pos_df,
        pos_ids,
        None,
        label_col=args.label_col,
        value_col=args.value_col,
        patient_col=args.patient_col,
        sampling_rate=args.sampling_rate,
        segment_length_seconds=args.segment_seconds,
    )
    neg_dataset = ECGDataset(
        neg_df,
        neg_ids,
        None,
        label_col=args.label_col,
        value_col=args.value_col,
        patient_col=args.patient_col,
        sampling_rate=args.sampling_rate,
        segment_length_seconds=args.segment_seconds,
    )
    return ConcatDataset([pos_dataset, neg_dataset])


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference for PocketED-K.")
    parser.add_argument("--data-csv", required=True, help="Matched cohort CSV.")
    parser.add_argument("--split-csv", required=True, help="Temporal split CSV.")
    parser.add_argument("--ckpt-path", required=True, help="Model checkpoint path.")
    parser.add_argument("--output-dir", default="outputs/inference_run", help="Inference output directory.")
    parser.add_argument("--split-name", default="test", choices=["train", "test"], help="Which split to evaluate.")
    parser.add_argument("--label-col", default="hypo_class")
    parser.add_argument("--value-col", default="Component_value")
    parser.add_argument("--patient-col", default="MRN")
    parser.add_argument("--time-col", default="Order_time")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)

    master_df = pd.read_csv(args.data_csv, low_memory=False)
    split_df = pd.read_csv(args.split_csv, low_memory=False)
    master_df[args.time_col] = master_df[args.time_col].astype(str)
    split_df[args.time_col] = split_df[args.time_col].astype(str)
    merged_df = pd.merge(
        master_df,
        split_df[[args.patient_col, args.time_col, "split"]],
        on=[args.patient_col, args.time_col],
        how="inner",
    )
    eval_df = merged_df[merged_df["split"] == args.split_name].reset_index(drop=True)

    dataset = build_dataset(eval_df, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Net1D(**DEFAULT_MODEL_KWARGS)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    all_probs, all_labels, all_values = [], [], []
    with torch.no_grad():
        for signals, labels, values in tqdm(loader, desc="Inference"):
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            signals = signals[valid_mask].to(device)
            logits = model(signals).squeeze(1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_values.extend(values[valid_mask].cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_values = np.array(all_values)

    auroc = roc_auc_score(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auprc = auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    preds = (all_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    pd.DataFrame(
        {"true_label": all_labels, "pred_prob": all_probs, "lab_value": all_values}
    ).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    pd.DataFrame(
        [
            {
                "samples_total": len(all_labels),
                "positive": int(all_labels.sum()),
                "negative": int(len(all_labels) - all_labels.sum()),
                "auroc": auroc,
                "auprc": auprc,
                "best_threshold": best_threshold,
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            }
        ]
    ).to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
