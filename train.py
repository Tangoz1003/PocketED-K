import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc as auc_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.config import DEFAULT_MODEL_KWARGS, resolve_device
from utils.dataset import BalancedBatchSampler, ECGDataset
from utils.focal_loss import FocalLoss
from utils.net1d import Net1D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained(model, ckpt_path, device):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    cleaned = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        if not key.startswith("dense."):
            cleaned[key] = v
    model.load_state_dict(cleaned, strict=False)


def build_patient_split(full_train_df, patient_col, seed):
    patient_ids = list(full_train_df[patient_col].unique())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_patients = patient_ids[:split_idx]
    val_patients = patient_ids[split_idx:]
    train_df = full_train_df[full_train_df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df = full_train_df[full_train_df[patient_col].isin(val_patients)].reset_index(drop=True)
    return train_df, val_df


def build_concat_dataset(df, args):
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
    dataset = ConcatDataset([pos_dataset, neg_dataset])
    pos_count = len(pos_dataset)
    neg_count = len(neg_dataset)
    pos_indices = list(range(0, pos_count))
    neg_indices = list(range(pos_count, pos_count + neg_count))
    return dataset, pos_indices, neg_indices


def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for signals, labels, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            signals = signals[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            probs = torch.sigmoid(model(signals).squeeze(1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        return {"auroc": -1.0, "auprc": -1.0, "f1": -1.0}
    auroc = roc_auc_score(all_labels, all_probs)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    auprc = auc_score(recall, precision)
    best_idx = np.argmax(2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8))
    best_threshold = thresholds[best_idx] if len(thresholds) else 0.5
    preds = (all_probs >= best_threshold).astype(int)
    f1 = f1_score(all_labels, preds)
    return {"auroc": auroc, "auprc": auprc, "f1": f1}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    for signals, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        valid_mask = labels != -1
        if not valid_mask.any():
            continue
        signals = signals[valid_mask].to(device)
        labels = labels[valid_mask].float().to(device)
        optimizer.zero_grad()
        logits = model(signals).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


def parse_args():
    parser = argparse.ArgumentParser(description="Train PocketED-K on matched ECG-potassium pairs.")
    parser.add_argument("--data-csv", required=True, help="Matched cohort CSV.")
    parser.add_argument("--split-csv", required=True, help="Temporal split CSV.")
    parser.add_argument("--output-dir", default="outputs/train_run", help="Training output directory.")
    parser.add_argument("--pretrained-ckpt", default=None, help="Optional pre-trained checkpoint.")
    parser.add_argument("--label-col", default="hypo_class", help="Binary label column.")
    parser.add_argument("--value-col", default="Component_value", help="Laboratory value column.")
    parser.add_argument("--patient-col", default="MRN", help="Patient identifier column.")
    parser.add_argument("--time-col", default="Order_time", help="Timestamp column.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
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

    full_train_df = merged_df[merged_df["split"] == "train"].reset_index(drop=True)
    train_df, val_df = build_patient_split(full_train_df, args.patient_col, args.seed)
    train_dataset, pos_indices, neg_indices = build_concat_dataset(train_df, args)
    val_dataset, _, _ = build_concat_dataset(val_df, args)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            balance_ratio=0.5,
            pos_indices=pos_indices,
            neg_indices=neg_indices,
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Net1D(**DEFAULT_MODEL_KWARGS)
    load_pretrained(model, args.pretrained_ckpt, device)
    model.to(device)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
    best_auroc = -1.0
    best_path = os.path.join(args.output_dir, "best_checkpoint.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": metrics["auroc"],
            "val_auprc": metrics["auprc"],
            "val_f1": metrics["f1"],
        }
        history.append(row)
        print(row)

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_auroc,
                    "args": vars(args),
                },
                best_path,
            )

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    print(f"Saved best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()
