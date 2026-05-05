import ast
import random

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, Sampler


class ECGDataset(Dataset):
    """Dataset for matched ECG-lab rows with WFDB-backed signal loading."""

    def __init__(
        self,
        df,
        patient_ids,
        num_samples_per_patient,
        label_col="hypo_class",
        value_col="Component_value",
        patient_col="MRN",
        sampling_rate=500,
        segment_length_seconds=10,
        target_samples=None,
    ):
        self.df = df
        self.patient_ids = patient_ids
        self.num_samples_per_patient = num_samples_per_patient
        self.label_col = label_col
        self.value_col = value_col
        self.patient_col = patient_col
        self.sampling_rate = sampling_rate
        self.segment_length_seconds = segment_length_seconds
        self.target_samples = target_samples
        self.segment_length_samples = self.sampling_rate * self.segment_length_seconds
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        all_samples = []
        use_all_segments = (
            self.num_samples_per_patient is None or self.num_samples_per_patient < 0
        )

        for patient_id in self.patient_ids:
            patient_df = self.df[self.df[self.patient_col] == patient_id]
            available_segments = []

            for _, row in patient_df.iterrows():
                component_value = row[self.value_col]
                label = row[self.label_col]
                num_matched_ecgs = row.get("num_matched_ecgs", 0)

                for i in range(1, num_matched_ecgs + 1):
                    dat_path = row.get(f"dat_path_{i}")
                    interval_str = row.get(f"index_interval_{i}")
                    if pd.isna(dat_path) or pd.isna(interval_str):
                        continue

                    try:
                        start_idx, end_idx = ast.literal_eval(interval_str)
                    except (ValueError, SyntaxError, TypeError):
                        continue

                    curr_idx = start_idx
                    while curr_idx < end_idx:
                        target_end_idx = curr_idx + self.segment_length_samples
                        valid_end_idx = min(target_end_idx, end_idx)
                        actual_length = valid_end_idx - curr_idx

                        if actual_length >= self.segment_length_samples:
                            available_segments.append(
                                (dat_path, curr_idx, component_value, label, valid_end_idx)
                            )
                        curr_idx += self.segment_length_samples

            if not available_segments:
                continue

            if use_all_segments:
                sampled = available_segments
            else:
                k = min(len(available_segments), self.num_samples_per_patient)
                sampled = (
                    random.sample(available_segments, k)
                    if len(available_segments) > k
                    else available_segments
                )

            all_samples.extend(sampled)

        if self.target_samples and len(all_samples) > self.target_samples:
            all_samples = random.sample(all_samples, self.target_samples)

        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dat_path, start_sample, component_value, label, valid_end_sample = self.samples[idx]
        record_name = dat_path.replace(".dat", "")

        try:
            signal = wfdb.rdrecord(
                record_name, sampfrom=start_sample, sampto=valid_end_sample
            ).p_signal
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0)
            normalized_signal = (signal - mean) / (std + 1e-8)

            current_length = normalized_signal.shape[0]
            if current_length < self.segment_length_samples:
                pad_length = self.segment_length_samples - current_length
                normalized_signal = np.pad(
                    normalized_signal,
                    ((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            signal_tensor = torch.from_numpy(normalized_signal.T.astype(np.float32))
            label_tensor = torch.tensor(label, dtype=torch.long)
            value_tensor = torch.tensor(component_value, dtype=torch.float32)
            return signal_tensor, label_tensor, value_tensor
        except Exception:
            return (
                torch.zeros((1, self.segment_length_samples), dtype=torch.float32),
                torch.tensor(-1, dtype=torch.long),
                torch.tensor(-1.0, dtype=torch.float32),
            )


class BalancedBatchSampler(Sampler):
    """Batch sampler that enforces a fixed positive/negative ratio per batch."""

    def __init__(self, dataset, batch_size, balance_ratio=0.5, pos_indices=None, neg_indices=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.balance_ratio = balance_ratio

        if pos_indices is not None and neg_indices is not None:
            self.positive_indices = list(pos_indices)
            self.negative_indices = list(neg_indices)
        else:
            self.positive_indices = []
            self.negative_indices = []
            for idx in range(len(dataset)):
                try:
                    label = dataset[idx][1]
                    if label.item() == 1:
                        self.positive_indices.append(idx)
                    elif label.item() == 0:
                        self.negative_indices.append(idx)
                except Exception:
                    continue

        self.pos_per_batch = int(batch_size * balance_ratio)
        self.neg_per_batch = batch_size - self.pos_per_batch
        self.num_batches = (
            len(self.positive_indices) // self.pos_per_batch if self.pos_per_batch > 0 else 0
        )

    def __iter__(self):
        random.shuffle(self.positive_indices)
        random.shuffle(self.negative_indices)
        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.num_batches):
            batch_indices = []
            batch_indices.extend(
                self.positive_indices[pos_ptr : pos_ptr + self.pos_per_batch]
            )
            pos_ptr += self.pos_per_batch
            batch_indices.extend(
                self.negative_indices[neg_ptr : neg_ptr + self.neg_per_batch]
            )
            neg_ptr += self.neg_per_batch
            yield batch_indices

    def __len__(self):
        return self.num_batches
