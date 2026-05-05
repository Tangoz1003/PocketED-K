import argparse

import pandas as pd


def create_temporal_split(data_csv, output_csv, time_col="Order_time", patient_col="MRN"):
    df = pd.read_csv(data_csv, low_memory=False)
    if time_col not in df.columns:
        raise ValueError(f"Missing required time column: {time_col}")
    if patient_col not in df.columns:
        raise ValueError(f"Missing required patient column: {patient_col}")

    df = df.sort_values(time_col).reset_index(drop=True)
    split_index = int(len(df) * 0.7)

    train_df = df.iloc[:split_index].copy()
    test_candidate_df = df.iloc[split_index:].copy()

    train_patients = set(train_df[patient_col].unique())
    test_df = test_candidate_df[~test_candidate_df[patient_col].isin(train_patients)].copy()

    train_df["split"] = "train"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, test_df], ignore_index=True)
    output_cols = [patient_col, time_col, "split"]
    if "CSN" in final_df.columns:
        output_cols.insert(1, "CSN")

    final_df[output_cols].to_csv(output_csv, index=False)
    print(f"Saved temporal split to: {output_csv}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Removed overlapping future-patient rows: {len(test_candidate_df) - len(test_df)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a strict temporal split for ECG-laboratory pairs."
    )
    parser.add_argument("--data-csv", required=True, help="Input matched cohort CSV.")
    parser.add_argument("--output-csv", required=True, help="Output split CSV path.")
    parser.add_argument("--time-col", default="Order_time", help="Temporal ordering column.")
    parser.add_argument("--patient-col", default="MRN", help="Patient identifier column.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_temporal_split(
        data_csv=args.data_csv,
        output_csv=args.output_csv,
        time_col=args.time_col,
        patient_col=args.patient_col,
    )
