import os
import pandas as pd
from sklearn.model_selection import train_test_split

# your existing functions
from baseline_models import (
    model_validation_fold_allTP,
    model_validation_fold_eachTP,
    current_protocol_validation_fold,
)

# -----------------------------
# Config for both datasets
# -----------------------------
DATASETS = [
    {
        "data_name": "NEO",
        "outcome_name": "resp.pCR",
        "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
        "remove_attributes": [
            "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "RCB.score",
            "RCB.category", "Chemo.NumCycles", "Chemo.first.Taxane",
            "Chemo.first.Anthracycline", "Chemo.second.Taxane",
            "Chemo.second.Anthracycline", "Chemo.any.Anthracycline",
            "Chemo.any.antiHER2"
        ],
    },
    {
        "data_name": "DUKE",
        "outcome_name": "pCR",
        "treatment_plans": ["TP1", "TP2"],
        "remove_attributes": [
            "pid", "Chemotheray.Adjuvant",
            "rfs_event", "rfs_time", "lrfs_event", "lrfs_time",
            "drfs_event", "drfs_time",
        ],
    },
]

# -----------------------------
# Global params
# -----------------------------
threshold = -999  # use all predicted treatment-effect values (threshold -999 is very small, below 0)
random_seed = 42
folds = 5
test_size = 0.30

base_folder = os.getcwd()
outputPath = os.path.join(base_folder, "output1")
os.makedirs(outputPath, exist_ok=True)

import os, random, numpy as np
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
seed_everything(42)
random.seed(random_seed)
np.random.seed(random_seed)

# -----------------------------
# Helpers
# -----------------------------
def split_data(base_folder: str, data_name: str, folds: int, test_size: float) -> None:
    """
    Create K (folds) random train/test splits and save them as CSVs:
    {data_name}_train_{i}.csv / {data_name}_test_{i}.csv
    Also saves a sanitized copy with '.' instead of spaces in column names.
    """
    inputPath = os.path.join(base_folder, "input", data_name)
    os.makedirs(inputPath, exist_ok=True)
    input_csv = os.path.join(inputPath, f"{data_name}.csv")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    dataset = pd.read_csv(input_csv, encoding="ISO-8859-1", engine="python")
    # sanitise column names (spaces -> '.')
    dataset.columns = dataset.columns.str.strip().str.replace(r"\s+", ".", regex=True)
    # dataset.to_csv(os.path.join(inputPath, f"{data_name}_sanitised.csv"), index=False)

    # If splits already exist, skip regeneration
    existing = all(
        os.path.exists(os.path.join(inputPath, f"{data_name}_train_{i}.csv"))
        and os.path.exists(os.path.join(inputPath, f"{data_name}_test_{i}.csv"))
        for i in range(1, folds + 1)
    )
    if existing:
        print(f"[{data_name}] Found existing {folds} splits. Skipping split generation.")
        return

    print(f"[{data_name}] Generating {folds} train/test splits...")
    fileCount = 1
    for seed in range(40, 40 + folds):
        Mtrain, Mtest = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)
        Mtrain.reset_index(drop=True, inplace=True)
        Mtest.reset_index(drop=True, inplace=True)
        Mtrain.to_csv(os.path.join(inputPath, f"{data_name}_train_{fileCount}.csv"), index=False)
        Mtest.to_csv(os.path.join(inputPath, f"{data_name}_test_{fileCount}.csv"), index=False)
        fileCount += 1
    print(f"[{data_name}] Done creating splits.")

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline_for_dataset(cfg: dict):
    data_name = cfg["data_name"]
    outcome_name = cfg["outcome_name"]
    treatment_plans = cfg["treatment_plans"]
    remove_attributes = cfg["remove_attributes"]

    inputPath = os.path.join(base_folder, "input", data_name)
    print("\n" + "=" * 70)
    print(f"Running pipeline for dataset: {data_name}")
    print(f"Outcome: {outcome_name}")
    print(f"Treatment plans: {treatment_plans}")
    print("=" * 70)

    # 1) Ensure splits exist (comment out if you already have them)
    split_data(base_folder, data_name, folds, test_size)

    # 2) Run baselines (allTP, eachTP) and current protocol
    #    These functions are expected to read the fold files from input/
    model_validation_fold_allTP(
        data_name, treatment_plans, outcome_name,
        remove_attributes, threshold, base_folder, random_seed
    )
    model_validation_fold_eachTP(
        data_name, treatment_plans, outcome_name,
        remove_attributes, threshold, base_folder, random_seed
    )
    # current_protocol_validation_fold(
    #     data_name, treatment_plans, outcome_name,
    #     remove_attributes, threshold, base_folder, random_seed
    # )
    print(f"[{data_name}] Completed.\n")

if __name__ == "__main__":
    for cfg in DATASETS:
        try:
            run_pipeline_for_dataset(cfg)
        except Exception as e:
            print(f"❌ Error while running {cfg['data_name']}: {e}")
