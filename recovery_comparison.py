import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def calculate_recovery_ratio(summary_stats, fullResultFolder, fold_nbr, prefileName, postfileName,
                             method_name, inputName, outcome_name):

    # Find files
    pattern = os.path.join(fullResultFolder, f"{prefileName}*{postfileName}.csv")
    files = sorted(glob.glob(pattern))[:fold_nbr]

    if len(files) == 0:
        print(f"❌ No file found in {fullResultFolder}")
        return None

    df_list = [pd.read_csv(f) for f in files]
    data = pd.concat(df_list, ignore_index=True)


    # Compute grouped stats
    rec_stats = data.groupby("FOLLOW_REC")[outcome_name].agg(['count', 'sum']).reset_index()

    for _, row in rec_stats.iterrows():
        count = int(row["count"])
        recovery = int(row["sum"])

        # ✅ Ensure count is at least 1
        if recovery == 0:
            print(f"⚠️  recovery is 0 for method {method_name}, FOLLOW_REC={int(row['FOLLOW_REC'])} — setting count to 1")
            recovery = 1

        summary_stats['recovery'].append({
            "Method": method_name,
            "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": count,
            "Recovery": recovery
        })
# ============================
# ✅ MAIN PIPELINE (DUKE + NEO)
# ============================

methods = ["LogrRe", "LogrRa", "SvmRe", "SvmRa", "RfRe", "RfRa", "CTR"]
fold_nbr = 5
threshold = -999
baseInFolder = "./output/"
testName = "test"

# dataset -> outcome name
DATASETS = {
    "DUKE": "pCR",
    "NEO": "resp.pCR",
}
# Current protocol recovery rate (fixed) per dataset
CURRENT_PR = {
    "NEO": 0.26,
    "DUKE": 0.21,

}

# ✅ Create PerformanceEval folder (single place for all outputs)
PerformanceEval_folder = os.path.join(baseInFolder, "PerformanceEval")
os.makedirs(PerformanceEval_folder, exist_ok=True)

# === Define custom colors and order ===
method_order_ratio = ['CTR', 'LogrRe', 'LogrRa', 'SvmRe', 'SvmRa', 'RfRe', 'RfRa']
method_order_rate  = method_order_ratio + ['CurrentPr']

colors = {
    'CTR': '#ff7f0e',
    'LogrRa': '#27ae60',
    'LogrRe': '#27ae60',
    'SvmRa': '#c8e526',
    'SvmRe': '#c8e526',
    'RfRa': '#f4d03f',
    'RfRe': '#f4d03f',
    'CurrentPr': '#1f77b4'
}

def process_dataset(inputName: str, outcome_name: str):
    """
    Run the recovery aggregation + plots for one dataset.
    """
    print(f"\n==== Processing dataset: {inputName}  (outcome: {outcome_name}) ====")

    # collect raw stats across methods
    summary_stats = {'recovery': []}

    # set outcome_name global if your calculate_recovery_ratio reads it globally
    globals()['outcome_name'] = outcome_name

    for method in methods:
        prefileName = f"{inputName}_{testName}_{method}_{threshold}_"

        if inputName == "DUKE":
            postfileName = "follow"
        # if inputName == "NEO":
        else:
            postfileName = ""
        fullResultFolder = os.path.join(baseInFolder, method, inputName)

        calculate_recovery_ratio(
            summary_stats,
            fullResultFolder, fold_nbr, prefileName, postfileName,
            method, inputName, outcome_name
        )

    # ---- Format results into DataFrame (robust to missing FOLLOW_REC levels) ----
    raw = pd.DataFrame(summary_stats['recovery'])

    # If nothing was collected, bail out gracefully
    if raw.empty:
        print(f"⚠️ No recovery stats available for dataset {inputName}. Skipping.")
        return

    # Pivot with fill_value=0 to ensure both FOLLOW_REC groups present
    pivot = raw.pivot_table(index="Method",
                            columns="FOLLOW_REC",
                            values=["Count", "Recovery"],
                            aggfunc="sum",
                            fill_value=0)

    # Ensure both columns 0 and 1 exist
    for lvl0 in ["Count", "Recovery"]:
        if lvl0 not in pivot.columns.levels[0]:
            # add missing level with zeros
            pivot[(lvl0, 0)] = 0
            pivot[(lvl0, 1)] = 0
    # Reindex level 1 (FOLLOW_REC) to [0,1] and sort columns
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([["Count","Recovery"], [0,1]]), fill_value=0)

    # Flatten to expected columns
    recovery_df = pd.DataFrame({
        "Method": pivot.index,
        "NotFollowing_Count":   pivot[("Count", 0)].values,
        "Following_Count":      pivot[("Count", 1)].values,
        "NotFollowing_Recovery":pivot[("Recovery", 0)].values,
        "Following_Recovery":   pivot[("Recovery", 1)].values,
    })

    recovery_df = recovery_df.reset_index(drop=True)
    # Save the raw recovery table
    # recovery_csv = os.path.join(PerformanceEval_folder, f"{inputName}{threshold}_Recovery.csv")
    # recovery_df.to_csv(recovery_csv, index=False)

    # === Compute Metrics (guard divide-by-zero) ===
    denom = recovery_df["NotFollowing_Recovery"].replace(0, np.nan)
    recovery_df["Recovery_Ratio"] = recovery_df["Following_Recovery"] / denom
    # if denom was zero, fallback to numerator (same behaviour you had)
    recovery_df["Recovery_Ratio"] = recovery_df["Recovery_Ratio"].fillna(recovery_df["Following_Recovery"])

    # denom2 = recovery_df["Following_Count"].replace(0, np.nan)
    # recovery_df["Recovery_Rate"] = (recovery_df["Following_Recovery"] / denom2).fillna(0.0)
    recovery_df["Recovery_Rate"] = (recovery_df["Following_Recovery"] / recovery_df["Following_Count"])

    # Save updated CSV (metrics)
    updated_csv_path = os.path.join(PerformanceEval_folder, f"{inputName}{threshold}_Recovery_Metrics.csv")
    recovery_df.to_csv(updated_csv_path, index=False)
    print(recovery_df)

    # === Plot 1: Recovery Ratio ===
    # keep only methods we want, and in that order (methods missing will be dropped)
    ratio_df = recovery_df.set_index("Method").reindex(method_order_ratio).dropna(subset=["Recovery_Ratio"]).reset_index()

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    plt.bar(
        ratio_df["Method"],
        ratio_df["Recovery_Ratio"],
        color=[colors[m] for m in ratio_df["Method"]]
    )
    plt.xlabel("Methods", fontsize=16)
    plt.ylabel("Recovery Ratio", fontsize=22)
    plt.title(f"{inputName} - Recovery Ratio Comparison", fontsize=26)
    plt.xticks(rotation=45, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path_ratio = os.path.join(PerformanceEval_folder, f"{inputName}{threshold}_Recovery_Ratio.png")
    plt.savefig(plot_path_ratio, dpi=300)
    plt.show()

    # === Plot 2: Recovery Rate + CurrentPr ===
    # Add CurrentPr with fixed value if we have it
    current_val = CURRENT_PR.get(inputName, None)
    if current_val is not None:
        extra_row = pd.DataFrame([{"Method": "CurrentPr", "Recovery_Rate": current_val}])
        rate_df = pd.concat([extra_row, recovery_df[["Method", "Recovery_Rate"]]], ignore_index=True)
    else:
        rate_df = recovery_df[["Method", "Recovery_Rate"]].copy()

    rate_df = rate_df.set_index("Method").reindex(method_order_rate).dropna(subset=["Recovery_Rate"]).reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plt.bar(
        rate_df["Method"],
        rate_df["Recovery_Rate"],
        color=[colors[m] for m in rate_df["Method"]]
    )
    plt.xlabel("Methods", fontsize=16)
    plt.ylabel("Recovery Rate", fontsize=22)
    plt.title(f"{inputName} - Recovery Rate Comparison", fontsize=26)
    plt.xticks(rotation=45, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path_rate = os.path.join(PerformanceEval_folder, f"{inputName}{threshold}_Recovery_Rate.png")
    plt.savefig(plot_path_rate, dpi=300)
    plt.show()

# ---- Run both datasets ----
for ds, outc in DATASETS.items():
    process_dataset(ds, outc)