import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def survival_curves(fullResultFolder, fold_nbr, prefileName, postfileName,
                    method_name, output_plot_path=None):
    # Find files
    pattern = os.path.join(fullResultFolder, f"{prefileName}*{postfileName}.csv")
    files = sorted(glob.glob(pattern))[:fold_nbr]

    if len(files) == 0:
        print(f"❌ No survival files found in {fullResultFolder}")
        return None

    df_list = [pd.read_csv(f) for f in files]
    data = pd.concat(df_list, ignore_index=True)

    summary_data_path = os.path.join(fullResultFolder, f"{method_name}_all_data.csv")
    data.to_csv(summary_data_path, index=False)

    # Find time and event columns

    if inputName == "DUKE":
        event_col_candidates = [c for c in data.columns if "rfs_event" in c]
        time_col_candidates = [c for c in data.columns if "rfs_time" in c ]

    if not time_col_candidates or not event_col_candidates:
        raise ValueError("Time or Event column not found!")

    time_col = time_col_candidates[0]
    event_col = event_col_candidates[0]

    # Convert to numeric
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[event_col] = pd.to_numeric(data[event_col], errors="coerce")

    # Drop rows with missing values
    data = data.dropna(subset=[time_col, event_col])

    followed = data[data["FOLLOW_REC"] == 1]
    not_followed = data[data["FOLLOW_REC"] == 0]

    # Check if groups are empty
    if followed.empty or not_followed.empty:
        print(f"⚠️ One group is empty (Followed={len(followed)}, NotFollowed={len(not_followed)}). Skipping KM plot.")
        return None

    # Fit Kaplan-Meier models
    km_followed = KaplanMeierFitter().fit(followed[time_col], followed[event_col], label="Followed")
    km_not_followed = KaplanMeierFitter().fit(not_followed[time_col], not_followed[event_col], label="Not Followed")

    # Log-rank test for p-value
    logrank_res = logrank_test(
        followed[time_col], not_followed[time_col],
        event_observed_A=followed[event_col],
        event_observed_B=not_followed[event_col]
    )
    p_value = logrank_res.p_value

    # Plot curves with confidence intervals
    plt.figure(figsize=(10, 6))
    km_followed.plot(ci_show=True, linewidth=2)
    km_not_followed.plot(ci_show=True, linewidth=2)

    plt.title(f"{method_name} (p = {p_value:.4f})", fontsize=32)
    plt.xlabel("Time (Days)",fontsize=20)
    plt.ylabel("Recurrence-Free Survival Probability", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()

    # ✅ Save or show plot
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ KM plot saved to {output_plot_path}")
    else:
        plt.show()

    return p_value


# ============================
# ✅ MAIN PIPELINE
# ============================

methods = ["LogrRe", "LogrRa", "SvmRe", "SvmRa", "RfRe", "RfRa", "CTR"]
fold_nbr = 5
threshold = -999
baseInFolder = "./output/"
testName = "test"

inputName = "DUKE"

# ✅ Create PerformanceEval folder
PerformanceEval_folder = os.path.join(baseInFolder, "PerformanceEval")
os.makedirs(PerformanceEval_folder, exist_ok=True)

results_summary = []

for method in methods:
    prefileName = f"{inputName}_{testName}_{method}_{threshold}_"
    postfileName = "follow"
    fullResultFolder = os.path.join(baseInFolder, method, inputName)

    # Define plot file path
    plot_filename = f"{inputName}{method}{threshold}_rfs_KM_plot.png"
    plot_path = os.path.join(PerformanceEval_folder, plot_filename)

    p_value = survival_curves(
        fullResultFolder, fold_nbr, prefileName, postfileName,
         method_name=method, output_plot_path=plot_path
    )

    results_summary.append({
        "Method": method,
        "p_value": p_value
    })

# ✅ Save summary table
summary_df = pd.DataFrame(results_summary)
summary_csv_path = os.path.join(PerformanceEval_folder, f"{inputName}{threshold}_rfs_p_value_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"\n✅ Summary table saved to {summary_csv_path}")
print(summary_df)
