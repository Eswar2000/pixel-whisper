import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === CONFIG ===
CSV_FOLDER = "summary"  # folder with your CSV files
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD ALL CSVs ===
csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {CSV_FOLDER}")

df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# === 1. BAR PLOTS: PSNR, SSIM, MSE, Entropy Diff ===
metrics = ["PSNR", "SSIM", "MSE", "Entropy Diff"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Phase", y=metric, data=df, ci=None, estimator=np.mean)
    plt.title(f"{metric} Comparison Across Phases (mean over images)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{metric}_barplot.png"))
    plt.close()

# === 2. SCATTER PLOT: PSNR vs Entropy Diff ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Entropy Diff", y="PSNR", hue="Phase", data=df, s=80)
plt.title("PSNR vs Entropy Difference by Phase")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "PSNR_vs_EntropyDiff_scatter.png"))
plt.close()

# === 3. RADAR PLOTS ===
def radar_plot(data, metrics, title, save_path, color="red"):
    labels = metrics
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, data, color=color, linewidth=2)
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

metrics_for_radar = ["PSNR", "SSIM", "MSE", "Entropy Diff"]

# Raw value radar plots
for phase in df["Phase"].unique():
    phase_df = df[df["Phase"] == phase]
    mean_values = [phase_df[m].mean() for m in metrics_for_radar]
    radar_plot(
        mean_values,
        metrics_for_radar,
        f"Radar Plot (Raw) - {phase}",
        os.path.join(OUTPUT_FOLDER, f"Radar_Raw_{phase}.png"),
        color="red"
    )

# Normalized radar plots
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[metrics_for_radar] = scaler.fit_transform(df_norm[metrics_for_radar])

for phase in df_norm["Phase"].unique():
    phase_df = df_norm[df_norm["Phase"] == phase]
    mean_values = [phase_df[m].mean() for m in metrics_for_radar]
    radar_plot(
        mean_values,
        metrics_for_radar,
        f"Radar Plot (Normalized) - {phase}",
        os.path.join(OUTPUT_FOLDER, f"Radar_Normalized_{phase}.png"),
        color="blue"
    )

print(f"[+] Plots saved in: {OUTPUT_FOLDER}")