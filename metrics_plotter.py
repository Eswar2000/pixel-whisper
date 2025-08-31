import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

METRICS_DIR = "summary"
STEGANALYSIS_DIR = "results"
PLOTS_DIR = "images/plot"

PHASE_MAP = {
        "phase1": "LSB",
        "phase3": "Chaotic LSB",
        "phase4_1": "Content Aware Chaotic LSB",
        "phase4_2": "Chaotic Channel-Adaptive LSB with Compensation",
        "phase4_3": "Fixed-Channel LSB with Cross-Channel Compensation",
        "phase4_4": "Edge-Chaotic LSB with Cross-Channel Compensation",
        "phase4_5": "Adaptive Chaotic Multi-Map Embedding"
    }

def read_csv_files(dir):
    csv_files = glob.glob(os.path.join(dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {dir}")
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def merge_dataframes(df1, df2):
    df1['File name'] = df1['Cover Image'].str.replace('.jpg', '', regex=False) + '_' + df1['Phase'] + '.png'
    merged_df = pd.merge(df1, df2, on='File name', how='inner')
    return merged_df

def prepare_ablation_table(df, group_col='Phase', exclude_cols=['Image']):
    df_abl = df.copy()
    df_abl = df_abl[df_abl[group_col] != 'phase2'] # Phase 2 is encryption of data and embedding
    # Get metric columns (all numeric columns except excluded ones)
    metric_cols = [col for col in df_abl.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_abl[col])]

    # Calculate mean and std for each metric grouped by algorithm
    mean_df = df_abl.groupby(group_col)[metric_cols].mean().round(5)
    std_df = df_abl.groupby(group_col)[metric_cols].std().round(7)
    
    # Create formatted table with mean ± std
    ablation_table = pd.DataFrame()
    for col in metric_cols:
        ablation_table[f"{col}"] = mean_df[col].astype(str) + " ± " + std_df[col].astype(str)
    
    # Reset index to make algorithm a column
    ablation_table = ablation_table.reset_index()
    
    return ablation_table

def prepare_radar_plot(df, cols, index='Phase'):
    radar_df = df.copy()
    radar_df = radar_df[radar_df[index] != 'phase2'] # Phase 2 is encryption of data and embedding
    radar_df = radar_df[[index] + cols]

    # Better plotting experience with proper legend
    radar_df[index] = radar_df[index].map(PHASE_MAP)

    mean_df = radar_df.groupby(index)[cols].mean()

    normalized_df = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min())
    categories = normalized_df.columns.tolist()
    N = len(categories)

    # Angles for axes (one per variable)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    # --- Step 4: Plot ---
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw=dict(polar=True))

    # Plot each algorithm
    for idx, row in normalized_df.iterrows():
        values = row.tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.1)

    # Set up the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Title and legend
    ax.set_title('Steganography Metric Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_DIR, 'steganography_metric_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def prepare_attack_metric_comparison(dir, distinct_cols = ['image_name', 'algorithm']):
    df = read_csv_files(dir)
    df.drop_duplicates(distinct_cols, inplace=True)
    metric_cols = [col for col in df.columns if col not in distinct_cols]

    # Calculate mean and std for each metric grouped by algorithm
    mean_df = df.groupby(['algorithm'])[metric_cols].mean().round(6)
    std_df = df.groupby(['algorithm'])[metric_cols].std().round(6)

    # Plot resilience error bar
    plt.figure(figsize=(12, 8))
    for algorithm in mean_df.index:
        plt.errorbar(metric_cols, mean_df.loc[algorithm], yerr=std_df.loc[algorithm], label=algorithm, capsize=4, marker='o')
    
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Attack Resilience of Steganography Algorithms")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig('images/plot/attack_resilience.png', dpi=300)
    plt.close()

    # Create formatted table with mean ± std
    ablation_table = pd.DataFrame()
    for col in metric_cols:
        ablation_table[f"{col}"] = mean_df[col].astype(str) + " ± " + std_df[col].astype(str)
    
    # Reset index to make algorithm a column
    ablation_table = ablation_table.reset_index()

    # Categorizing attack metrics
    compression_metric_cols = ['jpeg90']
    noise_metric_cols = ['noise_sigma2', 'salt_and_pepper', 'speckle']
    filtering_metric_cols = ['blur', 'median', 'bilateral', 'non_local_means', 'motion_blur']
    color_intensity_metric_cols = ['gamma09', 'histogram']
    
    # Print metrics as a table
    print(f"---------------- Compression Metrics ----------------\n{ablation_table[['algorithm'] + compression_metric_cols].to_markdown()}")
    print(f"---------------- Noise Metrics ----------------\n{ablation_table[['algorithm'] + noise_metric_cols].to_markdown()}")
    print(f"---------------- Filtering Metrics ----------------\n{ablation_table[['algorithm'] + filtering_metric_cols].to_markdown()}")
    print(f"---------------- Color Intensity Metrics ----------------\n{ablation_table[['algorithm'] + color_intensity_metric_cols].to_markdown()}")

if __name__ == "__main__":
    metric_df_cols = ['Cover Image', 'Phase', 'PSNR', 'SSIM', 'MSE', 'Entropy Diff', 'NCC', 'Noise Diff', 'Histogram Dist', 'Edge Diff', 'Skewness Diff', 'Kurtosis Diff']
    metrics_df = read_csv_files(METRICS_DIR)
    steganalysis_df_cols = ['Primary Sets', 'Chi Square', 'Sample Pairs', 'RS analysis']
    steganalysis_df = read_csv_files(STEGANALYSIS_DIR)

    print(f"Shape of Metrics Dataframe: {metrics_df.shape}")
    print(f"Shape of Steganalysis Dataframe: {steganalysis_df.shape}")

    df = merge_dataframes(metrics_df, steganalysis_df)
    print(f"Shape of Merged Dataframe: {df.shape}")

    df_cols = metric_df_cols + steganalysis_df_cols
    df = df[df_cols]

    ablation_table = prepare_ablation_table(df)

    print(f"Final Dataframe Shape: {ablation_table.shape}")
    quality_metric_cols = ['Phase', 'PSNR', 'SSIM', 'MSE', 'Entropy Diff']
    stat_similarity_cols = ['Phase', 'NCC', 'Noise Diff', 'Histogram Dist', 'Edge Diff', 'Skewness Diff', 'Kurtosis Diff']
    steg_cols = ['Phase', 'Primary Sets', 'Chi Square', 'Sample Pairs', 'RS analysis']
    
    print(f"---------------- Quality Metrics ----------------\n{ablation_table[quality_metric_cols].to_markdown()}")
    print(f"---------------- Statistical Similarity Metrics ----------------\n{ablation_table[stat_similarity_cols].to_markdown()}")
    print(f"---------------- Steganalysis Metrics ----------------\n{ablation_table[steg_cols].to_markdown()}")

    prepare_radar_plot(df, cols=['PSNR', 'Entropy Diff', 'Histogram Dist', 'Edge Diff', 'Chi Square', 'RS analysis'])

    prepare_attack_metric_comparison("outcomes")