import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal, getcontext, InvalidOperation

def extract_algorithm(file_name):
    # Assumes the pattern: <image_name>_phase<algorithm>.png
    match = re.search(r'_phase([^.]+)\.png$', file_name)
    return match.group(1) if match else 'Unknown'

def safe_decimal(x):
    try:
        return Decimal(x)
    except (InvalidOperation, ValueError):
        return Decimal('0.0')  # or use None if you want to filter later

# Set decimal precision
getcontext().prec = 10

# Load CSV with high float precision
def load_high_precision_csv(filepath):
    df = pd.read_csv(filepath, converters={
        'Primary Sets': safe_decimal,
        'Chi Square': safe_decimal,
        'Sample Pairs': safe_decimal,
        'RS analysis': safe_decimal,
        'Fusion (mean)': safe_decimal
    })

    # Extract algorithm from the filename
    df['Algorithm'] = df['File name'].apply(extract_algorithm)
    return df

def compute_algorithm_summary(df, threshold=Decimal('0.2')):
    summary = df.groupby('Algorithm').apply(lambda group: pd.Series({
        'Files analyzed': len(group),
        'Detected (True)': group['Above stego threshold?'].sum(),
        'Detection rate (%)': 100 * group['Above stego threshold?'].mean(),
        'Avg Fusion': float(sum(group['Fusion (mean)']) / len(group)),
        'Max Fusion': float(max(group['Fusion (mean)'])),
        'Avg Secret Size (bytes)': group['Secret message size in bytes (ignore for clean files)']
            .replace('?', 0)
            .astype(float).mean()
    })).reset_index()
    
    return summary.sort_values(by='Detection rate (%)')

# Main runner
if __name__ == "__main__":
    # Replace with your file path
    csv_file = "images/steganalysis1_4.csv"
    
    df = load_high_precision_csv(csv_file)
    summary_df = compute_algorithm_summary(df)
    print(summary_df)
    # plot_stegexpose_metrics(df)