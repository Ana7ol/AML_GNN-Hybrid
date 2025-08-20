import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

def analyze_sequential_results(results_dir):
    """
    Scans a directory for the sequential model's prediction CSVs,
    parses hyperparameters from filenames, calculates metrics,
    and returns a summary DataFrame.
    """
    all_results = []
    
    # Regex tailored to your exact filenames:
    # seq-label_tx-emb128_lr0.0001_epochs20_len10_step10_predictions.csv
    pattern = re.compile(
        r"seq-label_"
        r"tx-emb(?P<tx_emb>\d+)_"
        r"lr(?P<lr>[\d.]+)_"
        r"epochs(?P<epochs>\d+)_"
        r"len(?P<seq_len>\d+)_"
        r"step(?P<step_size>\d+)_"
        r"predictions.csv"
    )

    print(f"Scanning directory: {results_dir}")
    result_files = [f for f in os.listdir(results_dir) if f.endswith("_predictions.csv")]

    if not result_files:
        print("No prediction files found in the directory.")
        return None

    for filename in tqdm(result_files, desc="Processing experiment files"):
        match = pattern.match(filename)
        if not match:
            print(f"  - Warning: Skipping file with non-matching name: {filename}")
            continue

        # --- 1. Parse Hyperparameters ---
        params = match.groupdict()
        # Convert to correct numeric types
        for key in ['tx_emb', 'epochs', 'seq_len', 'step_size']:
            params[key] = int(params[key])
        params['lr'] = float(params['lr'])

        # --- 2. Load Data and Calculate Metrics ---
        file_path = os.path.join(results_dir, filename)
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df['y_true'].unique()) < 2:
                print(f"  - Warning: Skipping empty or single-class file: {filename}")
                continue

            y_true = df['y_true'].values
            y_pred_proba = df['y_pred_proba'].values

            # Calculate AUPRC (the main metric)
            params['auprc'] = average_precision_score(y_true, y_pred_proba)
            
            # Calculate the best F1-score and its corresponding Precision/Recall
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            # Add a small epsilon to avoid division by zero
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            best_f1_idx = np.argmax(f1_scores[:-1]) # Exclude the last value which can be ill-defined
            
            params['best_f1'] = f1_scores[best_f1_idx]
            params['precision_at_best_f1'] = precision[best_f1_idx]
            params['recall_at_best_f1'] = recall[best_f1_idx]
            
            all_results.append(params)
        except Exception as e:
            print(f"  - Error processing file {filename}: {e}")

    if not all_results:
        print("\nNo valid result files were successfully processed. Exiting.")
        return None
        
    return pd.DataFrame(all_results)

def create_dashboard(df):
    """
    Generates the 6-panel hyperparameter performance dashboard.
    """
    if df is None or df.empty:
        return
        
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Sequential Model: Hyperparameter Performance Dashboard', fontsize=24, weight='bold')

    df = df.sort_values('auprc', ascending=False)
    
    # --- 1. Practical Operating Points (Precision vs. Recall) ---
    ax = axes[0, 0]
    sns.scatterplot(
        data=df, x='recall_at_best_f1', y='precision_at_best_f1', hue='auprc',
        size='auprc', sizes=(50, 500), palette='viridis', ax=ax
    )
    ax.set_title('Practical Operating Points (at Best F1-Score)', fontsize=16)
    ax.set_xlabel('Recall at Best F1', fontsize=14)
    ax.set_ylabel('Precision at Best F1', fontsize=14)
    ax.legend(title='AUPRC')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- Boxplots for Key Hyperparameters ---
    # Define which hyperparameters to plot
    hyperparams_to_plot = {
        'tx_emb': 'GRU Hidden Dimension',
        'seq_len': 'Sequence Length',
        'step_size': 'Step Size',
        'lr': 'Learning Rate'
    }
    
    plot_axes = [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]
    palettes = ['BuGn', 'OrRd', 'PuBu', 'RdPu']

    for i, (param_key, param_name) in enumerate(hyperparams_to_plot.items()):
        ax = plot_axes[i]
        sns.boxplot(data=df, x=param_key, y='auprc', ax=ax, palette=palettes[i])
        sns.stripplot(data=df, x=param_key, y='auprc', ax=ax, color=".3")
        ax.set_title(f'Performance vs. {param_name}', fontsize=16)
        ax.set_xlabel(f'{param_name} Value', fontsize=14)
        ax.set_ylabel('Best AUPRC', fontsize=14)
    
    # --- Overall AUPRC Score Distribution ---
    ax = axes[2, 1]
    sns.histplot(data=df, x='auprc', ax=ax, color='crimson', bins=10)
    ax.set_title('Overall AUPRC Score Distribution', fontsize=16)
    ax.set_xlabel('AUPRC Score', fontsize=14)
    ax.set_ylabel('Count of Experiments', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    dashboard_path = 'sequential_model_performance_dashboard.png'
    plt.savefig(dashboard_path, dpi=300)
    print(f"\nDashboard saved to: {dashboard_path}")
    plt.show()

if __name__ == '__main__':
    RESULTS_DIRECTORY = 'hybrid_results'
    
    summary_df = analyze_sequential_results(RESULTS_DIRECTORY)
    
    if summary_df is not None:
        print("\n--- Experiment Summary (Ranked by AUPRC) ---")
        print(summary_df.head(10))
        create_dashboard(summary_df)
