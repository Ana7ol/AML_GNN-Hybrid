import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm
import glob

def analyze_gnn_results(results_dir):
    """
    Scans the GNN results directory, parses hyperparameters from directory names,
    finds the best epoch for each experiment based on AUPRC, and returns a summary DataFrame.
    """
    all_results = []
    
    # Regex tailored to your GNN's experiment directory names:
    # final_GNN_k100_L5_emb64x128_lr0.0001
    pattern = re.compile(
        r"k(?P<k_history>\d+)_"
        r"L(?P<gnn_layers>\d+)_"
        r"emb(?P<acc_emb>\d+)x(?P<tx_emb>\d+)_"
        r"lr(?P<lr>[\d.]+)"
    )

    print(f"Scanning directory: {results_dir}")
    experiment_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    if not experiment_dirs:
        print("No experiment directories found.")
        return None

    for exp_name in tqdm(experiment_dirs, desc="Processing GNN experiments"):
        match = pattern.search(exp_name)
        
        if "k1000" in exp_name or "k150" in exp_name or "k250" in exp_name:
            print(f"  - INFO: Explicitly skipping directory: {exp_name}")
            continue


        if not match:
            print(f"  - Warning: Skipping directory with non-matching name: {exp_name}")
            continue

        # --- 1. Parse Hyperparameters ---
        params = match.groupdict()
        # Convert to correct numeric types
        for key in ['k_history', 'gnn_layers', 'acc_emb', 'tx_emb']:
            params[key] = int(params[key])
        params['lr'] = float(params['lr'])

        exp_path = os.path.join(results_dir, exp_name)
        epoch_files = glob.glob(os.path.join(exp_path, "epoch_*_results.csv"))

        if not epoch_files:
            continue
            
        best_auprc = -1
        best_epoch_metrics = {}

        # --- 2. Find the Best Epoch based on AUPRC ---
        for epoch_file in epoch_files:
            try:
                df = pd.read_csv(epoch_file)
                if df.empty or len(df['y_true'].unique()) < 2:
                    continue

                auprc = average_precision_score(df['y_true'], df['y_pred_proba'])
                if auprc > best_auprc:
                    best_auprc = auprc
                    
                    # Store all metrics from this best epoch
                    y_true = df['y_true'].values
                    y_pred_proba = df['y_pred_proba'].values
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    best_f1_idx = np.argmax(f1_scores[:-1])
                    
                    best_epoch_metrics = {
                        'auprc': best_auprc,
                        'auroc': roc_auc_score(y_true, y_pred_proba),
                        'best_f1': f1_scores[best_f1_idx],
                        'precision_at_best_f1': precision[best_f1_idx],
                        'recall_at_best_f1': recall[best_f1_idx]
                    }
            except Exception as e:
                print(f"  - Error processing file {epoch_file}: {e}")
        
        if best_epoch_metrics:
            params.update(best_epoch_metrics)
            all_results.append(params)

    if not all_results:
        print("\nNo valid GNN result files were successfully processed. Exiting.")
        return None
        
    return pd.DataFrame(all_results)

def create_dashboard(df):
    """
    Generates the 6-panel hyperparameter performance dashboard for the GNN model.
    """
    if df is None or df.empty:
        return
        
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('GNN Model: Hyperparameter Performance Dashboard', fontsize=24, weight='bold')

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
        'k_history': 'K-History Size',
        'gnn_layers': 'GNN Layers',
        'acc_emb': 'Account Embedding Dim',
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
    
    dashboard_path = 'gnn_model_performance_dashboard.png'
    plt.savefig(dashboard_path, dpi=300)
    print(f"\nDashboard saved to: {dashboard_path}")
    plt.show()

if __name__ == '__main__':
    # The GNN script saves its results in the 'results' directory
    RESULTS_DIRECTORY = 'results'
    
    summary_df = analyze_gnn_results(RESULTS_DIRECTORY)
    
    if summary_df is not None:
        print("\n--- GNN Experiment Summary (Ranked by AUPRC) ---")
        print(summary_df.head(10))
        create_dashboard(summary_df)
