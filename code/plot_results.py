# plot_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import confusion_matrix

def analyze_epoch_file(filepath, thresholds_to_test):
    """
    Reads a single epoch results file and calculates metrics for various thresholds.
    """
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        return []

    y_true = df['y_true'].values
    y_pred_proba = df['y_pred_proba'].values

    results = []
    for th in thresholds_to_test:
        y_pred_binary = (y_pred_proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        results.append({
            'threshold': th,
            'TN': tn,
            'FP': fp,
            'TP': tp,
            'FN': fn
        })
    return results

def plot_experiment(exp_name, all_epoch_data):
    """
    Generates and saves plots for a single experiment run.
    """
    df = pd.DataFrame(all_epoch_data)
    
    # Let's find the best epoch based on the lowest FP at a high confidence threshold (e.g., 0.9)
    best_epoch_df = df[df['threshold'] == 0.9]
    if not best_epoch_df.empty:
        best_epoch = best_epoch_df.loc[best_epoch_df['FP'].idxmin()]['epoch']
        print(f"\n--- Analyzing Experiment: {exp_name} ---")
        print(f"Best performing epoch seems to be {best_epoch} (based on lowest FP at 0.9 threshold).")
        
        # Filter data for the best epoch to make our plots
        plot_df = df[df['epoch'] == best_epoch]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Metrics vs. Threshold for {exp_name} (Best Epoch: {best_epoch})', fontsize=16)

        # Plot 1: True Negatives vs. Threshold
        ax1.plot(plot_df['threshold'], plot_df['TN'], marker='o', linestyle='-', color='g')
        ax1.set_title('True Negatives vs. Decision Threshold')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Count of True Negatives')
        ax1.grid(True)

        # Plot 2: False Positives vs. Threshold
        ax2.plot(plot_df['threshold'], plot_df['FP'], marker='o', linestyle='-', color='r')
        ax2.set_title('False Positives vs. Decision Threshold')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Count of False Positives')
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plot_filename = f"plot_{exp_name}.png"
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")
        plt.close()


if __name__ == "__main__":
    results_dir = "results"
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    # Find all experiment directories
    experiment_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for exp_name in experiment_dirs:
        exp_path = os.path.join(results_dir, exp_name)
        epoch_files = glob.glob(os.path.join(exp_path, "epoch_*_results.csv"))
        
        if not epoch_files:
            continue

        all_results_for_exp = []
        for f in sorted(epoch_files):
            epoch_num = int(os.path.basename(f).split('_')[1])
            epoch_metrics = analyze_epoch_file(f, thresholds)
            for metric_row in epoch_metrics:
                metric_row['epoch'] = epoch_num
                all_results_for_exp.append(metric_row)

        if all_results_for_exp:
            plot_experiment(exp_name, all_results_for_exp)
