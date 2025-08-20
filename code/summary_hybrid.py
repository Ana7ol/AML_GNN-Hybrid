import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm

def find_best_f1_threshold(y_true, y_pred_proba):
    """
    Find the optimal threshold to maximize the F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # The last precision is 1, last recall is 0, which corresponds to no threshold.
    # We need to slice them to match the length of the thresholds array.
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    
    # Handle the case where precision + recall is 0, resulting in NaN
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find the index of the best F1 score
    best_f1_idx = np.argmax(f1_scores)
    
    # Get the corresponding values
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    precision_at_best_f1 = precisions[best_f1_idx]
    recall_at_best_f1 = recalls[best_f1_idx]
    
    return best_f1, best_threshold, precision_at_best_f1, recall_at_best_f1

def main():
    """
    Main function to process result files and generate a summary.
    """
    results_dir = 'hybrid_results/'
    summary_filename = 'hybrid_model_summary_auprc.csv'
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    # Find all prediction CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_predictions.csv')]
    
    if not csv_files:
        print(f"No prediction CSV files found in '{results_dir}'.")
        return

    all_results = []
    print(f"Found {len(csv_files)} result files. Processing...")

    # Regex to parse hyperparameters from the filename
    # Example: seq-label_tx-emb128_lr0.0001_epochs20_len10_step10_predictions.csv
    pattern = re.compile(r"tx-emb(\d+)_lr([\d.]+)_epochs(\d+)_len(\d+)_step(\d+)")

    for filename in tqdm(csv_files, desc="Summarizing Experiments"):
        match = pattern.search(filename)
        if not match:
            print(f"Warning: Could not parse hyperparameters from filename: {filename}")
            continue
            
        # Extract hyperparameters
        tx_emb, lr, epochs, seq_len, step_size = match.groups()
        
        # Load the prediction data
        file_path = os.path.join(results_dir, filename)
        df = pd.read_csv(file_path)
        y_true = df['y_true'].values
        y_pred_proba = df['y_pred_proba'].values
        
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        # Find best F1 and corresponding P/R
        best_f1, best_thresh, prec_at_f1, recall_at_f1 = find_best_f1_threshold(y_true, y_pred_proba)
        
        # Store results in a dictionary
        result_entry = {
        'experiment': filename.replace('_predictions.csv', ''),
        'tx_emb': int(tx_emb),
        'lr': float(lr),
        'epochs': int(epochs),
        'seq_len': int(seq_len),
        'step_size': int(step_size),
        'auroc': auroc,
        'auprc': auprc,
        'best_f1': best_f1,
        'best_threshold': best_thresh,
        'precision_at_best_f1': prec_at_f1,
        'recall_at_best_f1': recall_at_f1,
        }
        all_results.append(result_entry)
    # Create the final summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # Sort by the primary metric (AUPRC) in descending order
    summary_df = summary_df.sort_values(by='auprc', ascending=False)
    
    # Save to CSV
    summary_df.to_csv(summary_filename, index=False)
    
    print("\n" + "="*50)
    print(f"Summary successfully created at: {summary_filename}")
    print("Top 5 performing experiments (by AUPRC):")
    print(summary_df.head())
    print("="*50)

if __name__ == '__main__':
    main()
