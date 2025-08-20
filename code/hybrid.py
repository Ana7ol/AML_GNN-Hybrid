# --- 1. IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import math
import argparse

# --- 2. SETUP ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. MODEL ARCHITECTURES ---
class TransactionSequenceEncoder_MLP_GRU(nn.Module):
    def __init__(self, num_features: int, transaction_embedding_dim: int = 64, gru_hidden_size: int = 128):
        super().__init__()
        self.transaction_feature_embedder = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Linear(128, transaction_embedding_dim))
        self.gru = nn.GRU(input_size=transaction_embedding_dim, hidden_size=gru_hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(gru_hidden_size * 2, 1)
        print(f"Initialized TransactionSequenceEncoder_MLP_GRU (Sequence Labeling Version)")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x.shape
        x_flat = x.view(batch_size * seq_len, num_features)
        embedded_transactions = self.transaction_feature_embedder(x_flat)
        embedded_sequence = embedded_transactions.view(batch_size, seq_len, -1)
        gru_output, _ = self.gru(embedded_sequence)
        logits = self.fc_out(gru_output)
        return logits

class DownstreamClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 1))
        print(f"Initialized DownstreamClassifier with input dim {input_dim}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- 4. DATA FUNCTIONS ---
def _process_account_chunk(account_chunk, sequence_length, step_size):
    sequences, sequence_labels, account_ids = [], [], []
    feature_cols = [col for col in account_chunk[0][1].columns if col not in ['Account', 'Is Laundering']]
    for account_id, group in account_chunk:
        if len(group) >= sequence_length:
            for i in range(0, len(group) - sequence_length + 1, step_size):
                sequence_df_slice = group.iloc[i : i + sequence_length]
                sequences.append(sequence_df_slice[feature_cols].values)
                sequence_labels.append(sequence_df_slice['Is Laundering'].values)
                account_ids.append(account_id)
    return sequences, sequence_labels, account_ids

def create_sequences(df_processed: pd.DataFrame, sequence_length: int, step_size: int):
    print("Creating transaction sequences for each account...", flush=True)
    grouped_accounts = list(df_processed.groupby('Account'))
    n_jobs = os.cpu_count() or 1
    n_chunks = n_jobs * 4
    chunk_size = math.ceil(len(grouped_accounts) / n_chunks)
    chunks = [grouped_accounts[i:i + chunk_size] for i in range(0, len(grouped_accounts), chunk_size)]
    print(f"Processing {len(grouped_accounts)} accounts in {len(chunks)} chunks across {n_jobs} cores...", flush=True)
    results = Parallel(n_jobs=n_jobs)(delayed(_process_account_chunk)(chunk, sequence_length, step_size) for chunk in tqdm(chunks, desc="Creating Sequences"))
    all_sequences, all_labels, all_account_ids = [], [], []
    for seq_list, label_list, acc_id_list in results:
        all_sequences.extend(seq_list); all_labels.extend(label_list); all_account_ids.extend(acc_id_list)
    print(f"Created {len(all_sequences)} sequences.", flush=True)
    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int64), all_account_ids

# --- 5. MAIN EXECUTION BLOCK ---
def main(args):
    print(f"Using device: {DEVICE}"); print("--- Running Experiment with Configuration  ---")
    for key, value in vars(args).items(): print(f"  {key}: {value}")
    print("---------------------------------------------------------")

    # --- DATA PREPARATION ---
    print("\n" + "="*50 + "\nDATA PREPARATION\n" + "="*50, flush=True)
    df_raw = pd.read_csv(args.data_path)
    df_processed = df_raw.copy(); cols_to_drop_for_features = ['Account.1', 'Timestamp', 'Amount Paid', 'Payment Currency']
    df_processed.drop(columns=cols_to_drop_for_features, inplace=True, errors='ignore')
    df_processed = pd.get_dummies(df_processed, columns=['Receiving Currency', 'Payment Format'], dummy_na=False, dtype=float)
    feature_cols = [col for col in df_processed.columns if col not in ['Account', 'Is Laundering']]
    df_processed[feature_cols] = StandardScaler().fit_transform(df_processed[feature_cols])

    # --- STAGE 1: MLP-GRU ENCODER ---
    print("\n" + "="*50 + "\nSTAGE 1: MLP-GRU ENCODER\n" + "="*50, flush=True)
    sequences_np, sequence_labels_np, _ = create_sequences(df_processed, args.sequence_length, args.step_size)
    num_features = sequences_np.shape[2]
    
    model = TransactionSequenceEncoder_MLP_GRU(num_features, gru_hidden_size=args.gru_emb).to(DEVICE)

    # --- Training The Model ---
    X = torch.from_numpy(sequences_np).float()
    y = torch.from_numpy(sequence_labels_np).float()
    print("\n--- Splitting Data for Sequence Labeling Task ---", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_positives = y_train.sum(); num_negatives = y_train.numel() - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives], device=DEVICE) if num_positives > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    for epoch in range(1, args.epochs + 1):
        for data, target in tqdm(train_loader, desc=f"Downstream Epoch {epoch}/{args.epochs}"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target.unsqueeze(-1))
            loss.backward()
            optimizer.step()
        print(f"Downstream Epoch {epoch}/{args.epochs}, Final Batch Loss: {loss.item():.4f}", flush=True)

    # --- FINAL EVALUATION ---
    print("\n" + "="*50 + "\nFINAL EVALUATION (MLP-GRU -> MLP)\n" + "="*50, flush=True)
    model.eval(); test_probs_list = []
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(X_test), batch_size=args.batch_size * 2, shuffle=False)
        for batch_x, in tqdm(test_loader, desc="Evaluating on Test Set"):
            logits = model(batch_x.to(DEVICE)); test_probs_list.append(torch.sigmoid(logits).cpu())
    test_probs_tensor = torch.cat(test_probs_list)
    
    y_pred_prob = test_probs_tensor.flatten().numpy()
    y_true = y_test.flatten().numpy()

    # Create the results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    # Define a unique filename based on hyperparameters
    run_name = f"seq-label_tx-emb{args.gru_emb}_lr{args.lr}_epochs{args.epochs}_len{args.sequence_length}_step{args.step_size}"
    results_path = os.path.join(args.results_dir, f"{run_name}_predictions.csv")
    # Create a DataFrame and save it
    print(f"\nSaving per-transaction predictions to: {results_path}", flush=True)
    results_df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_prob})
    results_df.to_csv(results_path, index=False)
    
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    print(f"\nAUROC: {roc_auc_score(y_true, y_pred_prob):.4f}"); print(f"AUPRC: {average_precision_score(y_true, y_pred_prob):.4f}")
    print("\n--- Confusion Matrix ---\n", pd.DataFrame(confusion_matrix(y_true, y_pred_class), index=['Actual Normal', 'Actual Illicit'], columns=['Predicted Normal', 'Predicted Illicit']))
    print("\n--- Classification Report ---\n", classification_report(y_true, y_pred_class, target_names=['Normal', 'Illicit'], zero_division=0))
    print("\nPipeline finished successfully.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP-GRU for Transaction Sequence Labeling")
    
    parser.add_argument('--data_path', type=str, default='LI-Small_Trans.csv', help='Path to the transaction data CSV.')
    parser.add_argument('--results_dir', type=str, default='hybrid_results', help='Directory to save the final prediction results.')
    parser.add_argument('--gru_emb', type=int, default=128, help='Hidden dimension size for the GRU.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of transaction sequences.')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for creating sequences (overlap).')
    parser.add_argument('--force_retrain_encoder', action='store_true', help='Flag to force retraining. Currently unused in code but prevents script from crashing.')

    args = parser.parse_args()
    main(args)
