# run_benchmark_mlp.py

# --- 1. IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os
import argparse
from typing import Tuple

# --- 2. SETUP ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. REUSABLE COMPONENTS (Copied from main pipeline for consistency) ---

class DownstreamClassifier(nn.Module):
    """A simple MLP for the final downstream classification task."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        print(f"Initialized DownstreamClassifier with input dim {input_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Loads and preprocesses the transaction data from a CSV file."""
    print("\n" + "="*50 + "\nDATA PREPARATION\n" + "="*50, flush=True)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df_raw = pd.read_csv(file_path)
    class_counts = df_raw['Is Laundering'].value_counts()
    print(f"Normal Transactions (0): {class_counts.get(0, 0)}\nIllicit Transactions (1): {class_counts.get(1, 0)}\n" + "-"*45, flush=True)

    cols_to_drop = ['Is Laundering', 'Account', 'Account.1', 'Timestamp', 'Amount Paid', 'Payment Currency']
    df_features = pd.get_dummies(df_raw.drop(columns=cols_to_drop, errors='ignore'), columns=['Receiving Currency', 'Payment Format'], dummy_na=False, dtype=float)
    
    X_features_np = StandardScaler().fit_transform(df_features)
    y_labels_np = df_raw['Is Laundering'].values

    print(f"Data loaded. Feature shape: {X_features_np.shape}, Label shape: {y_labels_np.shape}")
    return X_features_np, y_labels_np, class_counts

def print_evaluation_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, model_name: str):
    """Prints a standard set of classification evaluation metrics."""
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    
    print(f"\n--- Final Evaluation ({model_name}) ---", flush=True)
    print(f"AUROC: {roc_auc_score(y_true, y_pred_prob):.4f}")
    print(f"AUPRC: {average_precision_score(y_true, y_pred_prob):.4f}")
    
    print(f"\n--- Confusion Matrix ({model_name}) ---", flush=True)
    cm = confusion_matrix(y_true, y_pred_class)
    print(pd.DataFrame(cm, index=['Actual Normal', 'Actual Illicit'], columns=['Predicted Normal', 'Predicted Illicit']))
    
    print(f"\n--- Classification Report ({model_name}) ---", flush=True)
    print(classification_report(y_true, y_pred_class, target_names=['Normal', 'Illicit'], zero_division=0))

# --- 4. MAIN EXECUTION BLOCK ---

def main(args: argparse.Namespace):
    """Main function to run the MLP benchmark pipeline."""
    print("Running MLP Benchmark...")
    print("This script trains a simple MLP on the raw transaction features ONLY.")
    print("It serves as a baseline to evaluate the performance gain from the GNN model.")
    print("Using device:", DEVICE)
    print("Current configuration:", args)

    # --- DATA PREP ---
    # This uses the exact same initial feature generation as the GNN pipeline
    X_features_np, y_labels_np, class_counts = load_and_preprocess_data(args.data_path)
    X_features_tensor = torch.tensor(X_features_np, dtype=torch.float32)
    y_labels_tensor = torch.tensor(y_labels_np, dtype=torch.float32).unsqueeze(1)

    # --- DATA SPLIT ---
    # CRITICAL: Use the same random_state and stratify as the GNN pipeline for a fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X_features_tensor, y_labels_tensor, test_size=0.3, random_state=42, stratify=y_labels_tensor
    )
    print(f"\nData split into training and testing sets.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- MODEL TRAINING ---
    print("\n" + "="*50 + "\nMLP BENCHMARK TRAINING\n" + "="*50, flush=True)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Use the same DownstreamClassifier, but on the raw features
    classifier_nn = DownstreamClassifier(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(classifier_nn.parameters(), lr=args.lr)
    
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=DEVICE)
    print(f"Using pos_weight for the rare class: {pos_weight.item():.2f}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.epochs + 1):
        classifier_nn.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = classifier_nn(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"MLP Benchmark Epoch {epoch:03d}/{args.epochs}, Avg Loss: {total_loss/len(train_loader):.4f}", flush=True)

    # --- EVALUATION ---
    classifier_nn.eval()
    with torch.no_grad():
        test_preds = classifier_nn(X_test.to(DEVICE))
        test_probs_nn = torch.sigmoid(test_preds).cpu().numpy()
        
    print_evaluation_metrics(y_test.numpy(), test_probs_nn, "MLP Benchmark")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP Benchmark for Fraud Detection")

    parser.add_argument('--data_path', type=str, default='LI-Small_Trans.csv', help='Path to the transaction data CSV file.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the MLP classifier.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training.')
    
    args = parser.parse_args()
    main(args)
