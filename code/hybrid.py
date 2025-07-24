# --- 1. IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os
import argparse
from typing import Tuple

# --- 2. SETUP ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. MODEL ARCHITECTURES ---

class TransactionFeatureEncoder(nn.Module):
    """Encodes raw transaction features into a dense embedding."""
    def __init__(self, num_features: int, final_embedding_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, final_embedding_dim)
        )
        print(f"Initialized TransactionFeatureEncoder with input dim {num_features} -> output dim {final_embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EncoderWithHead(nn.Module):
    """Wrapper for the encoder with a classification head for pre-training."""
    def __init__(self, num_features: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = TransactionFeatureEncoder(num_features, embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)
        print("Initialized training model: EncoderWithHead")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

class LinkPredictorGNN(nn.Module):
    """A GraphSAGE model for learning node embeddings via link prediction."""
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder_conv1 = SAGEConv(in_channels, hidden_dim)
        self.encoder_conv2 = SAGEConv(hidden_dim, hidden_dim)
        decoder_in_dim = 2 * hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        print(f"Initialized LinkPredictorGNN with input dim: {in_channels}, hidden dim: {hidden_dim}")

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv1(x, edge_index).relu()
        x = self.encoder_conv2(x, edge_index)
        return x

    def decode(self, h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        return self.decoder(torch.cat([src_emb, dst_emb], dim=-1))

class DownstreamClassifier(nn.Module):
    """A simple MLP for the final downstream classification task."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        print(f"Initialized DownstreamClassifier with input dim {input_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- 4. DATA & TRAINING FUNCTIONS ---

def load_and_preprocess_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    """Loads and preprocesses the transaction data from a CSV file."""
    print("\n" + "="*50 + "\nDATA PREPARATION\n" + "="*50, flush=True)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df_raw = pd.read_csv(file_path)
    df_gnn = df_raw.copy()
    class_counts = df_gnn['Is Laundering'].value_counts()
    print(f"Normal Transactions (0): {class_counts.get(0, 0)}\nIllicit Transactions (1): {class_counts.get(1, 0)}\n" + "-"*45, flush=True)

    cols_to_drop = ['Is Laundering', 'Account', 'Account.1', 'Timestamp', 'Amount Paid', 'Payment Currency']
    df_features = pd.get_dummies(df_gnn.drop(columns=cols_to_drop, errors='ignore'), columns=['Receiving Currency', 'Payment Format'], dummy_na=False, dtype=float)
    
    X_features_np = StandardScaler().fit_transform(df_features)
    y_labels_np = df_gnn['Is Laundering'].values

    print(f"Data loaded. Feature shape: {X_features_np.shape}, Label shape: {y_labels_np.shape}")
    return X_features_np, y_labels_np, df_gnn, class_counts

def train_or_load_encoder(args: argparse.Namespace, X_features: np.ndarray, y_labels: np.ndarray, class_counts: pd.Series) -> torch.Tensor:
    """Trains or loads a pre-trained transaction feature encoder."""
    print("\n" + "="*50 + "\nSTAGE 1 & 2: TRANSACTION ENCODER & EMBEDDING GENERATION\n" + "="*50, flush=True)
    
    encoder_path = args.encoder_path
    if not os.path.exists(encoder_path) or args.force_retrain_encoder:
        print(f"Training new transaction encoder (force_retrain_encoder={args.force_retrain_encoder}).", flush=True)
        tx_encoder_with_head = EncoderWithHead(num_features=X_features.shape[1], embedding_dim=args.tx_emb).to(DEVICE)
        
        ssl_dataset = TensorDataset(torch.from_numpy(X_features).float(), torch.from_numpy(y_labels).float().unsqueeze(1))
        ssl_loader = DataLoader(ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        ssl_optimizer = torch.optim.Adam(tx_encoder_with_head.parameters(), lr=args.lr)
        
        pos_weight_ssl = torch.tensor([class_counts[0] / class_counts[1]], device=DEVICE)
        ssl_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ssl)
        
        for epoch in range(args.ssl_epochs):
            tx_encoder_with_head.train()
            for data, target in ssl_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                ssl_optimizer.zero_grad()
                preds = tx_encoder_with_head(data)
                loss = ssl_criterion(preds, target)
                loss.backward()
                ssl_optimizer.step()
            print(f"Tx Encoder Training Epoch {epoch+1}/{args.ssl_epochs}, Loss: {loss.item():.4f}", flush=True)
            
        print(f"Training complete. Saving encoder weights to '{encoder_path}'", flush=True)
        torch.save(tx_encoder_with_head.encoder.state_dict(), encoder_path)

    tx_encoder = TransactionFeatureEncoder(num_features=X_features.shape[1], final_embedding_dim=args.tx_emb).to(DEVICE)
    print(f"Loading pre-trained encoder weights from '{encoder_path}'.", flush=True)
    tx_encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    
    print("\n--- Generating transaction embeddings using trained encoder ---", flush=True)
    tx_encoder.eval()
    with torch.no_grad():
        transaction_embeddings_ssl = tx_encoder(torch.from_numpy(X_features).float().to(DEVICE)).cpu()
    print(f"Generated SSL embeddings for {transaction_embeddings_ssl.shape[0]} transactions.", flush=True)
    
    return transaction_embeddings_ssl

def train_gnn_link_predictor(args: argparse.Namespace, ssl_embeddings: torch.Tensor, df_gnn: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Trains the GNN on a link prediction task."""
    print("\n" + "="*50 + "\nSTAGE 3: GNN TRAINING\n" + "="*50, flush=True)
    
    all_accounts = pd.concat([df_gnn['Account'], df_gnn['Account.1']]).unique()
    account_mapping = {acc_id: i for i, acc_id in enumerate(all_accounts)}
    num_unique_accounts = len(all_accounts)
    
    source_nodes = df_gnn['Account'].map(account_mapping).values
    dest_nodes = df_gnn['Account.1'].map(account_mapping).values
    full_edge_index = torch.from_numpy(np.stack([source_nodes, dest_nodes])).to(torch.long)
    
    print("\n--- Aggregating transaction embeddings to create initial node features ---", flush=True)
    initial_node_features = torch.zeros(num_unique_accounts, ssl_embeddings.shape[1])
    source_nodes_t = torch.from_numpy(source_nodes).long()
    dest_nodes_t = torch.from_numpy(dest_nodes).long()
    initial_node_features.scatter_add_(0, source_nodes_t.unsqueeze(1).repeat(1, ssl_embeddings.shape[1]), ssl_embeddings)
    initial_node_features.scatter_add_(0, dest_nodes_t.unsqueeze(1).repeat(1, ssl_embeddings.shape[1]), ssl_embeddings)
    
    degrees = torch.bincount(source_nodes_t, minlength=num_unique_accounts) + torch.bincount(dest_nodes_t, minlength=num_unique_accounts)
    degrees[degrees == 0] = 1
    initial_node_features = initial_node_features / degrees.unsqueeze(1)
    
    graph = Data(x=initial_node_features, edge_index=full_edge_index, num_nodes=num_unique_accounts)
    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=False, add_negative_train_samples=True)
    train_data, _, _ = transform(graph)
    train_data.to(DEVICE)
    
    gnn_model = LinkPredictorGNN(in_channels=initial_node_features.shape[1], hidden_dim=args.gnn_hidden_dim).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for GNN training.")
        gnn_model = nn.DataParallel(gnn_model)
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)

    print("\n--- Starting GNN Training on Link Prediction Task with Gradient Accumulation ---", flush=True)
    accumulation_steps = 8
    
    for epoch in range(1, args.gnn_epochs + 1):
        gnn_model.train()
        model_to_train = gnn_model.module if isinstance(gnn_model, nn.DataParallel) else gnn_model
        
        h = model_to_train.encode(train_data.x, train_data.edge_index)
        
        supervision_edges = train_data.edge_label_index
        supervision_labels = train_data.edge_label.float()
        
        gnn_optimizer.zero_grad()
        
        chunk_size = len(supervision_labels) // accumulation_steps
        for i in range(accumulation_steps):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < accumulation_steps - 1 else len(supervision_labels)
            
            chunk_edge_index = supervision_edges[:, start_idx:end_idx]
            chunk_labels = supervision_labels[start_idx:end_idx]
            
            preds = model_to_train.decode(h, chunk_edge_index)
            loss = F.binary_cross_entropy_with_logits(preds.squeeze(), chunk_labels) / accumulation_steps
            
            is_last_step = (i == accumulation_steps - 1)
            loss.backward(retain_graph=not is_last_step)
            
        gnn_optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}/{args.gnn_epochs}, Last Chunk Loss: {loss.item() * accumulation_steps:.4f}', flush=True)

    print("\n--- Generating Final GNN-enhanced Node Embeddings ---", flush=True)
    gnn_model.eval()
    with torch.no_grad():
        model_to_run = gnn_model.module if isinstance(gnn_model, nn.DataParallel) else gnn_model
        final_node_embeddings = model_to_run.encode(graph.x.to(DEVICE), graph.edge_index.to(DEVICE)).cpu()
        
    return final_node_embeddings, source_nodes, dest_nodes

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

def train_and_evaluate_nn_classifier(args: argparse.Namespace, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, class_counts: pd.Series):
    """Trains and evaluates the downstream PyTorch NN classifier."""
    print("\n" + "="*50 + "\nSTAGE 4: DOWNSTREAM CLASSIFICATION (PYTORCH NN)\n" + "="*50, flush=True)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    classifier_nn = DownstreamClassifier(input_dim=X_train.shape[1], hidden_dim=args.downstream_hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(classifier_nn.parameters(), lr=args.lr)
    
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=DEVICE)
    print(f"Using pos_weight for the rare class: {pos_weight.item():.2f}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.downstream_epochs + 1):
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
            print(f"Downstream NN Epoch {epoch:03d}/{args.downstream_epochs}, Avg Loss: {total_loss/len(train_loader):.4f}", flush=True)

    classifier_nn.eval()
    with torch.no_grad():
        test_preds = classifier_nn(X_test.to(DEVICE))
        test_probs_nn = torch.sigmoid(test_preds).cpu().numpy()
        
    print_evaluation_metrics(y_test.numpy(), test_probs_nn, "PyTorch NN")

def train_and_evaluate_rf_classifier(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor):
    """Trains and evaluates the downstream Random Forest classifier."""
    print("\n" + "="*50 + "\nSTAGE 5: DOWNSTREAM CLASSIFICATION (RANDOM FOREST)\n" + "="*50, flush=True)
    print("\n--- Training a Random Forest Classifier with Balanced Class Weights ---", flush=True)
    
    classifier_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    classifier_rf.fit(X_train.numpy(), y_train.numpy().ravel())
    
    y_prob_rf = classifier_rf.predict_proba(X_test.numpy())[:, 1]
    
    print_evaluation_metrics(y_test.numpy(), y_prob_rf, "Random Forest")

# --- 5. MAIN EXECUTION BLOCK ---

def main(args: argparse.Namespace):
    """Main function to run the entire pipeline."""
    print("Using device:", DEVICE)
    print("Current configuration:", args)

    # --- DATA PREP ---
    X_features_np, y_labels_np, df_gnn, class_counts = load_and_preprocess_data(args.data_path)

    # --- STAGE 1 & 2: ENCODER ---
    transaction_embeddings_ssl = train_or_load_encoder(args, X_features_np, y_labels_np, class_counts)

    # --- STAGE 3: GNN ---
    final_node_embeddings, source_nodes, dest_nodes = train_gnn_link_predictor(args, transaction_embeddings_ssl, df_gnn)

    # --- FINAL EMBEDDING CREATION & DATA SPLIT ---
    print("\n--- Creating Final Transaction Embeddings & Splitting Data ---", flush=True)
    src_embeddings = final_node_embeddings[source_nodes]
    dst_embeddings = final_node_embeddings[dest_nodes]
    final_transaction_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
    print(f"Final transaction embedding shape: {final_transaction_embeddings.shape}")

    y_labels_tensor = torch.tensor(y_labels_np, dtype=torch.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(
        final_transaction_embeddings, y_labels_tensor, test_size=0.3, random_state=42, stratify=y_labels_tensor)
    
    # --- STAGE 4: PYTORCH CLASSIFIER ---
    train_and_evaluate_nn_classifier(args, X_train, y_train, X_test, y_test, class_counts)

    # --- STAGE 5: RANDOM FOREST CLASSIFIER ---
    train_and_evaluate_rf_classifier(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-based Fraud Detection Pipeline")

    # Data and Path Arguments
    parser.add_argument('--data_path', type=str, default='LI-Small_Trans.csv', help='Path to the transaction data CSV file.')
    parser.add_argument('--encoder_path', type=str, default='ssl_encoder_refactored.pth', help='Path to save/load the pre-trained transaction encoder.')
    
    # Model Dimension Arguments
    parser.add_argument('--tx_emb', type=int, default=128, help='Embedding dimension for the initial transaction encoder.')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help='Hidden dimension for the GNN.')
    parser.add_argument('--downstream_hidden_dim', type=int, default=128, help='Hidden dimension for the downstream MLP classifier.')
    
    # Training Hyperparameters
    parser.add_argument('--ssl_epochs', type=int, default=10, help='Number of epochs for pre-training the transaction encoder.')
    parser.add_argument('--gnn_epochs', type=int, default=100, help='Number of epochs for training the GNN.')
    parser.add_argument('--downstream_epochs', type=int, default=100, help='Number of epochs for training the downstream classifier.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for all optimizers.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for encoder and downstream training.')
    
    # Control Arguments
    parser.add_argument('--force_retrain_encoder', action='store_true', help='If set, the script will retrain the encoder even if a saved file exists.')
    
    args = parser.parse_args()
    main(args)
