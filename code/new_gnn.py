# Filename: code/new_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as PyGDataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import time
import yaml
import os
import random
import socket
from contextlib import closing
import argparse
from tqdm import tqdm


# --- Argument Parsing ---
def get_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="GNN Training Script")
    parser.add_argument('--k_neighborhood', type=int, required=True, help='Value for k_neighborhood_transactions')
    parser.add_argument('--acc_emb_dim', type=int, required=True, help='Dimension for account embeddings')
    parser.add_argument('--tx_emb_dim', type=int, required=True, help='Dimension for transaction embeddings')
    parser.add_argument('--gnn_layers', type=int, required=True, help='Number of GNN layers')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for the optimizer')
    parser.add_argument('--config_path', type=str, default='./config/config.yaml', help='Path to a BASE config file')
    return parser.parse_args()


# --- Utility Functions ---
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# --- Data Preprocessing ---
def preprocess_to_numpy_parts_for_pyg(df_raw_input, config):
    print("Preprocessing data and converting to NumPy arrays (for PyG)...")
    df = df_raw_input.copy()
    all_accounts = pd.concat([df['Account'].astype(str), df['Account.1'].astype(str)]).unique()
    acc_to_id = {acc: i for i, acc in enumerate(all_accounts)}
    df['Sender_Global_ID'] = df['Account'].astype(str).map(acc_to_id)
    df['Receiver_Global_ID'] = df['Account.1'].astype(str).map(acc_to_id)
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp_dt'], inplace=True)
    df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce')
    df.dropna(subset=['Amount Received'], inplace=True)
    df['Amount_Log'] = np.log1p(df['Amount Received']).astype(np.float32)
    df = pd.get_dummies(df, columns=['Payment Format', 'Receiving Currency'], dummy_na=False, dtype=np.float32)
    feature_cols = [col for col in df.columns if col.startswith(('Amount_Log', 'Payment Format_', 'Receiving Currency_'))]
    df.sort_values(by='Timestamp_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    amount_log_idx = feature_cols.index('Amount_Log')
    return (df['Sender_Global_ID'].values.astype(np.int64), df['Receiver_Global_ID'].values.astype(np.int64), 
            df['Is Laundering'].values.astype(np.float32), df[feature_cols].values.astype(np.float32), 
            len(all_accounts), amount_log_idx)


# --- PyTorch Geometric Dataset Class ---
class PyGSnapshotDatasetOnline(PyGDataset):
    """Creates temporal graph snapshots on-the-fly."""
    def __init__(self, sender_ids, receiver_ids, labels, tx_features, k_history, amount_log_idx):
        super().__init__()
        self.sender_ids, self.receiver_ids, self.labels = sender_ids, receiver_ids, labels
        self.tx_features = tx_features
        self.k_history, self.amount_log_idx = k_history, amount_log_idx
        self._len = len(sender_ids) - k_history

    def len(self):
        return self._len

    def get(self, idx):
        target_idx = idx + self.k_history
        start_idx = max(0, target_idx - self.k_history)
        end_idx = target_idx + 1

        senders = self.sender_ids[start_idx:end_idx]
        receivers = self.receiver_ids[start_idx:end_idx]
        
        nodes, local_map = np.unique(np.concatenate([senders, receivers]), return_inverse=True)
        local_senders = local_map[:len(senders)]
        local_receivers = local_map[len(senders):]

        edge_index = torch.from_numpy(np.vstack([local_senders, local_receivers])).long()
        edge_weight = torch.from_numpy(self.tx_features[start_idx:end_idx, self.amount_log_idx]).float()
        
        target_sender_local = local_map[senders == self.sender_ids[target_idx]][0]
        target_receiver_local = local_map[receivers == self.receiver_ids[target_idx]][0]
        
        return Data(
            x_node_global_ids=torch.from_numpy(nodes).long(),
            edge_index=edge_index,
            edge_weight=edge_weight,
            target_tx_features=torch.from_numpy(self.tx_features[target_idx]).float().unsqueeze(0),
            target_node_idx=torch.tensor([target_sender_local, target_receiver_local], dtype=torch.long),
            y=torch.tensor(self.labels[target_idx], dtype=torch.float32).view(1, 1),
            num_nodes=len(nodes)
        )

# --- Model Definition with Skip Connections ---
class PyGTemporalGNN(nn.Module):
    """GNN model with robust skip connections and BatchNorm for stability."""
    def __init__(self, num_total_accounts, account_embedding_dim, num_transaction_features,
                 transaction_embedding_dim, gnn_hidden_dim, gnn_layers):
        super().__init__()
        self.account_node_embedding = nn.Embedding(num_total_accounts, account_embedding_dim)
        self.transaction_feat_embedder = nn.Linear(num_transaction_features, transaction_embedding_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_projections = nn.ModuleList()

        current_dim = account_embedding_dim
        for _ in range(gnn_layers):
            self.convs.append(GCNConv(current_dim, gnn_hidden_dim))
            self.bns.append(nn.BatchNorm1d(gnn_hidden_dim))
            # Projection for the skip connection if dimensions mismatch
            self.res_projections.append(nn.Linear(current_dim, gnn_hidden_dim) if current_dim != gnn_hidden_dim else nn.Identity())
            current_dim = gnn_hidden_dim

        classifier_input_dim = gnn_hidden_dim * 2 + transaction_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, gnn_hidden_dim), nn.BatchNorm1d(gnn_hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2), nn.BatchNorm1d(gnn_hidden_dim // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )

    def forward(self, data):
        h = self.account_node_embedding(data.x_node_global_ids)
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        for i, conv in enumerate(self.convs):
            h_prev = h # Store input for the skip connection
            h_conv = conv(h, data.edge_index, edge_weight=edge_weight)
            h_conv = self.bns[i](h_conv)
            h_conv = F.relu(h_conv)
            h_conv = F.dropout(h_conv, p=0.2, training=self.training)
            # Add the skip connection
            h = h_conv + self.res_projections[i](h_prev)

        target_embs = h[data.target_node_idx]
        sender_emb, receiver_emb = target_embs[0::2], target_embs[1::2]
        tx_emb = F.relu(self.transaction_feat_embedder(data.target_tx_features))
        
        final_embedding = torch.cat([sender_emb, receiver_emb, tx_emb], dim=1)
        return self.classifier(final_embedding)

# --- DDP Setup & Main Worker ---
def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, config, data_parts, port):
    is_ddp = world_size > 1
    device = rank
    set_seed(config['random_seed'] + rank)

    if is_ddp:
        setup_ddp(rank, world_size, port)
        torch.cuda.set_device(device)

    # --- Data Loading ---
    (sender_ids, receiver_ids, labels, tx_features, num_accounts, amount_idx) = data_parts
    train_end = int(len(sender_ids) * config['train_split_ratio'])
    val_end = train_end + int(len(sender_ids) * config['val_split_ratio'])
    
    ds_args = {'k_history': config['k_neighborhood_transactions'], 'amount_log_idx': amount_idx}
    train_dataset = PyGSnapshotDatasetOnline(sender_ids[:train_end], receiver_ids[:train_end], labels[:train_end], tx_features[:train_end], **ds_args)
    val_dataset = PyGSnapshotDatasetOnline(sender_ids[train_end:val_end], receiver_ids[train_end:val_end], labels[train_end:val_end], tx_features[train_end:val_end], **ds_args)
    test_dataset = PyGSnapshotDatasetOnline(sender_ids[val_end:], receiver_ids[val_end:], labels[val_end:], tx_features[val_end:], **ds_args)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    train_loader = PyGDataLoader(train_dataset, batch_size=config['batch_size_per_gpu'], shuffle=(train_sampler is None), num_workers=config['num_cpu_workers'], pin_memory=True, sampler=train_sampler)
    val_loader = PyGDataLoader(val_dataset, batch_size=config['batch_size_per_gpu'], num_workers=config['num_cpu_workers'])
    test_loader = PyGDataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], num_workers=config['num_cpu_workers'])

    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # --- Model, Optimizer, Loss ---
    model = PyGTemporalGNN(num_accounts, config['acc_emb_dim'], tx_features.shape[1], config['tx_emb_dim'], config['gnn_hidden_dim'], config['gnn_layers']).to(device)
    if is_ddp: model = DDP(model, device_ids=[device])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5))

    # Using your original, high pos_weight. Gradient clipping will help manage stability.
    pos_weight = (np.sum(labels[:train_end]==0) / np.sum(labels[:train_end]==1)) if np.sum(labels[:train_end]==1) > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    if rank == 0: print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight:.2f}")

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        model.train()
        if is_ddp: train_sampler.set_epoch(epoch)
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Training", disable=(rank!=0))
        for data in iterator:
            optimizer.zero_grad(set_to_none=True)
            data = data.to(device)
            
            logits = model(data)
            loss = criterion(logits, data.y)
            
            # This is the standard backward pass for a simple loop
            loss.backward()
            # CRITICAL: This clips large gradients and prevents model collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            iterator.set_postfix(loss=loss.item())

        # --- Validation after each epoch ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                probs = torch.sigmoid(model(data))
                val_preds.append(probs.cpu())
                val_labels.append(data.y.cpu())
        
        # Gather results if in DDP
        if is_ddp:
            dist.barrier()
            # In a real DDP validation, you'd gather these results. For simplicity, we only print Rank 0's results.
        
        if rank == 0:
            all_preds = torch.cat(val_preds).numpy().squeeze()
            all_labels = torch.cat(val_labels).numpy().squeeze()
            if len(np.unique(all_labels)) > 1:
                val_auroc = roc_auc_score(all_labels, all_preds)
                print(f"Epoch {epoch+1} Validation AUROC: {val_auroc:.4f}")
            else:
                print("Validation: Only one class present, cannot compute AUROC.")

    if is_ddp: cleanup_ddp()

# --- Script Entry Point ---
if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    config.update(vars(args))

    print("--- Running Experiment ---")
    print(f"K Neighborhood: {config['k_neighborhood']}")
    print(f"GNN Layers: {config['gnn_layers']}")
    print(f"Embeddings: Acc={config['acc_emb_dim']}, Tx={config['tx_emb_dim']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("--------------------------")

    world_size = torch.cuda.device_count()
    port = find_free_port()

    df_raw = pd.read_csv(config['data_path'])
    data_parts = preprocess_to_numpy_parts_for_pyg(df_raw, config)

    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        mp.spawn(main_worker, args=(world_size, config, data_parts, port), nprocs=world_size, join=True)
    else:
        print("Found 1 or 0 GPUs. Running in a single process.")
        main_worker(0, 1, config, data_parts, port)
