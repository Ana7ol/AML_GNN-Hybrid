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
        
        target_sender_local = local_senders[-1]
        target_receiver_local = local_receivers[-1]
        
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
            nn.Linear(classifier_input_dim, gnn_hidden_dim), nn.BatchNorm1d(gnn_hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2), nn.BatchNorm1d(gnn_hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
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

# In code/new_gnn.py, replace the old FocalLoss class with this one.

class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is used to address the issue of class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are the raw logits from the model
        # targets are the binary (0 or 1) labels
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # pt is the probability of the correct class
        pt = torch.exp(-BCE_loss)
        
        # ### THIS IS THE FIX ###
        # Create a tensor of alpha values based on the targets
        # alpha_t = alpha if target == 1, 1-alpha if target == 0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate the final Focal Loss
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss
        # ######################

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# --- DDP Setup & Main Worker ---
def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, config, data_parts, port, model_save_path):
    is_ddp = world_size > 1
    use_cuda = torch.cuda.is_available() and world_size > 0
    device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")

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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = PyGDataLoader(train_dataset, batch_size=config['batch_size_per_gpu'], shuffle=(train_sampler is None), num_workers=config['num_cpu_workers'], sampler=train_sampler)
    val_loader = PyGDataLoader(val_dataset, batch_size=config['batch_size_per_gpu'], num_workers=config['num_cpu_workers']) if len(val_dataset) > 0 else None
    test_loader = PyGDataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], num_workers=config['num_cpu_workers']) if len(test_dataset) > 0 else None

    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"Running on device: {device}")

    # --- Model and Optimizer ---
    model = PyGTemporalGNN(num_accounts, config['acc_emb_dim'], tx_features.shape[1], config['tx_emb_dim'], config['gnn_hidden_dim'], config['gnn_layers']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5))

    # --- Learning Rate Scheduler ---
    scheduler = None
    if config.get('scheduler', {}).get('enabled', False):
        scheduler_config = config['scheduler']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 0)
        )
        if rank == 0:
            print(f"Using ReduceLROnPlateau scheduler (patience={scheduler_config['patience']}, factor={scheduler_config['factor']})")

    # --- Resume from Checkpoint Logic ---
    start_epoch = 0
    best_val_auprc = 0.0
    if os.path.exists(model_save_path):
        if rank == 0:
            print(f"--- Found existing checkpoint at '{model_save_path}'. Resuming training. ---")
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auprc = checkpoint.get('best_val_auprc', 0.0) # Use .get for safety
        if rank == 0:
            print(f"Resuming from end of epoch {checkpoint['epoch']}. Best AUPRC so far: {best_val_auprc:.4f}")
    else:
        if rank == 0:
            print(f"--- No checkpoint found. Starting training from scratch. ---")

    if is_ddp: model = DDP(model, device_ids=[device])
    
    # --- Dynamic Loss Function Creation ---
    loss_config = config.get('loss', {})
    if loss_config.get('type') == 'FocalLoss':
        if rank == 0:
            print(f"Using FocalLoss (alpha={loss_config.get('focal_loss_alpha')}, gamma={loss_config.get('focal_loss_gamma')})")
        criterion = FocalLoss(
            alpha=loss_config.get('focal_loss_alpha', 0.25),
            gamma=loss_config.get('focal_loss_gamma', 2.0)
        )
    else: # Default to BCEWithLogitsLoss
        pos_weight = (np.sum(labels[:train_end]==0) / np.sum(labels[:train_end]==1)) if np.sum(labels[:train_end]==1) > 0 else 1.0
        if rank == 0:
            print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight:.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # --- Training Loop ---
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        if is_ddp: train_sampler.set_epoch(epoch)

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} Training", disable=(rank!=0))
        
        for i, data in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix(avg_loss=avg_loss)

        # --- Validation after each epoch ---
        if val_loader:
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    probs = torch.sigmoid(model(data))
                    val_preds.append(probs.cpu())
                    val_labels.append(data.y.cpu())
            if is_ddp: dist.barrier()
            if rank == 0:
                all_preds = torch.cat(val_preds).numpy().squeeze()
                all_labels = torch.cat(val_labels).numpy().squeeze()
                if len(np.unique(all_labels)) > 1:
                    val_auroc = roc_auc_score(all_labels, all_preds)
                    val_auprc = average_precision_score(all_labels, all_preds)
                    print(f"Epoch {epoch+1} Validation AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

                    if scheduler:
                        scheduler.step(val_auprc)

                    if val_auprc > best_val_auprc:
                        best_val_auprc = val_auprc
                        print(f"  -> New best validation AUPRC! Saving checkpoint to {model_save_path}")
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_auprc': float(best_val_auprc)
                        }
                        torch.save(checkpoint, model_save_path)
                else:
                    print("Validation: Only one class present, cannot compute metrics.")

    # --- FINAL TEST EVALUATION ---
    if rank == 0 and test_loader:
        if not os.path.exists(model_save_path):
             print(f"ERROR: Best model was not saved. Cannot run test evaluation.")
             if is_ddp: cleanup_ddp()
             return

        print(f"\n--- Loading best model checkpoint from '{model_save_path}' for final test evaluation ---")
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model_for_testing = PyGTemporalGNN(num_accounts, config['acc_emb_dim'], tx_features.shape[1], config['tx_emb_dim'], config['gnn_hidden_dim'], config['gnn_layers'])
        model_for_testing.load_state_dict(checkpoint['model_state_dict'])
        model_for_testing.to(device)
        model_for_testing.eval()
        
        test_preds, test_labels = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                data = data.to(device)
                probs = torch.sigmoid(model_for_testing(data))
                test_preds.append(probs.cpu())
                test_labels.append(data.y.cpu())

        all_preds = torch.cat(test_preds).numpy().squeeze()
        all_labels = torch.cat(test_labels).numpy().squeeze()
        if len(np.unique(all_labels)) > 1:
            test_auroc = roc_auc_score(all_labels, all_preds)
            test_auprc = average_precision_score(all_labels, all_preds)
            print("\n--- Final Test Results ---")
            print(f"Test AUROC: {test_auroc:.4f}")
            print(f"Test AUPRC: {test_auprc:.4f}")
            print("--------------------------")
        else:
            print("Test Set: Only one class present, cannot compute metrics.")

    if is_ddp: cleanup_ddp()


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    # This line below allows command-line args to override config file values
    config.update({k: v for k, v in vars(args).items() if v is not None})


    print("--- Running Experiment ---")
    print(f"K Neighborhood: {config['k_neighborhood']}")
    print(f"GNN Layers: {config['gnn_layers']}")
    print(f"Embeddings: Acc={config['acc_emb_dim']}, Tx={config['tx_emb_dim']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("--------------------------")

    # ## MODIFICATION: Create save directory and generate filename ##
    save_dir = config.get('model_save_dir', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_filename = (f"GNN_K{config['k_neighborhood']}_L{config['gnn_layers']}_"
                      f"Emb{config['acc_emb_dim']}x{config['tx_emb_dim']}_"
                      f"LR{config['learning_rate']}.pth")
    model_save_path = os.path.join(save_dir, model_filename)
    print(f"Models will be saved to: {model_save_path}")


    world_size = torch.cuda.device_count()
    port = find_free_port()

    df_raw = pd.read_csv(config['data_path'])
    data_parts = preprocess_to_numpy_parts_for_pyg(df_raw, config)

    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        # ## MODIFICATION: Pass the save path ##
        mp.spawn(main_worker, args=(world_size, config, data_parts, port, model_save_path), nprocs=world_size, join=True)
    else:
        print("Found 1 or 0 GPUs. Running in a single process.")
        # ## MODIFICATION: Pass the save path ##
        main_worker(0, 1, config, data_parts, port, model_save_path)
