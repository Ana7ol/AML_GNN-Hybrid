import os
import argparse
import random
import socket
from contextlib import closing
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score, confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm


# ==============================================================================
# 1. ARGUMENT PARSING & CONFIGURATION
# ==============================================================================

def get_args():
    """
    Parses command-line arguments for the training script.
    Batch_size, Gnn_hidden_dim and Epoch count can only be changed in the yaml file

    """
    parser = argparse.ArgumentParser(description="GNN Training Script for Transaction Classification")

    # --- Key Hyperparameters ---
    parser.add_argument('--k_neighborhood', type=int, required=True, help='Number of recent transactions to include in each graph snapshot.')
    parser.add_argument('--acc_emb_dim', type=int, required=True, help='Dimension for account node embeddings.')
    parser.add_argument('--tx_emb_dim', type=int, required=True, help='Dimension for embedded transaction features.')
    parser.add_argument('--gnn_layers', type=int, required=True, help='Number of GNN layers in the model.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for the Adam optimizer.')

    # --- Special Operation Modes ---
    parser.add_argument('--separate_banks', action='store_true',
                        help='If set, graph edges are only created for intra-bank transactions.')
    parser.add_argument('--isolate_bank', type=str, default=None,
                        help="Focus the entire analysis on a single bank. Filters the dataset to only include transactions involving this bank.")

    # --- File Paths ---
    parser.add_argument('--config_path', type=str, default='./config/config.yaml',
                        help='Path to the base YAML configuration file.')

    return parser.parse_args()


def load_config(config_path="./config/config.yaml"):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def find_free_port():
    """Finds a free port on localhost for DDP initialization."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def set_seed(seed):
    """Sets a random seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 3. DATA PREPROCESSING
# ==============================================================================

def preprocess_to_numpy_parts_for_pyg(df_raw_input):
    """
    Preprocesses the raw DataFrame into NumPy arrays for the PyG Dataset.

    This function performs:
    - Global integer ID mapping for all unique account strings.
    - Bank ID extraction from account strings.
    - Feature engineering (cyclical time features, log-transformed amounts).
    - One-hot encoding of categorical features.
    - Sorting by timestamp to preserve temporal order.
    - Conversion to final NumPy arrays for memory efficiency.
    """
    print("Preprocessing data and converting to NumPy arrays...")
    df = df_raw_input.copy()

    # --- Account and Bank ID Mapping ---
    df['Account_str'] = df['Account'].astype(str)
    df['Account.1_str'] = df['Account.1'].astype(str)
    all_account_strings = pd.concat([df['Account_str'], df['Account.1_str']]).unique()
    account_str_to_global_id_map = {acc_str: i for i, acc_str in enumerate(all_account_strings)}
    num_unique_accounts = len(all_account_strings)
    df['Sender_Global_ID'] = df['Account_str'].map(account_str_to_global_id_map).astype(np.int64)
    df['Receiver_Global_ID'] = df['Account.1_str'].map(account_str_to_global_id_map).astype(np.int64)

    # Assuming the first character of the account string is the bank ID (e.g., 'B' in 'B1234')
    df['Sender_Bank_ID'] = df['Account_str'].str[0].astype('category').cat.codes.values
    df['Receiver_Bank_ID'] = df['Account.1_str'].str[0].astype('category').cat.codes.values

    # --- Feature Engineering ---
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp_dt'], inplace=True)
    df['Time_Hour_Sin'] = np.sin(2 * np.pi * df['Timestamp_dt'].dt.hour / 24.0).astype(np.float32)
    df['Time_Hour_Cos'] = np.cos(2 * np.pi * df['Timestamp_dt'].dt.hour / 24.0).astype(np.float32)

    df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce').fillna(0)
    # Use log-transform for amount to handle skewed distributions; this will also serve as the edge weight.
    df['Amount_Log'] = np.log1p(df['Amount Received']).astype(np.float32)

    # --- One-Hot Encoding ---
    df['Payment Format'] = df['Payment Format'].astype(str)
    df['Receiving Currency'] = df['Receiving Currency'].astype(str)
    df = pd.get_dummies(df, columns=['Payment Format', 'Receiving Currency'],
                        prefix=['Format', 'Currency'], dummy_na=False, dtype=np.float32)

    # --- Final Assembly ---
    feature_cols_list = [col for col in df.columns if col.startswith(('Time_', 'Amount_', 'Format_', 'Currency_'))]
    
    # Sort transactions chronologically, essential for temporal snapshots
    df.sort_values(by='Timestamp_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert final columns to NumPy arrays
    sender_ids_np = df['Sender_Global_ID'].values
    receiver_ids_np = df['Receiver_Global_ID'].values
    labels_np = df['Is Laundering'].values.astype(np.float32)
    transaction_features_np = df[feature_cols_list].values.astype(np.float32)
    sender_bank_ids_np = df['Sender_Bank_ID'].values.astype(np.int64)
    receiver_bank_ids_np = df['Receiver_Bank_ID'].values.astype(np.int64)

    try:
        amount_log_idx = feature_cols_list.index('Amount_Log')
    except ValueError:
        raise ValueError("'Amount_Log' not found in features; required for edge weights.")

    if transaction_features_np.shape[0] == 0:
        raise ValueError("No data remains after preprocessing.")

    return (sender_ids_np, receiver_ids_np, labels_np, transaction_features_np,
            num_unique_accounts, sender_bank_ids_np, receiver_bank_ids_np, amount_log_idx)


# ==============================================================================
# 4. PYTORCH GEOMETRIC DATASET
# ==============================================================================

class PyGSnapshotDatasetOnline(PyGDataset):
    """
    A PyTorch Geometric Dataset that creates temporal graph snapshots on-the-fly.

    For each target transaction at index `i`, it constructs a graph consisting of
    the `k_history` most recent transactions (from `i-k` to `i`). The `Amount_Log`
    is used as the edge weight for the GCN, while other transaction features
    are used for classification.
    """
    def __init__(self, all_data_nps, k_history, amount_log_idx, separate_banks_mode,
                 is_train_split=True, split_name="unknown"):
        super().__init__(None)
        (self.sender_ids_np, self.receiver_ids_np, self.labels_np,
         self.transaction_features_np, self.sender_bank_ids_np,
         self.receiver_bank_ids_np) = all_data_nps

        self.k_history = k_history
        self.amount_log_idx = amount_log_idx
        self.separate_banks_mode = separate_banks_mode
        self.split_name = split_name
        self.num_total_source_transactions = len(self.sender_ids_np)
        self._len = self.num_total_source_transactions - self.k_history

        if self._len <= 0 and (is_train_split or self.num_total_source_transactions > 0):
            raise ValueError(f"Not enough transactions in '{self.split_name}' split "
                             f"({self.num_total_source_transactions}) to form a snapshot "
                             f"with k_history={self.k_history}.")

    def len(self):
        return self._len

    def get(self, idx):
        """Constructs and returns a single temporal graph `Data` object."""
        try:
            target_idx_global = idx + self.k_history
            start_idx = max(0, target_idx_global - self.k_history)
            end_idx = target_idx_global + 1

            # --- 1. Extract k-history window from the full dataset ---
            snapshot_senders_gid = self.sender_ids_np[start_idx:end_idx]
            snapshot_receivers_gid = self.receiver_ids_np[start_idx:end_idx]
            snapshot_edge_features_np = self.transaction_features_np[start_idx:end_idx]

            # --- 2. Filter edges for graph construction if in separate_banks_mode ---
            edge_senders_gid, edge_receivers_gid, edge_features = snapshot_senders_gid, snapshot_receivers_gid, snapshot_edge_features_np
            if self.separate_banks_mode:
                snapshot_sender_bank_ids = self.sender_bank_ids_np[start_idx:end_idx]
                snapshot_receiver_bank_ids = self.receiver_bank_ids_np[start_idx:end_idx]
                intra_bank_mask = (snapshot_sender_bank_ids == snapshot_receiver_bank_ids)
                
                edge_senders_gid = edge_senders_gid[intra_bank_mask]
                edge_receivers_gid = edge_receivers_gid[intra_bank_mask]
                edge_features = edge_features[intra_bank_mask]

            # --- 3. Identify all unique nodes in the window and map to local indices ---
            # Nodes are from the full window to ensure target sender/receiver are always present
            unique_nodes_gid = np.unique(np.concatenate([snapshot_senders_gid, snapshot_receivers_gid]))
            global_to_local_map = {gid: i for i, gid in enumerate(unique_nodes_gid)}

            # --- 4. Construct graph tensors ---
            if len(edge_senders_gid) > 0:
                source_nodes_local = torch.tensor([global_to_local_map[gid] for gid in edge_senders_gid], dtype=torch.long)
                target_nodes_local = torch.tensor([global_to_local_map[gid] for gid in edge_receivers_gid], dtype=torch.long)
                edge_index = torch.stack([source_nodes_local, target_nodes_local], dim=0)

                # Separate amount for edge_weight and other features for edge_attr
                edge_weight = torch.from_numpy(edge_features[:, self.amount_log_idx]).float()
                edge_attr_np = np.delete(edge_features, self.amount_log_idx, axis=1)
                edge_attr = torch.from_numpy(edge_attr_np).float()
            else: # Handle empty graph case
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty(0, dtype=torch.float)
                num_features = snapshot_edge_features_np.shape[1] - 1
                edge_attr = torch.empty((0, num_features), dtype=torch.float)

            # --- 5. Prepare target transaction and label tensors ---
            target_tx_features = torch.from_numpy(self.transaction_features_np[target_idx_global]).float().unsqueeze(0)
            target_sender_local_idx = torch.tensor(global_to_local_map[self.sender_ids_np[target_idx_global]], dtype=torch.long)
            target_receiver_local_idx = torch.tensor(global_to_local_map[self.receiver_ids_np[target_idx_global]], dtype=torch.long)
            label = torch.tensor(self.labels_np[target_idx_global], dtype=torch.float32).view(1, 1)

            return Data(
                x_node_global_ids=torch.from_numpy(unique_nodes_gid).long(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_weight=edge_weight,
                target_tx_features=target_tx_features,
                target_sender_local_idx=target_sender_local_idx,
                target_receiver_local_idx=target_receiver_local_idx,
                y=label,
                num_nodes=len(unique_nodes_gid)
            )
        except Exception as e:
            print(f"!!! FATAL ERROR IN DATALOADER GET (idx={idx}) !!!", flush=True)
            print(f"Error Type: {type(e).__name__}, Message: {e}", flush=True)
            raise


# ==============================================================================
# 5. GNN MODEL DEFINITION
# ==============================================================================

class PyGTemporalGNN(nn.Module):
    """
    Graph Neural Network for classifying transactions.

    Architecture:
    1. An Embedding layer to create initial vector representations for each account.
    2. A stack of GCNConv layers to perform message passing, using transaction
       amounts as edge weights to learn structural representations.
    3. A final MLP classifier that combines the embeddings of the target
       transaction's sender, receiver, and its own features to make a prediction.
    """
    def __init__(self, num_total_accounts, account_embedding_dim, num_transaction_features,
                 transaction_embedding_dim, gnn_hidden_dim, gnn_layers, num_classes=1):
        super().__init__()

        self.account_node_embedding = nn.Embedding(num_total_accounts, account_embedding_dim)
        self.transaction_feat_embedder = nn.Linear(num_transaction_features, transaction_embedding_dim)

        self.convs = nn.ModuleList()
        current_dim = account_embedding_dim
        for _ in range(gnn_layers):
            self.convs.append(GCNConv(current_dim, gnn_hidden_dim))
            current_dim = gnn_hidden_dim

        self.classifier_input_dim = gnn_hidden_dim * 2 + transaction_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, gnn_hidden_dim),
            nn.BatchNorm1d(gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.BatchNorm1d(gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(gnn_hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        # 1. Get initial account embeddings from their global IDs
        h_accounts = self.account_node_embedding(data.x_node_global_ids)

        # 2. Perform graph convolutions (message passing)
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        for conv_layer in self.convs:
            h_accounts = conv_layer(h_accounts, data.edge_index, edge_weight=edge_weight)
            h_accounts = F.relu(h_accounts)
            h_accounts = F.dropout(h_accounts, p=0.2, training=self.training)

        # 3. Extract final embeddings for the target transaction's sender and receiver
        if hasattr(data, 'batch') and data.batch is not None:
            # Handle batched graphs by using the `ptr` attribute to find offsets
            batch_offsets = data.ptr[:-1]
            sender_indices = data.target_sender_local_idx + batch_offsets
            receiver_indices = data.target_receiver_local_idx + batch_offsets
            target_sender_emb = h_accounts.index_select(0, sender_indices)
            target_receiver_emb = h_accounts.index_select(0, receiver_indices)
        else: # Handle a single, unbatched graph
            target_sender_emb = h_accounts[data.target_sender_local_idx].unsqueeze(0)
            target_receiver_emb = h_accounts[data.target_receiver_local_idx].unsqueeze(0)

        # 4. Embed the target transaction's own features
        embedded_target_tx = F.relu(self.transaction_feat_embedder(data.target_tx_features))

        # 5. Concatenate all embeddings and pass through the classifier
        classifier_input = torch.cat([target_sender_emb, target_receiver_emb, embedded_target_tx], dim=1)
        return self.classifier(classifier_input)


# ==============================================================================
# 6. LOSS FUNCTION
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification, designed to address extreme class imbalance.
    It down-weights the loss assigned to well-classified examples, focusing on hard,
    misclassified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets) # Probability of the correct class
        focal_modulator = (1.0 - pt) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * focal_modulator * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==============================================================================
# 7. DDP & MAIN WORKER
# ==============================================================================

def setup_ddp(rank, world_size, master_port):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def main_worker(rank, world_size, master_port, config, global_data_parts):
    """The main training and evaluation logic executed by each process."""
    is_ddp = world_size > 1
    is_main_process = (rank == 0)
    device = rank

    if 'random_seed' in config:
        set_seed(config['random_seed'] + rank)

    # --- Setup Directories & DDP ---
    if is_main_process:
        results_dir = "results"
        name_parts = [f"final_GNN",f"k{config['k_neighborhood_transactions']}", f"L{config['gnn_layers']}",
                      f"emb{config['account_embedding_dim']}x{config['transaction_embedding_dim']}",
                      f"lr{config['learning_rate']}"]
        if config.get('separate_banks_mode', False): name_parts.append("SepBanks")
        if config.get('isolate_bank'): name_parts.append(f"iso{config['isolate_bank']}")
        
        run_name = "_".join(name_parts)
        run_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving results for this run in: {run_dir}", flush=True)

    if is_ddp:
        print(f"Running DDP on rank {rank} / GPU {device}.")
        setup_ddp(rank, world_size, master_port)
        torch.cuda.set_device(device)
    else:
        print(f"Running in single-device mode on device {device}.")

    # --- Data Loading & Splitting ---
    (sender_ids, receiver_ids, labels, tx_features,
     num_accounts, sender_banks, receiver_banks, amount_idx) = global_data_parts

    train_end = int(len(sender_ids) * config['train_split_ratio'])
    val_end = train_end + int(len(sender_ids) * config['val_split_ratio'])

    dataset_args = {
        'k_history': config['k_neighborhood_transactions'],
        'amount_log_idx': amount_idx,
        'separate_banks_mode': config.get('separate_banks_mode', False)
    }

    # Slicing the numpy arrays is memory efficient (creates views, not copies)
    train_data_nps = (sender_ids[:train_end],
            receiver_ids[:train_end], labels[:train_end], 
            tx_features[:train_end], 
            sender_banks[:train_end], 
            receiver_banks[:train_end])

    val_data_nps = (sender_ids[train_end:val_end], 
            receiver_ids[train_end:val_end], 
            labels[train_end:val_end], 
            tx_features[train_end:val_end], 
            sender_banks[train_end:val_end], 
            receiver_banks[train_end:val_end])

    test_data_nps = (sender_ids[val_end:], 
            receiver_ids[val_end:], 
            labels[val_end:], 
            tx_features[val_end:], 
            sender_banks[val_end:], 
            receiver_banks[val_end:])

    train_dataset = PyGSnapshotDatasetOnline(train_data_nps, is_train_split=True, split_name="train", **dataset_args)
    val_dataset = PyGSnapshotDatasetOnline(val_data_nps, is_train_split=False, split_name="validation", **dataset_args) if len(val_data_nps[0]) > 0 else None
    test_dataset = PyGSnapshotDatasetOnline(test_data_nps, is_train_split=False, split_name="test", **dataset_args) if len(test_data_nps[0]) > 0 else None
    
    if is_main_process:
        print(f"Train set length: {len(train_dataset)}")
        print(f"Validation set length: {len(val_dataset) if val_dataset else 'N/A'}")
        print(f"Test set length: {len(test_dataset) if test_dataset else 'N/A'}")

    # --- DataLoaders ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_dataloader = PyGDataLoader(train_dataset, batch_size=config['batch_size_per_gpu'], shuffle=(train_sampler is None), num_workers=config['num_cpu_workers'], sampler=train_sampler)

    val_dataloader = PyGDataLoader(val_dataset, batch_size=config['batch_size_per_gpu'], shuffle=False, num_workers=config['num_cpu_workers']) if val_dataset else None
    test_dataloader = PyGDataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], shuffle=False, num_workers=config['num_cpu_workers']) if test_dataset else None

    # --- Model, Optimizer, and Loss Setup ---
    model = PyGTemporalGNN(num_total_accounts=num_accounts, account_embedding_dim=config['account_embedding_dim'],
                           num_transaction_features=tx_features.shape[1], transaction_embedding_dim=config['tx_emb_dim'],
                           gnn_hidden_dim=config['gnn_hidden_dim'], gnn_layers=config['gnn_layers']).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True) # find_unused_parameters can be needed if graph structure varies

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Loss function setup
    loss_config = config['loss']
    loss_type = loss_config.get('type', 'BCEWithLogitsLoss')
    if is_main_process: print(f"Using loss function: {loss_type}", flush=True)

    if loss_type == "FocalLoss":
        criterion = FocalLoss(alpha=loss_config.get('focal_loss_alpha', 0.25), gamma=loss_config.get('focal_loss_gamma', 2.0)).to(device)
    else: # Default to BCEWithLogitsLoss
        pos_weight_tensor = None
        if loss_config.get('pos_weight_enabled', True) and len(train_dataset) > 0:
            effective_labels = train_data_nps[2][config['k_neighborhood_transactions']:]
            pos_count = np.sum(effective_labels == 1)
            neg_count = len(effective_labels) - pos_count
            if pos_count > 0:
                pos_weight_value = neg_count / pos_count
                pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
                if is_main_process: print(f"BCEWithLogitsLoss using pos_weight: {pos_weight_value:.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)

    # ==================== MAIN TRAINING LOOP ====================
    for epoch in range(config['epochs']):
        model.train()
        if is_ddp: train_sampler.set_epoch(epoch)

        epoch_loss_sum, total_graphs = 0.0, 0
        
        # Use tqdm for progress bar only on the main process
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} Training", unit="batch") if is_main_process else train_dataloader

        for data_batch in train_iterator:
            optimizer.zero_grad(set_to_none=True)
            data_batch = data_batch.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                logits = model(data_batch)
                loss = criterion(logits, data_batch.y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.item() * data_batch.num_graphs
            total_graphs += data_batch.num_graphs
            
            if is_main_process: train_iterator.set_postfix(loss=loss.item())

        if is_main_process: train_iterator.close()
        
        # Aggregate loss across all processes
        if is_ddp: dist.barrier()
        # (Aggregation logic here if needed, but printing on main process is often sufficient)

        if is_main_process:
            avg_epoch_loss = epoch_loss_sum / total_graphs if total_graphs > 0 else 0
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} Summary ---")
            print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")

            # ==================== EVALUATION PHASE ====================
            model.eval()
            
            # --- Validation Set Evaluation ---
            if val_dataloader:
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = batch.to(device, non_blocking=True)
                        val_preds.append(torch.sigmoid(model(batch)).cpu())
                        val_labels.append(batch.y.cpu())
                
                val_preds = torch.cat(val_preds).numpy().squeeze()
                val_labels = torch.cat(val_labels).numpy().squeeze()
                if len(np.unique(val_labels)) > 1:
                    val_auroc = roc_auc_score(val_labels, val_preds)
                    val_auprc = average_precision_score(val_labels, val_preds)
                    print(f"  Validation ==> AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

            # --- Test Set Evaluation & Reporting ---
            if test_dataloader:
                test_preds, test_labels = [], []
                with torch.no_grad():
                    for batch in test_dataloader:
                        batch = batch.to(device, non_blocking=True)
                        test_preds.append(torch.sigmoid(model(batch)).cpu())
                        test_labels.append(batch.y.cpu())
                
                test_preds = torch.cat(test_preds).numpy().squeeze()
                test_labels = torch.cat(test_labels).numpy().squeeze()
                
                if test_preds.size > 0:
                    pd.DataFrame({'y_true': test_labels, 'y_pred_proba': test_preds}).to_csv(
                        os.path.join(run_dir, f"epoch_{epoch+1}_results.csv"), index=False)

                    if len(np.unique(test_labels)) > 1:
                        test_auroc = roc_auc_score(test_labels, test_preds)
                        test_auprc = average_precision_score(test_labels, test_preds)
                        print(f"  Test Eval    ==> AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
        
        if is_ddp: dist.barrier()

    if is_ddp: cleanup_ddp()


# ==============================================================================
# 8. SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Load Configs and Arguments ---
    args = get_args()
    CONFIG = load_config(args.config_path)
    
    # Override base config with command-line arguments for hyperparameter tuning
    CONFIG.update(vars(args))

    print("--- Running Experiment with Final Configuration ---")
    for key, value in CONFIG.items():
        if key in vars(args) or key in ['train_split_ratio', 'val_split_ratio']:
            print(f"  {key}: {value}")
    print("-------------------------------------------------")

    # --- 2. Setup DDP or Single-Device Execution ---
    world_size = torch.cuda.device_count() if torch.cuda.is_available() and CONFIG.get('use_gpu', True) else 0
    master_port = find_free_port() if world_size > 1 else "12355"
    if world_size == 0: world_size = 1 # Fallback to CPU

    # --- 3. Load and Preprocess Data ---
    df_raw = pd.read_csv(CONFIG['data_path'])

    # Optionally filter the entire dataset for a single bank before doing anything else
    if CONFIG.get('isolate_bank'):
        bank_id = CONFIG['isolate_bank']
        print(f"--- ISOLATING BANK: {bank_id} ---")
        mask = (df_raw['Account'].str[0] == bank_id) | (df_raw['Account.1'].str[0] == bank_id)
        df_raw = df_raw[mask].copy()
        if len(df_raw) == 0:
            raise ValueError(f"No transactions found for the isolated bank '{bank_id}'.")

    # Preprocess data once in the main process to be shared with workers
    global_data_parts = preprocess_to_numpy_parts_for_pyg(df_raw)

    # --- 4. Start Training Process ---
    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning DDP processes on port {master_port}.")
        mp.spawn(main_worker, args=(world_size, master_port, CONFIG, global_data_parts), nprocs=world_size, join=True)
    else:
        print(f"Found {world_size} device(s). Running in a single process.")
        main_worker(0, 1, master_port, CONFIG, global_data_parts)

    print("--- Training complete. ---")
