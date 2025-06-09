import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.data import Data, Dataset as PyGDataset, DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import time
import gc
import yaml
import os

def load_config(config_path="./config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_to_numpy_parts_for_pyg(df_raw_input, config):
    print("Preprocessing data and converting to NumPy arrays (for PyG)...")
    df = df_raw_input.copy()
    df['Account_str'] = df['Account'].astype(str)
    df['Account.1_str'] = df['Account.1'].astype(str)
    all_account_strings = pd.concat([df['Account_str'], df['Account.1_str']]).unique()
    account_str_to_global_id_map = {acc_str: i for i, acc_str in enumerate(all_account_strings)}
    num_unique_accounts = len(all_account_strings)
    df['Sender_Global_ID'] = df['Account_str'].map(account_str_to_global_id_map).astype(np.int64)
    df['Receiver_Global_ID'] = df['Account.1_str'].map(account_str_to_global_id_map).astype(np.int64)
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp_dt'], inplace=True)
    df['Time_Hour_Sin'] = np.sin(2 * np.pi * df['Timestamp_dt'].dt.hour / 24.0).astype(np.float32)
    df['Time_Hour_Cos'] = np.cos(2 * np.pi * df['Timestamp_dt'].dt.hour / 24.0).astype(np.float32)
    df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce')
    df.dropna(subset=['Amount Received'], inplace=True)
    df['Amount_Log'] = np.log1p(df['Amount Received']).astype(np.float32)
    df['Payment Format'] = df['Payment Format'].astype(str)
    df['Receiving Currency'] = df['Receiving Currency'].astype(str)
    df = pd.get_dummies(df, columns=['Payment Format', 'Receiving Currency'], prefix=['Format', 'Currency'], dummy_na=False, dtype=np.float32)
    feature_cols_list = [col for col in df.columns if col.startswith('Time_') or col.startswith('Amount_') or col.startswith('Format_') or col.startswith('Currency_')]
    for col in feature_cols_list:
        if df[col].isnull().any(): df[col].fillna(0.0, inplace=True)
        if df[col].dtype != np.float32: df[col] = df[col].astype(np.float32)
    df.sort_values(by='Timestamp_dt', inplace=True)
    df.reset_index(drop=True, inplace=True)
    sender_ids_np = df['Sender_Global_ID'].values
    receiver_ids_np = df['Receiver_Global_ID'].values
    labels_np = df['Is Laundering'].values.astype(np.float32)
    transaction_features_np = df[feature_cols_list].values
    if transaction_features_np.shape[0] == 0:
        raise ValueError("No data remains after preprocessing.")
    return (sender_ids_np, receiver_ids_np, labels_np, transaction_features_np, num_unique_accounts, account_str_to_global_id_map, feature_cols_list, df)

class PyGSnapshotDatasetOnline(PyGDataset):
    def __init__(self, sender_ids_np, receiver_ids_np, labels_np, transaction_features_np, k_history, is_train_split=True, split_name="unknown"):
        super().__init__(None)
        self.sender_ids_np = sender_ids_np
        self.receiver_ids_np = receiver_ids_np
        self.labels_np = labels_np
        self.transaction_features_np = transaction_features_np
        self.k_history = k_history
        self.num_total_source_transactions = len(sender_ids_np)
        self.split_name = split_name
        self._len = self.num_total_source_transactions - self.k_history
        if self._len <= 0:
            if is_train_split or self.num_total_source_transactions > 0:
                raise ValueError(f"Not enough transactions in the {self.split_name} split ({self.num_total_source_transactions}) to form even one snapshot with k_history={self.k_history}. Dataset length would be {self._len}.")
            else:
                print(f"Warning: {self.split_name} split has 0 source transactions. Dataset will be empty.")
    def len(self): return self._len
    def get(self, idx):
        target_idx_global = idx + self.k_history
        start_idx = max(0, target_idx_global - self.k_history)
        end_idx = target_idx_global + 1
        snapshot_sender_global_ids = self.sender_ids_np[start_idx:end_idx]
        snapshot_receiver_global_ids = self.receiver_ids_np[start_idx:end_idx]
        snapshot_edge_features_np = self.transaction_features_np[start_idx:end_idx]
        target_tx_features_np = self.transaction_features_np[target_idx_global]
        target_sender_global_id_val = self.sender_ids_np[target_idx_global]
        target_receiver_global_id_val = self.receiver_ids_np[target_idx_global]
        label_scalar_val = self.labels_np[target_idx_global]
        unique_nodes_global_ids_in_snapshot = np.unique(np.concatenate([snapshot_sender_global_ids, snapshot_receiver_global_ids]))
        x_node_global_ids = torch.from_numpy(unique_nodes_global_ids_in_snapshot).long()
        global_id_to_local_idx_map = {gid: i for i, gid in enumerate(unique_nodes_global_ids_in_snapshot)}
        source_nodes_local_np = np.array([global_id_to_local_idx_map[gid] for gid in snapshot_sender_global_ids], dtype=np.int64)
        target_nodes_local_np = np.array([global_id_to_local_idx_map[gid] for gid in snapshot_receiver_global_ids], dtype=np.int64)
        edge_index_snapshot = torch.from_numpy(np.array([source_nodes_local_np, target_nodes_local_np])).long()
        edge_attr_snapshot = torch.from_numpy(snapshot_edge_features_np).float()
        target_tx_features_tensor = torch.from_numpy(target_tx_features_np).float().unsqueeze(0)
        label_tensor = torch.tensor(label_scalar_val, dtype=torch.float32).view(1, 1)
        target_sender_local_idx_snapshot = torch.tensor(global_id_to_local_idx_map[target_sender_global_id_val], dtype=torch.long)
        target_receiver_local_idx_snapshot = torch.tensor(global_id_to_local_idx_map[target_receiver_global_id_val], dtype=torch.long)
        return Data(x_node_global_ids=x_node_global_ids, edge_index=edge_index_snapshot, edge_attr=edge_attr_snapshot, target_tx_features=target_tx_features_tensor, target_sender_local_idx=target_sender_local_idx_snapshot, target_receiver_local_idx=target_receiver_local_idx_snapshot, y=label_tensor, num_nodes=len(unique_nodes_global_ids_in_snapshot))

class PyGTemporalGNN(nn.Module):
    def __init__(self, num_total_accounts, account_embedding_dim, num_transaction_features, transaction_embedding_dim, gnn_hidden_dim, gnn_layers, num_classes=1):
        super().__init__()
        self.account_node_embedding = nn.Embedding(num_total_accounts, account_embedding_dim)
        self.transaction_feat_embedder = nn.Linear(num_transaction_features, transaction_embedding_dim)
        self.convs = nn.ModuleList()
        current_dim = account_embedding_dim
        for _ in range(gnn_layers):
            self.convs.append(SAGEConv(current_dim, gnn_hidden_dim))
            current_dim = gnn_hidden_dim
        self.classifier_input_dim = gnn_hidden_dim * 2 + transaction_embedding_dim
        #self.classifier = nn.Sequential(nn.Linear(self.classifier_input_dim, gnn_hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(gnn_hidden_dim // 2, num_classes))
        self.classifier = nn.Sequential(
    nn.Linear(self.classifier_input_dim, gnn_hidden_dim),
    nn.BatchNorm1d(gnn_hidden_dim), # Add BatchNorm
    nn.ReLU(),
    nn.Dropout(0.5), # Maybe increase dropout
    nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
    nn.BatchNorm1d(gnn_hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(gnn_hidden_dim // 2, num_classes)
)
    def forward(self, data):
        h_accounts_initial = self.account_node_embedding(data.x_node_global_ids)
        h_accounts_updated = h_accounts_initial
        for conv_layer in self.convs:
            h_accounts_updated = conv_layer(h_accounts_updated, data.edge_index)
            h_accounts_updated = F.relu(h_accounts_updated)
            h_accounts_updated = F.dropout(h_accounts_updated, p=0.2, training=self.training)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_offsets = data.ptr[:-1]
            global_sender_indices_in_batch = data.target_sender_local_idx + batch_offsets
            global_receiver_indices_in_batch = data.target_receiver_local_idx + batch_offsets
            target_sender_emb = h_accounts_updated.index_select(0, global_sender_indices_in_batch)
            target_receiver_emb = h_accounts_updated.index_select(0, global_receiver_indices_in_batch)
        else:
            target_sender_emb = h_accounts_updated[data.target_sender_local_idx].unsqueeze(0)
            target_receiver_emb = h_accounts_updated[data.target_receiver_local_idx].unsqueeze(0)
        embedded_target_transaction = F.relu(self.transaction_feat_embedder(data.target_tx_features))
        classifier_input = torch.cat([target_sender_emb, target_receiver_emb, embedded_target_transaction], dim=1)
        return self.classifier(classifier_input)

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Designed to address class imbalance.

    This loss was introduced in the 'Focal Loss for Dense Object Detection' paper.
    It's an extension of BCEWithLogitsLoss.

    Attributes:
        alpha (float): A balancing factor for the loss, typically set to the inverse
                       class frequency. For binary cases, it's the weight for the
                       positive class. Range: [0, 1].
        gamma (float): A focusing parameter that modifies the loss to focus on
                       hard-to-classify examples. Higher gamma means more focus
                       on hard examples. Range: [0, inf).
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): The model's raw output (pre-sigmoid).
                                   Shape: (batch_size, 1) or (batch_size,).
            targets (torch.Tensor): The ground truth labels (0 or 1).
                                    Shape should match logits.

        Returns:
            torch.Tensor: The calculated focal loss.
        """
        # Use BCEWithLogitsLoss to get the cross-entropy loss, but without reduction
        # This is numerically more stable than calculating it manually with sigmoids and logs
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Calculate the probability pt, which is p if target is 1, and 1-p if target is 0
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)

        # Calculate the focal loss modulating factor
        focal_modulator = (1.0 - pt) ** self.gamma

        # Calculate the alpha-balanced focal loss
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * focal_modulator * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

def setup_ddp(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(config['distributed_backend'], rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, config, global_data_parts):
    # This flag is crucial. It's true only when we are in a multi-GPU spawned process.
    is_ddp = world_size > 1
    device = rank # Use the rank as the device ID

    if is_ddp:
        print(f"Running DDP on rank {rank} / GPU {device}.")
        setup_ddp(rank, world_size, config)
        torch.cuda.set_device(device)
    else:
        print(f"Running in single-device mode on device {device}.")
        # No DDP setup for single device to avoid CUDA/multiprocessing deadlocks

    (sender_ids_all_np, receiver_ids_all_np, labels_all_np, transaction_features_all_np,
            num_total_unique_accounts, _, feature_cols, df_processed_for_viz) = global_data_parts

    NUM_FEATURES_PER_TRANSACTION = transaction_features_all_np.shape[1]
    num_total_processed_samples = len(sender_ids_all_np)
    train_split_idx = int(num_total_processed_samples * config['train_split_ratio'])

    train_dataset = PyGSnapshotDatasetOnline(sender_ids_all_np[:train_split_idx], receiver_ids_all_np[:train_split_idx], labels_all_np[:train_split_idx], transaction_features_all_np[:train_split_idx], config['k_neighborhood_transactions'], is_train_split=True, split_name="train")
    
    test_dataset = None
    if len(sender_ids_all_np[train_split_idx:]) > config['k_neighborhood_transactions']:
        test_dataset = PyGSnapshotDatasetOnline(sender_ids_all_np[train_split_idx:], receiver_ids_all_np[train_split_idx:], labels_all_np[train_split_idx:], transaction_features_all_np[train_split_idx:], config['k_neighborhood_transactions'], is_train_split=False, split_name="test")
        if len(test_dataset) == 0: test_dataset = None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_dataloader = PyGDataLoader(train_dataset, batch_size=config['batch_size_per_gpu'], shuffle=(train_sampler is None), num_workers=config['num_cpu_workers'], pin_memory=True, sampler=train_sampler)

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp and test_dataset else None
    test_dataloader = PyGDataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], shuffle=False, num_workers=config['num_cpu_workers'], pin_memory=True, sampler=test_sampler) if test_dataset else None

    model = PyGTemporalGNN(num_total_accounts=num_total_unique_accounts, account_embedding_dim=config['account_embedding_dim'], num_transaction_features=NUM_FEATURES_PER_TRANSACTION, transaction_embedding_dim=config['transaction_embedding_dim'], gnn_hidden_dim=config['gnn_hidden_dim'], gnn_layers=config['gnn_layers'], num_classes=config['model_output_classes']).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[device])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    criterion = None
    loss_config = config['loss']
    loss_type = loss_config['type']

    print(f"Using loss function: {loss_type}", flush=True)

    if loss_type == "BCEWithLogitsLoss":
        pos_weight_tensor = None
        if loss_config.get('pos_weight_enabled', True):
            # Calculate positive weight (same logic as before)
            if len(train_dataset) > 0:
                # Assuming labels_all_np is still available from global_data_parts
                train_labels_np = labels_all_np[:train_split_idx]
                effective_train_labels = train_labels_np[config['k_neighborhood_transactions']:]
                if len(effective_train_labels) > 0:
                    train_illicit_count = np.sum(effective_train_labels == 1)
                    train_licit_count = len(effective_train_labels) - train_illicit_count
                    if train_illicit_count > 0 and train_licit_count > 0:
                        pos_weight_value = train_licit_count / train_illicit_count
                        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
                        if rank == 0:
                            print(f"BCEWithLogitsLoss using pos_weight: {pos_weight_value:.2f}", flush=True)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)

    elif loss_type == "FocalLoss":
        alpha = loss_config.get('focal_loss_alpha', 0.25)
        gamma = loss_config.get('focal_loss_gamma', 2.0)
        if rank == 0:
            print(f"FocalLoss using alpha={alpha}, gamma={gamma}", flush=True)
        criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean').to(device)

    else:
        raise ValueError(f"Unknown loss function type: {loss_type}. Choose from 'BCEWithLogitsLoss', 'FocalLoss'.")

    if criterion is None:
        raise RuntimeError("Criterion (loss function) was not created.")



    if config['pos_weight_enabled'] and len(train_dataset) > 0:
        effective_train_labels = labels_all_np[:train_split_idx][config['k_neighborhood_transactions']:]
        train_illicit_count = np.sum(effective_train_labels == 1)
        train_licit_count = len(effective_train_labels) - train_illicit_count
        if train_illicit_count > 0 and train_licit_count > 0:
            pos_weight_value = train_licit_count / train_illicit_count
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(config['epochs']):
        model.train()
        if is_ddp: train_sampler.set_epoch(epoch)
        
        epoch_loss_sum = 0.0
        total_graphs_processed = 0
        epoch_start_time = time.time()
        
        for i, data_batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            data_batch = data_batch.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                logits = model(data_batch)
                loss = criterion(logits, data_batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.item() * data_batch.num_graphs
            total_graphs_processed += data_batch.num_graphs

        if is_ddp: dist.barrier()

        # In DDP, aggregate losses. In single-GPU, it's just the local loss.
        epoch_loss_tensor = torch.tensor([epoch_loss_sum], device=device)
        total_graphs_tensor = torch.tensor([total_graphs_processed], device=device)
        if is_ddp:
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_graphs_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_epoch_loss = epoch_loss_tensor.item() / total_graphs_tensor.item() if total_graphs_tensor.item() > 0 else float('nan')
            print(f"--- Epoch {epoch+1}/{config['epochs']} Summary (Rank 0) ---")
            print(f"  Avg Train Loss: {avg_epoch_loss:.4f}, Duration: {time.time() - epoch_start_time:.2f}s")

        # --- Evaluation ---
        model.eval()
        local_preds = []
        local_labels = []
        if test_dataloader:
            with torch.no_grad():
                for data_batch_eval in test_dataloader:
                    data_batch_eval = data_batch_eval.to(device)
                    logits_eval = model(data_batch_eval)
                    probs_eval = torch.sigmoid(logits_eval)
                    local_preds.append(probs_eval.cpu())
                    local_labels.append(data_batch_eval.y.cpu())

        # Gather results for evaluation on Rank 0
        if is_ddp:
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_preds, torch.cat(local_preds) if local_preds else torch.empty(0))
            dist.all_gather_object(gathered_labels, torch.cat(local_labels) if local_labels else torch.empty(0))
        else:
            gathered_preds = [torch.cat(local_preds)] if local_preds else []
            gathered_labels = [torch.cat(local_labels)] if local_labels else []

        if rank == 0:
            all_preds_proba = torch.cat(gathered_preds).numpy().squeeze() if gathered_preds and gathered_preds[0].numel() > 0 else np.array([])
            all_labels = torch.cat(gathered_labels).numpy().squeeze() if gathered_labels and gathered_labels[0].numel() > 0 else np.array([])

            if all_preds_proba.size > 0 and len(np.unique(all_labels)) > 1:
                auroc = roc_auc_score(all_labels, all_preds_proba)
                auprc = average_precision_score(all_labels, all_preds_proba)
                print(f"  Global Eval ==> AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
                
                # convert probabilities to binary outcomes
                threshold = 0.98
                all_preds_binary = (all_preds_proba >= threshold).astype(int)

                # --- Metrics that use binary predictions ---
                # Now use 'all_preds_binary' instead of 'all_preds_proba'
                f1 = f1_score(all_labels, all_preds_binary, zero_division=0)
                report = classification_report(all_labels, all_preds_binary, target_names=["Licit (0)", "Illicit (1)"], zero_division=0)
                cm = confusion_matrix(all_labels, all_preds_binary)

                print(f"    F1 Score (at threshold {threshold}): {f1:.4f}")
                print("    Classification Report:")
                print(report)
                print("    Confusion matrix:")
                print(cm)

                print("\n--- Optimizing Threshold for Best TP/TN Balance (ROC Curve) ---", flush=True)
            
                # Calculate points on the ROC curve
                fpr, tpr, thresholds_roc = roc_curve(all_labels, all_preds_proba)

                # We want to find the point on the curve closest to the top-left corner (0, 1)
                # This point represents the best balance of high TPR (recall) and low FPR.
                # The distance from the top-left corner is sqrt((1-tpr)^2 + fpr^2).
                # We want to minimize this distance.
                optimal_idx = np.argmin(np.sqrt((1 - tpr)**2 + fpr**2))
                optimal_threshold_roc = thresholds_roc[optimal_idx]

                # Calculate metrics at this new threshold
                all_preds_roc_binary = (all_preds_proba >= optimal_threshold_roc).astype(int)
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_roc_binary).ravel()

                recall_at_optimal = tp / (tp + fn)
                specificity_at_optimal = tn / (tn + fp)

                print(f"  Optimal threshold for TP/TN balance: {optimal_threshold_roc:.4f}", flush=True)
                print(f"    Recall (TP Rate) at this threshold: {recall_at_optimal:.4f}", flush=True)
                print(f"    Specificity (TN Rate) at this threshold: {specificity_at_optimal:.4f}", flush=True)

                print("\n    --- Classification Report (at Best ROC Threshold) ---")
                print(classification_report(all_labels, all_preds_roc_binary, target_names=["Licit (0)", "Illicit (1)"], zero_division=0))
                print("    --- Confusion Matrix (at Best ROC Threshold) ---")
                print(confusion_matrix(all_labels, all_preds_roc_binary))

            elif all_preds.size > 0:
                print("  Global Eval - Only one class present in test labels. Metrics not available.")
            else:
                print("  Global Eval - Test set is empty or too small.")
        
        if is_ddp: dist.barrier()

    if is_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    CONFIG = load_config()
    world_size = torch.cuda.device_count() if torch.cuda.is_available() and CONFIG.get('use_gpu', True) else 0
    if world_size == 0: # Handle CPU case
        world_size = 1 # Run as a single process
        print("No GPUs found or 'use_gpu' is false. Running on CPU.")

    # Preprocess data once in the main process
    df_raw_main = pd.read_csv(CONFIG['data_path'])
    global_data_parts_main = preprocess_to_numpy_parts_for_pyg(df_raw_main, CONFIG)

    if world_size > 1:
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        mp.spawn(main_worker, args=(world_size, CONFIG, global_data_parts_main), nprocs=world_size, join=True)
    else:
        print(f"Found {world_size} device(s). Running in a single process.")
        main_worker(0, 1, CONFIG, global_data_parts_main)

    # Calculate and print model parameters in the main process after everything is done
    temp_model = PyGTemporalGNN(
        num_total_accounts=global_data_parts_main[4],
        account_embedding_dim=CONFIG['account_embedding_dim'],
        num_transaction_features=global_data_parts_main[3].shape[1],
        transaction_embedding_dim=CONFIG['transaction_embedding_dim'],
        gnn_hidden_dim=CONFIG['gnn_hidden_dim'], gnn_layers=CONFIG['gnn_layers'],
        num_classes=CONFIG['model_output_classes']
    )
    print("parameter amount: " + str(sum(p.numel() for p in temp_model.parameters())))
