import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp # For spawning processes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler # For DDP with DataLoader

# PyTorch Geometric Imports
from torch_geometric.data import Data, Dataset as PyGDataset, DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv # Or any other PyG GNN layer

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import time
import gc
import yaml
import os

# --- Configuration Loading Function ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- preprocess_to_numpy_parts_for_pyg (Keep as is) ---
def preprocess_to_numpy_parts_for_pyg(df_raw_input, config): # Pass config for data_path
    # ... (same as your previous version, uses config['data_path'])
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
    # ... (other feature engineering)

    df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce')
    df.dropna(subset=['Amount Received'], inplace=True)
    df['Amount_Log'] = np.log1p(df['Amount Received']).astype(np.float32)

    df['Payment Format'] = df['Payment Format'].astype(str)
    df['Receiving Currency'] = df['Receiving Currency'].astype(str)
    df = pd.get_dummies(df, columns=['Payment Format', 'Receiving Currency'], prefix=['Format', 'Currency'], dummy_na=False, dtype=np.float32)

    feature_cols_list = [col for col in df.columns if col.startswith('Time_') or \
                         col.startswith('Amount_') or \
                         col.startswith('Format_') or \
                         col.startswith('Currency_')]

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
    return (sender_ids_np, receiver_ids_np, labels_np, transaction_features_np,
            num_unique_accounts, account_str_to_global_id_map, feature_cols_list, df) # Return df for viz


# --- PyGSnapshotDatasetOnline (Keep as is) ---
class PyGSnapshotDatasetOnline(PyGDataset):
    # ... (same as your last working version) ...
    def __init__(self, sender_ids_np, receiver_ids_np, labels_np,
                 transaction_features_np, k_history,
                 is_train_split=True, split_name="unknown"):
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
            if is_train_split or self.num_total_source_transactions > 0 :
                 raise ValueError(
                    f"Not enough transactions in the {self.split_name} split ({self.num_total_source_transactions}) "
                    f"to form even one snapshot with k_history={self.k_history}. "
                    f"Dataset length would be {self._len}."
                )
            else:
                print(f"Warning: {self.split_name} split has 0 source transactions. Dataset will be empty.")
        # print(f"PyGSnapshotDatasetOnline ({self.split_name}): Usable snapshots: {self._len} from {self.num_total_source_transactions} source txns (k={k_history}).")

    def len(self): return self._len
    def get(self, idx):
        # ... (same logic as before to construct and return a PyG Data object) ...
        target_idx_global = idx + self.k_history
        start_idx = max(0, target_idx_global - self.k_history)
        end_idx = target_idx_global + 1
        # ... (rest of the Data object creation)
        snapshot_sender_global_ids = self.sender_ids_np[start_idx:end_idx]
        snapshot_receiver_global_ids = self.receiver_ids_np[start_idx:end_idx]
        snapshot_edge_features_np = self.transaction_features_np[start_idx:end_idx]
        target_tx_features_np = self.transaction_features_np[target_idx_global]
        target_sender_global_id_val = self.sender_ids_np[target_idx_global]
        target_receiver_global_id_val = self.receiver_ids_np[target_idx_global]
        label_scalar_val = self.labels_np[target_idx_global]
        unique_nodes_global_ids_in_snapshot = np.unique(
            np.concatenate([snapshot_sender_global_ids, snapshot_receiver_global_ids]))
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
        return Data(
            x_node_global_ids=x_node_global_ids, edge_index=edge_index_snapshot, edge_attr=edge_attr_snapshot,
            target_tx_features=target_tx_features_tensor, target_sender_local_idx=target_sender_local_idx_snapshot,
            target_receiver_local_idx=target_receiver_local_idx_snapshot, y=label_tensor,
            num_nodes=len(unique_nodes_global_ids_in_snapshot))


# --- PyGTemporalGNN Model (Keep as is) ---
class PyGTemporalGNN(nn.Module):
    # ... (same as your last working version) ...
    def __init__(self, num_total_accounts, account_embedding_dim,
                 num_transaction_features, transaction_embedding_dim,
                 gnn_hidden_dim, gnn_layers, num_classes=1):
        super().__init__()
        self.account_node_embedding = nn.Embedding(num_total_accounts, account_embedding_dim)
        self.transaction_feat_embedder = nn.Linear(num_transaction_features, transaction_embedding_dim)
        self.convs = nn.ModuleList()
        current_dim = account_embedding_dim
        for _ in range(gnn_layers):
            self.convs.append(SAGEConv(current_dim, gnn_hidden_dim))
            current_dim = gnn_hidden_dim
        self.classifier_input_dim = gnn_hidden_dim * 2 + transaction_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, gnn_hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(gnn_hidden_dim // 2, num_classes)
        )
    def forward(self, data):
        # ... (same forward pass logic) ...
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


# --- DDP Setup and Main Training Function ---
def setup_ddp(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Ensure this port is free
    dist.init_process_group(config['distributed_backend'], rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Assigns default GPU for this process

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, config, global_data_parts):
    print(f"Running DDP on rank {rank} / GPU {rank}.")
    setup_ddp(rank, world_size, config)

    # Unpack globally shared preprocessed data
    (sender_ids_all_np, receiver_ids_all_np, labels_all_np, transaction_features_all_np,
     num_total_unique_accounts, _account_s_to_g_id_map, feature_cols, df_processed_for_viz) = global_data_parts
    
    NUM_FEATURES_PER_TRANSACTION = transaction_features_all_np.shape[1]

    # Temporal split on NumPy arrays (done by each process, but it's fast)
    num_total_processed_samples = len(sender_ids_all_np)
    train_split_idx = int(num_total_processed_samples * config['train_split_ratio'])

    train_sender_ids_np = sender_ids_all_np[:train_split_idx]
    train_receiver_ids_np = receiver_ids_all_np[:train_split_idx]
    train_labels_np = labels_all_np[:train_split_idx]
    train_transaction_features_np = transaction_features_all_np[:train_split_idx]

    test_sender_ids_np = sender_ids_all_np[train_split_idx:]
    test_receiver_ids_np = receiver_ids_all_np[train_split_idx:]
    test_labels_np = labels_all_np[train_split_idx:]
    test_transaction_features_np = transaction_features_all_np[train_split_idx:]

    train_dataset = PyGSnapshotDatasetOnline(
        train_sender_ids_np, train_receiver_ids_np, train_labels_np,
        train_transaction_features_np, config['k_neighborhood_transactions'], is_train_split=True, split_name="train"
    )
    test_dataset = None
    if len(test_sender_ids_np) > config['k_neighborhood_transactions']:
        test_dataset = PyGSnapshotDatasetOnline(
            test_sender_ids_np, test_receiver_ids_np, test_labels_np,
            test_transaction_features_np, config['k_neighborhood_transactions'], is_train_split=False, split_name="test"
        )
        if len(test_dataset) == 0: test_dataset = None
    
    # DistributedSampler ensures each GPU gets a different part of the data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = None
    if test_dataset:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = PyGDataLoader(
        train_dataset, batch_size=config['batch_size_per_gpu'], shuffle=False, # Shuffle is handled by sampler
        num_workers=config['num_cpu_workers'], pin_memory=True, sampler=train_sampler
    )
    test_dataloader = []
    if test_dataset and test_sampler:
        test_dataloader = PyGDataLoader(
            test_dataset, batch_size=config['batch_size_per_gpu'], shuffle=False,
            num_workers=config['num_cpu_workers'], pin_memory=True, sampler=test_sampler
        )

    model = PyGTemporalGNN(
        num_total_accounts=num_total_unique_accounts,
        account_embedding_dim=config['account_embedding_dim'],
        num_transaction_features=NUM_FEATURES_PER_TRANSACTION,
        transaction_embedding_dim=config['transaction_embedding_dim'],
        gnn_hidden_dim=config['gnn_hidden_dim'], gnn_layers=config['gnn_layers'],
        num_classes=config['model_output_classes']
    ).to(rank) # Move model to the GPU assigned to this rank

    model = DDP(model, device_ids=[rank], find_unused_parameters=False) # find_unused_parameters can be True if complex model

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(True)) # Enable AMP for DDP

    criterion = None
    if config['pos_weight_enabled']:
        if len(train_labels_np) > config['k_neighborhood_transactions']:
            effective_train_labels = train_labels_np[config['k_neighborhood_transactions']:]
            train_illicit_count = np.sum(effective_train_labels == 1)
            train_licit_count = len(effective_train_labels) - train_illicit_count
            if train_illicit_count > 0 and train_licit_count > 0:
                 pos_weight_value = train_licit_count / train_illicit_count
                 pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(rank)
                 criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if criterion is None: # Default if not enabled or counts are zero
        criterion = nn.BCEWithLogitsLoss().to(rank)

    if rank == 0: print(f"Criterion: {criterion}")

    # --- Training Loop ---
    epoch_training_losses = [] # For rank 0 to collect
    for epoch in range(config['epochs']):
        model.train()
        train_sampler.set_epoch(epoch) # Important for shuffling with DDP
        
        epoch_loss_sum_rank = 0.0
        total_graphs_processed_rank = 0
        epoch_start_time_rank = time.time()
        optimizer.zero_grad()

        for i, data_batch in enumerate(train_dataloader):
            data_batch = data_batch.to(rank) # Move batch to current GPU
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                logits = model(data_batch) # DDP model forward
                loss = criterion(logits, data_batch.y)
            
            loss_to_backward = loss / config['gradient_accumulation_steps']
            scaler.scale(loss_to_backward).backward() # DDP handles gradient synchronization
            
            epoch_loss_sum_rank += loss.item() * data_batch.num_graphs
            total_graphs_processed_rank += data_batch.num_graphs

            if (i + 1) % config['gradient_accumulation_steps'] == 0 or (i + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if rank == 0 and ((i // config['gradient_accumulation_steps']) + 1) % config['print_interval_batches'] == 0 :
                # Note: This loss is local to rank 0's processed items if not aggregated
                # For a more accurate global average loss, you'd need to dist.all_reduce
                if total_graphs_processed_rank > 0:
                    avg_loss_so_far_rank = epoch_loss_sum_rank / total_graphs_processed_rank
                    print(f"GPU {rank} | Epoch {epoch+1}, Batch Step { (i // config['gradient_accumulation_steps']) +1 } / {len(train_dataloader)//config['gradient_accumulation_steps']}, "
                          f"Avg Loss: {avg_loss_so_far_rank:.4f}")
        
        # Aggregate losses from all GPUs for accurate epoch loss (on rank 0)
        epoch_loss_tensor_rank = torch.tensor([epoch_loss_sum_rank], device=rank)
        total_graphs_tensor_rank = torch.tensor([total_graphs_processed_rank], device=rank)
        dist.all_reduce(epoch_loss_tensor_rank, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_graphs_tensor_rank, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_epoch_loss_global = epoch_loss_tensor_rank.item() / total_graphs_tensor_rank.item() if total_graphs_tensor_rank.item() > 0 else float('nan')
            epoch_training_losses.append(avg_epoch_loss_global) # Store for viz
            epoch_total_time = time.time() - epoch_start_time_rank # Rank 0's time, roughly represents all
            print(f"--- Epoch {epoch+1}/{config['epochs']} Summary (Rank 0) ---")
            print(f"  Global Avg Train Loss (per graph): {avg_epoch_loss_global:.4f}, Duration: {epoch_total_time:.2f}s")

        # --- Evaluation Loop (Run on all ranks, collect results on rank 0) ---
        model.eval()
        all_preds_proba_eval_rank = []
        all_labels_eval_rank = []
        
        if test_dataloader and len(test_dataloader) > 0:
            if test_sampler: test_sampler.set_epoch(epoch) # Good practice though shuffle=False
            with torch.no_grad():
                for data_batch_eval in test_dataloader:
                    data_batch_eval = data_batch_eval.to(rank)
                    logits_eval = model(data_batch_eval) # DDP model
                    probs_eval = torch.sigmoid(logits_eval)
                    all_preds_proba_eval_rank.append(probs_eval.cpu()) # Collect on CPU
                    all_labels_eval_rank.append(data_batch_eval.y.cpu())

        # Gather results from all ranks to rank 0
        # This is a common pattern for DDP evaluation.
        # It can be slow if the eval set is huge.
        # Alternative: Each rank saves its part, then combine offline.
        
        # Pad lists to same size for gather (important!)
        # This is a simplified gather. For robust gathering of tensors of varying sizes,
        # you might need dist.all_gather_object or more complex logic.
        # For simplicity, we assume each GPU processes roughly the same number of test samples.
        # For a more robust solution, use torch.distributed.gather_object if available,
        # or save parts to disk and combine.
        
        # Let's do a simpler aggregation for metrics calculation on rank 0
        # Each rank calculates metrics on its part, then average metrics (approximate)
        # OR, gather all predictions and labels to rank 0 (can be memory intensive)

        # Simplified: Each rank computes its own metrics, rank 0 prints.
        # This is NOT a global evaluation unless test_sampler ensures no overlap and full coverage.
        # For true global eval, all_preds and all_labels must be gathered.
        
        if rank == 0 and all_labels_eval_rank and all_preds_proba_eval_rank:
            # This part is tricky with DDP. For a true global eval, you need to gather
            # all_preds_proba_eval and all_labels_eval from all ranks to rank 0.
            # This requires careful handling of list/tensor sizes.
            # Example (conceptual, may need torch.cat and padding/unpadding):
            
            gathered_preds_list = [None for _ in range(world_size)]
            gathered_labels_list = [None for _ in range(world_size)]
            
            # Convert local lists of tensors to single tensors
            if all_preds_proba_eval_rank:
                local_preds_tensor = torch.cat(all_preds_proba_eval_rank, dim=0)
                local_labels_tensor = torch.cat(all_labels_eval_rank, dim=0)
            else: # Handle empty test set on a rank
                local_preds_tensor = torch.empty(0, dtype=torch.float32)
                local_labels_tensor = torch.empty(0, dtype=torch.float32)

            dist.gather_object(
                obj=(local_preds_tensor.numpy(), local_labels_tensor.numpy()),
                object_gather_list=gathered_preds_list if rank == 0 else None, # Only rank 0 receives
                dst=0
            )

            if rank == 0:
                final_preds_np = []
                final_labels_np = []
                for item in gathered_preds_list:
                    if item is not None:
                        preds_part, labels_part = item
                        if preds_part.ndim > 0 and preds_part.shape[0] > 0: final_preds_np.append(preds_part)
                        if labels_part.ndim > 0 and labels_part.shape[0] > 0: final_labels_np.append(labels_part)
                
                if final_preds_np and final_labels_np:
                    all_preds_proba_np_eval_global = np.concatenate(final_preds_np).squeeze()
                    all_labels_np_eval_global = np.concatenate(final_labels_np).squeeze()

                    if len(all_preds_proba_np_eval_global.shape) == 0: all_preds_proba_np_eval_global = np.array([all_preds_proba_np_eval_global])
                    if len(all_labels_np_eval_global.shape) == 0: all_labels_np_eval_global = np.array([all_labels_np_eval_global])

                    if len(np.unique(all_labels_np_eval_global)) > 1:
                        auroc_eval = roc_auc_score(all_labels_np_eval_global, all_preds_proba_np_eval_global)
                        auprc_eval = average_precision_score(all_labels_np_eval_global, all_preds_proba_np_eval_global)
                        print(f"  Epoch {epoch+1} Global Eval - AUROC: {auroc_eval:.4f}, AUPRC: {auprc_eval:.4f}")
                        # ... (rest of classification report, confusion matrix using global results) ...
                        threshold = config.get('plot_threshold', 0.5)
                        y_pred_binary_eval_global = (all_preds_proba_np_eval_global >= threshold).astype(int)
                        print("\n Classification Report (Global Test Set):")
                        print(classification_report(all_labels_np_eval_global, y_pred_binary_eval_global, target_names=["Licit (0)", "Illicit (1)"], zero_division=0))
                    else:
                        print(f"  Epoch {epoch+1} Global Eval - Only one class present in gathered test labels. Full metrics not available.")
                else:
                    print(f"  Epoch {epoch+1} Global Eval - No predictions or labels gathered.")
        
        # Synchronize all processes before starting next epoch
        dist.barrier()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if rank == 0:
        print("\n--- DDP GNN Training Finished (Rank 0) ---")
        # Rank 0 can save the model
        # For DDP, it's common to save model.module.state_dict()
        # torch.save(model.module.state_dict(), "ddp_model_final.pth")
        # print("Model saved by Rank 0.")

        # --- Pass data to visualization (if enabled) ---
        # This part would be outside main_worker, called only by rank 0 after mp.spawn finishes
        # For simplicity, if viz runs in the same script, these variables need to be "returned" or made accessible
        # For now, we assume visualization script is separate and would load saved model/metrics.
        # Or, you can call the visualization function from rank 0 here.
        # For this example, let's just print that it would happen.
        if config.get('visualization_enabled', False):
            print("\n--- Data for Visualization (Rank 0) ---")
            print(f"Epoch Training Losses: {epoch_training_losses}")
            # For eval metrics, you'd use the 'all_preds_proba_np_eval_global' and 'all_labels_np_eval_global'
            # and 'y_pred_binary_eval_global' from the last epoch if you store them.
            print("Evaluation metrics from the last epoch would be used for visualization.")
            print(f"DataFrame for dataset nature plots: df_processed_for_viz.shape={df_processed_for_viz.shape}")
            print(f"Feature columns: {feature_cols}")


    cleanup_ddp()


if __name__ == "__main__":
    CONFIG = load_config()
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1 # Number of GPUs

    if world_size == 0: # CPU only
        print("No GPUs found. Running on CPU in a single process (DDP setup skipped).")
        # Load data once
        df_raw_main = pd.read_csv(CONFIG['data_path'])
        global_data_parts_main = preprocess_to_numpy_parts_for_pyg(df_raw_main, CONFIG)
        main_worker(0, 1, CONFIG, global_data_parts_main) # rank 0, world_size 1
    elif world_size == 1 and torch.cuda.is_available() : # Single GPU
        print("1 GPU found. Running in a single process (DDP setup skipped).")
        df_raw_main = pd.read_csv(CONFIG['data_path'])
        global_data_parts_main = preprocess_to_numpy_parts_for_pyg(df_raw_main, CONFIG)
        main_worker(0, 1, CONFIG, global_data_parts_main) # rank 0, world_size 1, will use cuda:0
    else: # Multi-GPU
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        # Preprocess data ONCE in the main process and share it
        # This avoids each spawned process re-reading and re-prepping the large CSV.
        # However, passing large NumPy arrays via mp.spawn args can be tricky.
        # A common pattern is to load/preprocess and then let child processes access shared memory
        # or simply let each process load (if preprocessing is fast enough and not memory prohibitive per process).
        # For this dataset size, let's try loading once.
        df_raw_main = pd.read_csv(CONFIG['data_path'])
        global_data_parts_main = preprocess_to_numpy_parts_for_pyg(df_raw_main, CONFIG)
        
        mp.spawn(main_worker,
                 args=(world_size, CONFIG, global_data_parts_main),
                 nprocs=world_size,
                 join=True)