# =============================================================================
#
# FULL SCRIPT - BALANCED PARALLEL PREPROCESSING (Version 8)
#
# Key Improvements in this version:
# 1. True Parallel Load Balancing: The sequence creation step now pre-chunks
#    the account groups before sending them to worker processes. This prevents
#    the main process from becoming a bottleneck and ensures all CPU cores
#    are utilized effectively, leading to a major speedup.
# 2. Optimized Worker Function: The helper function now processes a large
#    chunk of accounts, minimizing the overhead of parallelization.
#
# =============================================================================

# --- 1. IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv, DataParallel as PyGDataParallel

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, classification_report)

from lightly.loss import NTXentLoss

import pandas as pd
import numpy as np
import time
import gc
from tqdm import tqdm
import os
# Import for parallel processing
from joblib import Parallel, delayed

# --- 2. SETUP AND HYPERPARAMETERS ---
# Check for available GPUs
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    NUM_GPUS = torch.cuda.device_count()
    print(f"Using {NUM_GPUS} GPU(s). Primary device: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    NUM_GPUS = 0
    print(f"Using device: {DEVICE}")


# General Hyperparameters
ENCODER_EMBEDDING_DIM = 128
PROJECTION_DIM = 32
# Adjust batch size based on number of GPUs to maintain effective batch size per GPU
BATCH_SIZE = 128 * max(1, NUM_GPUS)
print(f"Effective Batch Size: {BATCH_SIZE}")
EPOCHS = 10 # SSL Epochs
GNN_EPOCHS = 100 # GNN Epochs
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1

# Sequence Creation Hyperparameters
SEQUENCE_LENGTH = 30
STEP_SIZE = 5

# --- 3. PREPROCESSING AND DATASET CLASSES (Parallelized Version) ---

def preprocess_features(df_input):
    print("Starting feature preprocessing...")
    df = df_input.copy()
    if 'Timestamp' in df.columns:
        df['Hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24.0)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24.0)
        df['Minute_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.minute / 60.0)
        df['Minute_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.minute / 60.0)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.dayofweek / 7.0)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.dayofweek / 7.0)
        df = df.drop(['Timestamp'], axis=1)
    if 'Account' in df.columns and 'Account.1' in df.columns:
        def hex_to_int_safe(hex_str):
            try: return int(str(hex_str), 16)
            except: return -1
        df['Account_Num'] = df['Account'].apply(hex_to_int_safe)
        df['Account.1_Num'] = df['Account.1'].apply(hex_to_int_safe)
        df = df.drop(['Account', 'Account.1'], axis=1)
    if 'Amount Received' in df.columns:
        df['Amount_Log'] = np.log1p(df['Amount Received'])
        df = df.drop(['Amount Received'], axis=1)
    currency_col_to_use = 'Receiving Currency'
    if currency_col_to_use in df.columns:
        categorical_to_encode = [currency_col_to_use, 'Payment Format']
        for col in categorical_to_encode:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df = pd.get_dummies(df, columns=categorical_to_encode, prefix=['Currency', 'Format'], dummy_na=False, dtype=float)
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"\nPreprocessing finished. Final feature shape: {df.shape}")
    return df.values.astype(np.float32)

def _process_chunk_of_accounts(groups_chunk, sequence_length, step_size, feature_cols):
    """
    Helper function to process a large chunk of account groups.
    This function is called by a single parallel worker.
    """
    chunk_sequences = []
    chunk_labels = []
    
    # This loop runs inside the worker process
    for _, group in groups_chunk:
        transactions = group[feature_cols].values
        labels = group['Is Laundering'].values
        num_transactions = len(transactions)

        if num_transactions < sequence_length:
            padding_needed = sequence_length - num_transactions
            padded_features = np.pad(transactions, ((0, padding_needed), (0, 0)), 'constant', constant_values=0)
            chunk_sequences.append(padded_features)
            chunk_labels.append(labels[-1])
        else:
            for i in range(0, num_transactions - sequence_length + 1, step_size):
                chunk_sequences.append(transactions[i : i + sequence_length])
                chunk_labels.append(labels[i + sequence_length - 1])
                
    return chunk_sequences, chunk_labels

def create_transaction_sequences_parallel(df_original_with_ts, features_processed, sequence_length=10, step_size=5):
    """
    Transforms a flat list of transactions into sequences using balanced parallel processing.
    """
    print(f"Creating sequences with length {sequence_length} and step {step_size}...")
    df_temp = df_original_with_ts.copy()
    feature_cols = [f'feature_{i}' for i in range(features_processed.shape[1])]
    df_temp[feature_cols] = features_processed
    df_temp.sort_values(by=['Account', 'Timestamp'], inplace=True)
    
    n_jobs = -1  # Use all available cores
    cpu_count = os.cpu_count()
    print(f"Starting balanced parallel processing for sequence creation on {cpu_count} cores...")
    
    # Group data in memory
    grouped = df_temp.groupby('Account')
    
    # Pre-chunk the groups to ensure balanced workload
    # We create a number of chunks proportional to the number of cores
    n_chunks = cpu_count * 4  # Create more chunks than cores for better dynamic scheduling
    all_groups = list(grouped)
    chunk_size = len(all_groups) // n_chunks + 1
    chunks = [all_groups[i:i + chunk_size] for i in range(0, len(all_groups), chunk_size)]
    
    print(f"Split {len(all_groups)} account groups into {len(chunks)} chunks for parallel processing.")
    
    # Process the chunks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_chunk_of_accounts)(chunk, sequence_length, step_size, feature_cols) for chunk in chunks
    )
    
    # Collect results from all parallel processes
    all_sequences = []
    all_labels = []
    print("Collecting results from parallel workers...")
    for chunk_sequences, chunk_labels in results:
        all_sequences.extend(chunk_sequences)
        all_labels.extend(chunk_labels)
        
    final_sequences = np.array(all_sequences, dtype=np.float32)
    final_labels = np.array(all_labels, dtype=np.int64)
    print(f"Finished creating sequences. Final shape: {final_sequences.shape}")
    return final_sequences, final_labels

class ContrastiveDataset(Dataset):
    def __init__(self, sequence_array):
        self.sequences = torch.tensor(sequence_array, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.sequences[idx]

class Simple2DDataset(Dataset):
    def __init__(self, features_np_array):
        self.features = torch.tensor(features_np_array, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx]

# --- 4. SSL MODEL ARCHITECTURES ---
class TransactionFeatureCNN(nn.Module):
    def __init__(self, num_input_features, cnn_channels=[64, 128], kernel_sizes=[5, 3], output_embedding_dim=128):
        super().__init__()
        cnn_layers = []
        in_channels = 1
        for i, out_c in enumerate(cnn_channels):
            cnn_layers.append(nn.Conv1d(in_channels, out_c, kernel_sizes[i], padding='same'))
            cnn_layers.append(nn.ReLU())
            in_channels = out_c
        self.cnn_block = nn.Sequential(*cnn_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out_cnn = nn.Linear(cnn_channels[-1], output_embedding_dim)
        print(f"TransactionFeatureCNN: InputFeat={num_input_features}, OutputEmb={output_embedding_dim}")
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn_block(x)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.fc_out_cnn(x)

class TransactionSequenceEncoder_CNNthenGRU(nn.Module):
    def __init__(self, num_features, transaction_embedding_dim=128, gru_hidden_size=256, gru_layers=5, final_sequence_embedding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.transaction_cnn_embedder = TransactionFeatureCNN(num_features, output_embedding_dim=transaction_embedding_dim)
        self.gru = nn.GRU(input_size=transaction_embedding_dim, hidden_size=gru_hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out_sequence = nn.Linear(gru_hidden_size * 2, final_sequence_embedding_dim)
        print(f"TransactionSequenceEncoder_CNNthenGRU: FinalEmb={final_sequence_embedding_dim}, Dropout={dropout_rate}")
    def forward(self, x):
        b, s, f = x.shape
        x_flat = x.reshape(b * s, f)
        emb_flat = self.transaction_cnn_embedder(x_flat)
        emb_seq = emb_flat.reshape(b, s, -1)
        if torch.cuda.device_count() > 1:
            self.gru.flatten_parameters()
        _, h_n = self.gru(emb_seq)
        h_n_last = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        h_n_last = self.dropout(h_n_last)
        sequence_embedding = self.fc_out_sequence(h_n_last)
        return F.normalize(sequence_embedding, p=2, dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        print(f"Projection Head: Input={input_dim}, Hidden={hidden_dim}, Output={output_dim}")
    def forward(self, x):
        return self.head(x)

# --- 5. GNN MODEL ARCHITECTURE AND HELPERS ---
def create_pyg_graph(df, transaction_embeddings, labels, num_nodes):
    print("--- Starting Graph Construction ---")
    all_accounts = pd.concat([df['Account'], df['Account.1']]).unique()
    account_mapping = {acc_id: i for i, acc_id in enumerate(all_accounts)}
    print(f"Found {len(all_accounts)} unique accounts (nodes).")
    source_nodes = df['Account'].map(account_mapping).values
    dest_nodes = df['Account.1'].map(account_mapping).values
    edge_index = torch.from_numpy(np.stack([source_nodes, dest_nodes])).to(torch.long)
    edge_attr = torch.tensor(transaction_embeddings, dtype=torch.float)
    edge_label = torch.tensor(labels, dtype=torch.long)
    graph = Data(num_nodes=num_nodes, edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
    print("\n--- Applying Train/Val/Test Split to Edges ---")
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.2, is_undirected=False,
        add_negative_train_samples=True, key="edge_label", split_labels=True
    )
    train_data, val_data, test_data = transform(graph)
    print("\n--- Graph Construction Complete ---")
    return train_data, val_data, test_data

class EdgeClassifierGNN(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim, hidden_dim, out_channels=1):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, node_feat_dim)
        self.conv1 = SAGEConv(node_feat_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        classifier_in_dim = 2 * hidden_dim + edge_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )
        print("Initialized GNN that uses both node and edge features (EdgeClassifierGNN)")
    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h
    def decode(self, h, edge_label_index, edge_attr):
        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        final_edge_emb = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.classifier(final_edge_emb)
    # This special forward is for PyG's DataParallel, which is not used here for simplicity
    # but kept for potential future use. The direct train/test calls are clearer.
    def forward(self, data):
        h = self.encode(self.node_emb.weight, data.edge_index)
        return self.decode(h, data.edge_label_index, data.edge_attr)


# --- 6. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    use_amp = (DEVICE.type == 'cuda')
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    
    # --- DATA PREPARATION (Done Once) ---
    print("\n--- PREPARING DATASET (ONCE) ---")
    df_raw = pd.read_csv("LI-Small_Trans.csv")
    df_gnn = df_raw.copy()
    df_gnn['Timestamp'] = pd.to_datetime(df_gnn['Timestamp'], errors='coerce')
    df_gnn['Amount Received'] = pd.to_numeric(df_gnn['Amount Received'], errors='coerce')
    df_gnn['Amount Paid'] = pd.to_numeric(df_gnn['Amount Paid'], errors='coerce')
    df_gnn.dropna(subset=['Timestamp', 'Amount Received', 'Amount Paid'], inplace=True)
    df_gnn.drop(columns=['Amount Paid', 'Payment Currency'], inplace=True, errors='ignore')
    
    cols_to_drop_for_features = ['Is Laundering', 'Account', 'Account.1']
    X_features_np = preprocess_features(df_gnn.drop(columns=cols_to_drop_for_features))
    y_labels_np = df_gnn['Is Laundering'].values
    assert len(df_gnn) == len(X_features_np), "Row count mismatch!"
    
    # === STAGE 1: SELF-SUPERVISED TRAINING OF THE ENCODER ===
    print("\n" + "="*50 + "\nSTAGE 1: SELF-SUPERVISED LEARNING (SSL) PRE-TRAINING\n" + "="*50 + "\n")
    X_sequences_np, _ = create_transaction_sequences_parallel(df_gnn, X_features_np, SEQUENCE_LENGTH, STEP_SIZE)
    if X_sequences_np.size == 0:
        print("\nERROR: Sequence creation failed. Halting.")
        exit()
    
    N_FEATURES_PROC = X_sequences_np.shape[2]
    ssl_dataset = ContrastiveDataset(X_sequences_np)
    ssl_dataloader = DataLoader(ssl_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    encoder = TransactionSequenceEncoder_CNNthenGRU(N_FEATURES_PROC, final_sequence_embedding_dim=ENCODER_EMBEDDING_DIM).to(DEVICE)
    projection_head = ProjectionHead(ENCODER_EMBEDDING_DIM, ENCODER_EMBEDDING_DIM, PROJECTION_DIM).to(DEVICE)
    
    if NUM_GPUS > 1:
        print(f"Wrapping SSL models for {NUM_GPUS} GPUs...")
        encoder = nn.DataParallel(encoder)
        projection_head = nn.DataParallel(projection_head)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(ssl_dataloader) * EPOCHS)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    print("\n--- Starting SSL Training (with LR Scheduler) ---")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        encoder.train()
        projection_head.train()
        for view1, view2 in tqdm(ssl_dataloader, desc=f"SSL Epoch {epoch+1}/{EPOCHS}"):
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                p1 = projection_head(encoder(view1))
                p2 = projection_head(encoder(view2))
                loss = criterion(p1, p2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {epoch_loss/len(ssl_dataloader):.4f}")

    # === STAGE 2: EMBEDDING GENERATION FOR GNN ---
    print("\n" + "="*50 + "\nSTAGE 2: EMBEDDING GENERATION FOR GNN\n" + "="*50 + "\n")
    individual_dataset = Simple2DDataset(X_features_np)
    individual_dataloader = DataLoader(individual_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    encoder_to_eval = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
    encoder_to_eval.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for batch_features in tqdm(individual_dataloader, desc="Generating Embeddings"):
            batch_features = batch_features.to(DEVICE).unsqueeze(1)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                embeddings = encoder_to_eval(batch_features)
            all_embeddings.append(embeddings.cpu())
    final_embeddings_np_individual = torch.cat(all_embeddings).numpy()
    print(f"Final shape of transaction embeddings: {final_embeddings_np_individual.shape}")

    # === STAGE 3: GNN TRAINING & EVALUATION ===
    print("\n" + "="*50 + "\nSTAGE 3: GRAPH NEURAL NETWORK (GNN) TRAINING & EVALUATION\n" + "="*50 + "\n")
    num_unique_accounts = pd.concat([df_gnn['Account'], df_gnn['Account.1']]).nunique()
    train_graph, val_graph, test_graph = create_pyg_graph(df_gnn, final_embeddings_np_individual, y_labels_np, num_unique_accounts)
    
    train_graph.edge_label = train_graph.edge_label.float()
    val_graph.edge_label = val_graph.edge_label.float()
    test_graph.edge_label = test_graph.edge_label.float()
    
    gnn_model = EdgeClassifierGNN(
        num_nodes=num_unique_accounts, node_feat_dim=64,
        edge_feat_dim=final_embeddings_np_individual.shape[1], hidden_dim=128
    ).to(DEVICE)
    
    if NUM_GPUS > 1:
        print(f"Wrapping GNN model for {NUM_GPUS} GPUs...")
        gnn_model = nn.DataParallel(gnn_model)

    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    gnn_criterion = torch.nn.BCEWithLogitsLoss()

    # The GNN trains on a single large graph, so we move it once
    train_graph.to(DEVICE)
    
    def train_gnn():
        gnn_model.train()
        gnn_optimizer.zero_grad()
        model_to_train = gnn_model.module if isinstance(gnn_model, nn.DataParallel) else gnn_model
        h = model_to_train.encode(model_to_train.node_emb.weight, train_graph.edge_index)
        preds = model_to_train.decode(h, train_graph.edge_label_index, train_graph.edge_attr)
        loss = gnn_criterion(preds.squeeze(), train_graph.edge_label)
        loss.backward()
        gnn_optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_gnn(graph):
        gnn_model.eval()
        graph_on_device = graph.to(DEVICE)
        model_to_test = gnn_model.module if isinstance(gnn_model, nn.DataParallel) else gnn_model
        h = model_to_test.encode(model_to_test.node_emb.weight, graph_on_device.edge_index)
        preds = model_to_test.decode(h, graph_on_device.edge_label_index, graph_on_device.edge_attr)
        probs = preds.squeeze().sigmoid()
        y_true = graph_on_device.edge_label.cpu().numpy()
        y_prob = probs.cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        if len(np.unique(y_true)) < 2:
            return 0.5, np.mean(y_true), y_true, y_pred
        return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob), y_true, y_pred

    print("\n--- Starting GNN Training ---")
    for epoch in range(1, GNN_EPOCHS + 1):
        loss = train_gnn()
        if epoch % 10 == 0 or epoch == 1:
            val_auroc, val_auprc, _, _ = test_gnn(val_graph)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}')
    print("\n--- GNN Training Finished ---")

    # --- FINAL GNN EVALUATION ON TEST SET ---
    print("\n" + "="*50 + "\nFINAL GNN EVALUATION ON TEST SET\n" + "="*50 + "\n")
    test_auroc, test_auprc, y_true_test, y_pred_test = test_gnn(test_graph)
    
    print(f"Final Test AUROC: {test_auroc:.4f}")
    print(f"Final Test AUPRC: {test_auprc:.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true_test, y_pred_test)
    print(pd.DataFrame(cm, 
                       index=['Actual Licit (0)', 'Actual Illicit (1)'], 
                       columns=['Predicted Licit (0)', 'Predicted Illicit (1)']))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true_test, y_pred_test, target_names=['Licit (0)', 'Illicit (1)'], zero_division=0))
