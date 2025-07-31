import pandas as pd
import numpy as np
import torch
import h5py
import os
import yaml
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# --- load_config is the same ---
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- The worker function is almost the same ---
def process_chunk_for_h5(indices, config, sender_ids, receiver_ids, labels, tx_features, amount_log_idx):
    # This function is now perfect and needs no changes.
    # It correctly processes a local chunk.
    k_history = config['k_neighborhood_transactions']
    all_x_node_ids, all_edge_indices, all_edge_weights, all_target_tx_feats, all_target_node_idx, all_y = [], [], [], [], [], []
    x_node_ptr, edge_ptr = [0], [0]
    
    for i in indices:
        # ... (rest of the function is identical to the previous HDF5 version)
        target_idx = i + k_history
        start_idx = max(0, target_idx - k_history)
        end_idx = target_idx + 1

        senders = sender_ids[start_idx:end_idx]
        receivers = receiver_ids[start_idx:end_idx]
        
        nodes, local_map = np.unique(np.concatenate([senders, receivers]), return_inverse=True)
        local_senders = local_map[:len(senders)]
        local_receivers = local_map[len(senders):]

        edge_index = np.vstack([local_senders, local_receivers])
        edge_weight = tx_features[start_idx:end_idx, amount_log_idx]
        
        target_sender_local = local_senders[-1]
        target_receiver_local = local_receivers[-1]
        
        all_x_node_ids.append(nodes)
        all_edge_indices.append(edge_index.T)
        all_edge_weights.append(edge_weight)
        all_target_tx_feats.append(tx_features[target_idx])
        all_target_node_idx.append(np.array([target_sender_local, target_receiver_local]))
        all_y.append(labels[target_idx])

        x_node_ptr.append(x_node_ptr[-1] + len(nodes))
        edge_ptr.append(edge_ptr[-1] + len(edge_weight))

    return {
        "x_node_ids": np.concatenate(all_x_node_ids, axis=0).astype(np.int64),
        "edge_indices": np.concatenate(all_edge_indices, axis=0).astype(np.int64),
        "edge_weights": np.concatenate(all_edge_weights, axis=0).astype(np.float32),
        "target_tx_feats": np.stack(all_target_tx_feats, axis=0).astype(np.float32),
        "target_node_idx": np.stack(all_target_node_idx, axis=0).astype(np.int64),
        "y": np.array(all_y, dtype=np.float32),
        "x_node_ptr": np.array(x_node_ptr[1:], dtype=np.int64),
        "edge_ptr": np.array(edge_ptr[1:], dtype=np.int64),
    }

def preprocess_and_save_h5(config):
    """Main function to pre-process data and save to a single HDF5 file."""
    print("--- Starting OFFLINE HDF5 Pre-processing (FIXED) ---")
    
    # --- 1. Load Raw Data (same as before) ---
    # ... (omitted for brevity, copy from previous version)
    print(f"Loading raw data from: {config['data_path']}")
    df_raw = pd.read_csv(config['data_path'])
    print("Performing initial feature engineering and ID mapping...")
    df = df_raw.copy(); all_accounts = pd.concat([df['Account'].astype(str), df['Account.1'].astype(str)]).unique(); acc_to_id = {acc: i for i, acc in enumerate(all_accounts)}; num_accounts = len(all_accounts)
    df['Sender_Global_ID'] = df['Account'].astype(str).map(acc_to_id); df['Receiver_Global_ID'] = df['Account.1'].astype(str).map(acc_to_id); df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce'); df.dropna(subset=['Timestamp_dt'], inplace=True)
    df['Amount Received'] = pd.to_numeric(df['Amount Received'], errors='coerce'); df.dropna(subset=['Amount Received'], inplace=True); df['Amount_Log'] = np.log1p(df['Amount Received']).astype(np.float32); df = pd.get_dummies(df, columns=['Payment Format', 'Receiving Currency'], dummy_na=False, dtype=np.float32)
    feature_cols = [col for col in df.columns if col.startswith(('Amount_Log', 'Payment Format_', 'Receiving Currency_'))]; df.sort_values(by='Timestamp_dt', inplace=True); df.reset_index(drop=True, inplace=True)
    sender_ids = df['Sender_Global_ID'].values.astype(np.int64); receiver_ids = df['Receiver_Global_ID'].values.astype(np.int64); labels = df['Is Laundering'].values.astype(np.float32); tx_features = df[feature_cols].values.astype(np.float32); amount_log_idx = feature_cols.index('Amount_Log'); num_tx_features = tx_features.shape[1]

    # --- 2. Process Data in Parallel and Write to a Single HDF5 File ---
    k_history = config['k_neighborhood_transactions']
    num_samples = len(sender_ids) - k_history
    output_path = os.path.join(config['processed_data_path'], 'data.h5')
    os.makedirs(config['processed_data_path'], exist_ok=True)
    
    num_workers = max(1, mp.cpu_count() - 2)
    WORKER_CHUNK_SIZE = 50000
    index_chunks = [range(i, min(i + WORKER_CHUNK_SIZE, num_samples)) for i in range(0, num_samples, WORKER_CHUNK_SIZE)]
    
    worker_func = partial(process_chunk_for_h5,
                          config=config, sender_ids=sender_ids, receiver_ids=receiver_ids,
                          labels=labels, tx_features=tx_features, amount_log_idx=amount_log_idx)
    
    print(f"Processing {num_samples} samples and writing to {output_path}")
    
    # --- CHANGE IS HERE: Track cumulative offsets in the main process ---
    last_x_node_offset = 0
    last_edge_offset = 0
    with h5py.File(output_path, 'w') as f:
        # Create resizable datasets (same as before)
        dsets = {
            "x_node_ids": f.create_dataset("x_node_ids", (0,), maxshape=(None,), dtype=np.int64),
            "edge_indices": f.create_dataset("edge_indices", (0, 2), maxshape=(None, 2), dtype=np.int64),
            "edge_weights": f.create_dataset("edge_weights", (0,), maxshape=(None,), dtype=np.float32),
            "target_tx_feats": f.create_dataset("target_tx_feats", (0, num_tx_features), maxshape=(None, num_tx_features), dtype=np.float32),
            "target_node_idx": f.create_dataset("target_node_idx", (0, 2), maxshape=(None, 2), dtype=np.int64),
            "y": f.create_dataset("y", (0,), maxshape=(None,), dtype=np.float32),
            "x_node_ptr": f.create_dataset("x_node_ptr", (0,), maxshape=(None,), dtype=np.int64),
            "edge_ptr": f.create_dataset("edge_ptr", (0,), maxshape=(None,), dtype=np.int64)
        }

        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=num_samples, desc="Processing Samples") as pbar:
                for result_dict in pool.imap_unordered(worker_func, index_chunks):
                    # --- FIX IS HERE: Adjust the pointers before saving ---
                    result_dict['x_node_ptr'] += last_x_node_offset
                    result_dict['edge_ptr'] += last_edge_offset

                    # Append data from the finished worker
                    for key, data in result_dict.items():
                        dset = dsets[key]
                        dset.resize(dset.shape[0] + data.shape[0], axis=0)
                        dset[-data.shape[0]:] = data

                    # --- UPDATE THE CUMULATIVE OFFSETS ---
                    last_x_node_offset = result_dict['x_node_ptr'][-1]
                    last_edge_offset = result_dict['edge_ptr'][-1]

                    pbar.update(len(result_dict['y']))

    # --- 3. Save Metadata (same as before) ---
    metadata = { 'num_accounts': num_accounts, 'num_tx_features': num_tx_features, 'num_samples': num_samples }
    with open(os.path.join(config['processed_data_path'], 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)

    print("\n--- HDF5 Pre-processing Complete ---")

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Offline HDF5 Data Pre-processing")
    parser.add_argument('--config_path', type=str, default='./config/config.yaml', help='Path to config')
    args = parser.parse_args()
    config = load_config(args.config_path)
    preprocess_and_save_h5(config)
