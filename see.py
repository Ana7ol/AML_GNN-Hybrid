import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os # For file path joining
import kagglehub

# --- Configuration ---
# Set the path to the directory containing the 'elliptic_bitcoin_dataset' folder
# Example: If your structure is /home/user/data/elliptic_bitcoin_dataset/
DATA_DIR = "./" # CHANGE THIS to the directory *containing* elliptic_bitcoin_dataset

ELLIPTIC_DIR = os.path.join(DATA_DIR, "elliptic_bitcoin_dataset")

N_FEATURES = 166  # Time step + 165 features
EMBEDDING_DIM = 64
BATCH_SIZE = 256 # Can increase if GPU memory allows
EPOCHS = 15 # Adjust as needed, can take time on full dataset
LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
DBSCAN_EPS = 0.7   # *** Needs tuning based on embedding distribution ***
DBSCAN_MIN_SAMPLES = 10 # *** Needs tuning ***

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Load and Prepare Elliptic Data ---
print("Loading Elliptic dataset...")
try:
    
    
    # edgelist_df = pd.read_csv(os.path.join(ELLIPTIC_DIR, "elliptic_txs_edgelist.csv")) # Not used by CNN-GRU
except FileNotFoundError:
    #print(f"Error: Dataset files not found in {ELLIPTIC_DIR}")
    print("Please download the Elliptic dataset and place it in the correct directory,")
    print("or update the DATA_DIR variable.")
    exit()

# Assign feature headers (txId, Time step, f1 to f165)
feature_headers = ['txId', 'Time step'] + [f'f{i}' for i in range(1, 166)]
features_df.columns = feature_headers

# Merge features and classes
df = pd.merge(features_df, classes_df, on='txId', how='left') # Keep all transactions

# Map classes for evaluation: 1 (illicit) -> 1, 2 (licit) -> 0, unknown -> NaN
df['class'] = df['class'].map({'1': 1, '2': 0})

# Separate features (X) and labels (y)
# Features: Time step + f1 to f165
X_cols = ['Time step'] + [f'f{i}' for i in range(1, 166)]
X_all = df[X_cols].values
y_all = df['class'].values # Contains NaN for unknowns

# Identify known labels for evaluation later
known_indices = df.index[df['class'].notna()].tolist()
y_known = df.loc[known_indices, 'class'].values.astype(int)
txId_known = df.loc[known_indices, 'txId'].values # Keep track of txId if needed

print(f"Total transactions: {len(df)}")
print(f"Transactions with known labels: {len(known_indices)}")
print(f"Illicit (1): {np.sum(y_known == 1)}, Licit (0): {np.sum(y_known == 0)}")

# Scale features (fit only on training data, but here we apply to all for unsupervised)
print("Scaling features...")
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)


# --- 2. PyTorch Dataset and DataLoader ---
class EllipticDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return single view for simplified contrastive loss
        return self.features[idx]

# Use all data for unsupervised training
dataset = EllipticDataset(X_all_scaled)
# drop_last=True is important for the artificial pairing in contrastive_loss
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# --- 3. CNN-GRU Encoder Model (same as previous) ---
class CNNGRUEncoder(nn.Module):
    def __init__(self, input_dim, cnn_channels1=32, cnn_channels2=64,
                 kernel_size=5, pool_kernel=2,
                 gru_hidden_size=128, gru_layers=1, bidirectional=True,
                 embedding_dim=64):
        super(CNNGRUEncoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(1, cnn_channels1, kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool_kernel)
        self.conv2 = nn.Conv1d(cnn_channels1, cnn_channels2, kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(pool_kernel)

        l_out1 = math.floor(input_dim / pool_kernel)
        l_out2 = math.floor(l_out1 / pool_kernel)
        self.cnn_output_length = l_out2
        self.cnn_output_channels = cnn_channels2

        self.gru = nn.GRU(self.cnn_output_channels, gru_hidden_size, gru_layers,
                          batch_first=True, bidirectional=bidirectional)
        gru_output_dim = gru_hidden_size * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(gru_output_dim, embedding_dim)
        print(f"CNN Output Sequence Length: {self.cnn_output_length}, Channels: {self.cnn_output_channels}")


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        if self.gru.bidirectional:
            gru_out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            gru_out = h_n[-1,:,:]
        embedding = self.fc_out(gru_out)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

# Instantiate encoder
encoder = CNNGRUEncoder(input_dim=N_FEATURES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
print("\nCNN-GRU Encoder Architecture:")
print(encoder)


# --- 4. Contrastive Loss Function (InfoNCE like - same as previous) ---
def contrastive_loss(z, temperature):
    n = z.shape[0]
    if n % 2 != 0:
         return torch.tensor(0.0, device=DEVICE, requires_grad=True) # Skip odd batches

    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    mask = torch.eye(n, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    n_pairs = n // 2
    indices_i = torch.arange(n_pairs).to(DEVICE)
    indices_j = torch.arange(n_pairs, n).to(DEVICE)

    sim_pos = torch.exp(sim_matrix[indices_i, indices_j] / temperature)
    sim_all_neg_i = torch.exp(sim_matrix[indices_i] / temperature).sum(dim=1)
    sim_all_neg_j = torch.exp(sim_matrix[indices_j] / temperature).sum(dim=1)

    loss_i = -torch.log(sim_pos / (sim_all_neg_i + 1e-8))
    loss_j = -torch.log(sim_pos / (sim_all_neg_j + 1e-8))
    loss = (loss_i.mean() + loss_j.mean()) / 2
    return loss


# --- 5. Training Loop ---
print("\nStarting Unsupervised Training with CNN-GRU Encoder...")
encoder.train()
losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    batches_processed = 0
    for batch_idx, data in enumerate(dataloader):
        # Artificial pairing for simplified contrastive loss demo
        actual_batch_size = data.shape[0] # Could be smaller than BATCH_SIZE if drop_last=False used
        if actual_batch_size < 2 or actual_batch_size % 2 != 0 : continue # Ensure we have pairs

        data = data.to(DEVICE)
        view1 = data[:actual_batch_size//2]
        view2 = data[actual_batch_size//2:]

        z1 = encoder(view1)
        z2 = encoder(view2)
        z_combined = torch.cat([z1, z2], dim=0)

        loss = contrastive_loss(z_combined, TEMPERATURE)

        optimizer.zero_grad()
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
             continue
        else:
             loss.backward()
             optimizer.step()
             epoch_loss += loss.item()
             batches_processed += 1

        # Print progress occasionally
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Current Loss: {loss.item():.4f}")


    if batches_processed > 0:
        avg_epoch_loss = epoch_loss / batches_processed
        losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Training Loss: {avg_epoch_loss:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{EPOCHS}], No valid batches processed.")
        losses.append(float('nan'))

print("Training Finished.")

# Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(losses) + 1), losses) # Adjust x-axis range based on actual epochs run
plt.title("Training Loss over Epochs (CNN-GRU on Elliptic)")
plt.xlabel("Epoch")
plt.ylabel("Contrastive Loss")
plt.grid(True)
plt.savefig("elliptic_training_loss_cnngru.png") # Save the plot
plt.show()


# --- 6. Generate Embeddings for ALL data ---
print("Generating final embeddings for all data...")
encoder.eval()
all_embeddings_list = []
# Use a dataloader for all data, no shuffling, potentially larger batch
eval_dataset = EllipticDataset(X_all_scaled)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

with torch.no_grad():
    for data in eval_dataloader:
        data = data.to(DEVICE)
        embeddings = encoder(data)
        all_embeddings_list.append(embeddings.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings_list, axis=0)
print(f"Generated embeddings shape: {all_embeddings.shape}")

# Ensure embeddings count matches original data count
if len(all_embeddings) != len(df):
    print(f"Warning: Embedding count ({len(all_embeddings)}) does not match dataframe rows ({len(df)}). Check dataloading.")
    # Attempt to truncate if slightly off due to batching, but investigate if large difference
    all_embeddings = all_embeddings[:len(df)]


# --- 7. Anomaly Detection using Clustering (DBSCAN) ---
print("Performing DBSCAN clustering...")
# *** Tuning eps and min_samples is CRITICAL ***
# Start with values based on previous synthetic data, but expect to adjust.
# Consider using OPTICS or analyzing k-distance plot to help choose eps.
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine', n_jobs=-1)
cluster_labels = dbscan.fit_predict(all_embeddings)

# Predicted anomalies are those labeled -1 by DBSCAN
predicted_anomalies_all = (cluster_labels == -1).astype(int) # 1=anomaly, 0=normal (in cluster)

print(f"Number of predicted anomalies (outliers): {np.sum(predicted_anomalies_all)}")
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters found: {n_clusters}")


# --- 8. Evaluation (on known labels only) ---
print("\n--- Evaluating on known licit/illicit transactions ---")

# Select embeddings and predictions corresponding to known labels
embeddings_known = all_embeddings[known_indices]
predicted_anomalies_known = predicted_anomalies_all[known_indices]
# True labels: y_known (0=licit/normal, 1=illicit/anomaly)

if len(predicted_anomalies_known) == 0:
     print("No known labels found or predicted. Cannot evaluate.")
else:
    accuracy = accuracy_score(y_known, predicted_anomalies_known)
    precision, recall, f1, _ = precision_recall_fscore_support(y_known, predicted_anomalies_known, average='binary', zero_division=0) # Focus on illicit class (1)

    try:
        auroc = roc_auc_score(y_known, predicted_anomalies_known)
    except ValueError as e:
        print(f"Could not calculate AUROC (likely only one class predicted): {e}")
        auroc = float('nan')

    print(f"Accuracy (known): {accuracy:.4f}")
    print(f"AUROC (known): {auroc:.4f}")
    print(f"Precision (for illicit): {precision:.4f}")
    print(f"Recall (for illicit): {recall:.4f}")
    print(f"F1-Score (for illicit): {f1:.4f}")

    print("\nClassification Report (known):")
    print(classification_report(y_known, predicted_anomalies_known, target_names=["Licit (0)", "Illicit (1)"], zero_division=0))


# --- 9. Visualization (Optional - PCA reduction on known data) ---
print("Visualizing embeddings for known data using PCA...")

# Only visualize the subset with known labels for clarity
pca = PCA(n_components=2)
embeddings_known_2d = pca.fit_transform(embeddings_known)

df_viz = pd.DataFrame({
    'PCA1': embeddings_known_2d[:, 0],
    'PCA2': embeddings_known_2d[:, 1],
    'True Label': y_known,
    # Cluster labels for known nodes need filtering from original cluster_labels
    'Cluster': cluster_labels[known_indices],
    'Predicted Anomaly': predicted_anomalies_known
})

plt.figure(figsize=(12, 10))
plt.suptitle("CNN-GRU Embeddings Visualization (Known Elliptic Data)", fontsize=16)

# Plot True Labels
plt.subplot(2, 2, 1)
sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='True Label', palette={0: 'blue', 1: 'red'}, s=10, alpha=0.5)
plt.title('Embeddings Colored by True Label (0=Licit, 1=Illicit)')
plt.grid(True)

# Plot DBSCAN Cluster Labels (including outliers -1)
plt.subplot(2, 2, 2)
# Use a qualitative palette, maybe map -1 to black
unique_clusters = sorted(df_viz['Cluster'].unique())
palette = sns.color_palette("deep", n_colors=len(unique_clusters)-1 if -1 in unique_clusters else len(unique_clusters))
cluster_colors = {cluster: color for cluster, color in zip(sorted([c for c in unique_clusters if c!=-1]), palette)}
if -1 in unique_clusters: cluster_colors[-1] = 'black' # Outliers in black
sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='Cluster', palette=cluster_colors, s=10, alpha=0.5)
plt.title(f'DBSCAN Clusters (eps={DBSCAN_EPS}, min={DBSCAN_MIN_SAMPLES})')
plt.grid(True)


# Plot Predicted Anomalies
plt.subplot(2, 2, 3)
sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='Predicted Anomaly', palette={0: 'green', 1: 'orange'}, s=10, alpha=0.5)
plt.title('Predicted Anomalies (1=Anomaly/Outlier)')
plt.grid(True)

# Highlight incorrect predictions
df_viz['Correct'] = (df_viz['True Label'] == df_viz['Predicted Anomaly'])
plt.subplot(2, 2, 4)
sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='Correct', palette={True: 'green', False: 'red'}, style='True Label', s=15, alpha=0.7)
plt.title('Correct (Green) vs Incorrect (Red) Predictions')
plt.grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("elliptic_embeddings_vis_cnngru.png")
plt.show()

print("\n--- NOTE: CNN-GRU model does not use graph structure (edges). Consider GNNs for better performance on Elliptic. ---")
print("--- NOTE: DBSCAN parameters (eps, min_samples) likely require significant tuning. ---")