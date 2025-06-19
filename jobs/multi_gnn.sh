#!/bin/bash
#$ -cwd
#$ -m beas

# === Job configuration ===
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=1

# === Job name and output ===
module load python
#virtualenv pytorch_env
source pytorch_env/bin/activate
#pip install torch torch_geometric numpy scikit-learn pandas seaborn matplotlib PyYaml

YQ=~/bin/yq
k=$($YQ '.k_neighborhood_transactions' config/config.yaml)
layers=$($YQ '.gnn_layers' config/config.yaml)
lr=$($YQ '.learning_rate' config/config.yaml)
loss_type=$($YQ '.loss.type' config/config.yaml)
acc_emb=$($YQ '.account_embedding_dim' config/config.yaml)
tx_emb=$($YQ '.transaction_embedding_dim' config/config.yaml)
threshold=$($YQ '.evaluation.monitoring_threshold' config/config.yaml)

log_name="logs/GNN_K${k}_L${layers}_emb${acc_emb}x${tx_emb}_${loss_type}_lr${lr}_thresh${threshold}.txt"


nvidia-smi

#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

# === Job script ===
python code/gnn.py > "$log_name" 2>&1

