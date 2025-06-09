#!/bin/bash
#$ -cwd
#$ -m beas

# === Job configuration ===
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=1
#$ -o ./logs/GNN_output.txt

# === Job name and output ===
module load python
#virtualenv pytorch_env
source pytorch_env/bin/activate
#pip install torch torch_geometric numpy scikit-learn pandas seaborn matplotlib PyYaml

nvidia-smi

#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL 

# === Job script ===
python GNN.py
