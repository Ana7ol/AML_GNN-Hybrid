#!/bin/bash
#$ -cwd
#$ -m beas

# === Job configuration ===
#$ -pe smp 16
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=2
#$ -o ./logs/GNN2_output.txt

# === Job name and output ===
module load python
#virtualenv pytorch_env
source pytorch_env/bin/activate
#pip install torch torch_geometric numpy scikit-learn pandas seaborn matplotlib PyYaml

# === Job script ===
python CNN_GRU.py
nvidia-smi:
