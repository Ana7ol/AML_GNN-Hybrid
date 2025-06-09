#!/bin/bash
#$ -cwd
#$ -m beas

# === Job configuration ===
#$ -pe smp 32
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=4
#$ -o ./logs/MultiGPU_GNN2_output.txt

# === Job name and output ===
module load python
#virtualenv pytorch_env
source pytorch_env/bin/activate
#pip install torch torch_geometric numpy scikit-learn pandas seaborn matplotlib PyYaml
rm ./logs/MultiGPU_GNN_output.txt
nvidia-smi

#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL 

# === Job script ===
python code/gnn.py
