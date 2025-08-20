#!/bin/bash


#$ -cwd                         # Start job in the current working directory
#$ -pe smp 8                    # Request 8 parallel CPU cores
#$ -l h_vmem=11G                # Request 11GB of virtual memory per core (total 88G)
#$ -l h_rt=1:0:0                # Request a walltime of 1 hour
#$ -l gpu=1                     # Request 1 GPU
#$ -N MLP                       # Job name


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"
echo "Working Directory: $(pwd)"
echo "---"

# --- 1. Set up the Environment ---
module load python
source "$HOME/AML/pytorch_env/bin/activate"



# Control variable to select which script to run
# Defaults to 'gnn' if not specified
RUN_MODE=${MODE:-gnn}

# Shared hyperparameters
LEARNING_RATE=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-4096}
DATA_PATH=${DATA_PATH:-"LI-Small_Trans.csv"}

# GNN-specific hyperparameters
TX_EMBEDDING=${TX_EMB:-128}
GNN_EPOCHS=${GNN_EPOCHS:-100}

# MLP-specific / Downstream-specific hyperparameters
DOWNSTREAM_EPOCHS=${DS_EPOCHS:-100}
MLP_EPOCHS=${MLP_EPOCHS:-150}
HIDDEN_DIM=${HDIM:-128}




if [ "$RUN_MODE" = "mlp" ]; then
    # --- Execute the MLP Benchmark ---
    echo "Running the MLP benchmark (run_benchmark_mlp.py)"
    echo "Parameters:"
    echo "  --lr: $LEARNING_RATE"
    echo "  --batch_size: $BATCH_SIZE"
    echo "  --epochs: $MLP_EPOCHS"
    echo "  --hidden_dim: $HIDDEN_DIM"
    echo "  --data_path: $DATA_PATH"
    echo "---"

    python code/MLP.py \
        --lr $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --epochs $MLP_EPOCHS \
        --hidden_dim $HIDDEN_DIM \
        --data_path $DATA_PATH

else
    echo "Error: Invalid mode specified: '$RUN_MODE'. Must be 'gnn' or 'mlp'."
    exit 1
fi

echo "---"
echo "Job finished at $(date)"
