#!/bin/bash

# ==============================================================================
#           Hyperparameter Grid Search Controller for GNN Experiments
# ==============================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
echo "Controller has set working directory to: $(pwd)"


# --- Configuration ---
JOB_SCRIPT="jobs/new_gnn.sh"
JOB_NAME="new_gnn_worker_job"

# --- Define Your Experiment Grid ---
SEP_BANKS_FLAG=False
ISOLATE_BANK_NAME=null 
K_VALUES=(20 50 100)
ACC_EMB_VALUES=(32 64 128)
TX_EMB_VALUES=(64 128 256)
GNN_LAYERS_VALUES=(2 3 5)
LR_VALUES=(0.0001)

# Calculate total number of experiments for a pre-run sanity check
TOTAL_EXPERIMENTS=$(( ${#K_VALUES[@]} * ${#ACC_EMB_VALUES[@]} * ${#TX_EMB_VALUES[@]} * ${#GNN_LAYERS_VALUES[@]} * ${#LR_VALUES[@]} ))

echo "Grid Search Configuration:"
echo "--------------------------"
echo "K Values: ${K_VALUES[*]}"
echo "Account Embedding Dims: ${ACC_EMB_VALUES[*]}"
echo "Transaction Embedding Dims: ${TX_EMB_VALUES[*]}"
echo "GNN Layers: ${GNN_LAYERS_VALUES[*]}"
echo "Learning Rates: ${LR_VALUES[*]}"
echo "--------------------------"
echo "TOTAL EXPERIMENTS TO RUN: $TOTAL_EXPERIMENTS"

# --- Main Experiment Loop (Nested Grid Search) ---
EXPERIMENT_COUNT=1
for k_val in "${K_VALUES[@]}"; do
  for acc_emb_val in "${ACC_EMB_VALUES[@]}"; do
    for tx_emb_val in "${TX_EMB_VALUES[@]}"; do
      for layers_val in "${GNN_LAYERS_VALUES[@]}"; do
        for lr_val in "${LR_VALUES[@]}"; do ### ADDED THIS LOOP

          echo "======================================================================"
          echo "Preparing Experiment $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS"
          echo "Parameters: K=$k_val, AccEmb=$acc_emb_val, TxEmb=$tx_emb_val, Layers=$layers_val, LR=$lr_val"
          echo "======================================================================"

          # 1. WAIT until no job with our specific name is running
          while [ $(qstat -u "$USER" | grep -c "$JOB_NAME") -gt 0 ]; do
              echo -n "A '$JOB_NAME' job is active. Waiting... (Checked at $(date +%T))"
              echo -ne "\r"
              sleep 60
          done
          echo -e "\nNo active job found. Proceeding with submission."

          # 2. CONSTRUCT the variable list for qsub
          QSUB_VARS="K_VAL=${k_val},ACC_EMB_VAL=${acc_emb_val},TX_EMB_VAL=${tx_emb_val},LAYERS_VAL=${layers_val},LR_VAL=${lr_val}"

          # 3. SUBMIT the job script using the -v flag
          echo "Submitting job with variables: $QSUB_VARS"
          qsub -v "$QSUB_VARS" "$JOB_SCRIPT"

          sleep 5
          ((EXPERIMENT_COUNT++))

          echo "Job submitted. The controller will now wait for it to complete."
          echo

        done
      done
    done
  done
done

echo "******************************************************************"
echo "All $TOTAL_EXPERIMENTS experiment combinations have been submitted."
echo "This controller script will now exit."
echo "******************************************************************"
