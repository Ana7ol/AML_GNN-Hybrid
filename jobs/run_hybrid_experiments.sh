#!/bin/bash

# ==============================================================================
#      Hyperparameter Grid Search Controller for HYBRID EXPERIMENT
# ==============================================================================

# Self-locating script logic to ensure it runs from the project root.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
echo "Controller has set working directory to: $(pwd)"

# --- Configuration ---
JOB_SCRIPT="jobs/hybrid.sh"
JOB_NAME="hybrid_worker_job" # MUST match the #$ -N name in hybrid.sh

# --- Control the retraining from here ---
FORCE_RETRAIN_ENCODER="true"

# --- Define Your Experiment Grid ---
GRU_EMB_VALUES=(64 128 256)
SEQUENCE_LENGTH=(5 10 15)
STEP_SIZE=(2 5 10)
LR_VALUES=(0.0001)

# Calculate total number of experiments
TOTAL_EXPERIMENTS=$(( ${#GRU_EMB_VALUES[@]} * ${#SEQUENCE_LENGTH[@]} * ${#STEP_SIZE[@]} * ${#LR_VALUES[@]} ))

echo "Grid Search Configuration for Hybrid Model:"
echo "-------------------------------------------"
echo "GRU Embedding Dims:         ${GRU_EMB_VALUES[*]}"
echo "Sequence length:            ${SEQUENCE_LENGTH[*]}"
echo "Step Size:                  ${STEP_SIZE[*]}"
echo "Learning Rates:             ${LR_VALUES[*]}"
echo "Force Retrain Encoder:      ${FORCE_RETRAIN_ENCODER}"
echo "-------------------------------------------"
echo "TOTAL EXPERIMENTS TO RUN: $TOTAL_EXPERIMENTS"
echo

# --- Main Experiment Loop (Nested Grid Search) ---
EXPERIMENT_COUNT=1
for gru_emb_val in "${GRU_EMB_VALUES[@]}"; do
  for seq_length_val in "${SEQUENCE_LENGTH[@]}"; do
    for step_size_val in "${STEP_SIZE[@]}"; do
      for lr_val in "${LR_VALUES[@]}"; do

        echo "======================================================================"
        echo "Preparing Experiment $EXPERIMENT_COUNT / $TOTAL_EXPERIMENTS"
        echo "Parameters: GruEmb=$gru_emb_val, Sequence length=$seq_length_val, Step_size=$step_size_val, LR=$lr_val"
        
        # Build the variable string for qsub
        QSUB_VARS="GRU_EMB=${gru_emb_val},SEQ_LEN=${seq_length_val},STEP_SIZE=${step_size_val},LR=${lr_val},RETRAIN=${FORCE_RETRAIN_ENCODER}"

        # Submit the job
        echo "Submitting job with variables: $QSUB_VARS"
        qsub -v "$QSUB_VARS" "$JOB_SCRIPT"

        echo "Job submitted. The controller will now wait for it to complete."
        # Add a small initial sleep to give the scheduler time to register the job
        sleep 10 

        # Now, wait until the job is no longer in the queue
        while [ $(qstat -u "$USER" | grep -c "$JOB_NAME") -gt 0 ]; do
            echo -n "A '$JOB_NAME' job is active. Waiting... (Checked at $(date +%T))"
            echo -ne "\r"
            sleep 30 # Check every 30 seconds
        done
        
        # Add a newline for clean logging after the wait is over
        echo -e "\nJob has completed. Proceeding to the next experiment.\n"

        ((EXPERIMENT_COUNT++))
      done
    done
  done
done

echo "******************************************************************"
echo "All $TOTAL_EXPERIMENTS experiment combinations have been submitted."
echo "******************************************************************"
