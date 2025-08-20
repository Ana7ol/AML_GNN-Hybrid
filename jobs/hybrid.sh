#!/bin/bash
#$ -cwd
#$ -m beas
#$ -N hybrid_worker_job

# === Job configuration ===
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=1
#$ -o ./logs/hybrid_output.$JOB_ID.txt
#$ -e ./logs/hybrid_error.$JOB_ID.txt

# --- 1. Environment Setup ---
module load python
source pytorch_env/bin/activate

# --- 2. Define Parameters with Defaults ---
LEARNING_RATE=${LR:-0.0001}
GRU_EMBEDDING=${GRU_EMB:-128}
SEQ_LENGTH=${SEQ_LEN:-10}
STEP_SIZE_VAL=${STEP_SIZE:-5}
FORCE_RETRAIN=${RETRAIN:-false}

# --- 3. Conditionally set the retrain flag ---
RETRAIN_FLAG="" # Start with an empty flag
if [ "$FORCE_RETRAIN" = "true" ]; then
  RETRAIN_FLAG="--force_retrain_encoder"
  echo "Retrain flag is SET. The encoder will be retrained."
else
  echo "Retrain flag is NOT set. Using existing encoder if available."
fi

# --- 4. Construct and Run the Python Command ---
echo "========================================================"
echo "Job started at $(date)"
echo "Job ID: $JOB_ID"
echo "Running Python script with the following parameters:"
echo "--------------------------------------------------------"
echo "Learning Rate:             $LEARNING_RATE"
echo "GRU Embedding Dim:          $GRU_EMBEDDING"
echo "Sequence length:           $SEQ_LENGTH"
echo "Step size:                 $STEP_SIZE_VAL"
echo "========================================================"

# Execute the python script, including the (possibly empty) retrain flag
python code/hybrid.py \
    --lr $LEARNING_RATE \
    --gru_emb $GRU_EMBEDDING \
    --sequence_length $SEQ_LENGTH \
    --step_size $STEP_SIZE_VAL \
    $RETRAIN_FLAG # This will either be "--force_retrain_encoder" or an empty string

echo "---"
echo "Job finished at $(date)"
