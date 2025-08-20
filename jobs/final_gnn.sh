
#submit job through
#qsub -v "K_VAL=100,LAYERS_VAL=10,ACC_EMB_VAL=64,TX_EMB_VAL=64,LR_VAL=0.0001" jobs/new_gnn.sh

#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -l gpu=1

# Check if required environment variables are set
if [ -z "$K_VAL" ] || [ -z "$ACC_EMB_VAL" ] || [ -z "$TX_EMB_VAL" ] || [ -z "$LAYERS_VAL" ] || [ -z "$LR_VAL" ]; then
  echo "Error: One or more required environment variables are not set."
  echo "This job must be submitted via a controller script or 'qsub -v ...'"
  exit 1
fi

# Activate Python environment
module load python
source "$HOME/AML/pytorch_env/bin/activate"

# --- FLAG & LOG NAME LOGIC ---

# 1. Handle the --separate_banks flag (boolean)
sep_banks_log_part=""
if [ -n "$SEP_BANKS_FLAG" ]; then
  sep_banks_log_part="_SepBanks"
fi

# 2. Handle the --isolate_bank argument (takes a value)
iso_bank_log_part=""
ISOLATE_BANK_ARG="" # This will hold the full argument string, e.g., "--isolate_bank B"
if [ -n "$ISOLATE_BANK_NAME" ]; then
  # If the variable ISOLATE_BANK_NAME is set...
  iso_bank_log_part="_iso${ISOLATE_BANK_NAME}"
  ISOLATE_BANK_ARG="--isolate_bank ${ISOLATE_BANK_NAME}"
fi

# 3. Construct the final log name using the parts from above
log_name="logs/final_GNN_K${K_VAL}_Layers${LAYERS_VAL}_emb${ACC_EMB_VAL}x${TX_EMB_VAL}_lr${LR_VAL}${sep_banks_log_part}${iso_bank_log_part}.txt"

echo "Starting job with parameters from environment:"
echo "Log file: $log_name"
echo "---"
echo "K = $K_VAL"
echo "AccEmb = $ACC_EMB_VAL"
echo "TxEmb = $TX_EMB_VAL"
echo "Layers = $LAYERS_VAL"
echo "LR = $LR_VAL"
echo "Separate Banks Flag being passed to Python: '${SEP_BANKS_FLAG}'"
echo "Isolate Bank Argument being passed to Python: '${ISOLATE_BANK_ARG}'"
echo "---"

nvidia-smi

# --- PYTHON COMMAND ---
# The $SEP_BANKS_FLAG and $ISOLATE_BANK_ARG variables will be replaced by the shell.
# If they are empty strings, they disappear. Otherwise, they are added as arguments.
python code/final_gnn.py \
    --k_neighborhood "$K_VAL" \
    --acc_emb_dim "$ACC_EMB_VAL" \
    --tx_emb_dim "$TX_EMB_VAL" \
    --gnn_layers "$LAYERS_VAL" \
    --learning_rate "$LR_VAL" \
    $SEP_BANKS_FLAG \
    $ISOLATE_BANK_ARG \
    > "$log_name" 2>&1
