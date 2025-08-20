#!/bin/bash
#$ -cwd
#$ -m beas

# === Job configuration (RIGHT-SIZED for plotting) ===
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=0:15:0

# === Job name and output ===
#$ -o plot_job.o$JOB_ID
#$ -e plot_job.e$JOB_ID

echo "========================================================================"
echo "Job Started: $JOB_ID"
echo "Running on host: $(hostname)"
echo "Running at: $(date)"
echo "========================================================================"

# === Environment setup ===
echo "Loading Python environment..."
module load python
source pytorch_env/bin/activate
pip install plotly
echo "Environment activated."

# === Job script ===
echo "Running the plotting script: code/plot_results.py"
python code/plot_results.py
echo "Plotting script finished."

echo "========================================================================"
echo "Job Finished at: $(date)"
echo "========================================================================"
