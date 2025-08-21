# A Comparative Study of Graph vs. Sequential Models for Money Laundering Detection

This repository contains the code and experimental setup for the research paper comparing a Dynamic Temporal Graph Convolutional Network (Model A) against an End-to-End Sequential Profiler (Model B) for detecting illicit transactions in a highly imbalanced, synthetic financial dataset. The project also includes a simple MLP baseline for comparison.

## Project Overview

The core objective of this research is to determine which deep learning paradigm is more effective for Anti-Money Laundering (AML) detection:

*   **Model A (GCN):** A graph-based approach that models the relational structure of transactions by creating dynamic, temporal graph snapshots for each transaction.
*   **Model B (MLP-GRU):** A sequence-based approach that learns behavioral signatures from the ordered history of an account's transactions, ignoring the explicit graph structure.
*   **Model C (MLP):** A simple baseline that classifies each transaction based on its own features, without relational or temporal context.

The code is designed for reproducibility and hyperparameter exploration, with a workflow tailored for execution on a High-Performance Computing (HPC) cluster using a job scheduler.

## File Structure

-   `code/`: Contains all core Python scripts for models, data processing, and analysis.
-   `config/`: Holds YAML configuration files for experiments.
-   `jobs/`: Contains shell scripts for submitting and managing training and plotting jobs.
-   `results/` & `hybrid_results/`: Directories where raw prediction CSVs are saved after model runs.
-   `plots_report/`: Directory where final performance dashboards are saved.
-   `LI-Small_Trans.csv`: The input dataset file (must be placed in the root directory).
-   `requirements.txt`: A list of all required Python dependencies.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ana7ol/AML_GNN-Hybrid.git
cd AML_GNN-Hybrid
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv pytorch_env
source pytorch_env/bin/activate
```

### 3. Install Dependencies
Install all required packages using the `requirements.txt` file. Ensure you have the correct PyTorch version for your CUDA drivers if using a GPU.

```bash
pip install -r requirements.txt
```

### 4. Place the Dataset
Download the `LI-Small_Trans.csv` dataset and place it in the root directory of the project.

## Running the Experiments

The experiments are designed to be run via the master shell scripts located in the `jobs/` directory. These scripts iterate through a predefined set of hyperparameters and submit an individual training job for each configuration.

To run full hyperparameter test a job schedular is necessary due to the ```qsub``` commands!

### Model A: Dynamic Temporal GCN
This experiment sweeps through different values for k-history, GNN layers, and embedding dimensions.

To run a specific test:
```bash
python code/final_gnn.py --k_neighborhood 100 --acc_emb_dim 64 --tx_emb_dim 64 --gnn_layers 5 --learning_rate 0.0001
```

To run the full hyperparameter sweep for Model A:
```bash
bash jobs/run_experiment.sh
```

### Model B: End-to-End Sequential Profiler (MLP-GRU)
This experiment sweeps through different values for sequence length, step size, and GRU hidden dimensions.

To run a specific test:
```bash
python code/hybrid.py --sequence_length 10 --step_size 5 --gru_emb_dim 64 --learning_rate 0.0001
```

To run the full hyperparameter sweep for Model B:
```bash
bash jobs/run_hybrid_experiments.sh
```

### Model C: Simple MLP Baseline
To run the simple MLP baseline model:
```bash
bash jobs/Mlp.sh
```

## Processing Results and Generating Plots

After the training jobs have completed and saved their prediction files, you can process the results to generate summary CSVs and performance dashboards.

### 1. Summarize Results
To aggregate the results from all individual runs into a single, sorted CSV file:

```bash
# For Model B (MLP-GRU)
python code/summary_hybrid.py

# (A similar script would be needed for Model A's results)
```
This will create `hybrid_model_summary_auprc.csv`.

### 2. Generate Performance Dashboards
Use the plotting scripts to create the visual dashboards summarizing the hyperparameter performance.

```bash
# For Model A (GCN)
bash jobs/plot.sh

# For Model B (MLP-GRU)
bash jobs/hybrid_plot.sh
```
This will generate `gnn_model_performance_dashboard.png` and `sequential_model_performance_dashboard.png`.

## Key Script Explanations

### `code/` Directory
*   `final_gnn.py`: The complete implementation of **Model A (Dynamic Temporal GCN)**. Handles on-the-fly graph creation, training, and evaluation.
*   `hybrid.py`: The complete implementation of **Model B (End-to-End Sequential Profiler)**. Handles sequence creation, training, and evaluation.
*   `MLP.py`: The implementation of the simple MLP baseline model.
*   `plot_results.py` / `plot_hybrid.py`: Scripts that read summary CSVs and generate the performance dashboard plots.
*   `summary_hybrid.py`: Aggregates raw prediction files from Model B experiments into a single summary CSV with key metrics.

### `jobs/` Directory
*   `run_experiment.sh` / `run_hybrid_experiments.sh`: **Master scripts** that contain loops to iterate through different hyperparameter combinations and submit individual training jobs. **These are the primary entry points for reproducing the research.**
*   `final_gnn.sh` / `hybrid.sh` / `Mlp.sh`: Individual job scripts that execute a single run of a model with a specific set of hyperparameters. These are called by the master scripts.
*   `plot.sh` / `hybrid_plot.sh`: Job scripts to execute the plotting code.

### `config/` Directory
*   `config.yaml`: A base configuration file, primarily used by `final_gnn.py`, which can be overridden by command-line arguments.
