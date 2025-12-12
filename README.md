# ARCADIA Reproducibility

Reproducible code and scripts for ARCADIA benchmarking, simulation, and paper figures.

> **Note**: This repository is currently under active development. Some components may be incomplete or subject to change. We recommend checking back periodically for updates.

## Overview

This repository contains reproducible code and scripts used in the ARCADIA paper. ARCADIA is a method for integrating single-cell RNA-seq and spatial proteomics data using archetype-based cross-modal alignment. This repository provides:

- **Benchmarking**: Comparison of ARCADIA against state-of-the-art methods (scMODAL, MaxFuse)
- **Simulation**: Synthetic dataset generation - CITE-seq dataset with paired protein markers, scRNA-seq data, and synthetic spatial locations for cells
- **Datasets**: Real dataset - Tonsil dataset with CODEX spatial proteomics and scRNA-seq data
- **Analysis**: Notebooks for reproducing paper figures and robustness analyses

Please check out our main [ARCADIA repository](https://github.com/azizilab/ARCADIA_public) for the latest code and tutorials.

## Repository Structure

```
.
├── README.md                    # This file - main reproducibility repository README
├── run_pipeline_direct.sh      # Script to run ARCADIA pipeline (direct Python execution, no plots)
├── run_pipeline_notebooks.sh   # Script to run ARCADIA pipeline (notebook-based execution, with plots)
├── generate_publication_figures_github.ipynb  # Notebook for generating publication figures
│
├── model_comparison/            # Model comparison scripts for scMODAL and MaxFuse
│   ├── model_scmodal_dataset_cite_seq.py
│   ├── model_scmodal_dataset_tonsil.py
│   ├── model_maxfuse_dataset_cite_seq.py
│   ├── model_maxfuse_dataset_tonsil.py
│   └── scMODAL_main/            # scMODAL implementation included
│
├── ARCADIA/                     # ARCADIA model implementation repository (git submodule)
│   ├── scripts/                 # Pipeline execution scripts
│   │   ├── _0_preprocess_cite_seq.py
│   │   ├── _0_preprocess_tonsil.py
│   │   ├── _1_align_datasets.py
│   │   ├── _2_spatial_integrate.py
│   │   ├── _3_generate_archetypes.py
│   │   ├── _4_prepare_training.py
│   │   ├── _5_train_vae.py
│   │   └── hyperparameter_search.py
│   │
│   ├── src/arcadia/             # ARCADIA package source code
│   │   ├── training/            # Training utilities and dual VAE training plan
│   │   ├── archetypes/          # Archetype generation and matching
│   │   ├── data_utils/          # Data loading and preprocessing
│   │   ├── plotting/            # Visualization functions
│   │   ├── spatial/             # Spatial analysis utilities
│   │   └── utils/               # General utilities (including scVI patching)
│   │
│   ├── environments/            # Conda environment files and Docker setup
│   │   ├── environment_cpu.yaml
│   │   ├── environment_gpu_cuda11.8.yaml
│   │   ├── environment_gpu_cuda12.1.yaml
│   │   ├── environment_scmodal.yaml
│   │   ├── environment_maxfuse.yaml
│   │   ├── requirements.txt
│   │   └── docker/              # Docker setup for reproducible execution
│   │
│   ├── configs/                 # Configuration files
│   │   └── config.json          # Main configuration file
│   │
│   ├── run_pipeline.sh          # ARCADIA pipeline script (used by run_pipeline_notebooks.sh)
│   └── README.md                # ARCADIA repository README
```

## Datasets

### Tonsil Dataset
- **Source**: Real spatial proteomics data from tonsil tissue
- **Modalities**: scRNA-seq and CODEX spatial proteomics
- **Cell Types**: B cells, T cells, dendritic cells, macrophages
- **Spatial**: spatial coordinates from CODEX imaging
- **Location**: `data/tonsil/`

### CITE-seq Dataset
- **Source**: CITE-seq spleen lymph node data with synthetic spatial information
- **Modalities**: scRNA-seq, protein markers (ADT), and synthetic spatial coordinates
- **Cell Types**: CD4+ T, CD8+ T, B cells, dendritic cells, T cells
- **Spatial**: Synthetic spatial coordinates (simulated)
- **Location**: `data/cite_seq/`

## Models

### ARCADIA
ARCADIA (Archetype-based Cross-modal Data Integration and Alignment) integrates single-cell RNA-seq and spatial proteomics data using:

- **Archetype Generation**: Identifies representative cell states across modalities
- **Cross-modal Matching**: Aligns archetypes between RNA and protein spaces
- **Dual VAE Training**: Jointly trains variational autoencoders for both modalities
- **Spatial Integration**: Incorporates spatial neighborhood information

**Implementation**: `ARCADIA/`

### scMODAL
scMODAL (Single-Cell Multi-Omics Data Alignment) is a baseline method for multi-modal integration.

**Paper**: [scMODAL: a general deep learning framework for comprehensive single-cell multi-omics data alignment with feature links](https://www.nature.com/articles/s41467-025-60333-z)

**Repository**: [scMODAL](https://github.com/gefeiwang/scMODAL)

**Comparison Scripts**: `model_comparison/model_scmodal_dataset_*.py`

### MaxFuse
MaxFuse is a method for integrating single-cell multi-omics data using maximum mean discrepancy.

**Paper**: [MaxFuse: an integration framework for multimodal single-cell data](https://www.nature.com/articles/s41592-023-01966-0)

**Repository**: [MaxFuse](https://github.com/shuxiaoc/maxfuse)

**Comparison Scripts**: `model_comparison/model_maxfuse_dataset_*.py`

## Getting Started

### Prerequisites

**ARCADIA is included as a git submodule.** Clone this repository with submodules:

```bash
git clone --recursive https://github.com/your-repo/arcadia_reproducibility.git
```

The `ARCADIA/` folder will contain the ARCADIA repository. **You do not need to separately clone ARCADIA_public** - it's already included as a submodule.

Then follow the instructions in the ARCADIA README to set up the environment.

> **📖 For detailed ARCADIA installation instructions** (conda environments, Docker, manual setup), see [`ARCADIA/README.md`](ARCADIA/README.md#installation).

### Running the ARCADIA Pipeline

There are two scripts available to run the complete ARCADIA pipeline end-to-end:

#### `run_pipeline_direct.sh` - Direct Python Execution

Runs Python scripts directly without notebook conversion. Faster execution, suitable for production runs. **No intermediate plots or visualizations are generated during pipeline execution.**

**Usage:**
```bash
bash run_pipeline_direct.sh [dataset_name]
```

**Supported datasets:**
- `cite_seq` (default): CITE-seq spleen lymph node data with synthetic spatial information
- `tonsil`: Real spatial proteomics data from tonsil tissue with CODEX imaging

**Examples:**
```bash
# Run with cite_seq dataset (default)
bash run_pipeline_direct.sh cite_seq

# Run with tonsil dataset
bash run_pipeline_direct.sh tonsil
```

**What it does:**
Executes the complete ARCADIA pipeline (preprocessing → alignment → spatial integration → archetype generation → training preparation → VAE training).

> **📖 For detailed pipeline step descriptions**, see [`ARCADIA/README.md`](ARCADIA/README.md#detailed-workflow).

#### `run_pipeline_notebooks.sh` - Notebook-based Execution

Converts Python scripts to Jupyter notebooks and executes them with papermill. **All intermediate plots and visualizations are generated and saved as Jupyter notebooks**, allowing you to review and inspect the results at each pipeline step. Notebooks are saved with timestamps for easy tracking.

**Usage:**
```bash
bash run_pipeline_notebooks.sh [dataset_name]
```

**Supported datasets:** Same as `run_pipeline_direct.sh` (cite_seq, tonsil)

**Examples:**
```bash
# Run with cite_seq dataset (default)
bash run_pipeline_notebooks.sh cite_seq

# Run with tonsil dataset
bash run_pipeline_notebooks.sh tonsil
```

**What it does:**
Executes the same pipeline steps as `run_pipeline_direct.sh`, but converts scripts to notebooks and saves all intermediate plots and visualizations for review.

- Converts each Python script to a Jupyter notebook using `jupytext`
- Executes notebooks with `papermill` (allows parameter injection)
- **Generates and saves all intermediate plots and visualizations** in the executed notebooks
- Saves executed notebooks with timestamps in `ARCADIA/notebooks/${dataset_name}/` for review

**Key difference:** Use `run_pipeline_notebooks.sh` when you need to inspect intermediate results, plots, and visualizations at each step. Use `run_pipeline_direct.sh` for faster execution without visualization outputs.

**Requirements:** Both scripts automatically change to the `ARCADIA/` directory and activate the `scvi` conda environment. For `run_pipeline_notebooks.sh`, you also need `jupytext` and `papermill` installed in the environment.

### Viewing Training Results with MLflow

During VAE training, all training metrics, loss curves, and visualizations are automatically logged to MLflow. You can view these results in an interactive web interface.

**Starting MLflow UI:**

```bash
# Navigate to the ARCADIA directory (where mlruns folder is located)
cd ARCADIA

# Start MLflow UI server
mlflow ui

# Or specify a custom port if default (5000) is in use
mlflow ui --port 5001
```

**Accessing MLflow in Browser:**

Once MLflow UI is running, open your web browser and navigate to:
- **Local machine:** `http://localhost:5000`
- **Remote server:** `http://<server-ip>:5000` (or your specified port)

**What You Can View:**

- **Training Metrics:** Loss curves (reconstruction, similarity, matching, cell type clustering, etc.)
- **Training Plots:** 
  - Combined latent space PCA/UMAP visualizations
  - CN-specific cell type UMAPs
  - Modality-specific visualizations
  - Counterfactual generation plots
- **Model Parameters:** Hyperparameters used for each training run
- **Artifacts:** Saved model checkpoints, logs, and visualization PDFs
- **Experiment Comparison:** Compare multiple training runs side-by-side

**Note:** MLflow experiments are organized by dataset name (e.g., `cite_seq`, `tonsil`, or `default` if no dataset name is specified). Navigate to the experiment matching your dataset to view the training results. Training plots are only generated when `plot_x_times > 0` in your training configuration. Set `plot_x_times: 0` in the training parameters (see `ARCADIA/scripts/_5_train_vae.py`) to disable **all** plotting during training (including first batch plots, latent space visualizations, counterfactual plots, etc.) for faster execution. When `plot_x_times=0`, no plots will be generated regardless of the `plot_first_step` setting.

**💡 Tip: Fast Debugging Runs**

For faster execution during development or debugging, you can reduce the number of cells processed and training epochs:

1. **Edit `ARCADIA/configs/config.json`:**
```json
{
  "subsample": {
    "num_rna_cells": 500,
    "num_protein_cells": 500
  },
  "plot_flag": false,
  "training": {
    "max_epochs": 10
  }
}
```

2. **Disable training plots** by setting `plot_x_times: 0` in `ARCADIA/scripts/_5_train_vae.py` (line 310). This disables all training visualizations including first batch plots, latent space plots, and counterfactual plots.

This will significantly speed up pipeline execution while still allowing you to test the full workflow. The default values are 2000 cells for each modality and 400 epochs for training. **For debugging and testing, set `max_epochs` to a low number (e.g., 10-50) and `plot_x_times: 0` to quickly verify the pipeline works correctly.**

### Running with Docker

Docker provides a containerized environment for reproducible execution. Recommended when you want to avoid local environment setup.

**Quick Start:**

```bash
# From the ARCADIA root directory
cd ARCADIA
bash environments/docker/run_docker.sh test              # Build and test
bash environments/docker/run_docker.sh pipeline cite_seq  # Run pipeline
bash environments/docker/run_docker.sh bash               # Interactive session
```

**Rebuilding Docker Image:**

After changing `requirements.txt` or other dependencies, rebuild the Docker image without cache:

```bash
# Force rebuild without cache (ensures all dependencies are updated)
bash environments/docker/run_docker.sh --no-cache pipeline

# Or use --rebuild (same as --no-cache)
bash environments/docker/run_docker.sh --rebuild bash
```

**Note:** The script should be run from the `ARCADIA` root directory. It will automatically mount the parent repository root so that `run_pipeline_direct.sh` is accessible.

**Prerequisites:** Docker installed (NVIDIA Docker runtime optional for GPU support)

**Note:** The Docker image automatically patches the scVI library (`scvi-tools==1.2.2.post2`) to support custom training plans. This patch adds `self._training_plan = training_plan` to the scVI training mixin, which is required for ARCADIA's dual VAE training. The patching happens automatically during Docker image build, so no manual intervention is needed.

> **📖 For detailed Docker documentation** including setup instructions, usage examples, troubleshooting, and advanced usage, see [`ARCADIA/environments/docker/DOCKER_README.md`](ARCADIA/environments/docker/DOCKER_README.md).

### Environment Setup for Benchmarking Methods

To run the benchmarking comparisons, you'll need to install scMODAL and MaxFuse following their official installation instructions:

**scMODAL:**
- Repository: [scMODAL](https://github.com/gefeiwang/scMODAL)
- Follow the installation instructions in the scMODAL repository
- The scMODAL implementation is also included in `model_comparison/scMODAL_main/` for convenience

**MaxFuse:**
- Repository: [MaxFuse](https://github.com/shuxiaoc/maxfuse)
- Follow the installation instructions in the MaxFuse repository

### Running Simulations

Generate synthetic datasets:

```bash
cd simulation
bash run_simulation.sh
```

This will generate synthetic spatial datasets for both tonsil and cite_seq.

### Running Model Comparisons

Compare ARCADIA against baseline methods (scMODAL and MaxFuse) on the same datasets. The comparison scripts are located in `model_comparison/` and run the external models using the same preprocessed data as ARCADIA.

#### Prerequisites

Before running comparisons, ensure you have:

1. **Run ARCADIA preprocessing scripts first** to generate the filtered datasets:
   ```bash
   # For CITE-seq dataset
   cd ARCADIA
   conda activate scvi  # or your ARCADIA environment name
   python scripts/_0_preprocess_cite_seq.py
   
   # For Tonsil dataset
   python scripts/_0_preprocess_tonsil.py
   ```
   This ensures that all comparison methods use the same filtered dataset as ARCADIA for fair comparison.

2. **Installed comparison methods** following their official installation instructions:
   - **scMODAL**: Follow installation instructions at [scMODAL repository](https://github.com/gefeiwang/scMODAL)
   - **MaxFuse**: Follow installation instructions at [MaxFuse repository](https://github.com/shuxiaoc/maxfuse)
   - The scMODAL implementation is included in `model_comparison/scMODAL_main/` for convenience

#### Running scMODAL Comparisons

scMODAL comparisons use the included implementation in `model_comparison/scMODAL_main/`:

**Option 1: Using Docker (Recommended for scMODAL)**

A Dockerfile is provided for scMODAL at `model_comparison/Dockerfile.scmodal`:

```bash
cd model_comparison

# Build the Docker image
docker build -f Dockerfile.scmodal -t scmodal:latest .

# Run the container (mount your workspace)
docker run -it --gpus all -v /path/to/arcadia_reproducibility:/workspace scmodal:latest

# Inside the container, run scMODAL
python model_scmodal_dataset_cite_seq.py
python model_scmodal_dataset_tonsil.py
```

**Option 2: Using Conda Environment**

```bash
cd model_comparison

# Activate your scMODAL conda environment (set up following scMODAL installation instructions)
conda activate scmodal  # or your scMODAL environment name

# Run scMODAL on cite_seq dataset
python model_scmodal_dataset_cite_seq.py

# Run scMODAL on tonsil dataset
python model_scmodal_dataset_tonsil.py
```

**What it does:**
- Loads preprocessed data from ARCADIA pipeline (same filtered data used for ARCADIA training)
- Runs scMODAL integration on RNA and protein modalities
- Computes evaluation metrics (matching accuracy, integration quality, etc.)
- Saves results and visualizations for comparison with ARCADIA

#### Running MaxFuse Comparisons

MaxFuse comparisons use the official MaxFuse package:

```bash
cd model_comparison

# Activate your MaxFuse conda environment (set up following MaxFuse installation instructions)
conda activate maxfuse  # or your MaxFuse environment name

# Run MaxFuse on cite_seq dataset
python model_maxfuse_dataset_cite_seq.py

# Run MaxFuse on tonsil dataset
python model_maxfuse_dataset_tonsil.py
```

**What it does:**
- Loads preprocessed data from ARCADIA pipeline (same filtered data used for ARCADIA training)
- Runs MaxFuse integration on RNA and protein modalities
- Computes evaluation metrics (matching accuracy, integration quality, etc.)
- Saves results and visualizations for comparison with ARCADIA

#### Comparing ARCADIA Against Baseline Models

After running both ARCADIA and baseline models (scMODAL or MaxFuse), use `compare_results.py` to compute and compare performance metrics between ARCADIA and the baseline model.

**Prerequisites:**

1. **Completed ARCADIA training** - You need a trained ARCADIA checkpoint (saved in MLflow)
2. **Completed baseline model run** - The baseline model outputs should exist in `model_comparison/outputs/{model_name}_{dataset_name}/`

**Usage:**

The script can be run in two ways:

**Option 1: Using MLflow experiment name (Recommended)**

```bash
cd model_comparison

# Activate your ARCADIA conda environment
conda activate scvi  # or your ARCADIA environment name

# Compare ARCADIA against scMODAL on cite_seq dataset
python compare_results.py --experiment_name "cite_seq" --other_model_name "scmodal"

# Compare ARCADIA against MaxFuse on tonsil dataset
python compare_results.py --experiment_name "tonsil" --other_model_name "maxfuse"
```

**Option 2: Using checkpoint path directly**

```bash
cd model_comparison
conda activate scvi

# Specify the checkpoint path explicitly
python compare_results.py --checkpoint_path "/path/to/mlruns/.../epoch_0499" --other_model_name "scmodal"
```

**Arguments:**

- `--experiment_name`: MLflow experiment name (e.g., "cite_seq", "tonsil") - automatically finds the latest checkpoint
- `--checkpoint_path`: Direct path to ARCADIA checkpoint folder (alternative to experiment_name)
- `--other_model_name`: Baseline model name to compare against (default: "maxfuse", options: "scmodal", "maxfuse")
- `--experiment_id`: MLflow experiment ID (auto-inferred if using experiment_name)
- `--run_id`: MLflow run ID (auto-inferred if using experiment_name)

**What it does:**

- Loads ARCADIA checkpoint data from MLflow (automatically finds checkpoint if using experiment_name)
- Loads baseline model outputs from `model_comparison/outputs/{model_name}_{dataset_name}/`
- Computes comprehensive evaluation metrics for both models:
  - **Integration Quality**: iLISI, kBET, silhouette scores
  - **Matching Accuracy**: Cross-modal cell type and CN matching
  - **Spatial Metrics**: Moran's I, pair distance, FOSCTTM, FOSKNN
  - **Cell Type Preservation**: ARI, F1 scores, cell type silhouette
- Generates comparison visualizations (if `plot_flag=True` in config):
  - Combined UMAP plots for both models
  - Per-cell-type UMAPs
  - Moran's I spatial autocorrelation plots
- **Automatically generates files required for publication figures:**
  - **CN Assignment CSVs**: `{dataset}_protein_CN_assignments.csv`, `{dataset}_rna_CN_assignments.csv`
  - **Confusion Matrices**: `data/confusion_matrices/{dataset}_ct_matching_{model}.csv`, `data/confusion_matrices/all.csv`, `data/confusion_matrices/bcells.csv` (for cite_seq)
  - **UMAP Embeddings**: `data/{dataset}_{model}_umap.csv` or `data/{dataset}_{model}_UMAP.csv`
- Saves results to CSV files:
  - Individual model results: `metrics/results_comparison_{model}_{dataset}_{timestamp}.csv`
  - Comparison results: `metrics/results_comparison_{dataset}_{model}.csv` (appended with each run)

**Output:**

The script prints a comparison table to the console showing all metrics side-by-side for ARCADIA vs. the baseline model. Results are also saved to CSV files in the `metrics/` directory, making it easy to track performance across multiple experiments and runs.

**Note:** The dataset name is automatically inferred from the ARCADIA checkpoint metadata, so you don't need to specify it manually.

#### Comparison Results

Results from model comparisons are saved in `model_comparison/` with dataset-specific outputs. You can compare:

- **Integration Quality**: iLISI scores, kBET rejection rates
- **Matching Accuracy**: Cross-modal cell matching performance
- **Latent Space Quality**: UMAP/PCA visualizations
- **Cell Type Preservation**: Cell type clustering metrics

**Important:** These comparison scripts use the same filtered dataset as ARCADIA to ensure fair comparison. **You must run the ARCADIA preprocessing scripts first** (`_0_preprocess_cite_seq.py` for CITE-seq or `_0_preprocess_tonsil.py` for Tonsil) before running any comparison scripts to generate the required preprocessed data files.

### Generating Publication Figures

The `generate_publication_figures_github.py` script (or `generate_publication_figures_github.ipynb` notebook) contains code to reproduce all publication figures from the ARCADIA paper. This includes:

- **Spatial visualizations**: Spatial embeddings colored by cell types, CN assignments, and spatial grid indices
- **Integration quality plots**: UMAP visualizations of integrated RNA and protein modalities
- **Confusion matrices**: Cell type matching accuracy for ARCADIA, scMODAL, and MaxFuse
- **Differential expression analysis**: Counterfactual spatial neighborhood DEG analysis for tonsil dataset
- **Dot plots**: Gene expression patterns across CNs and cell types

**Note:** The script automatically loads the latest ARCADIA checkpoint data from MLflow runs for both `cite_seq` and `tonsil` datasets. It uses `find_checkpoint_from_experiment_name()` to locate the most recent checkpoint for each experiment, ensuring you always use the latest trained models without needing to specify hardcoded file paths or timestamps.

**Complete Workflow for Generating Publication Figures:**

To generate all publication figures, follow these steps in order:

**Step 1: Run ARCADIA Pipeline for Both Datasets**

First, train ARCADIA models for both datasets:

```bash
# Run ARCADIA pipeline for cite_seq dataset
bash run_pipeline_direct.sh cite_seq

# Run ARCADIA pipeline for tonsil dataset
bash run_pipeline_direct.sh tonsil
```

**Step 2: Run Baseline Models (scMODAL and MaxFuse)**

Run the baseline comparison models for both datasets (see [Running Model Comparisons](#running-model-comparisons) section above):

```bash
cd model_comparison

# Run scMODAL for cite_seq
conda activate scmodal  # or your scMODAL environment
python model_scmodal_dataset_cite_seq.py

# Run scMODAL for tonsil
python model_scmodal_dataset_tonsil.py

# Run MaxFuse for cite_seq
conda activate maxfuse  # or your MaxFuse environment
python model_maxfuse_dataset_cite_seq.py

# Run MaxFuse for tonsil
python model_maxfuse_dataset_tonsil.py
```

**Step 3: Run Model Comparisons to Generate Required Files**

Run `compare_results.py` for both datasets and both baseline models. This will automatically generate all CN assignments, confusion matrices, and UMAP embeddings needed for the publication figures:

```bash
cd model_comparison
conda activate scvi  # or your ARCADIA environment

# Compare ARCADIA vs scMODAL on cite_seq
python compare_results.py --experiment_name "cite_seq" --other_model_name "scmodal"

# Compare ARCADIA vs MaxFuse on cite_seq
python compare_results.py --experiment_name "cite_seq" --other_model_name "maxfuse"

# Compare ARCADIA vs scMODAL on tonsil
python compare_results.py --experiment_name "tonsil" --other_model_name "scmodal"

# Compare ARCADIA vs MaxFuse on tonsil
python compare_results.py --experiment_name "tonsil" --other_model_name "maxfuse"
```

**What gets generated:**

After running all four comparisons, the following files will be created:

- **CN Assignments**: 
  - `synthetic_protein_CN_assignments.csv`, `synthetic_rna_CN_assignments.csv` (from cite_seq runs)
  - `tonsil_protein_CN_assignments.csv`, `tonsil_rna_CN_assignments.csv` (from tonsil runs)

- **Confusion Matrices** (`data/confusion_matrices/`):
  - `cite_seq_ct_matching_arcadia.csv`, `synthetic_ct_matching_maxfuse.csv`, `synthetic_ct_matching_scmodal.csv`
  - `tonsil_ct_matching_arcadia.csv`, `tonsil_ct_matching_maxfuse.csv`, `tonsil_ct_matching_scmodal.csv`
  - `all.csv`, `bcells.csv` (from cite_seq runs)

- **UMAP Embeddings** (`data/`):
  - `scModal_umap.csv`, `maxfuse_umap.csv` (from cite_seq runs)
  - `tonsil_scModal_umap.csv`, `tonsil_maxfuse_UMAP.csv` (from tonsil runs)

**Step 4: Generate Publication Figures**

Once all required files are generated, run the publication figure generation script:

```bash
# Using Python script
python generate_publication_figures_github.py

# Or using Jupyter notebook
jupyter notebook generate_publication_figures_github.ipynb
```

**Prerequisites:**

Before running the figure generation script, ensure you have:
1. ✅ Completed the ARCADIA pipeline for both `cite_seq` and `tonsil` datasets (Step 1)
2. ✅ Run baseline models (scMODAL and MaxFuse) for both datasets (Step 2)
3. ✅ Run `compare_results.py` for both datasets and both baseline models (Step 3)
4. ✅ MLflow runs exist in `ARCADIA/mlruns/` with checkpoints for both experiments (automatically created by Step 1)

**How it works:**

The script automatically:
- Loads the latest ARCADIA checkpoint data from MLflow experiments (`cite_seq` and `tonsil`)
- Uses the most recent checkpoint for each dataset (no need to specify file paths or timestamps)
- Loads RNA and protein AnnData objects with trained latent representations from the checkpoints

The script saves all generated figures to the `fig_khh/` directory.

## Benchmarking Metrics

TBD Benchmarking Metrics

## Directory Descriptions

### `run_pipeline_direct.sh`
Script that runs the complete ARCADIA pipeline end-to-end using direct Python execution. Supports multiple datasets (cite_seq, tonsil) and executes all pipeline steps from preprocessing to VAE training with hyperparameter search. Faster execution, suitable for production runs. **No intermediate plots or visualizations are generated.**

### `run_pipeline_notebooks.sh`
Script that runs the complete ARCADIA pipeline end-to-end using notebook-based execution. Converts Python scripts to Jupyter notebooks and executes them with papermill. **All intermediate plots and visualizations are generated and saved in the notebooks**, which are saved with timestamps in `ARCADIA/notebooks/${dataset_name}/` for review and inspection. Useful for development, analysis, and when you need to examine intermediate results.

### `model_comparison/`
Contains scripts for running comparative benchmarks between ARCADIA, scMODAL, and MaxFuse on tonsil and cite_seq datasets. Each script:
- Loads preprocessed data from ARCADIA pipeline (same filtered data used for ARCADIA training)
- Runs the respective method
- Computes evaluation metrics
- Saves results and visualizations

To use the scMODAL implementation, clone it into the [model_comparison](model_comparison) directory:

```bash
cd model_comparison
git clone https://github.com/gefeiwang/scMODAL.git scMODAL_main
```

### `generate_publication_figures_github.ipynb`
Jupyter notebook for generating all publication figures from the ARCADIA paper. Includes spatial visualizations, integration quality plots, confusion matrices, differential expression analysis, and dot plots. See the [Generating Publication Figures](#generating-publication-figures) section for usage instructions.

### `ARCADIA/`
Complete ARCADIA implementation including:
- **Pipeline scripts** (`scripts/`): Step-by-step execution scripts for preprocessing, alignment, spatial integration, archetype generation, and training
- **Source code** (`src/arcadia/`): Core implementation including training plans, archetype algorithms, data utilities, and visualization functions
- **Environment files** (`environments/`): Conda environment YAML files and Docker setup for reproducible execution
- **Configuration** (`configs/`): Configuration files for pipeline parameters

## Citation

If you use ARCADIA or this reproducibility repository, please cite:

**Plain text citation:**
```
Rozenman, B., Hoffer-Hawlik, K., Djedjos, N., Azizi, E.
ARCADIA reveals spatially dependent transcriptional programs through integration of scRNA-seq and spatial proteomics.
bioRxiv (2025).
https://doi.org/10.1101/2025.11.20.689521
```

**BibTeX:**
```bibtex
@article {Rozenman2025.11.20.689521,
	author = {Rozenman, Bar and Hoffer-Hawlik, Kevin and Djedjos, Nicholas and Azizi, Elham},
	title = {ARCADIA Reveals Spatially Dependent Transcriptional Programs through Integration of scRNA-seq and Spatial Proteomics},
	elocation-id = {2025.11.20.689521},
	year = {2025},
	doi = {10.1101/2025.11.20.689521},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/21/2025.11.20.689521},
	eprint = {https://www.biorxiv.org/content/early/2025/11/21/2025.11.20.689521.full.pdf},
	journal = {bioRxiv}
}
```

## License

BSD 3-Clause License

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

- Bar Rozenman - br2783@columbia.edu
- Kevin Hoffer-Hawlik - kh3205@columbia.edu
- Elham Azizi - elham@azizilab.com
