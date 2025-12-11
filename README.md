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
├── run_pipeline_direct.sh      # Script to run ARCADIA pipeline (direct Python execution, no plots)
├── run_pipeline_notebooks.sh   # Script to run ARCADIA pipeline (notebook-based execution, with plots)
├── benchmark/              # Benchmarking scripts for comparison methods
│   ├── test_scmodal_tonsil.py
│   ├── test_scmodal_cite_seq.py
│   ├── test_maxfuse_tonsil.py
│   ├── test_maxfuse_cite_seq.py
│   ├── test_arcadia_tonsil.py
│   ├── test_arcadia_cite_seq.py
│   └── README.md
│
├── data/                   # Dataset information and metadata
│   ├── tonsil/
│   │   ├── README.md
│   │   └── metadata.json
│   ├── cite_seq/
│   │   ├── README.md
│   │   └── metadata.json
│   └── README.md
│
├── notebooks/              #  analysis notebooks
│   ├── arcadia_simulation.ipynb
│   ├── arcadia_archetype_analysis.ipynb
│   ├── arcadia_cross_modal_matching.ipynb
│   ├── arcadia_benchmark_comparison.ipynb
│   ├── arcadia_spatial_integration.ipynb
│   ├── arcadia_archetype_validation.ipynb
│   └── arcadia_robustness_analysis.ipynb
│
├── simulation/             # Synthetic dataset generation
│   ├── generate_synthetic_cite_seq.py
│   ├── spatial_simulation.py
│   ├── utils.py
│   ├── run_simulation.sh
│   └── README.md
│
├── environments/           # Conda environment files for benchmarking methods
│   ├── environment_scmodal.yaml
│   └── environment_maxfuse.yaml
│
├── ARCADIA/                # ARCADIA model implementation repository (cloned)
│   ├── model_comparison/   # Model comparison scripts for scMODAL and MaxFuse
│   │   ├── model_scmodal_dataset_cite_seq.py
│   │   ├── model_scmodal_dataset_tonsil.py
│   │   ├── model_maxfuse_dataset_cite_seq.py
│   │   ├── model_maxfuse_dataset_tonsil.py
│   │   └── scMODAL_main/   # scMODAL implementation included
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

**Environment**: `environments/environment_scmodal.yaml`

**Benchmark Scripts**: `benchmark/test_scmodal_*.py`

### MaxFuse
MaxFuse is a method for integrating single-cell multi-omics data using maximum mean discrepancy.

**Paper**: [MaxFuse: an integration framework for multimodal single-cell data](https://www.nature.com/articles/s41592-023-01966-0)

**Repository**: [MaxFuse](https://github.com/shuxiaoc/maxfuse)

**Environment**: `environments/environment_maxfuse.yaml`

**Benchmark Scripts**: `benchmark/test_maxfuse_*.py`

## Getting Started

### Prerequisites

Clone the ARCADIA repository to the `ARCADIA/` folder and follow the instructions in its README to set up the environment.

> **📖 For detailed ARCADIA installation instructions** (conda environments, Docker, manual setup), see [`ARCADIA/README.md`](ARCADIA/README.md#installation).

### Running the ARCADIA Pipeline

There are two scripts available to run the complete ARCADIA pipeline end-to-end:

#### `run_pipeline_direct.sh` - Direct Python Execution

Runs Python scripts directly without notebook conversion. Faster execution, suitable for production runs. **No intermediate plots or visualizations are generated during pipeline execution.**

**Usage:**
```bash
./run_pipeline_direct.sh [dataset_name]
```

**Supported datasets:**
- `cite_seq` (default): CITE-seq spleen lymph node data with synthetic spatial information
- `tonsil`: Real spatial proteomics data from tonsil tissue with CODEX imaging

**Examples:**
```bash
# Run with cite_seq dataset (default)
./run_pipeline_direct.sh cite_seq

# Run with tonsil dataset
./run_pipeline_direct.sh tonsil
```

**What it does:**
Executes the complete ARCADIA pipeline (preprocessing → alignment → spatial integration → archetype generation → training preparation → VAE training).

> **📖 For detailed pipeline step descriptions**, see [`ARCADIA/README.md`](ARCADIA/README.md#detailed-workflow).

#### `run_pipeline_notebooks.sh` - Notebook-based Execution

Converts Python scripts to Jupyter notebooks and executes them with papermill. **All intermediate plots and visualizations are generated and saved as Jupyter notebooks**, allowing you to review and inspect the results at each pipeline step. Notebooks are saved with timestamps for easy tracking.

**Usage:**
```bash
./run_pipeline_notebooks.sh [dataset_name]
```

**Supported datasets:** Same as `run_pipeline_direct.sh` (cite_seq, tonsil)

**Examples:**
```bash
# Run with cite_seq dataset (default)
./run_pipeline_notebooks.sh cite_seq

# Run with tonsil dataset
./run_pipeline_notebooks.sh tonsil
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

**Note:** Training plots are only generated when `plot_x_times > 0` in your training configuration. Set `plot_x_times: 0` in the training parameters (see `ARCADIA/scripts/_5_train_vae.py`) to disable **all** plotting during training (including first batch plots, latent space visualizations, counterfactual plots, etc.) for faster execution. When `plot_x_times=0`, no plots will be generated regardless of the `plot_first_step` setting.

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
./environments/docker/run_docker.sh test              # Build and test
DATASET_NAME=cite_seq ./environments/docker/run_docker.sh pipeline  # Run pipeline
./environments/docker/run_docker.sh bash               # Interactive session
```

**Rebuilding Docker Image:**

After changing `requirements.txt` or other dependencies, rebuild the Docker image without cache:

```bash
# Force rebuild without cache (ensures all dependencies are updated)
./environments/docker/run_docker.sh --no-cache pipeline

# Or use --rebuild (same as --no-cache)
./environments/docker/run_docker.sh --rebuild bash
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
- The scMODAL implementation is also included in `ARCADIA/model_comparison/scMODAL_main/` for convenience

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

Compare ARCADIA against baseline methods (scMODAL and MaxFuse) on the same datasets. The comparison scripts are located in `ARCADIA/model_comparison/` and run the external models using the same preprocessed data as ARCADIA.

#### Prerequisites

Before running comparisons, ensure you have:

1. **Completed ARCADIA pipeline** for your dataset (preprocessing through training)
2. **Installed comparison methods** following their official installation instructions:
   - **scMODAL**: Follow installation instructions at [scMODAL repository](https://github.com/gefeiwang/scMODAL)
   - **MaxFuse**: Follow installation instructions at [MaxFuse repository](https://github.com/shuxiaoc/maxfuse)
   - The scMODAL implementation is included in `ARCADIA/model_comparison/scMODAL_main/` for convenience

#### Running scMODAL Comparisons

scMODAL comparisons use the included implementation in `ARCADIA/model_comparison/scMODAL_main/`:

```bash
cd ARCADIA/model_comparison

# Activate your scMODAL conda environment (set up following scMODAL installation instructions)
conda activate scmodal  # or your scMODAL environment name

# Run scMODAL on cite_seq dataset
python model_scmodal_dataset_cite_seq.py

# Run scMODAL on tonsil dataset
python model_scmodal_dataset_tonsil.py
```

**What it does:**
- Loads preprocessed data from ARCADIA pipeline (same data used for ARCADIA training)
- Runs scMODAL integration on RNA and protein modalities
- Computes evaluation metrics (matching accuracy, integration quality, etc.)
- Saves results and visualizations for comparison with ARCADIA

#### Running MaxFuse Comparisons

MaxFuse comparisons use the official MaxFuse package:

```bash
cd ARCADIA/model_comparison

# Activate your MaxFuse conda environment (set up following MaxFuse installation instructions)
conda activate maxfuse  # or your MaxFuse environment name

# Run MaxFuse on cite_seq dataset
python model_maxfuse_dataset_cite_seq.py

# Run MaxFuse on tonsil dataset
python model_maxfuse_dataset_tonsil.py
```

**What it does:**
- Runs MaxFuse integration on RNA and protein modalities
- Computes evaluation metrics (matching accuracy, integration quality, etc.)
- Saves results and visualizations for comparison with ARCADIA

#### Comparison Results

Results from model comparisons are saved in `ARCADIA/model_comparison/` with dataset-specific outputs. You can compare:

- **Integration Quality**: iLISI scores, kBET rejection rates
- **Matching Accuracy**: Cross-modal cell matching performance
- **Latent Space Quality**: UMAP/PCA visualizations
- **Cell Type Preservation**: Cell type clustering metrics

**Note:** These comparison scripts use the same preprocessed data as ARCADIA to ensure fair comparison. Make sure you've run the ARCADIA pipeline first to generate the required preprocessed data files.





## Benchmarking Metrics

TBD Benchmarking Metrics

## Directory Descriptions

### `run_pipeline_direct.sh`
Script that runs the complete ARCADIA pipeline end-to-end using direct Python execution. Supports multiple datasets (cite_seq, tonsil) and executes all pipeline steps from preprocessing to VAE training with hyperparameter search. Faster execution, suitable for production runs. **No intermediate plots or visualizations are generated.**

### `run_pipeline_notebooks.sh`
Script that runs the complete ARCADIA pipeline end-to-end using notebook-based execution. Converts Python scripts to Jupyter notebooks and executes them with papermill. **All intermediate plots and visualizations are generated and saved in the notebooks**, which are saved with timestamps in `ARCADIA/notebooks/${dataset_name}/` for review and inspection. Useful for development, analysis, and when you need to examine intermediate results.



### `ARCADIA/`
Complete ARCADIA implementation including:
- Core archetype generation and matching algorithms
- Dual VAE training plan
- Data loading and preprocessing utilities
- Visualization functions
- Pipeline scripts for end-to-end execution
- Model comparison scripts (`model_comparison/`) for running scMODAL and MaxFuse comparisons

### `outputs/`
Generated outputs organized by type:
- `benchmark_results/`: CSV files and plots from benchmarking
- `figures/`: Generated paper figures
- `processed_data/`: Preprocessed datasets ready for analysis

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

For questions or issues, please open an issue on GitHub or contact elham@azizilab.com.
