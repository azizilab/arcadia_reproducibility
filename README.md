# ARCADIA Reproducibility

Reproducible code and scripts for ARCADIA benchmarking, simulation, and paper figures.

> **Note**: This repository is currently under active development. Some components may be incomplete or subject to change. We recommend checking back periodically for updates.

## Overview

This repository contains reproducible code and scripts used in the ARCADIA paper. ARCADIA is a method for integrating single-cell RNA-seq and spatial proteomics data using archetype-based cross-modal alignment. This repository provides:

- **Benchmarking**: Comparison of ARCADIA against state-of-the-art methods (scMODAL, MaxFuse)
- **Simulation**: Synthetic dataset generation - CITE-seq dataset with paired protein markers, scRNA-seq data and synthetic spatial locations for cells
- **Datasets**: Real dataset - Tonsil dataset with CODEX spatial proteomics and scRNA-seq data
- **Analysis**: Notebooks for reproducing paper figures and robustness analyses

Please check out our main [ARCADIA repository](https://github.com/azizilab/ARCADIA) for the latest code and tutorials.

## Repository Structure

```
.
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

### Environment Setup for Benchmarking Methods

To run the benchmarking comparisons, you'll need separate conda environments for scMODAL and MaxFuse. Run these commands from the root of the reproducibility repository:

**scMODAL Environment:**
```bash
conda env create -f ARCADIA/environments/environment_scmodal.yaml
conda activate scmodal
# Install scMODAL from source (clone from https://github.com/gefeiwang/scMODAL)
```

**MaxFuse Environment:**
```bash
conda env create -f ARCADIA/environments/environment_maxfuse.yaml
conda activate maxfuse
```

### Running Simulations

Generate synthetic datasets:

```bash
cd simulation
bash run_simulation.sh
```

This will generate synthetic spatial datasets for both tonsil and cite_seq.

### Running Benchmarks

Compare ARCADIA against baseline methods:

```bash
cd benchmark

# Run ARCADIA
python test_arcadia_tonsil.py
python test_arcadia_cite_seq.py

# Run scMODAL
python test_scmodal_tonsil.py
python test_scmodal_cite_seq.py

# Run MaxFuse
python test_maxfuse_tonsil.py
python test_maxfuse_cite_seq.py
```

### Reproducing Paper Figures

All figure notebooks are in the `notebooks/` directory:

cd notebooks

# Run individual figure notebooks
jupyter nbconvert --execute arcadia_fig1_simulation.ipynb
jupyter nbconvert --execute arcadia_fig2_archetype_analysis.ipynb


## Benchmarking Metrics

TBD Benchmarking Metrics

## Directory Descriptions

### `benchmark/`
Contains scripts for running comparative benchmarks between ARCADIA, scMODAL, and MaxFuse on tonsil and cite_seq datasets. Each script:
- Loads preprocessed data
- Runs the respective method
- Computes evaluation metrics
- Saves results and visualizations

### `data/`
Contains dataset metadata, preprocessing information, and download instructions. Each dataset subdirectory includes:
- `README.md`: Dataset description and preprocessing steps
- `metadata.json`: Dataset metadata (cell types, modalities, spatial info)

### `notebooks/`
Jupyter notebooks for reproducing paper figures and analyses:
- Simulation results and synthetic data generation
- Archetype analysis and visualization
- Cross-modal matching results
- Benchmark comparison across methods
- Spatial integration analysis
- **archetype_validation**: Validation of archetype generation
- **robustness_analysis**: Robustness to hyperparameters and data variations

### `simulation/`
Scripts for generating synthetic spatial datasets:
- `generate_synthetic_*.py`: Dataset-specific generation scripts
- `spatial_simulation.py`: Core spatial simulation functions
- `utils.py`: Utility functions for simulation
- `run_simulation.sh`: Batch script to run all simulations

### `ARCADIA/`
Complete ARCADIA implementation including:
- Core archetype generation and matching algorithms
- Dual VAE training plan
- Data loading and preprocessing utilities
- Visualization functions
- Pipeline scripts for end-to-end execution

### `outputs/`
Generated outputs organized by type:
- `benchmark_results/`: CSV files and plots from benchmarking
- `figures/`: Generated paper figures
- `processed_data/`: Preprocessed datasets ready for analysis

<!-- ## Citation

If you use ARCADIA or this reproducibility repository, please cite:

```bibtex
@article{arcadia2024,
  title={ARCADIA: Archetype-based Cross-modal Data Integration and Alignment},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
``` -->

## License

BSD 3-Clause License

## Contact

For questions or issues, please open an issue on GitHub or contact elham@azizilab.com.
