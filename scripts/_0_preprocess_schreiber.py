# %% This script preprocesses raw CODEX spatial proteomics and RNA-seq data from the Schreiber lab dataset:

# Key Operations:
# Load raw data files:
# RNA: rna_umap.h5ad (5,546 cells × 13,447 genes)
# Protein: codex_cn_tumor.h5ad (spatial proteomics data)
# Metadata: codex_meta.csv
# Filter to mutual cell types between RNA and protein datasets
# Quality control and outlier removal using MAD (Median Absolute Deviation)
# Normalize protein data using either z-normalization or log-double-z-normalization (selected based on silhouette score)
# Select highly variable genes for RNA data (using knee detection)
# Perform spatial analysis on protein data
# Save processed data with timestamps

# Outputs:
# preprocessed_adata_rna_[timestamp].h5ad
# preprocessed_adata_prot_[timestamp].h5ad


# %% --- Imports and Config ---
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")


# Helper function to work in both scripts and notebooks
def here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# Determine ROOT based on whether we're running as script or notebook
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent

THIS_DIR = here()

# Add src to path for arcadia package
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Update sys.path and cwd
sys.path.append(str(ROOT))
sys.path.append(str(THIS_DIR))
os.chdir(str(ROOT))

import numpy as np
import pandas as pd
import scanpy as sc

from arcadia.data_utils import (
    filter_unwanted_cell_types,
    harmonize_cell_types_names,
    mad_outlier_removal,
    preprocess_rna_initial_steps,
    read_legacy_adata,
    save_processed_data,
)
from arcadia.plotting import preprocessing as pp_plots
from arcadia.utils import metadata as pipeline_metadata_utils

# Load config_ if exists
config_path = Path("configs/config.json")
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

start_time = datetime.now()
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
sc.settings.set_figure_params(dpi=50, facecolor="white")

# %% --- Directory Structure Definition ---
base_path = "."
dataset_name = "schreiber"
FILENAME = f"_0_preprocess_{dataset_name}.py"
directories = [
    "processed_data",
    f"raw_datasets/{dataset_name}",
]
print(f"Creating directory structure under: {os.path.abspath(base_path)}")
dataset_path = f"raw_datasets/{dataset_name}"
for rel_path in directories:
    full_path = os.path.join(base_path, rel_path)
    os.makedirs(full_path, exist_ok=True)
# %%
rna_annotation_cell_kevin = pd.read_csv(
    f"{dataset_path}/annotated_cells_for_Kevin.tsv", sep="\t", index_col=0
)
harmonization_mapping_path = f"{dataset_path}/modality_cell_type_harmonization.yaml"
with open(harmonization_mapping_path, "r") as f:
    harmonization_mapping = yaml.safe_load(f)
file_path = f"{dataset_path}/mouse_ICB_scRNA_umap_labelled.h5ad"
adata_rna = read_legacy_adata(file_path)
adata_rna.X = adata_rna.layers["counts"]
# to sparse matrix if needed
adata_rna.X = adata_rna.X.tocsr()
adata_rna.obs["cell_types"] = pd.Categorical(
    rna_annotation_cell_kevin.loc[adata_rna.obs.index, "annotation"]
)
adata_rna.obs["batch"] = "batch_0"
adata_rna.uns["pipeline_metadata"] = pipeline_metadata_utils.initialize_pipeline_metadata(
    timestamp_str, FILENAME, dataset_name
)
# %%
total_elements = adata_rna.X.shape[0] * adata_rna.X.shape[1]
print(
    f"Number of zero elements proportion in new dataset: num of genes is {adata_rna.X.shape[1]} and the proportion of zero elements is {(total_elements - adata_rna.X.nnz)/total_elements}"
)
# %%
pp_plots.plot_count_distribution(adata_rna, plot_flag)
pp_plots.plot_expression_heatmap(adata_rna, plot_flag)
adata_rna.X = adata_rna.X.tocsr()

# %%
assert np.allclose(adata_rna.X.data, np.round(adata_rna.X.data))
adata_rna = preprocess_rna_initial_steps(adata_rna, min_genes=200, min_cells=3, plot_flag=plot_flag)


print(f"\nMerged dataset shape: {adata_rna.shape}")
print(f"Cell type distribution in merged dataset:")
print(adata_rna.obs["cell_types"].value_counts())
print(f"Dataset source distribution:")
print(adata_rna.obs["batch"].value_counts())

# %% Processing
batch_means = (
    pd.DataFrame(adata_rna.X.toarray() if hasattr(adata_rna.X, "toarray") else adata_rna.X)
    .groupby(adata_rna.obs["batch"].values)
    .mean()
)

# plot heatmap of subsample of the adata_rna.X
adata_rna.X = adata_rna.X.astype(float)
pp_plots.plot_merged_dataset_analysis(adata_rna, plot_flag)
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
adata_rna.uns["pipeline_metadata"]["log1p"] = True
adata_rna.uns["pipeline_metadata"]["normalization"] = {}
adata_rna.uns["pipeline_metadata"]["normalization"]["log1p_applied"] = False
adata_rna.layers["log1p"] = adata_rna.X.copy()
# %% load protein data

adata_prot = sc.read(f"{dataset_path}/codex_cn_tumor.h5ad")
prot_metadata = pd.read_csv(f"{dataset_path}/codex_meta.csv")
adata_rna = filter_unwanted_cell_types(adata_rna, ["tumor", "dead"])
adata_prot = filter_unwanted_cell_types(adata_prot, ["tumor", "dead"])

# Assign metadata and labels from codex csv to adata
adata_prot.obs["cell_types"] = pd.Categorical(adata_prot.obs["cell_type"])
adata_prot.obs["granular_cell_types"] = pd.Categorical(prot_metadata["cell_type"])
adata_prot.obs["Image"] = prot_metadata["Image"].astype(str).values
adata_prot.obs["lab_CN"] = pd.Categorical(adata_prot.obs["neighborhood"])
adata_prot.obs["lab_CN"] = (
    adata_prot.obs["lab_CN"]
    .replace(harmonization_mapping["cn_to_codex_mapping"])
    .astype("category")
)
adata_prot.layers["raw"] = adata_prot.X
pp_plots.plot_protein_analysis(adata_prot, plot_flag=plot_flag)
# # todo: remove later
# sc.pp.subsample(adata_prot, n_obs=min(10000, adata_prot.n_obs))
# sc.pp.subsample(adata_rna, n_obs=min(10000, adata_rna.n_obs))
adata_prot_subsampled = adata_prot[
    np.random.choice(adata_prot.n_obs, size=min(4000, adata_prot.n_obs), replace=False)
]

sc.pp.pca(adata_prot_subsampled)
sc.pp.neighbors(adata_prot_subsampled, use_rep="X_pca")
sc.tl.umap(adata_prot_subsampled)
if plot_flag:
    sc.pl.umap(adata_prot_subsampled, color="cell_types")
    sc.pl.umap(adata_prot_subsampled, color="condition")
    sc.pl.umap(
        adata_prot_subsampled[adata_prot_subsampled.obs["condition"] == "ICT",], color="cell_types"
    )

adata_rna_subsampled = adata_rna[
    np.random.choice(adata_rna.n_obs, size=min(4000, adata_rna.n_obs), replace=False)
]
sc.pp.pca(adata_rna_subsampled)
sc.pp.neighbors(adata_rna_subsampled)
sc.tl.umap(adata_rna_subsampled)
if plot_flag:
    sc.pl.umap(adata_rna_subsampled, color="cell_types")

# %%

if plot_flag:
    images = set(adata_prot.obs["Image"])
    images_ict = [image for image in images if image.startswith("ict")]
    images_ict = [images_ict[0]]
    images_cntrl = []  # [image for image in images if image.startswith("cntrl")]
    # all unordered pairs (ict, cntrl) for matching images by suffix
    image_pairs = list(product(images_ict, images_cntrl))
    for i, (image_ict, image_cntrl) in enumerate(image_pairs):
        if i > 3:
            break
        adata_prot_temp = adata_prot[adata_prot.obs["Image"].isin([image_ict, image_cntrl])].copy()
        sc.pp.subsample(adata_prot_temp, n_obs=min(800, adata_prot_temp.n_obs))
        sc.pp.pca(adata_prot_temp, copy=False)
        sc.pp.neighbors(adata_prot_temp, use_rep="X_pca")
        sc.tl.umap(adata_prot_temp)
        pp_plots.plot_umap_analysis(adata_prot_temp, "cell_types", "Cell Types", plot_flag)
        pp_plots.plot_umap_analysis(adata_prot_temp, "Image", "Image", plot_flag)

# %%
protein_sample_names = [
    "cntrl_n109_d10",
    "cntrl_n130_d10",
    "cntrl_n131_d10",
    "cntrl_n140_d10",
    "cntrl_n251_d10",
    "ict_n112_d10",
    "ict_n113_d10",
    "ict_n205_d10",
    "ict_n212_d10",
    "ict_n55_d10",
]
valid_protein_sample_names = ["cntrl_n251_d10", "ict_n205_d10"]
# plot the barplot of
for sample in sorted(protein_sample_names):
    counts = adata_prot[adata_prot.obs["Image"] == sample].obs["cell_types"].value_counts()
    # drop tumor cells
    counts = counts.sort_index()  # sort bars alphabetically by cell type
    ax = counts.plot.bar(title=f"Proportion of cell types in sample {sample}", rot=45)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cell Type")
    plt.tight_layout()
    if plot_flag:
        plt.show()
    plt.close()

# take only specific samples
adata_prot = adata_prot[adata_prot.obs["Image"].isin(valid_protein_sample_names)].copy()
adata_prot.obs["batch"] = adata_prot.obs["Image"]

# %% Treatment filtering has already been applied during merge process:
# The merged adata_rna now contains only treated cells from both datasets
print(f"Shape: {adata_rna.shape}")
print(f'Dataset source distribution: {adata_rna.obs["batch"].value_counts().to_dict()}')
if "Sample" in adata_rna.obs.columns:
    print(f'Sample values: {adata_rna.obs["Sample"].value_counts().to_dict()}')
if "treated" in adata_rna.obs.columns:
    print(f'Treated values: {adata_rna.obs["treated"].value_counts().to_dict()}')

# %% Debug/Logging
print(f"\n[DEBUG] Before harmonize_cell_types_names:")
print(f"RNA shape: {adata_rna.shape}, Protein shape: {adata_prot.shape}")
print(
    f"RNA, Protein cell types: {sorted(set(adata_rna.obs['cell_types'])), sorted(set(adata_prot.obs['cell_types']))}"
)
print("[DEBUG] Calling harmonize_cell_types_names...")
adata_rna, adata_prot = harmonize_cell_types_names(adata_rna, adata_prot, harmonization_mapping)
print("[DEBUG] harmonize_cell_types_names completed successfully")
print(
    f"RNA, Protein cell types: {sorted(set(adata_rna.obs['cell_types']))}, {sorted(set(adata_prot.obs['cell_types']))}"
)

# %%
pp_plots.plot_protein_violin(adata_prot, plot_flag)
sc.pp.scale(adata_prot)
adata_prot.layers["z_normalized"] = adata_prot.X.copy()
adata_prot_subsampled = sc.pp.subsample(
    adata_prot, n_obs=min(9000, adata_prot.n_obs), copy=True
).copy()
sc.pp.pca(adata_prot_subsampled)
sc.pp.neighbors(adata_prot_subsampled, use_rep="X_pca")
sc.tl.umap(adata_prot_subsampled)
sc.pl.umap(adata_prot_subsampled, color="cell_types")


adata_prot = mad_outlier_removal(adata_prot)

# Finalize pipeline metadata before saving
pipeline_metadata_utils.finalize_preprocess_metadata(adata_rna, adata_prot, ["batch_0"])

# Plot protein feature distributions
pp_plots.plot_protein_feature_distributions(
    adata_prot, layer="z_normalized", n_features=30, n_cells=1000, plot_flag=plot_flag
)
pp_plots.plot_protein_feature_distributions(
    adata_prot, layer=None, n_features=30, n_cells=1000, plot_flag=plot_flag
)

save_processed_data(adata_rna, adata_prot, "processed_data", caller_filename=FILENAME)
# # take a subsample of the data
# adata_prot_subsampled = adata_prot_normalized[
#     np.random.choice(
#         adata_prot_normalized.n_obs, size=min(6000, adata_prot_normalized.n_obs), replace=False
#     )
# ].copy()
# sc.pp.pca(adata_prot_subsampled, copy=False)
# sc.pp.neighbors(adata_prot_subsampled, use_rep="X_pca")
# sc.tl.umap(adata_prot_subsampled)
# pp_plots.plot_spatial_analysis(adata_prot_subsampled, plot_flag)
# pp_plots.plot_heatmap_analysis(adata_prot_subsampled, plot_flag)
# # same for rna
# adata_rna_subsampled = adata_rna_normalized[
#     np.random.choice(
#         adata_rna_normalized.n_obs, size=min(6000, adata_rna_normalized.n_obs), replace=False
#     )
# ].copy()
# sc.pp.pca(adata_rna_subsampled, copy=False)
# sc.pp.neighbors(adata_rna_subsampled, use_rep="X_pca")
# sc.tl.umap(adata_rna_subsampled)
# pp_plots.plot_spatial_analysis(adata_rna_subsampled, plot_flag)
# pp_plots.plot_heatmap_analysis(adata_rna_subsampled, plot_flag)
# # %%

# %%
