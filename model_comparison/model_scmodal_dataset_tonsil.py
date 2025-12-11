# %%
# scMODAL application to tonsil CODEX + RNA-seq data
# Based on model_maxfuse_dataset_tonsil.py but using scMODAL instead of MaxFuse

import sys

sys.path.append("/workspace/model_comparison/scMODAL_main")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scmodal

scmodal.model.Model
from scipy.io import mmread

plt.rcParams["figure.figsize"] = (6, 4)

import os
import warnings
from datetime import datetime

import anndata as ad
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# %%
scmodal

# %%
# Load protein (CODEX) data
root_dir = "/workspace"
protein = pd.read_csv(
    f"{root_dir}/CODEX_RNA_seq/data/raw_data/tonsil/tonsil_codex.csv"
)  # ~178,000 codex cells

# %%
# Visualize spatial distribution (optional)
sns.scatterplot(
    data=protein.sample(frac=0.1), x="centroid_x", y="centroid_y", hue="cluster.term", s=1
)

# %%
# Extract protein features
protein_features = [
    "CD38",
    "CD19",
    "CD31",
    "Vimentin",
    "CD22",
    "Ki67",
    "CD8",
    "CD90",
    "CD123",
    "CD15",
    "CD3",
    "CD152",
    "CD21",
    "cytokeratin",
    "CD2",
    "CD66",
    "collagen IV",
    "CD81",
    "HLA-DR",
    "CD57",
    "CD4",
    "CD7",
    "CD278",
    "podoplanin",
    "CD45RA",
    "CD34",
    "CD54",
    "CD9",
    "IGM",
    "CD117",
    "CD56",
    "CD279",
    "CD45",
    "CD49f",
    "CD5",
    "CD16",
    "CD63",
    "CD11b",
    "CD1c",
    "CD40",
    "CD274",
    "CD27",
    "CD104",
    "CD273",
    "FAPalpha",
    "Ecadherin",
]

# Convert to AnnData
adata_ADT = ad.AnnData(
    protein[protein_features].to_numpy(),
    dtype=np.float32,
    obsm={"spatial": protein[["centroid_x", "centroid_y"]].to_numpy()},
)
adata_ADT.var_names = protein[protein_features].columns

# %%
# Load RNA data
base_folder = f"{root_dir}/CODEX_RNA_seq/data/raw_data/tonsil"
rna = mmread(f"{base_folder}/tonsil_rna_counts.txt")  # ~10k cells
rna_names = pd.read_csv(f"{base_folder}/tonsil_rna_names.csv")["names"].to_numpy()

# Convert to AnnData
adata_RNA = ad.AnnData(rna.tocsr(), dtype=np.float32)
adata_RNA.var_names = rna_names

# %%
print(f"RNA dataset: {adata_RNA.shape}")
print(f"Protein dataset: {adata_ADT.shape}")

# %%
# Read in cell type labels
df_rna_metadata = pd.read_csv(f"{base_folder}/tonsil_rna_meta.csv", index_col=0)
labels_rna = df_rna_metadata["cluster.info"].to_numpy()
labels_codex = protein["cluster.term"].to_numpy()
adata_ADT.obs["celltype"] = labels_codex
adata_RNA.obs["celltype"] = labels_rna
adata_RNA.obs.index = df_rna_metadata.index.to_numpy()

# Add celltype.l1 and celltype.l2 for consistency
adata_RNA.obs["celltype.l1"] = adata_RNA.obs["celltype"]
adata_RNA.obs["celltype.l2"] = adata_RNA.obs["celltype"]
adata_ADT.obs["celltype.l1"] = adata_ADT.obs["celltype"]
adata_ADT.obs["celltype.l2"] = adata_ADT.obs["celltype"]

# %%
# Optional subsampling for faster processing
num_cells = 500000000
sc.pp.subsample(adata_RNA, n_obs=min(num_cells, adata_RNA.shape[0]))
sc.pp.subsample(adata_ADT, n_obs=min(num_cells, adata_ADT.shape[0]))
adata_RNA = adata_RNA.copy()
adata_ADT = adata_ADT.copy()

# %%
# Load correspondence file
correspondence = pd.read_csv(f"{base_folder}/protein_gene_conversion.csv")
correspondence["Protein name"] = correspondence["Protein name"].replace(
    to_replace={"CD11a-CD18": "CD11a/CD18", "CD66a-c-e": "CD66a/c/e"}
)
print(correspondence.head())

# %%


# %%
# Build RNA-protein correspondence
rna_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]
    if curr_protein_name not in adata_ADT.var_names:
        continue
    if curr_rna_names.find("Ignore") != -1:
        continue
    curr_rna_names = curr_rna_names.split("/")
    for r in curr_rna_names:
        if r in adata_RNA.var_names:
            rna_protein_correspondence.append([r, curr_protein_name])

rna_protein_correspondence = np.array(rna_protein_correspondence)
print(f"Found {len(rna_protein_correspondence)} RNA-protein correspondences")

# %%
# Create shared features datasets
RNA_shared = adata_RNA[:, rna_protein_correspondence[:, 0]].copy()
ADT_shared = adata_ADT[:, rna_protein_correspondence[:, 1]].copy()
RNA_shared.var["feature_name"] = RNA_shared.var.index.values
ADT_shared.var["feature_name"] = ADT_shared.var.index.values
RNA_shared.var_names_make_unique()
ADT_shared.var_names_make_unique()

# %%
# Create unshared features datasets
rna_unshared_genes = list(set(adata_RNA.var.index) - set(rna_protein_correspondence[:, 0]))
# adt_unshared_proteins = list(set(adata_ADT.var.index) - set(rna_protein_correspondence[:, 1]))

RNA_unshared = adata_RNA[:, sorted(rna_unshared_genes)].copy()
# ADT_unshared = adata_ADT[:, sorted(adt_unshared_proteins)].copy()

# Select highly variable genes for RNA unshared
# Convert sparse to dense if needed
if issparse(RNA_unshared.X):
    RNA_unshared.X = RNA_unshared.X.toarray()

sc.pp.highly_variable_genes(RNA_unshared, flavor="seurat_v3", n_top_genes=1000)
RNA_unshared = RNA_unshared[:, RNA_unshared.var.highly_variable].copy()

RNA_unshared.var["feature_name"] = RNA_unshared.var.index.values
# ADT_unshared.var["feature_name"] = ADT_unshared.var.index.values

# %%
# Convert sparse matrices to dense for scMODAL
if issparse(RNA_shared.X):
    RNA_shared.X = RNA_shared.X.toarray()
if issparse(ADT_shared.X):
    ADT_shared.X = ADT_shared.X.toarray()
if issparse(RNA_unshared.X):
    RNA_unshared.X = RNA_unshared.X.toarray()
# if issparse(ADT_unshared.X):
#     ADT_unshared.X = ADT_unshared.X.toarray()

print(f"RNA_shared shape: {RNA_shared.shape}")
print(f"ADT_shared shape: {ADT_shared.shape}")
print(f"RNA_unshared shape: {RNA_unshared.shape}")
# print(f"ADT_unshared shape: {ADT_unshared.shape}")

# %% [markdown]
# ## Normalization

# %%
# Normalize following original scMODAL tutorial approach
# Note: CODEX data is assumed to be raw counts here, so we normalize it first
# to get the target sum for RNA normalization

# First normalize and log-transform ADT_shared to match tutorial's assumption
sc.pp.normalize_total(ADT_shared)
sc.pp.log1p(ADT_shared)

# Calculate target sum from CODEX shared features (reverse log transform)
target_sum = np.median((np.exp(ADT_shared.X) - 1).sum(axis=1))

# Normalize RNA_shared to match CODEX distribution
sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
sc.pp.log1p(RNA_shared)

# Normalize unshared features
sc.pp.normalize_total(RNA_unshared)
sc.pp.log1p(RNA_unshared)

# Concatenate shared and unshared features
adata1 = ad.concat([RNA_shared, RNA_unshared], axis=1)
adata1.obs["celltype"] = RNA_shared.obs["celltype"]

# CODEX data do not contain unlinked features with RNA data
adata2 = ADT_shared.copy()
adata2.obs["celltype"] = ADT_shared.obs["celltype"]

# Scale
sc.pp.scale(adata1, max_value=10)
sc.pp.scale(adata2, max_value=10)

print(f"adata1 (RNA) shape: {adata1.shape}")
print(f"adata2 (ADT) shape: {adata2.shape}")

# %%
ADT_shared

# %% [markdown]
# ## Running scMODAL
# import torch
#
# print("cuda available: ", torch.cuda.is_available())

# %%
# Train model following original tutorial approach with explicit hyperparameters
model = scmodal.model.Model(
    training_steps=10000, lambdaMNN=5, lambdaGAN=0.5, model_path="./scMODAL_tonsil"
)

# Use integrate_datasets_feats with paired MNN inputs for shared features
model.integrate_datasets_feats(
    input_feats=[adata2.X, adata1.X],
    paired_input_MNN=[[adata2.X[:, : RNA_shared.shape[1]], adata1.X[:, : RNA_shared.shape[1]]]],
)

# %%
# Create integrated AnnData
# Note: order is [CODEX, RNA] based on input_feats order
adata_integrated = ad.AnnData(X=model.latent)
adata_integrated.obs = pd.concat([adata_ADT.obs, adata_RNA.obs])
adata_integrated.obs["modality"] = ["CODEX"] * adata_ADT.shape[0] + ["RNA"] * adata_RNA.shape[0]

scmodal.utils.compute_umap(adata_integrated)

# %%
sc.pl.umap(adata_integrated, color=["modality", "celltype"])

# %%
sc.pl.umap(adata_integrated[adata_integrated.obs["modality"] == "RNA"], color=["celltype"])
sc.pl.umap(adata_integrated[adata_integrated.obs["modality"] == "CODEX"], color=["celltype"])

# %%
print(adata_integrated)

# %% [markdown]
# ## Benchmarking

# %%
# Label transfer from RNA to CODEX (matching original tutorial)
from scipy.spatial.distance import cdist

dist_mtx = cdist(
    model.latent[: adata2.shape[0], :],  # CODEX
    model.latent[adata2.shape[0] : (adata1.shape[0] + adata2.shape[0]), :],  # RNA
    metric="euclidean",
)

matching = dist_mtx.argsort()[:, :1]

codex_labels = adata_ADT.obs["celltype"].values
rna_labels = adata_RNA.obs["celltype"].values

print(
    "Label transfer accuracy (RNA->CODEX): ",
    np.sum(codex_labels == rna_labels[matching.reshape(-1)]) / adata_ADT.shape[0],
)

# %%
adata_integrated.obs["modality"].value_counts()

# %% [markdown]
# ## Visualize latent space

# %%
# Create combined latent space visualization
dim_use = min(15, model.latent.shape[1])  # use first 15 dimensions or all if less

latent_adata = ad.AnnData(
    np.concatenate(
        (
            model.latent[: adata2.shape[0], :dim_use],
            model.latent[adata2.shape[0] :, :dim_use],
        ),
        axis=0,
    ),
    dtype=np.float32,
)
latent_adata.obs["data_type"] = ["CODEX"] * adata_ADT.shape[0] + ["RNA"] * adata_RNA.shape[0]
latent_adata.obs["celltype"] = list(adata_ADT.obs["celltype"]) + list(adata_RNA.obs["celltype"])

# %%
# Compute UMAP on latent space
sc.pp.neighbors(latent_adata, n_neighbors=15)
sc.tl.umap(latent_adata)
sc.pl.umap(latent_adata, color="data_type", title="scMODAL Latent Space - Modality (Tonsil)")

# %%
# Plot by cell types
sc.pl.umap(latent_adata, color="celltype", title="scMODAL Latent Space - Cell Types (Tonsil)")

# %%
# Prepare separate RNA and Protein AnnData objects with latent embeddings
# Order is [CODEX, RNA] based on input_feats
protein_latent = model.latent[: adata2.shape[0], :]
rna_latent = model.latent[adata2.shape[0] :, :]

# %%
# Create RNA AnnData with original data and latent embedding
if issparse(adata_RNA.X):
    adata_RNA.X = adata_RNA.X.toarray()

rna_adata_output = ad.AnnData(X=adata_RNA.X.copy())
rna_adata_output.obs = adata_RNA.obs.copy()
rna_adata_output.var = adata_RNA.var.copy()

# Add metadata fields
rna_adata_output.obs["batch_indices"] = 0
rna_adata_output.obs["n_genes"] = (rna_adata_output.X > 0).sum(axis=1)
rna_adata_output.obs["percent_mito"] = 0
rna_adata_output.obs["leiden_subclusters"] = "unknown"
rna_adata_output.obs["cell_types"] = rna_adata_output.obs["celltype"]
rna_adata_output.obs["tissue"] = "tonsil"
rna_adata_output.obs["batch"] = "scmodal_tonsil"
rna_adata_output.obs["minor_cell_types"] = rna_adata_output.obs["celltype"]
rna_adata_output.obs["major_cell_types"] = rna_adata_output.obs["celltype"]
rna_adata_output.obs["total_counts"] = np.array(rna_adata_output.X.sum(axis=1)).flatten()
rna_adata_output.obs["n_genes_by_counts"] = (rna_adata_output.X > 0).sum(axis=1)
rna_adata_output.obs["pct_counts_mt"] = 0
rna_adata_output.obs["index_col"] = np.arange(rna_adata_output.n_obs)

# var fields
rna_adata_output.var["n_cells"] = (rna_adata_output.X > 0).sum(axis=0)
rna_adata_output.var["mt"] = False
rna_adata_output.var["ribo"] = False
rna_adata_output.var["hb"] = False
rna_adata_output.var["total_counts"] = np.array(rna_adata_output.X.sum(axis=0)).flatten()
rna_adata_output.var["n_cells_by_counts"] = (rna_adata_output.X > 0).sum(axis=0)

# uns fields
rna_adata_output.uns["dataset_name"] = "scmodal_tonsil"
rna_adata_output.uns["processing_stage"] = "scmodal_integrated"
rna_adata_output.uns["file_generated_from"] = "model_scmodal_dataset_tonsil.py"

# obsm fields - clear and add latent
rna_adata_output.obsm.clear()
rna_adata_output.obsm["latent"] = rna_latent

# layers
rna_adata_output.layers["counts"] = rna_adata_output.X.copy()

print(f"rna_adata_output shape: {rna_adata_output.shape}")
print(f"rna_adata_output.obs columns: {list(rna_adata_output.obs.columns)}")

# %%
# Create Protein AnnData with original data and latent embedding
if issparse(adata_ADT.X):
    adata_ADT.X = adata_ADT.X.toarray()

protein_adata_output = ad.AnnData(X=adata_ADT.X.copy())
protein_adata_output.obs = adata_ADT.obs.copy()
protein_adata_output.var = adata_ADT.var.copy()

protein_adata_output.obs["batch_indices"] = 0
protein_adata_output.obs["percent_mito"] = 0
protein_adata_output.obs["leiden_subclusters"] = "unknown"
protein_adata_output.obs["cell_types"] = protein_adata_output.obs["celltype"]
protein_adata_output.obs["tissue"] = "tonsil"
protein_adata_output.obs["batch"] = "scmodal_tonsil"
protein_adata_output.obs["minor_cell_types"] = protein_adata_output.obs["celltype"]
protein_adata_output.obs["major_cell_types"] = protein_adata_output.obs["celltype"]

# Add spatial coordinates
protein_adata_output.obs["condition"] = "tonsil"
protein_adata_output.obs["Image"] = "tonsil_image"
protein_adata_output.obs["Sample"] = "tonsil_sample"
protein_adata_output.obs["total_counts"] = protein_adata_output.X.sum(axis=1)
protein_adata_output.obs["outlier"] = False
protein_adata_output.obs["n_genes_by_counts"] = (protein_adata_output.X > 0).sum(axis=1)
protein_adata_output.obs["log1p_n_genes_by_counts"] = np.log1p(
    protein_adata_output.obs["n_genes_by_counts"]
)
protein_adata_output.obs["log1p_total_counts"] = np.log1p(protein_adata_output.obs["total_counts"])
protein_adata_output.obs["CN"] = "CN_unknown"
protein_adata_output.obs["index_col"] = np.arange(protein_adata_output.n_obs)

# var fields
protein_adata_output.var["feature_type"] = "protein"

# uns fields
protein_adata_output.uns["dataset_name"] = "scmodal_tonsil"
protein_adata_output.uns["processing_stage"] = "scmodal_integrated"
protein_adata_output.uns["file_generated_from"] = "model_scmodal_dataset_tonsil.py"

# obsm fields - clear and add latent, keep spatial
protein_adata_output.obsm.clear()
protein_adata_output.obsm["latent"] = protein_latent
protein_adata_output.obsm["spatial"] = adata_ADT.obsm["spatial"].copy()

# layers
protein_adata_output.layers["counts"] = protein_adata_output.X.copy()

print(f"protein_adata_output shape: {protein_adata_output.shape}")
print(f"protein_adata_output.obs columns: {list(protein_adata_output.obs.columns)}")

# %%
# Visualize spatial coordinates with cell types
sc.pl.embedding(protein_adata_output, "spatial", color="cell_types")

# Save the formatted AnnData objects
output_dir = f"{root_dir}/model_comparison/outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rna_output = f"{output_dir}/scmodal_tonsil/7_rna_{timestamp}.h5ad"
protein_output = f"{output_dir}/scmodal_tonsil/7_protein_{timestamp}.h5ad"

os.makedirs(f"{output_dir}/scmodal_tonsil", exist_ok=True)

rna_adata_output.write(rna_output)
protein_adata_output.write(protein_output)

print(f"Saved rna_adata to: {rna_output}")
print(f"Saved protein_adata to: {protein_output}")
print(f"\nrna_adata: {rna_adata_output}")
print(f"\nprotein_adata: {protein_adata_output}")

# %%
# Display summary
print("=" * 80)
print("RNA AnnData Summary:")
print("=" * 80)
print(f"Shape: {rna_adata_output.shape}")
print(f"\nobs fields ({len(rna_adata_output.obs.columns)}):")
print(list(rna_adata_output.obs.columns))
print(f"\nvar fields ({len(rna_adata_output.var.columns)}):")
print(list(rna_adata_output.var.columns))
print(f"\nuns fields ({len(rna_adata_output.uns.keys())}):")
print(list(rna_adata_output.uns.keys()))
print(f"\nobsm fields ({len(rna_adata_output.obsm.keys())}):")
print(list(rna_adata_output.obsm.keys()))
print(f"\nlayers ({len(rna_adata_output.layers.keys())}):")
print(list(rna_adata_output.layers.keys()))
print(f"\nobsp fields ({len(rna_adata_output.obsp.keys()) if rna_adata_output.obsp else 0}):")
print(list(rna_adata_output.obsp.keys()) if rna_adata_output.obsp else [])

print("\n" + "=" * 80)
print("Protein AnnData Summary:")
print("=" * 80)
print(f"Shape: {protein_adata_output.shape}")
print(f"\nobs fields ({len(protein_adata_output.obs.columns)}):")
print(list(protein_adata_output.obs.columns))
print(f"\nvar fields ({len(protein_adata_output.var.columns)}):")
print(list(protein_adata_output.var.columns))
print(f"\nuns fields ({len(protein_adata_output.uns.keys())}):")
print(list(protein_adata_output.uns.keys()))
print(f"\nobsm fields ({len(protein_adata_output.obsm.keys())}):")
print(list(protein_adata_output.obsm.keys()))
print(f"\nlayers ({len(protein_adata_output.layers.keys())}):")
print(list(protein_adata_output.layers.keys()))
print(
    f"\nobsp fields ({len(protein_adata_output.obsp.keys()) if protein_adata_output.obsp else 0}):"
)
print(list(protein_adata_output.obsp.keys()) if protein_adata_output.obsp else [])

# %%
