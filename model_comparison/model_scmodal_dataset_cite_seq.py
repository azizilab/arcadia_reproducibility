# %%
# add /home/barroz/projects/ARCADIA/model_comparison/scMODAL_main tot he PYTHONPATH
import sys

sys.path.append("/workspace/model_comparison/scMODAL_main")
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import anndata as ad

# print all modules in scmodal
import numpy as np
import pandas as pd
import scanpy as sc
import scmodal

warnings.filterwarnings("ignore")


# %%
def here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


ROOT = here().parent.parent
THIS_DIR = here()
print(f"ROOT: {ROOT}")
print(f"THIS_DIR: {THIS_DIR}")

# Update sys.path and cwd
sys.path.append(str(ROOT))
sys.path.append(str(THIS_DIR))
os.chdir(str(ROOT))
print(f"Working directory: {os.getcwd()}")

# %%
# Load data using the same approach as MaxFuse
dataset_name = "cite_seq"
print("Loading CITE-seq spleen lymph node data from h5ad files...")


def _get_latest_from_numbered_files(folder, files, index_from_end=0, exact_step=None):
    """Helper function to get latest file from numbered files."""

    def get_step_number(filename):
        step_part = filename.split("_")[0]
        try:
            return int(step_part)
        except ValueError:
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return int(mtime)

    if exact_step is not None:
        filtered_files = []
        for f in files:
            step_num = get_step_number(f)
            if isinstance(step_num, int) and step_num == exact_step:
                filtered_files.append(f)
        files = filtered_files

    if not files:
        return None

    def get_timestamp_from_filename_helper(filename):
        parts = filename.split("_")
        if len(parts) >= 3:
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return str(int(mtime))

    files.sort(key=get_timestamp_from_filename_helper, reverse=True)
    latest_file = files[index_from_end]
    return os.path.join(folder, latest_file)


def get_latest_file(
    folder, prefix, index_from_end=0, dataset_name=None, exact_step=None
) -> Optional[str]:
    """Get the latest file with given prefix from folder."""
    base_folder = Path(folder)

    if dataset_name:
        dataset_folder = base_folder / dataset_name
        if dataset_folder.exists():
            files = [f for f in os.listdir(dataset_folder) if f.endswith(".h5ad")]
            pattern_files = []
            for f in files:
                if f[0].isdigit() and "_" in f:
                    parts = f.split("_")
                    if len(parts) >= 2:
                        step_part = parts[0]
                        prefix_part = parts[1]
                        if step_part.isdigit() and prefix_part == prefix:
                            pattern_files.append(f)
            if pattern_files:
                return _get_latest_from_numbered_files(
                    dataset_folder, pattern_files, index_from_end, exact_step
                )

    all_files = []

    if base_folder.exists():
        main_files = [f for f in os.listdir(base_folder) if f.endswith(".h5ad")]
        for f in main_files:
            if f[0].isdigit() and "_" in f:
                parts = f.split("_")
                if len(parts) >= 2:
                    step_part = parts[0]
                    prefix_part = parts[1]
                    if step_part.isdigit() and prefix_part == prefix:
                        all_files.append((base_folder, f))

    if base_folder.exists():
        for subdir in base_folder.iterdir():
            if subdir.is_dir():
                sub_files = [f for f in os.listdir(subdir) if f.endswith(".h5ad")]
                for f in sub_files:
                    if f[0].isdigit() and "_" in f:
                        parts = f.split("_")
                        if len(parts) >= 2:
                            step_part = parts[0]
                            prefix_part = parts[1]
                            if step_part.isdigit() and prefix_part == prefix:
                                all_files.append((subdir, f))

    if not all_files:
        return None

    if exact_step is not None:
        filtered_files = []
        for folder_path, filename in all_files:
            step_part = filename.split("_")[0]
            try:
                step_num = int(step_part)
                if step_num == exact_step:
                    filtered_files.append((folder_path, filename))
            except ValueError:
                pass
        all_files = filtered_files

    if not all_files:
        return None

    def get_timestamp_from_filename(folder_file_tuple):
        folder_path, filename = folder_file_tuple
        parts = filename.split("_")
        if len(parts) >= 3:
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            mtime = os.path.getmtime(folder_path / filename)
            return str(int(mtime))

    all_files.sort(key=get_timestamp_from_filename, reverse=True)

    folder_path, latest_file = all_files[index_from_end]
    full_path = folder_path / latest_file

    file_time = datetime.fromtimestamp(os.path.getmtime(full_path))
    time_diff = datetime.now() - file_time

    if time_diff.days > 0:
        print(
            f"{full_path} was created {time_diff.days} days, {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    elif time_diff.seconds > 3600:
        print(
            f"{full_path} was created {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    else:
        print(f"{full_path} was created {time_diff.seconds} seconds ago")

    return str(full_path)


rna_file = get_latest_file(
    "CODEX_RNA_seq/data/processed_data", "rna", exact_step=1, dataset_name=dataset_name
)
protein_file = get_latest_file(
    "CODEX_RNA_seq/data/processed_data",
    "protein",
    exact_step=1,
    dataset_name=dataset_name,
)
adata_RNA = sc.read(rna_file)
adata_ADT = sc.read(protein_file)
adata_RNA.X = adata_RNA.layers["counts"].copy()
adata_ADT.X = adata_ADT.layers["counts"].copy()
# %%
# Cell type labels
adata_RNA.obs["celltype.l1"] = adata_RNA.obs["cell_types"]
adata_RNA.obs["celltype.l2"] = adata_RNA.obs["minor_cell_types"]

adata_ADT.obs["celltype.l1"] = adata_ADT.obs["cell_types"]
adata_ADT.obs["celltype.l2"] = adata_ADT.obs["minor_cell_types"]

print(f"RNA dataset: {adata_RNA.shape[0]} cells")
print(f"Protein dataset: {adata_ADT.shape[0]} cells")

# %%
# Find common cells between RNA and protein datasets
common_cells = adata_RNA.obs.index.intersection(adata_ADT.obs.index)
print(f"Common cells between RNA and protein: {len(common_cells)}")
print(f"RNA unique cells: {len(adata_RNA.obs.index) - len(common_cells)}")
print(f"Protein unique cells: {len(adata_ADT.obs.index) - len(common_cells)}")

# Subset both datasets to common cells
adata_RNA = adata_RNA[common_cells].copy()
adata_ADT = adata_ADT[common_cells].copy()

print(f"After filtering to common cells:")
print(f"RNA dataset: {adata_RNA.shape}")
print(f"Protein dataset: {adata_ADT.shape}")

# Now subsample if needed (using the same indices for both)
num_cells = 1000000000000
max_cells = min(num_cells, len(common_cells))
if len(common_cells) > max_cells:
    np.random.seed(42)
    subsample_indices = np.random.choice(len(common_cells), max_cells, replace=False)
    subsample_cells = common_cells[subsample_indices]
    adata_RNA = adata_RNA[subsample_cells].copy()
    adata_ADT = adata_ADT[subsample_cells].copy()
    print(f"Subsampled to {max_cells} cells")

# Verify matching cells
print(f"Final RNA dataset: {adata_RNA.shape[0]} cells")
print(f"Final Protein dataset: {adata_ADT.shape[0]} cells")
print(f"Indices match: {all(adata_RNA.obs.index == adata_ADT.obs.index)}")

# %%
# Use raw counts from layers for scMODAL processing
from scipy.sparse import issparse

if "counts" in adata_RNA.layers:
    adata_RNA.X = adata_RNA.layers["counts"].copy()
    if issparse(adata_RNA.X):
        adata_RNA.X = adata_RNA.X.toarray()
    print("Using raw counts from adata_RNA.layers['counts']")

if "counts" in adata_ADT.layers:
    adata_ADT.X = adata_ADT.layers["counts"].copy()
    if issparse(adata_ADT.X):
        adata_ADT.X = adata_ADT.X.toarray()
    print("Using raw counts from adata_ADT.layers['counts']")

# %%
# Load correspondence file
data_dir = "/workspace/CODEX_RNA_seq/data/raw_data"
correspondence = pd.read_csv(f"{data_dir}/tonsil/protein_gene_conversion.csv")
correspondence["Protein name"] = correspondence["Protein name"].replace(
    to_replace={"CD11a-CD18": "CD11a/CD18", "CD66a-c-e": "CD66a/c/e"}
)
print(correspondence.head())

# %%
# Create protein name mapping
protein_name_mapping = {}
for var_name in adata_ADT.var_names:
    if var_name.startswith("ADT_"):
        parts = var_name.split("_")
        if len(parts) >= 2:
            clean_name = parts[1].split("(")[0]
            protein_name_mapping[clean_name] = var_name

print(f"Created mapping for {len(protein_name_mapping)} proteins")

# Create RNA gene mapping (case-insensitive)
rna_gene_mapping = {gene.upper(): gene for gene in adata_RNA.var_names}

rna_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]

    actual_protein_name = protein_name_mapping.get(curr_protein_name)
    if actual_protein_name is None:
        continue

    if curr_rna_names.find("Ignore") != -1:
        continue
    curr_rna_names = curr_rna_names.split("/")
    for r in curr_rna_names:
        actual_rna_name = rna_gene_mapping.get(r.upper())
        if actual_rna_name is not None:
            rna_protein_correspondence.append([actual_rna_name, actual_protein_name])

rna_protein_correspondence = np.array(rna_protein_correspondence)
print(f"Found {len(rna_protein_correspondence)} RNA-protein correspondences")

# %%
RNA_shared = adata_RNA[:, rna_protein_correspondence[:, 0]].copy()
ADT_shared = adata_ADT[:, rna_protein_correspondence[:, 1]].copy()
RNA_shared.var["feature_name"] = RNA_shared.var.index.values
ADT_shared.var["feature_name"] = ADT_shared.var.index.values
RNA_shared.var_names_make_unique()
ADT_shared.var_names_make_unique()

# %%
RNA_unshared = adata_RNA[
    :, sorted(set(adata_RNA.var.index) - set(rna_protein_correspondence[:, 0]))
].copy()
ADT_unshared = adata_ADT[
    :, sorted(set(adata_ADT.var.index) - set(rna_protein_correspondence[:, 1]))
].copy()

sc.pp.highly_variable_genes(RNA_unshared, flavor="seurat_v3", n_top_genes=3000)
RNA_unshared = RNA_unshared[:, RNA_unshared.var.highly_variable].copy()

RNA_unshared.var["feature_name"] = RNA_unshared.var.index.values
ADT_unshared.var["feature_name"] = ADT_unshared.var.index.values

# %% [markdown]
# ## Normalization

# %%
RNA_counts = RNA_shared.X.sum(axis=1)
ADT_counts = ADT_shared.X.sum(axis=1)
target_sum = np.maximum(np.median(RNA_counts.copy()), 20)

sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
sc.pp.log1p(RNA_shared)

sc.pp.normalize_total(ADT_shared, target_sum=target_sum)
sc.pp.log1p(ADT_shared)

sc.pp.normalize_total(RNA_unshared)
sc.pp.log1p(RNA_unshared)

sc.pp.normalize_total(ADT_unshared)
sc.pp.log1p(ADT_unshared)

adata1 = ad.concat([RNA_shared, RNA_unshared], axis=1)
adata2 = ad.concat([ADT_shared, ADT_unshared], axis=1)

sc.pp.scale(adata1, max_value=10)
sc.pp.scale(adata2, max_value=10)
# plot a umap of subset of cells from adata1 and adata2


# %% [markdown]
# ## Running scMODAL

# %%
model = scmodal.model.Model(model_path="./CITE-seq_PBMC")  # ,training_steps=100)

model.preprocess(adata1, adata2, shared_gene_num=RNA_shared.shape[1])

# %%
model.train()
model.eval()

# %%
adata_integrated = ad.AnnData(X=model.latent)
adata_integrated.obs = pd.concat([adata_RNA.obs, adata_ADT.obs])
adata_integrated.obs["modality"] = ["RNA"] * adata_RNA.shape[0] + ["ADT"] * adata_ADT.shape[0]

scmodal.utils.compute_umap(adata_integrated)

# %%
sc.pl.umap(adata_integrated, color=["modality", "celltype.l2"])


# %%
adata_integrated

# %%


# %% [markdown]
# ## benchmarking

# %%
from scipy.spatial.distance import cdist

dist_mtx = cdist(
    model.latent[adata1.shape[0] :, :],
    model.latent[: adata1.shape[0], :],
    metric="euclidean",
)  # Transfer labels from RNA to ADT

matching = dist_mtx.argsort()[:, :1]

df1_labels = adata_RNA.obs["celltype.l1"].values
df2_labels = adata_ADT.obs["celltype.l1"].values

print(
    "Label transfer accuracy for L1: ",
    np.sum(df1_labels == df2_labels[matching.reshape(-1)]) / adata_RNA.shape[0],
)

# %%
dist_mtx = cdist(
    model.latent[adata1.shape[0] :, :],
    model.latent[: adata1.shape[0], :],
    metric="euclidean",
)

matching = dist_mtx.argsort()[:, :1]

df1_labels = adata_RNA.obs["celltype.l2"].values
df2_labels = adata_ADT.obs["celltype.l2"].values

print(
    "Label transfer accuracy for L2: ",
    np.sum(df1_labels == df2_labels[matching.reshape(-1)]) / adata_RNA.shape[0],
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
        (model.latent[: adata1.shape[0], :dim_use], model.latent[adata1.shape[0] :, :dim_use]),
        axis=0,
    ),
    dtype=np.float32,
)
latent_adata.obs["data_type"] = ["RNA"] * adata_RNA.shape[0] + ["Protein"] * adata_ADT.shape[0]
latent_adata.obs["celltype.l1"] = list(adata_RNA.obs["celltype.l1"]) + list(
    adata_ADT.obs["celltype.l1"]
)
latent_adata.obs["celltype.l2"] = list(adata_RNA.obs["celltype.l2"]) + list(
    adata_ADT.obs["celltype.l2"]
)

# %%
# Compute UMAP on latent space
sc.pp.neighbors(latent_adata, n_neighbors=15)
sc.tl.umap(latent_adata)
sc.pl.umap(latent_adata, color="data_type", title="scMODAL Latent Space - Modality")

# %%
# Plot by cell types
sc.pl.umap(latent_adata, color=["celltype.l1", "celltype.l2"], ncols=2)

# %%
# Prepare separate RNA and Protein AnnData objects with latent embeddings
rna_latent = model.latent[: adata1.shape[0], :]
protein_latent = model.latent[adata1.shape[0] :, :]

# %%
# Create RNA AnnData with original data and latent embedding
rna_adata_output = ad.AnnData(X=adata_RNA.X.copy())
rna_adata_output.obs = adata_RNA.obs.copy()
rna_adata_output.var = adata_RNA.var.copy()

# Add metadata fields
rna_adata_output.obs["batch_indices"] = 0
rna_adata_output.obs["n_genes"] = (rna_adata_output.X > 0).sum(axis=1)
rna_adata_output.obs["percent_mito"] = 0
rna_adata_output.obs["leiden_subclusters"] = "unknown"
rna_adata_output.obs["cell_types"] = rna_adata_output.obs["celltype.l1"]
rna_adata_output.obs["tissue"] = "pbmc"
rna_adata_output.obs["batch"] = "scmodal_cite_seq"
rna_adata_output.obs["minor_cell_types"] = rna_adata_output.obs["celltype.l2"]
rna_adata_output.obs["major_cell_types"] = rna_adata_output.obs["celltype.l1"]
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
rna_adata_output.uns["dataset_name"] = "scmodal_cite_seq"
rna_adata_output.uns["processing_stage"] = "scmodal_integrated"
rna_adata_output.uns["file_generated_from"] = "scmodal_cite_seq.py"

# obsm fields - clear and add latent
rna_adata_output.obsm.clear()
rna_adata_output.obsm["latent"] = rna_latent

# layers
rna_adata_output.layers["counts"] = rna_adata_output.X.copy()

print(f"rna_adata_output shape: {rna_adata_output.shape}")
print(f"rna_adata_output.obs columns: {list(rna_adata_output.obs.columns)}")

# %%
# Create Protein AnnData with original data and latent embedding
protein_adata_output = ad.AnnData(X=adata_ADT.X.copy())
protein_adata_output.obs = adata_ADT.obs.copy()
protein_adata_output.var = adata_ADT.var.copy()

protein_adata_output.obs["batch_indices"] = 0
protein_adata_output.obs["percent_mito"] = 0
protein_adata_output.obs["leiden_subclusters"] = "unknown"
protein_adata_output.obs["cell_types"] = protein_adata_output.obs["celltype.l1"]
protein_adata_output.obs["tissue"] = "pbmc"
protein_adata_output.obs["batch"] = "scmodal_cite_seq"
protein_adata_output.obs["minor_cell_types"] = protein_adata_output.obs["celltype.l2"]
protein_adata_output.obs["major_cell_types"] = protein_adata_output.obs["celltype.l1"]
protein_adata_output.obs["total_counts"] = protein_adata_output.X.sum(axis=1)
protein_adata_output.obs["n_genes_by_counts"] = (protein_adata_output.X > 0).sum(axis=1)
protein_adata_output.obs["index_col"] = np.arange(protein_adata_output.n_obs)

# var fields
protein_adata_output.var["feature_type"] = "protein"

# uns fields
protein_adata_output.uns["dataset_name"] = "scmodal_cite_seq"
protein_adata_output.uns["processing_stage"] = "scmodal_integrated"
protein_adata_output.uns["file_generated_from"] = "scmodal_cite_seq.py"

# obsm fields - clear and add latent
protein_adata_output.obsm.clear()
protein_adata_output.obsm["latent"] = protein_latent

# layers
protein_adata_output.layers["counts"] = protein_adata_output.X.copy()

print(f"protein_adata_output shape: {protein_adata_output.shape}")
print(f"protein_adata_output.obs columns: {list(protein_adata_output.obs.columns)}")

# %%
# Save the formatted AnnData objects
output_dir = "/workspace/model_comparison/outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rna_output = f"{output_dir}/scmodal_cite_seq/7_rna_{timestamp}.h5ad"
protein_output = f"{output_dir}/scmodal_cite_seq/7_protein_{timestamp}.h5ad"

os.makedirs(f"{output_dir}/scmodal_cite_seq", exist_ok=True)

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

# %%
