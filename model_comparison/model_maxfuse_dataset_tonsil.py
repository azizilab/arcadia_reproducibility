# %%
# need to use python version 3.8 with conda as that's the requirement for maxfuse
# inspired from https://github.com/shuxiaoc/maxfuse/blob/main/docs/tonsil_codex_rnaseq.ipynb
# preprocess tonsil data for maxfuse and runs maxfuse, saves the results to a file to be compared with arcadia

import os
import warnings
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import maxfuse as mf
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from comparison_utils import get_latest_file, here
import sys

# Suppress warnings about log-transformed data since we're using raw counts from layers
warnings.filterwarnings('ignore', message='.*adata.X seems to be already log-transformed.*', category=UserWarning)
# Suppress warnings about variable names not being unique (we handle this explicitly)
warnings.filterwarnings('ignore', message='.*Variable names are not unique.*', category=UserWarning)
# Suppress warnings about cells with zero counts (we're using raw counts which may have zeros)
warnings.filterwarnings('ignore', message='.*Some cells have zero counts.*', category=UserWarning)
plt.rcParams["figure.figsize"] = (6, 4)
if here().parent.name == "notebooks":
    os.chdir("../../")

ROOT = here().parent
THIS_DIR = here()
print(ROOT)
# Update sys.path and cwd
sys.path.append(str(ROOT))
sys.path.append(str(THIS_DIR))
os.chdir(str(ROOT))

# %%
# Load the tonsil data directly from h5ad files (matching preprocessing pipeline)
dataset_name = "tonsil"
print("Loading tonsil CODEX + RNA-seq data from h5ad files...")

rna_file = get_latest_file(
    "ARCADIA/processed_data", "rna", exact_step=1, dataset_name=dataset_name
)
protein_file = get_latest_file(
    "ARCADIA/processed_data", "protein", exact_step=1, dataset_name=dataset_name
)
adata_rna = sc.read(str(rna_file))
adata_prot = sc.read(str(protein_file))

# Make variable names unique immediately after loading to avoid warnings
adata_rna.var_names_make_unique()
adata_prot.var_names_make_unique()

# Check what's available in the loaded data
print(f"\nLoaded data structure:")
print(f"RNA obsm keys: {list(adata_rna.obsm.keys())}")
print(f"Protein obsm keys: {list(adata_prot.obsm.keys())}")
print(f"RNA obs columns sample: {list(adata_rna.obs.columns[:10])}")
print(f"Protein obs columns sample: {list(adata_prot.obs.columns[:10])}")

# Use raw counts from layers for maxfuse (it will do its own normalization)
if "counts" in adata_rna.layers:
    adata_rna.X = adata_rna.layers["counts"].copy()
    print("Using raw counts from adata_rna.layers['counts']")
if "counts" in adata_prot.layers:
    adata_prot.X = adata_prot.layers["counts"].copy()
    print("Using raw counts from adata_prot.layers['counts']")

# Check if spatial coordinates are already present (from ARCADIA preprocessing)
if "spatial" in adata_prot.obsm:
    print(f"Spatial coordinates found in obsm['spatial']: shape {adata_prot.obsm['spatial'].shape}")
elif "X" in adata_prot.obs.columns and "Y" in adata_prot.obs.columns:
    # Reconstruct spatial coordinates from obs columns if needed
    print("Reconstructing spatial coordinates from obs['X'] and obs['Y']")
    adata_prot.obsm["spatial"] = np.column_stack([adata_prot.obs["X"].values, adata_prot.obs["Y"].values])
else:
    print("Warning: No spatial coordinates found in protein data")

# %%
# Cell type labels are already in the obs from preprocessed data
# For compatibility with maxfuse tutorial, create celltype.l1 and celltype.l2
# cell_types is guaranteed to exist from ARCADIA preprocessing
adata_rna.obs["celltype.l1"] = adata_rna.obs["cell_types"]
adata_rna.obs["celltype"] = adata_rna.obs["cell_types"]
adata_rna.obs["celltype.l2"] = adata_rna.obs.get("minor_cell_types", adata_rna.obs["cell_types"])

adata_prot.obs["celltype.l1"] = adata_prot.obs["cell_types"]
adata_prot.obs["celltype"] = adata_prot.obs["cell_types"]
adata_prot.obs["celltype.l2"] = adata_prot.obs.get("minor_cell_types", adata_prot.obs["cell_types"])

# Extract labels for evaluation
labels_l1_rna = adata_rna.obs["celltype.l1"].to_numpy()
labels_l2_rna = adata_rna.obs["celltype.l2"].to_numpy()
labels_l1_prot = adata_prot.obs["celltype.l1"].to_numpy()
labels_l2_prot = adata_prot.obs["celltype.l2"].to_numpy()

print(f"RNA dataset: {adata_rna.shape[0]} cells")
print(f"Protein dataset: {adata_prot.shape[0]} cells")
print(f"RNA cell barcodes sample: {list(adata_rna.obs.index[:5])}")
print(f"Protein cell barcodes sample: {list(adata_prot.obs.index[:5])}")
print(f"RNA cell types: {sorted(set(labels_l1_rna))}")
print(f"Protein cell types: {sorted(set(labels_l1_prot))}")

# %%
# Load correspondence file (still from raw_datasets as it's a mapping file)
base_folder = "ARCADIA/raw_datasets/tonsil"
correspondence = pd.read_csv(f"{base_folder}/protein_gene_conversion.csv")
correspondence.head()

# %%
# Check protein variable names format
print(f"Sample protein var names: {list(adata_prot.var_names[:10])}")
print(f"Sample RNA var names: {list(adata_rna.var_names[:10])}")

# Create a mapping from clean protein names to actual var_names
# For tonsil CODEX, protein names are typically direct (not ADT_ prefixed)
protein_name_mapping = {}
for var_name in adata_prot.var_names:
    # Try to match directly first
    protein_name_mapping[var_name] = var_name
    # Also try without any prefixes if they exist
    clean_name = var_name.split("(")[0].strip()  # Handle cases like 'CD115(CSF-1R)'
    if clean_name != var_name:
        protein_name_mapping[clean_name] = var_name

print(f"Created mapping for {len(protein_name_mapping)} proteins")
print(f"Sample mappings: {list(protein_name_mapping.items())[:5]}")

rna_protein_correspondence = []

# Create a case-insensitive mapping for RNA gene names
rna_gene_mapping = {gene.upper(): gene for gene in adata_rna.var_names}

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]

    # Try to find the protein in our mapping
    actual_protein_name = protein_name_mapping.get(curr_protein_name)
    if actual_protein_name is None:
        continue

    if (
        curr_rna_names.find("Ignore") != -1
    ):  # some correspondence ignored eg. protein isoform to one gene
        continue
    curr_rna_names = curr_rna_names.split("/")  # eg. one protein to multiple genes
    for r in curr_rna_names:
        # Try to find the RNA gene (case-insensitive match)
        actual_rna_name = rna_gene_mapping.get(r.upper())
        if actual_rna_name is not None:
            rna_protein_correspondence.append([actual_rna_name, actual_protein_name])

if len(rna_protein_correspondence) == 0:
    raise ValueError("No RNA-protein correspondences found! Check protein variable names.")

rna_protein_correspondence = np.array(rna_protein_correspondence)
print(f"Found {len(rna_protein_correspondence)} RNA-protein correspondences")

# %%
# Note: For tonsil dataset, RNA and protein cells are from different samples
# (not matched by barcode like cite_seq), so we work with all cells from both datasets
print(f"RNA dataset: {adata_rna.shape[0]} cells")
print(f"Protein dataset: {adata_prot.shape[0]} cells")
print(f"RNA index sample: {list(adata_rna.obs.index[:5])}")
print(f"Protein index sample: {list(adata_prot.obs.index[:5])}")

# Extract labels for evaluation
labels_l1_rna = adata_rna.obs["celltype.l1"].to_numpy()
labels_l2_rna = adata_rna.obs["celltype.l2"].to_numpy()
labels_l1_prot = adata_prot.obs["celltype.l1"].to_numpy()
labels_l2_prot = adata_prot.obs["celltype.l2"].to_numpy()

# %%
# Columns rna_shared and protein_shared are matched.
# One may encounter "Variable names are not unique" warning,
# this is fine and is because one RNA may encode multiple proteins and vice versa.
rna_shared = adata_rna[:, rna_protein_correspondence[:, 0]].copy()
protein_shared = adata_prot[:, rna_protein_correspondence[:, 1]].copy()
# Use raw counts from layers if they exist (similar to lines 54-61)
if "counts" in rna_shared.layers:
    rna_shared.X = rna_shared.layers["counts"].copy()
if "counts" in protein_shared.layers:
    protein_shared.X = protein_shared.layers["counts"].copy()

# %%
# Make sure no column is static, only use protein features
# that are variable (larger than a certain threshold)
# mask = (rna_shared.X.std(axis=0) > 0.5) & (protein_shared.X.std(axis=0) > 0.1)
if issparse(rna_shared.X):
    rna_shared.X = rna_shared.X.toarray()
if issparse(protein_shared.X):
    protein_shared.X = protein_shared.X.toarray()
mask = (rna_shared.X.std(axis=0) > 0.3) & (protein_shared.X.std(axis=0) > 0.05)

rna_shared = rna_shared[:, mask].copy()
protein_shared = protein_shared[:, mask].copy()
print([rna_shared.shape, protein_shared.shape])

# %%
# process rna_shared
# Note: We don't filter cells here to avoid batch index misalignment issues
# MaxFuse will handle any zero-count cells internally
sc.pp.normalize_total(rna_shared)
sc.pp.log1p(rna_shared)
sc.pp.scale(rna_shared)

# %%

# plot UMAP of rna cells based only on rna markers with protein correspondence
sc.pp.pca(rna_shared)
sc.pp.neighbors(rna_shared, n_neighbors=15)
sc.tl.umap(rna_shared)
sc.pl.umap(rna_shared, color="celltype")

# %%
# # plot UMAPs of codex cells based only on protein markers with rna correspondence
# # due to a large number of codex cells, this can take a while. uncomment below to plot.
sc.pp.pca(protein_shared)
sc.pp.neighbors(protein_shared, n_neighbors=15)
sc.tl.umap(protein_shared)
sc.pl.umap(protein_shared, color="celltype")

# %%
# make sure no feature is static
# Process full adata_rna FIRST
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=5000)
adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()
sc.pp.scale(adata_rna)

# Extract active arrays
rna_active = adata_rna.X
protein_active = adata_prot.X
rna_active = rna_active[:, rna_active.std(axis=0) > 1e-5]
protein_active = protein_active[:, protein_active.std(axis=0) > 1e-5]
print(rna_active.shape, protein_active.shape, rna_shared.shape, protein_shared.shape)


# %%
# Convert shared arrays to numpy
rna_shared = rna_shared.X.copy()
protein_shared = protein_shared.X.copy()

# Convert to dense if sparse
if issparse(rna_shared):
    rna_shared = rna_shared.toarray()
if issparse(protein_shared):
    protein_shared = protein_shared.toarray()

# Check for and handle inf/NaN values in shared arrays
if np.any(~np.isfinite(rna_shared)):
    print(f"Warning: Found {np.sum(~np.isfinite(rna_shared))} inf/NaN values in rna_shared, replacing with 0")
    rna_shared = np.nan_to_num(rna_shared, nan=0.0, posinf=0.0, neginf=0.0)
if np.any(~np.isfinite(protein_shared)):
    print(f"Warning: Found {np.sum(~np.isfinite(protein_shared))} inf/NaN values in protein_shared, replacing with 0")
    protein_shared = np.nan_to_num(protein_shared, nan=0.0, posinf=0.0, neginf=0.0)

# Now create Fusor object
fusor = mf.model.Fusor(
    shared_arr1=rna_shared,
    shared_arr2=protein_shared,
    active_arr1=rna_active,
    active_arr2=protein_active,
    labels1=None,
    labels2=None,
)

# %%
fusor.split_into_batches(max_outward_size=8000, matching_ratio=4, metacell_size=2, verbose=True)

# %%
# plot top singular values of avtive_arr1 on a random batch
try:
    fusor.plot_singular_values(
        target="active_arr1", n_components=None  # can also explicitly specify the number of components
    )
except (IndexError, KeyError) as e:
    print(f"Warning: Could not plot singular values for active_arr1: {e}")
    print("Skipping visualization (this is non-critical)")

# %%
# plot top singular values of avtive_arr2 on a random batch
try:
    fusor.plot_singular_values(target="active_arr2", n_components=None)
except (IndexError, KeyError) as e:
    print(f"Warning: Could not plot singular values for active_arr2: {e}")
    print("Skipping visualization (this is non-critical)")

# %%
fusor.construct_graphs(
    n_neighbors1=15,
    n_neighbors2=15,
    svd_components1=40,
    svd_components2=15,
    resolution1=2,
    resolution2=2,
    # if two resolutions differ less than resolution_tol
    # then we do not distinguish between then
    resolution_tol=0.1,
    verbose=True,
)

# %%
# plot top singular values of shared_arr1 on a random batch
try:
    fusor.plot_singular_values(
        target="shared_arr1",
        n_components=None,
    )
except (IndexError, KeyError) as e:
    print(f"Warning: Could not plot singular values for shared_arr1: {e}")
    print("Skipping visualization (this is non-critical)")

# %%
# plot top singular values of shared_arr2 on a random batch
try:
    fusor.plot_singular_values(target="shared_arr2", n_components=None)
except (IndexError, KeyError) as e:
    print(f"Warning: Could not plot singular values for shared_arr2: {e}")
    print("Skipping visualization (this is non-critical)")

# %%
fusor.find_initial_pivots(wt1=0.3, wt2=0.3, svd_components1=25, svd_components2=20)

# %%
fusor.plot_canonical_correlations(svd_components1=50, svd_components2=None, cca_components=45)

# %%
fusor.refine_pivots(
    wt1=0.5,
    wt2=0.5,
    svd_components1=40,
    svd_components2=None,
    cca_components=25,
    n_iters=1,
    randomized_svd=False,
    svd_runs=1,
    verbose=True,
)

# %%
fusor.filter_bad_matches(target="pivot", filter_prop=0.5)


# %%
pivot_matching = fusor.get_matching(order=(2, 1), target="pivot")
lv1_acc = mf.metrics.get_matching_acc(
    matching=pivot_matching, labels1=labels_l1_rna, labels2=labels_l1_prot, order=(2, 1)
)
lv2_acc = mf.metrics.get_matching_acc(
    matching=pivot_matching, labels1=labels_l2_rna, labels2=labels_l2_prot, order=(2, 1)
)
print(f"lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.")

# %%
# We can inspect the first pivot pair.
[pivot_matching[0][0], pivot_matching[1][0], pivot_matching[2][0]]

# %%
cm = confusion_matrix(
    labels_l1_rna[pivot_matching[0]], labels_l1_prot[pivot_matching[1]]
)
ConfusionMatrixDisplay(
    confusion_matrix=np.round((cm.T / np.sum(cm, axis=1)).T * 100),
    display_labels=np.unique(labels_l1_rna),
).plot()

# %%
fusor.propagate(
    svd_components1=40,
    svd_components2=None,
    wt1=0.7,
    wt2=0.7,
)

# %%
fusor.filter_bad_matches(target="propagated", filter_prop=0.3)

# %%
full_matching = fusor.get_matching(order=(2, 1), target="full_data")

# %%
pd.DataFrame(
    list(zip(full_matching[0], full_matching[1], full_matching[2])),
    columns=["mod1_indx", "mod2_indx", "score"],
)

# %%
# compute the cell type level matching accuracy, for the full (filtered version) dataset
lv1_acc = mf.metrics.get_matching_acc(
    matching=full_matching, labels1=labels_l1_rna, labels2=labels_l1_prot
)
lv2_acc = mf.metrics.get_matching_acc(
    matching=full_matching, labels1=labels_l2_rna, labels2=labels_l2_prot
)
print(f"lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.")

# %%
rna_cca, protein_cca_sub = fusor.get_embedding(
    active_arr1=fusor.active_arr1,
    active_arr2=fusor.active_arr2[full_matching[1], :],  # cells in codex remained after filtering
)

# %%
np.random.seed(42)
subs = min(13000, protein_cca_sub.shape[0], adata_rna.shape[0])
randix = np.random.choice(protein_cca_sub.shape[0], subs, replace=False)

dim_use = 15  # dimensions of the CCA embedding to be used for UMAP etc

cca_adata = ad.AnnData(
    np.concatenate((rna_cca[:, :dim_use], protein_cca_sub[randix, :dim_use]), axis=0),
    dtype=np.float32,
)
cca_adata.obs["data_type"] = ["rna"] * rna_cca.shape[0] + ["protein"] * subs
cca_adata.obs["celltype.l1"] = list(labels_l1_rna) + list(
    labels_l1_prot[full_matching[1]][randix]
)
cca_adata.obs["celltype.l2"] = list(labels_l2_rna) + list(
    labels_l2_prot[full_matching[1]][randix]
)
cca_adata.obs["cell_type"] = cca_adata.obs["celltype.l1"]

# %%
sc.pp.neighbors(cca_adata, n_neighbors=15)
sc.tl.umap(cca_adata)
sc.pl.umap(cca_adata, color="data_type")

# %%
sc.pl.umap(cca_adata, color="cell_type")

# %%
# Get full embedding for all cells (not just the subsampled ones used for visualization)
rna_cca_full, codex_cca_full = fusor.get_embedding(
    active_arr1=fusor.active_arr1, active_arr2=fusor.active_arr2
)

# %%
# Prepare RNA AnnData object
# obs fields - preserve existing fields from ARCADIA preprocessing where possible
# obs fields - only set fields needed for maxfuse compatibility that don't already exist
# cell_types, major_cell_types, total_counts, n_genes, n_genes_by_counts, pct_counts_mt are already set from preprocessing
if "batch_indices" not in adata_rna.obs.columns:
    adata_rna.obs["batch_indices"] = 0
if "percent_mito" not in adata_rna.obs.columns:
    adata_rna.obs["percent_mito"] = 0
if "leiden_subclusters" not in adata_rna.obs.columns:
    adata_rna.obs["leiden_subclusters"] = "unknown"
# Update batch name for maxfuse output
adata_rna.obs["batch"] = "maxfuse_tonsil"
# Set minor_cell_types if not already present
if "minor_cell_types" not in adata_rna.obs.columns:
    adata_rna.obs["minor_cell_types"] = adata_rna.obs["celltype.l2"]
adata_rna.obs["tissue"] = "tonsil"
adata_rna.obs["index_col"] = np.arange(adata_rna.n_obs)

# var fields - preserve existing fields from ARCADIA preprocessing
# Most var fields (n_cells, mt, ribo, hb, total_counts, n_cells_by_counts) are already set during preprocessing

# uns fields
adata_rna.uns["dataset_name"] = "maxfuse_tonsil"
adata_rna.uns["processing_stage"] = "maxfuse_integrated"
adata_rna.uns["file_generated_from"] = "model_maxfuse_dataset_tonsil.py"

# obsm fields - clear and only keep what we need
adata_rna.obsm.clear()
adata_rna.obsm["latent"] = rna_cca_full

# layers
adata_rna.layers["counts"] = adata_rna.X.copy()

print(f"adata_rna shape: {adata_rna.shape}")
print(f"adata_rna.obs columns: {list(adata_rna.obs.columns)}")

# %%
# Prepare Protein AnnData object with spatial coordinates
# obs fields - only set fields needed for maxfuse compatibility that don't already exist
# cell_types, major_cell_types, total_counts, X, Y, condition, Image, Sample, n_genes_by_counts are already set
if "batch_indices" not in adata_prot.obs.columns:
    adata_prot.obs["batch_indices"] = 0
if "percent_mito" not in adata_prot.obs.columns:
    adata_prot.obs["percent_mito"] = 0
if "leiden_subclusters" not in adata_prot.obs.columns:
    adata_prot.obs["leiden_subclusters"] = "unknown"
# Update batch name for maxfuse output
adata_prot.obs["batch"] = "maxfuse_tonsil"
# Set minor_cell_types if not already present
if "minor_cell_types" not in adata_prot.obs.columns:
    adata_prot.obs["minor_cell_types"] = adata_prot.obs["celltype.l2"]
adata_prot.obs["tissue"] = "tonsil"
# total_counts, n_genes_by_counts, log1p fields, X, Y, condition, Image, Sample are already set from preprocessing
if "outlier" not in adata_prot.obs.columns:
    adata_prot.obs["outlier"] = False
if "CN" not in adata_prot.obs.columns:
    adata_prot.obs["CN"] = "CN_unknown"
adata_prot.obs["index_col"] = np.arange(adata_prot.n_obs)

# var fields - only set if not already present
if "feature_type" not in adata_prot.var.columns:
    adata_prot.var["feature_type"] = "protein"
# uns fields
adata_prot.uns["dataset_name"] = "maxfuse_tonsil"
adata_prot.uns["processing_stage"] = "maxfuse_integrated"
adata_prot.uns["file_generated_from"] = "model_maxfuse_dataset_tonsil.py"
# obsm fields - clear and add latent, preserve spatial coordinates
spatial_coords = adata_prot.obsm.get("spatial", None)
adata_prot.obsm.clear()
adata_prot.obsm["latent"] = codex_cca_full
if spatial_coords is not None:
    adata_prot.obsm["spatial"] = spatial_coords
    print("Preserved spatial coordinates in obsm['spatial']")

print(f"adata_prot shape: {adata_prot.shape}")
print(f"adata_prot.obs columns: {list(adata_prot.obs.columns)}")

# %%
# plot spatial coordinates with sc.pl.embedding with celltype as color (if spatial coordinates available)
if "spatial" in adata_prot.obsm:
    sc.pl.embedding(adata_prot, "spatial", color="cell_types")

# %%
# Save the formatted AnnData objects


output_dir = "model_comparison/outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rna_output = f"{output_dir}/maxfuse_tonsil/7_rna_{timestamp}.h5ad"
prot_output = f"{output_dir}/maxfuse_tonsil/7_protein_{timestamp}.h5ad"

adata_rna.write(rna_output)
adata_prot.write(prot_output)

print(f"Saved adata_rna to: {rna_output}")
print(f"Saved adata_prot to: {prot_output}")
print(f"\nadata_rna: {adata_rna}")
print(f"\nadata_prot: {adata_prot}")

# %%
# Display summary of created objects
print("=" * 80)
print("RNA AnnData Summary:")
print("=" * 80)
print(f"Shape: {adata_rna.shape}")
print(f"\nobs fields ({len(adata_rna.obs.columns)}):")
print(list(adata_rna.obs.columns))
print(f"\nvar fields ({len(adata_rna.var.columns)}):")
print(list(adata_rna.var.columns))
print(f"\nuns fields ({len(adata_rna.uns.keys())}):")
print(list(adata_rna.uns.keys()))
print(f"\nobsm fields ({len(adata_rna.obsm.keys())}):")
print(list(adata_rna.obsm.keys()))
print(f"\nlayers ({len(adata_rna.layers.keys())}):")
print(list(adata_rna.layers.keys()))
print(f"\nobsp fields ({len(adata_rna.obsp.keys()) if adata_rna.obsp else 0}):")
print(list(adata_rna.obsp.keys()) if adata_rna.obsp else [])

print("\n" + "=" * 80)
print("Protein AnnData Summary:")
print("=" * 80)
print(f"Shape: {adata_prot.shape}")
print(f"\nobs fields ({len(adata_prot.obs.columns)}):")
print(list(adata_prot.obs.columns))
print(f"\nvar fields ({len(adata_prot.var.columns)}):")
print(list(adata_prot.var.columns))
print(f"\nuns fields ({len(adata_prot.uns.keys())}):")
print(list(adata_prot.uns.keys()))
print(f"\nobsm fields ({len(adata_prot.obsm.keys())}):")
print(list(adata_prot.obsm.keys()))
print(f"\nlayers ({len(adata_prot.layers.keys())}):")
print(list(adata_prot.layers.keys()))
print(f"\nobsp fields ({len(adata_prot.obsp.keys()) if adata_prot.obsp else 0}):")
print(list(adata_prot.obsp.keys()) if adata_prot.obsp else [])

# %%
