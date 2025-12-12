# %% [markdown]
# # Example of MaxFuse usage between RNA and Protein modality.

# %% [markdown]
# In this tutorial, we demonstrate the application of MaxFuse integration and matching across weak-linked modalities. Here we showcase an example between RNA and Protein modality. For testing reason, we uses a CITE-seq pbmc data with 228 antibodies from Hao et al. (2021), and we use the Protein and RNA information but __disregard the fact they are multiome data__.

import os
import sys
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import maxfuse as mf
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from comparison_utils import get_latest_file, here
import sys

plt.rcParams["figure.figsize"] = (6, 4)

if here().parent.name == "notebooks":
    os.chdir("../../")

ROOT = here().parent
THIS_DIR = here()
print(f"ROOT: {ROOT}")
print(f"THIS_DIR: {THIS_DIR}")
# Update sys.path and cwd
sys.path.append(str(ROOT))
sys.path.append(str(THIS_DIR))
os.chdir(str(ROOT))
print(f"Working directory: {os.getcwd()}")


# %% [markdown]
# ## Data acquire

# %% [markdown]
# Since the example data we are uisng in the tutorial excedes the size limit for github repository files, we have uploaded them onto a server and can be easily donwloaded with the code below. Also this code only need to run **once** for both of the tutorial examples.

# %%
# import requests, zipfile, io
# r = requests.get("http://stat.wharton.upenn.edu/~zongming/maxfuse/data.zip")
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall("../")

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# We begin by reading in protein measurements and RNA measurements.
#
# Note that the two modalities in this example have *matching rows* since CITE-Seq measures proteins and RNAs simultaneously.
# But we will ignore this fact and treat the two modalities as if they are measured separately.
#
# The file format for MaxFuse to read in is ```adata```. In this tutorial we read in the original RNA counts or Protein counts where each row is a cell and each column is a feature, then turn them into ```adata``` objects.

# %%
# Load the CITE-seq spleen lymph node data directly from h5ad files (matching preprocessing pipeline)
dataset_name = "cite_seq"
print("Loading CITE-seq spleen lymph node data from h5ad files...")

rna_file = get_latest_file(
    "ARCADIA/processed_data", "rna", exact_step=1, dataset_name=dataset_name
)
protein_file = get_latest_file(
    "ARCADIA/processed_data", "protein", exact_step=1, dataset_name=dataset_name
)
rna_adata = sc.read(str(rna_file))
protein_adata = sc.read(str(protein_file))

# %% [markdown]
# **Optional**: meta data for the cells. In this case we are using them to **evaluate the integration results**, but for actual running, MaxFuse does not require you have this information.

# %%
# Cell type labels are already in the obs from preprocessed data
# For compatibility with maxfuse tutorial, create celltype.l1 and celltype.l2
# Use cell_types as the main label (l1), and minor_cell_types as l2
rna_adata.obs["celltype.l1"] = rna_adata.obs["cell_types"]
rna_adata.obs["celltype.l2"] = rna_adata.obs["minor_cell_types"]

protein_adata.obs["celltype.l1"] = protein_adata.obs["cell_types"]
protein_adata.obs["celltype.l2"] = protein_adata.obs["minor_cell_types"]

# Extract labels for evaluation
labels_l1_rna = rna_adata.obs["celltype.l1"].to_numpy()
labels_l2_rna = rna_adata.obs["celltype.l2"].to_numpy()
labels_l1_prot = protein_adata.obs["celltype.l1"].to_numpy()
labels_l2_prot = protein_adata.obs["celltype.l2"].to_numpy()

print(f"RNA dataset: {rna_adata.shape[0]} cells")
print(f"Protein dataset: {protein_adata.shape[0]} cells")
print(f"RNA cell barcodes sample: {list(rna_adata.obs.index[:5])}")
print(f"Protein cell barcodes sample: {list(protein_adata.obs.index[:5])}")
print(f"RNA cell types: {sorted(set(labels_l1_rna))}")
print(f"Protein cell types: {sorted(set(labels_l1_prot))}")

# %% [markdown]
# Here we are integrating protein and RNA data, and most of the time there are name differences between protein (antibody) and their corresponding gene names.
#
# These "weak linked" features will be used during initialization (we construct two arrays, `rna_shared` and `protein_shared`, whose columns are matched, and the two arrays can be used to obtain the initial matching).
#
# To construct the feature correspondence in straight forward way, we prepared a ```.csv``` file containing most of the antibody name (seen in cite-seq or codex etc) and their corresponding gene names:
#

# %%
# Use the protein_gene_conversion from the tonsil dataset (it's a general mapping file)
data_dir = "ARCADIA/raw_datasets"
correspondence = pd.read_csv(f"{data_dir}/tonsil/protein_gene_conversion.csv")
correspondence.head()

# %% [markdown]
# But of course this files does contain all names including custom names in new assays. If a certain correspondence is not found, either it is missing in the other modality, or you should customly add the name conversion to this ```.csv``` file.

# %%
# Check protein variable names format
print(f"Sample protein var names: {list(protein_adata.var_names[:10])}")
print(f"Sample RNA var names: {list(rna_adata.var_names[:10])}")

# Create a mapping from clean protein names to actual var_names
# Protein names are in format 'ADT_CD102_A0104', we need to extract 'CD102'
protein_name_mapping = {}
for var_name in protein_adata.var_names:
    if var_name.startswith("ADT_"):
        # Extract the middle part (e.g., 'CD102' from 'ADT_CD102_A0104')
        parts = var_name.split("_")
        if len(parts) >= 2:
            clean_name = parts[1].split("(")[0]  # Handle cases like 'CD115(CSF-1R)'
            protein_name_mapping[clean_name] = var_name

print(f"Created mapping for {len(protein_name_mapping)} proteins")
print(f"Sample mappings: {list(protein_name_mapping.items())[:5]}")

rna_protein_correspondence = []

# Create a case-insensitive mapping for RNA gene names (human uppercase -> mouse titlecase)
rna_gene_mapping = {gene.upper(): gene for gene in rna_adata.var_names}

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
        # Try to find the RNA gene (case-insensitive match for human->mouse conversion)
        actual_rna_name = rna_gene_mapping.get(r.upper())
        if actual_rna_name is not None:
            rna_protein_correspondence.append([actual_rna_name, actual_protein_name])

if len(rna_protein_correspondence) == 0:
    raise ValueError("No RNA-protein correspondences found! Check protein variable names.")

rna_protein_correspondence = np.array(rna_protein_correspondence)
print(f"Found {len(rna_protein_correspondence)} RNA-protein correspondences")

# %%
# Find common cells between RNA and protein datasets
common_cells = rna_adata.obs.index.intersection(protein_adata.obs.index)
print(f"Common cells between RNA and protein: {len(common_cells)}")
print(f"RNA unique cells: {len(rna_adata.obs.index) - len(common_cells)}")
print(f"Protein unique cells: {len(protein_adata.obs.index) - len(common_cells)}")

# Subset both datasets to common cells
rna_adata = rna_adata[common_cells].copy()
protein_adata = protein_adata[common_cells].copy()

print(f"After filtering to common cells:")
print(f"RNA dataset: {rna_adata.shape}")
print(f"Protein dataset: {protein_adata.shape}")

# Now subsample if needed (using the same indices for both)
num_cells = 300000000
max_cells = min(num_cells, len(common_cells))
if len(common_cells) > max_cells:
    np.random.seed(42)
    subsample_indices = np.random.choice(len(common_cells), max_cells, replace=False)
    subsample_cells = common_cells[subsample_indices]
    rna_adata = rna_adata[subsample_cells].copy()
    protein_adata = protein_adata[subsample_cells].copy()
    print(f"Subsampled to {max_cells} cells")

# Update labels after subsampling (extract from the subsampled obs)
labels_l1_rna = rna_adata.obs["celltype.l1"].to_numpy()
labels_l2_rna = rna_adata.obs["celltype.l2"].to_numpy()
labels_l1_prot = protein_adata.obs["celltype.l1"].to_numpy()
labels_l2_prot = protein_adata.obs["celltype.l2"].to_numpy()

# Verify that RNA and protein datasets now have matching cells
print(f"Final RNA dataset: {rna_adata.shape[0]} cells")
print(f"Final Protein dataset: {protein_adata.shape[0]} cells")
print(f"RNA index sample: {list(rna_adata.obs.index[:5])}")
print(f"Protein index sample: {list(protein_adata.obs.index[:5])}")
print(f"Indices match: {all(rna_adata.obs.index == protein_adata.obs.index)}")

# %%
# Columns rna_shared and protein_shared are matched.
# One may encounter "Variable names are not unique" warning,
# this is fine and is because one RNA may encode multiple proteins and vice versa.
rna_shared = rna_adata[:, rna_protein_correspondence[:, 0]].copy()
protein_shared = protein_adata[:, rna_protein_correspondence[:, 1]].copy()

# Use raw counts from layers for maxfuse (it will do its own normalization)
if "counts" in rna_shared.layers:
    rna_shared.X = rna_shared.layers["counts"].copy()
    print("Using raw counts from rna_shared.layers['counts']")
if "counts" in protein_shared.layers:
    protein_shared.X = protein_shared.layers["counts"].copy()
    print("Using raw counts from protein_shared.layers['counts']")

# %%
# Make sure no column is static
# Handle both sparse and dense matrices
rna_X = rna_shared.X.toarray() if issparse(rna_shared.X) else rna_shared.X
protein_X = protein_shared.X.toarray() if issparse(protein_shared.X) else protein_shared.X
mask = (rna_X.std(axis=0) > 1e-5) & (protein_X.std(axis=0) > 1e-5)
rna_shared = rna_shared[:, mask].copy()
protein_shared = protein_shared[:, mask].copy()

# %% [markdown]
# We apply standard Scanpy preprocessing steps to `rna_shared` and `protein_shared`.
# One modification we do is that we normalize the rows of the two arrays to be a common `target_sum`.
# If the input data is already pre-processed (normalized etc), we suggest skipping the standardized processing steps below.

# %%
# row sum for RNA
rna_X_for_sum = rna_shared.X.toarray() if issparse(rna_shared.X) else rna_shared.X
rna_counts = rna_X_for_sum.sum(axis=1)
# row sum for protein
protein_X_for_sum = protein_shared.X.toarray() if issparse(protein_shared.X) else protein_shared.X
protein_counts = protein_X_for_sum.sum(axis=1)
# take median of each and then take mean
target_sum = (np.median(rna_counts.copy()) + np.median(protein_counts.copy())) / 2

# %%
# process rna_shared
sc.pp.normalize_total(rna_shared, target_sum=target_sum)
sc.pp.log1p(rna_shared)
sc.pp.scale(rna_shared)


# %%
# plot UMAPs of rna cells based only on rna markers with protein correspondence
sc.pp.pca(rna_shared)
sc.pp.neighbors(rna_shared, n_neighbors=15)
sc.tl.umap(rna_shared)
sc.pl.umap(rna_shared, color=["celltype.l1", "celltype.l2"])

# %%
rna_shared = rna_shared.X.copy()

# %%
# process protein_shared
sc.pp.normalize_total(protein_shared, target_sum=target_sum)
sc.pp.log1p(protein_shared)
sc.pp.scale(protein_shared)

# Handle any NaN values that might have been introduced
if np.isnan(protein_shared.X).any():
    print(f"Warning: Found NaN values in protein_shared after normalization, replacing with 0")
    protein_shared.X = np.nan_to_num(protein_shared.X, nan=0.0)


# %%
# plot UMAPs of protein cells based only on protein markers with rna correspondence
sc.pp.pca(protein_shared)
sc.pp.neighbors(protein_shared, n_neighbors=15)
# sc.tl.umap(protein_shared)
# sc.pl.umap(protein_shared, color=['celltype.l1','celltype.l2'])

# %%
protein_shared = protein_shared.X.copy()

# %% [markdown]
# We again apply standard Scanpy preprocessing steps to **all available RNA measurements and protein measurements** (not just the shared ones) to get two arrays, `rna_active` and `protein_active`, which are used for iterative refinement. Again if the input data is already processed, these steps can be skipped.

# %%
# process all RNA features
# Use raw counts for maxfuse processing
if "counts" in rna_adata.layers:
    rna_adata.X = rna_adata.layers["counts"].copy()
    print("Using raw counts from rna_adata.layers['counts']")

sc.pp.normalize_total(rna_adata)
sc.pp.log1p(rna_adata)
sc.pp.highly_variable_genes(rna_adata)
# only retain highly variable genes
rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()
sc.pp.scale(rna_adata)

# %%
# plot UMAPs of rna cells based on all active rna markers

sc.pp.neighbors(rna_adata, n_neighbors=15)
# sc.tl.umap(rna_adata)
# sc.pl.umap(rna_adata, color=['celltype.l1','celltype.l2'])

# %%
# process all protein features
# Use raw counts for maxfuse processing
if "counts" in protein_adata.layers:
    protein_adata.X = protein_adata.layers["counts"].copy()
    print("Using raw counts from protein_adata.layers['counts']")

sc.pp.normalize_total(protein_adata)
sc.pp.log1p(protein_adata)
sc.pp.scale(protein_adata)

# Handle any NaN values in protein_adata
if np.isnan(protein_adata.X).any():
    print(f"Warning: Found NaN values in protein_adata after normalization, replacing with 0")
    protein_adata.X = np.nan_to_num(protein_adata.X, nan=0.0)

# %%
# plot UMAPs of protein cells based on all active protein markers

sc.pp.neighbors(protein_adata, n_neighbors=15)
sc.tl.umap(protein_adata)
sc.pl.umap(protein_adata, color=["celltype.l1", "celltype.l2"])

# %%
# make sure no feature is static
rna_active = rna_adata.X
protein_active = protein_adata.X
rna_active = rna_active[:, rna_active.std(axis=0) > 1e-5]
protein_active = protein_active[:, protein_active.std(axis=0) > 1e-5]

# %%
# inspect shape of the four matrices
print(rna_active.shape)
print(protein_active.shape)
print(rna_shared.shape)
print(protein_shared.shape)

# %% [markdown]
# ## Fitting MaxFuse

# %% [markdown]
# ### Step I: preparations

# %% [markdown]
# We now have four arrays. `rna_shared` and `protein_shared` are used for finding initial pivots, whereas `rna_active` and `protein_active` are used for iterative refinement.
#
# The main object for running MaxFuse pipeline is `mf.model.Fusor`, and its constructor takes the above four arrays as input.
#
# If your data have not been clustered and annotated, you can leave `labels1` and `labels2` to be `None`, then MaxFuse will automatically run clustering algorithms to fill them in.
#
# **Optional**: If your data have already been clustered (and you trust your annotation is optimal and should be used to guide the MaxFuse smoothing steps), you could supply them as ```numpy``` arrays to `labels1` and `labels2`.

# %%
# call constructor for Fusor object
# which is the main object for running MaxFuse pipeline
fusor = mf.model.Fusor(
    shared_arr1=rna_shared,
    shared_arr2=protein_shared,
    active_arr1=rna_active,
    active_arr2=protein_active,
    labels1=None,
    labels2=None,
)

# %% [markdown]
# To reduce computational complexity, we call `split_into_batches` to fit the batched version of MaxFuse.
#
# Internally, MaxFuse will solve a few linear assignment problems of size $n_1 \times n_2$, where $n_1$ and $n_2$ (with $n_1\leq n_2$ by convention) are the sample sizes of the two modalities (after batching and metacell construction).
# `max_outward_size` specifis the maximum value of $n_1$.
#
# `matching_ratio` specifies approximately the ratio of $n_2/n_1$.
# The larger it is, the more candidate cells in the second modality MaxFuse will seek for to match each cell/metacell in the first modality.
#
# `metacell_size` specifies the average size of the metacells in the first modality.

# %%
fusor.split_into_batches(max_outward_size=5000, matching_ratio=3, metacell_size=2, verbose=True)

# %% [markdown]
# The next step is to construct appropriate nearest-neighbor graphs for each modality with all features available.
# But before that, we plot the singular values of the two active arrays to determine how many principal components (PCs) to keep when doing graph construction.

# %%
# plot top singular values of avtive_arr1 on a random batch
fusor.plot_singular_values(
    target="active_arr1", n_components=None  # can also explicitly specify the number of components
)

# %%
# plot top singular values of avtive_arr2 on a random batch
fusor.plot_singular_values(target="active_arr2", n_components=None)

# %% [markdown]
# Inspecting the "elbows", we choose the number of PCs to be **30** for both RNA and protein active data.
# We then call `construct_graphs` to compute nearest-neighbor graphs as needed.

# %%
fusor.construct_graphs(
    n_neighbors1=15,
    n_neighbors2=15,
    svd_components1=30,
    svd_components2=30,
    resolution1=2,
    resolution2=2,
    # if two resolutions differ less than resolution_tol
    # then we do not distinguish between then
    resolution_tol=0.1,
    verbose=True,
)

# %% [markdown]
# ### Step II: finding initial pivots

# %% [markdown]
# We then use shared arrays whose columns are matched to find initial pivots.
# Before we do so, we plot top singular values of two shared arrays to determine how many PCs to use.

# %%
# plot top singular values of shared_arr1 on a random batch
fusor.plot_singular_values(
    target="shared_arr1",
    n_components=None,
)

# %%
# plot top singular values of shared_arr2 on a random batch
fusor.plot_singular_values(target="shared_arr2", n_components=None)

# %% [markdown]
# We choose to use **25** PCs for ``rna_shared`` and **20** PCs for ``protein_shared``.
#
# We then call ``find_initial_pivots`` to compute initial set of matched pairs.
# In this function, ``wt1`` (resp. ``wt2``) is a number between zero and one that specifies the weight on the smoothing target for the first (resp. second) modality.
# The smaller it is, the greater the strength of fuzzy smoothing.
# When the weight is one, then there is no smoothing at all, meaning the original data will be used.

# %%
fusor.find_initial_pivots(wt1=0.7, wt2=0.7, svd_components1=25, svd_components2=20)

# %% [markdown]
# Now, we have a set of *initial pivots* that store the matched pairs when only the information in the shared arrays is used. The information on initial pivots are stored in the internal field ``fusor._init_matching`` that is invisible to users.

# %% [markdown]
# ### Step III: finding refined pivots

# %% [markdown]
# We now use the information in active arrays to iteratively refine initial pivots. Recall we chose the number of PCs for the active arrays to be **30**.
# We plot the top canonical correlations to choose the best number of components in canonical correlation analysis (CCA).

# %%
# plot top canonical correlations in a random batch
fusor.plot_canonical_correlations(svd_components1=30, svd_components2=30, cca_components=30)

# %% [markdown]
# From the "elblow" above, we choose retain top **20** canonical scores.
#
# We then call `refine_pivots` to get the refined pivots.
# Here, `wt1` and `wt2` admit their usual interpretation of controling the strength of smoothing.
# `n_iters` specifies the number of iterations, which we choose to be **3**.
# We recommend setting `n_iters` to be *less than 5*, as higher iterations will be slower and may make the algorithm diverge when the signal-to-noise ratio is ultra low.

# %%
fusor.refine_pivots(
    wt1=0.7,
    wt2=0.7,
    svd_components1=30,
    svd_components2=30,
    cca_components=20,
    n_iters=3,
    randomized_svd=False,
    svd_runs=1,
    verbose=True,
)

# %% [markdown]
# The function `filter_bad_matches` filters away unreliable pivots and is helpful for the propagation step.
# `filter_prop` specifies approximately the proportion of pivots to be filtered away.

# %%
fusor.filter_bad_matches(target="pivot", filter_prop=0.3)

# %% [markdown]
# We can extract the matched pairs in refined pivots by calling `get_matching` function.
# The resulting `pivot_matching` is a nested list.
# `pivot_matching[0][i]` and `pivot_matching[1][i]` constitute the matched pair from the first and the second modality;
# `pivot_matching[2][i]` is a *quality score* (between zero and one) assigned to this matched pair.

# %%
pivot_matching = fusor.get_matching(target="pivot")

# %%
# We can inspect the first pivot pair.
[pivot_matching[0][0], pivot_matching[1][0], pivot_matching[2][0]]

# %% [markdown]
# We now compute the cell type level accuracy to evaluate the performance. (This step is not required for actual MaxFuse running)

# %%
lv1_acc = mf.metrics.get_matching_acc(
    matching=pivot_matching, labels1=labels_l1_rna, labels2=labels_l1_rna
)
lv2_acc = mf.metrics.get_matching_acc(
    matching=pivot_matching, labels1=labels_l2_rna, labels2=labels_l2_rna
)
print(f"lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.")

# %% [markdown]
# We can also compute the confusion matrix to see where the pivot matching goes wrong.

# %%
cm = confusion_matrix(labels_l1_rna[pivot_matching[0]], labels_l1_rna[pivot_matching[1]])
ConfusionMatrixDisplay(
    confusion_matrix=np.round((cm.T / np.sum(cm, axis=1)).T * 100),
    display_labels=np.unique(labels_l1_rna),
).plot()

# %% [markdown]
# As long as the refined pivots have been obtained, we can get joint embedding of the *full* datasets (active arrays).

# %%
rna_cca, protein_cca = fusor.get_embedding(
    active_arr1=fusor.active_arr1, active_arr2=fusor.active_arr2
)

# %% [markdown]
# Since we know the ground truth matching (it is the identity matching as we manually cut CITE-Seq data into halves), we can compute *fraction of samples closer than the true match* (FOSCTTM). The smaller this metric is, the better the joint embeddings.
# We refer the readers to our paper for more metrics.

# %%
dim_use = 15  # dimensions of the CCA embedding to be used for UMAP etc

mf.metrics.get_foscttm(
    dist=mf.utils.cdist_correlation(rna_cca[:, :dim_use], protein_cca[:, :dim_use]),
    true_matching="identity",
)

# %% [markdown]
# We can also plot the UMAP visualizations of the joint embeddings and we can see that: (1) the two datasets mix well; and (2) the cell types are preseved.
#
# Empirically, we find *10-20* dimensions of the joint embeddings best represents the data, similar to choosing PCA components to plot UMAPs in the conventional pipelines.

# %%
cca_adata = ad.AnnData(
    np.concatenate((rna_cca[:, :dim_use], protein_cca[:, :dim_use]), axis=0), dtype=np.float32
)
cca_adata.obs["data_type"] = ["rna"] * rna_cca.shape[0] + ["protein"] * protein_cca.shape[0]
cca_adata.obs["celltype.l1"] = list(protein_adata.obs["celltype.l1"]) * 2
cca_adata.obs["celltype.l2"] = list(protein_adata.obs["celltype.l2"]) * 2

# %%
cca_adata.obs["celltype.l1"]

# %%
sc.pp.neighbors(cca_adata, n_neighbors=15)
sc.tl.umap(cca_adata)
sc.pl.umap(cca_adata, color="data_type")

# %%
sc.pl.umap(cca_adata, color=["celltype.l1", "celltype.l2"])

# %% [markdown]
# ### Step IV: propagation

# %% [markdown]
# Refined pivots can only give us a pivot matching that captures a subset of cells. In order to get a *full* matching that involves all cells during input, we need to call `propagate`.
#
# Propagation uses active arrays, so we set the SVD components to be **30**.

# %%
fusor.propagate(
    svd_components1=30,
    svd_components2=30,
    wt1=0.7,
    wt2=0.7,
)

# %% [markdown]
# We call `filter_bad_matches` with `target=propagated` to optionally filter away a few matched pairs from propagation.
# Here, we want a full matching, so we do not do any filtering and set `filter_prop=0`. But in other cases where you believe some proportion of cells from the original input can be removed, you could increase this value.

# %%
fusor.filter_bad_matches(target="propagated", filter_prop=0)

# %% [markdown]
# We use `get_matching` with `target='full_data'` to extract the full matching.
#
# Because of the batching operation, the resulting matching may contain duplicates. The `order` argument determines how those duplicates are dealt with.
# `order=None` means doing nothing and returning the matching with potential duplicates;
# `order=(1, 2)` means returning a matching where each cell in the first modality contains *at least one match* in the second modality;
# `order=(2, 1)` means returning a matching where each cell in the second modality contains *at least one match* in the first modality.

# %%
full_matching = fusor.get_matching(order=(2, 1), target="full_data")

# %% [markdown]
# Since we are doing `order=(2, 1)` here, the matching info is all the cells (10k) in mod 2 (protein) has at least one match cell in the RNA modality. Note that the matched cell in RNA could be duplicated, as different protein cells could be matched to the same RNA cell. For a quick check on matching format:

# %%
pd.DataFrame(
    list(zip(full_matching[0], full_matching[1], full_matching[2])),
    columns=["mod1_indx", "mod2_indx", "score"],
)
# columns: cell idx in mod1, cell idx in mod2, and matching scores

# %%
# compute the cell type level matching accuracy
lv1_acc = mf.metrics.get_matching_acc(
    matching=full_matching, labels1=labels_l1_rna, labels2=labels_l1_rna
)
lv2_acc = mf.metrics.get_matching_acc(
    matching=full_matching, labels1=labels_l2_rna, labels2=labels_l2_rna
)
print(f"lv1 matching acc: {lv1_acc:.3f},\nlv2 matching acc: {lv2_acc:.3f}.")

# %%
# confusion matrix for full matching
cm = confusion_matrix(labels_l1_rna[full_matching[0]], labels_l1_rna[full_matching[1]])
ConfusionMatrixDisplay(
    confusion_matrix=np.round((cm.T / np.sum(cm, axis=1)).T * 100),
    display_labels=np.unique(labels_l1_rna),
).plot()

# %%
# what is full matching?

cca_adata.X

# %%


# %%


# %%


# %%
adata_1 = ad.AnnData(rna_cca)
adata_2 = ad.AnnData(protein_cca)

# %%
adata_1.obs["cell_types"] = protein_adata.obs["celltype.l1"]

# %%
adata_2.obs["cell_types"] = protein_adata.obs["celltype.l2"]

# %%


# %%
cca_adata.obs["data_type"]

# %%
cca_adata.obs["cell_types"] = cca_adata.obs["celltype.l1"]
cca_adata.obs["modality"] = cca_adata.obs["data_type"]

# %%


# %%
cca_adata.obs["cell_types"] = cca_adata.obs["celltype.l2"]
cca_adata.obs["modality"] = cca_adata.obs["data_type"]


# %%
# from scib.metrics import ilisi_graph, clisi_graph

# # neighbors already computed with scanpy
# sc.pp.neighbors(cca_adata, use_rep="X", n_neighbors=30)

# # compute iLISI (integration) -- requires 'batch_key'
# ilisi = ilisi_graph(
#     cca_adata,
#     batch_key="data_type",
#     type_="knn"    # <-- required
# )

# # compute cLISI (conservation) -- requires 'label_key'
# clisi = clisi_graph(
#     cca_adata,
#     label_key="celltype.l1",
#     type_="knn"    # <-- required
# )

# print("Mean iLISI:", ilisi.mean())
# print("Mean cLISI:", clisi.mean())


# %%
# Get full embedding for all cells (not just the subsampled ones used for visualization)
rna_cca_full, protein_cca_full = fusor.get_embedding(
    active_arr1=fusor.active_arr1, active_arr2=fusor.active_arr2
)

# %%
# Prepare RNA AnnData object
# obs fields
rna_adata.obs["batch_indices"] = 0
rna_adata.obs["n_genes"] = (rna_adata.X > 0).sum(axis=1)
rna_adata.obs["percent_mito"] = 0  # not available in this data
rna_adata.obs["leiden_subclusters"] = "unknown"
rna_adata.obs["cell_types"] = rna_adata.obs["celltype.l1"]
rna_adata.obs["tissue"] = "pbmc"
rna_adata.obs["batch"] = "maxfuse_cite_seq"
rna_adata.obs["minor_cell_types"] = rna_adata.obs["celltype.l2"]
rna_adata.obs["major_cell_types"] = rna_adata.obs["celltype.l1"]
rna_adata.obs["total_counts"] = np.array(rna_adata.X.sum(axis=1)).flatten()
rna_adata.obs["n_genes_by_counts"] = (rna_adata.X > 0).sum(axis=1)
rna_adata.obs["pct_counts_mt"] = 0
rna_adata.obs["index_col"] = np.arange(rna_adata.n_obs)

# var fields
rna_adata.var["n_cells"] = (rna_adata.X > 0).sum(axis=0)
rna_adata.var["mt"] = False
rna_adata.var["ribo"] = False
rna_adata.var["hb"] = False
rna_adata.var["total_counts"] = np.array(rna_adata.X.sum(axis=0)).flatten()
rna_adata.var["n_cells_by_counts"] = (rna_adata.X > 0).sum(axis=0)

# uns fields
rna_adata.uns["dataset_name"] = "maxfuse_cite_seq"
rna_adata.uns["processing_stage"] = "maxfuse_integrated"
rna_adata.uns["file_generated_from"] = "model_maxfuse_dataset_cite_seq.py"

# obsm fields - clean up and only keep what we need
rna_adata.obsm.clear()
rna_adata.obsm["latent"] = rna_cca_full

# layers
rna_adata.layers["counts"] = rna_adata.X.copy()

print(f"rna_adata shape: {rna_adata.shape}")
print(f"rna_adata.obs columns: {list(rna_adata.obs.columns)}")

# %%
# Prepare Protein AnnData object

protein_adata.obs["batch_indices"] = 0
protein_adata.obs["percent_mito"] = 0
protein_adata.obs["leiden_subclusters"] = "unknown"
protein_adata.obs["cell_types"] = protein_adata.obs["celltype.l1"]
protein_adata.obs["tissue"] = "pbmc"
protein_adata.obs["batch"] = "maxfuse_cite_seq"
protein_adata.obs["minor_cell_types"] = protein_adata.obs["celltype.l2"]
protein_adata.obs["major_cell_types"] = protein_adata.obs["celltype.l1"]
protein_adata.obs["total_counts"] = protein_adata.X.sum(axis=1)
protein_adata.obs["n_genes_by_counts"] = (protein_adata.X > 0).sum(axis=1)
protein_adata.obs["index_col"] = np.arange(protein_adata.n_obs)

# var fields
protein_adata.var["feature_type"] = "protein"

# uns fields
protein_adata.uns["dataset_name"] = "maxfuse_cite_seq"
protein_adata.uns["processing_stage"] = "maxfuse_integrated"
protein_adata.uns["file_generated_from"] = "model_maxfuse_dataset_cite_seq.py"

# obsm fields - clean up and only keep what we need
protein_adata.obsm.clear()
protein_adata.obsm["latent"] = protein_cca_full

# layers
protein_adata.layers["counts"] = protein_adata.X.copy()

print(f"protein_adata shape: {protein_adata.shape}")
print(f"protein_adata.obs columns: {list(protein_adata.obs.columns)}")

# %%
# Save the formatted AnnData objects
output_dir = "model_comparison/outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rna_output = f"{output_dir}/maxfuse_cite_seq/7_rna_{timestamp}.h5ad"
protein_output = f"{output_dir}/maxfuse_cite_seq/7_protein_{timestamp}.h5ad"

os.makedirs(f"{output_dir}/maxfuse_cite_seq", exist_ok=True)

rna_adata.write(rna_output)
protein_adata.write(protein_output)

print(f"Saved rna_adata to: {rna_output}")
print(f"Saved protein_adata to: {protein_output}")
print(f"\nrna_adata: {rna_adata}")
print(f"\nprotein_adata: {protein_adata}")

# %%
# Display summary of created objects
print("=" * 80)
print("RNA AnnData Summary:")
print("=" * 80)
print(f"Shape: {rna_adata.shape}")
print(f"\nobs fields ({len(rna_adata.obs.columns)}):")
print(list(rna_adata.obs.columns))
print(f"\nvar fields ({len(rna_adata.var.columns)}):")
print(list(rna_adata.var.columns))
print(f"\nuns fields ({len(rna_adata.uns.keys())}):")
print(list(rna_adata.uns.keys()))
print(f"\nobsm fields ({len(rna_adata.obsm.keys())}):")
print(list(rna_adata.obsm.keys()))
print(f"\nlayers ({len(rna_adata.layers.keys())}):")
print(list(rna_adata.layers.keys()))
print(f"\nobsp fields ({len(rna_adata.obsp.keys()) if rna_adata.obsp else 0}):")
print(list(rna_adata.obsp.keys()) if rna_adata.obsp else [])

print("\n" + "=" * 80)
print("Protein AnnData Summary:")
print("=" * 80)
print(f"Shape: {protein_adata.shape}")
print(f"\nobs fields ({len(protein_adata.obs.columns)}):")
print(list(protein_adata.obs.columns))
print(f"\nvar fields ({len(protein_adata.var.columns)}):")
print(list(protein_adata.var.columns))
print(f"\nuns fields ({len(protein_adata.uns.keys())}):")
print(list(protein_adata.uns.keys()))
print(f"\nobsm fields ({len(protein_adata.obsm.keys())}):")
print(list(protein_adata.obsm.keys()))
print(f"\nlayers ({len(protein_adata.layers.keys())}):")
print(list(protein_adata.layers.keys()))
print(f"\nobsp fields ({len(protein_adata.obsp.keys()) if protein_adata.obsp else 0}):")
print(list(protein_adata.obsp.keys()) if protein_adata.obsp else [])

# %%
