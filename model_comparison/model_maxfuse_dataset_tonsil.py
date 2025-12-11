# %%
# need to use python version 3.8 with conda as that's the requirement for maxfuse
# inspired from https://github.com/shuxiaoc/maxfuse/blob/main/docs/tonsil_codex_rnaseq.ipynb
# preprocess tonsil data for maxfuse and runs maxfuse, saves the results to a file to be compared with other arcadia
# import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

plt.rcParams["figure.figsize"] = (6, 4)

import os
from datetime import datetime

import anndata as ad
import maxfuse as mf
import scanpy as sc
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# %%
# !/usr/bin/python3 -m pip install ipykernel -U --user --force-reinstall

# %%
# this cell only needs to be run once
# import requests, zipfile, io
# r = requests.get("http://stat.wharton.upenn.edu/~zongming/maxfuse/data.zip")
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall("../")

# %%


# %%
protein = pd.read_csv(
    "/home/barroz/projects/ARCADIA/CODEX_RNA_seq/data/raw_data/tonsil/tonsil_codex.csv"
)  # ~178,000 codex cells

# %%
# not needed to run maxfuse
sns.scatterplot(
    data=protein.sample(frac=0.1), x="centroid_x", y="centroid_y", hue="cluster.term", s=1
)

# %%
# input csv contains meta info, take only protein features
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
# convert to AnnData
adata_prot = ad.AnnData(
    protein[protein_features].to_numpy(),
    dtype=np.float32,
    obsm={"spatial": protein[["centroid_x", "centroid_y"]].to_numpy()},
)
adata_prot.var_names = protein[protein_features].columns

# %%
base_folder = "/home/barroz/projects/ARCADIA/CODEX_RNA_seq/data/raw_data/tonsil"
# read in RNA data
rna = mmread(f"{base_folder}/tonsil_rna_counts.txt")  # rna count as sparse matrix, 10k cells (RNA)
rna_names = pd.read_csv(f"{base_folder}/tonsil_rna_names.csv")["names"].to_numpy()
# convert to AnnData
adata_rna = ad.AnnData(rna.tocsr(), dtype=np.float32)
adata_rna.var_names = rna_names
adata_rna.X = adata_rna.X.toarray()

# %%
adata_rna, adata_prot


# %%


# %%
# read in celltyle labels
df_rna_metadata = pd.read_csv(f"{base_folder}/tonsil_rna_meta.csv", index_col=0)
labels_rna = df_rna_metadata["cluster.info"].to_numpy()
labels_codex = protein["cluster.term"].to_numpy()
adata_prot.obs["celltype"] = labels_codex
adata_rna.obs["celltype"] = labels_rna
adata_rna.obs.index = df_rna_metadata.index.to_numpy()


# %%
# sc.pp.subsample(adata_rna, n_obs=1500)
# sc.pp.subsample(adata_prot, n_obs=1500)
labels_rna = adata_rna.obs["celltype"].to_numpy()
labels_codex = adata_prot.obs["celltype"].to_numpy()

# %%
correspondence = pd.read_csv(f"{base_folder}/protein_gene_conversion.csv")
correspondence.head()

# %%
rna_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]
    if curr_protein_name not in adata_prot.var_names:
        continue
    if (
        curr_rna_names.find("Ignore") != -1
    ):  # some correspondence ignored eg. protein isoform to one gene
        continue
    curr_rna_names = curr_rna_names.split("/")  # eg. one protein to multiple genes
    for r in curr_rna_names:
        if r in adata_rna.var_names:
            rna_protein_correspondence.append([r, curr_protein_name])

rna_protein_correspondence = np.array(rna_protein_correspondence)

# %%
# Columns rna_shared and protein_shared are matched.
# One may encounter "Variable names are not unique" warning,
# this is fine and is because one RNA may encode multiple proteins and vice versa.
rna_shared = adata_rna[:, rna_protein_correspondence[:, 0]].copy()
protein_shared = adata_prot[:, rna_protein_correspondence[:, 1]].copy()

# %%
# Make sure no column is static, only use protein features
# that are variable (larger than a certain threshold)
# mask = (rna_shared.X.std(axis=0) > 0.5) & (protein_shared.X.std(axis=0) > 0.1)
mask = (rna_shared.X.toarray().std(axis=0) > 0.3) & (protein_shared.X.std(axis=0) > 0.05)

rna_shared = rna_shared[:, mask].copy()
protein_shared = protein_shared[:, mask].copy()
print([rna_shared.shape, protein_shared.shape])

# %%
# process rna_shared
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
fusor.plot_singular_values(
    target="active_arr1", n_components=None  # can also explicitly specify the number of components
)

# %%
# plot top singular values of avtive_arr2 on a random batch
fusor.plot_singular_values(target="active_arr2", n_components=None)

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
fusor.plot_singular_values(
    target="shared_arr1",
    n_components=None,
)

# %%
# plot top singular values of shared_arr2 on a random batch
fusor.plot_singular_values(target="shared_arr2", n_components=None)

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
    matching=pivot_matching, labels1=labels_rna, labels2=labels_codex, order=(2, 1)
)
lv1_acc

# %%
# We can inspect the first pivot pair.
[pivot_matching[0][0], pivot_matching[1][0], pivot_matching[2][0]]

# %%
cm = confusion_matrix(
    adata_rna.obs["celltype"].to_numpy()[pivot_matching[0]],
    adata_prot.obs["celltype"].to_numpy()[pivot_matching[1]],
)
ConfusionMatrixDisplay(
    confusion_matrix=np.round((cm.T / np.sum(cm, axis=1)).T * 100),
    display_labels=np.unique(adata_rna.obs["celltype"].to_numpy()),
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
    matching=full_matching,
    labels1=adata_rna.obs["celltype"].to_numpy(),
    labels2=adata_prot.obs["celltype"].to_numpy(),
)
lv1_acc

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
cca_adata.obs["cell_type"] = list(
    np.concatenate(
        (
            adata_rna.obs["celltype"].to_numpy(),
            adata_prot.obs["celltype"].to_numpy()[full_matching[1]][randix],
        ),
        axis=0,
    )
)

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
# obs fields
adata_rna.obs["batch_indices"] = 0
adata_rna.obs["n_genes"] = (adata_rna.X > 0).sum(axis=1)
adata_rna.obs["percent_mito"] = 0  # not available in this data
adata_rna.obs["leiden_subclusters"] = "unknown"
adata_rna.obs["cell_types"] = adata_rna.obs["celltype"]
adata_rna.obs["tissue"] = "tonsil"
adata_rna.obs["batch"] = "tonsil"
adata_rna.obs["minor_cell_types"] = adata_rna.obs["celltype"]
adata_rna.obs["major_cell_types"] = adata_rna.obs["celltype"]
adata_rna.obs["total_counts"] = np.array(adata_rna.X.sum(axis=1)).flatten()
adata_rna.obs["n_genes_by_counts"] = (adata_rna.X > 0).sum(axis=1)
adata_rna.obs["pct_counts_mt"] = 0
adata_rna.obs["index_col"] = np.arange(adata_rna.n_obs)

# var fields
adata_rna.var["n_cells"] = (adata_rna.X > 0).sum(axis=0)
adata_rna.var["mt"] = False
adata_rna.var["ribo"] = False
adata_rna.var["hb"] = False
adata_rna.var["total_counts"] = np.array(adata_rna.X.sum(axis=0)).flatten()
adata_rna.var["n_cells_by_counts"] = (adata_rna.X > 0).sum(axis=0)

# uns fields
adata_rna.uns["dataset_name"] = "tonsil"
adata_rna.uns["processing_stage"] = "maxfuse_integrated"
adata_rna.uns["file_generated_from"] = "maxfuse_tut.py"

# obsm fields
adata_rna.obsm["latent"] = rna_cca_full

# layers
adata_rna.layers["counts"] = adata_rna.X.copy()

print(f"adata_rna shape: {adata_rna.shape}")
print(f"adata_rna.obs columns: {list(adata_rna.obs.columns)}")

# %%
# Prepare Protein AnnData object with spatial coordinates

# Extract spatial coordinates from original protein dataframe

adata_prot.obs["batch_indices"] = 0
adata_prot.obs["percent_mito"] = 0
adata_prot.obs["leiden_subclusters"] = "unknown"
adata_prot.obs["cell_types"] = adata_prot.obs["celltype"]
adata_prot.obs["tissue"] = "tonsil"
adata_prot.obs["batch"] = "tonsil"
adata_prot.obs["minor_cell_types"] = adata_prot.obs["celltype"]
adata_prot.obs["major_cell_types"] = adata_prot.obs["celltype"]

# Add spatial coordinates from original protein dataframe
adata_prot.obs["condition"] = "tonsil"
adata_prot.obs["Image"] = "tonsil_image"
adata_prot.obs["Sample"] = "tonsil_sample"
adata_prot.obs["total_counts"] = adata_prot.X.sum(axis=1)
adata_prot.obs["outlier"] = False
adata_prot.obs["n_genes_by_counts"] = (adata_prot.X > 0).sum(axis=1)
adata_prot.obs["log1p_n_genes_by_counts"] = np.log1p(adata_prot.obs["n_genes_by_counts"])
adata_prot.obs["log1p_total_counts"] = np.log1p(adata_prot.obs["total_counts"])
adata_prot.obs["CN"] = "CN_unknown"
adata_prot.obs["index_col"] = np.arange(adata_prot.n_obs)
# var fields
adata_prot.var["feature_type"] = "protein"
# uns fields
adata_prot.uns["dataset_name"] = "tonsil"
adata_prot.uns["processing_stage"] = "maxfuse_integrated"
adata_prot.uns["file_generated_from"] = "maxfuse_tut.py"
# obsm fields - add spatial coordinates
adata_prot.obsm["latent"] = codex_cca_full

print(f"adata_prot shape: {adata_prot.shape}")
print(f"adata_prot.obs columns: {list(adata_prot.obs.columns)}")

# %%
# plot spatial coordinates with sc.pl.embedding with celltype as color
sc.pl.embedding(adata_prot, "spatial", color="cell_types")

# %%
# Save the formatted AnnData objects


output_dir = "/home/barroz/projects/ARCADIA/model_comparison/outputs"
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
