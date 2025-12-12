#!/usr/bin/env python
# coding: utf-8

# # Generate publication figures for RECOMB submission

# ## load data

# In[ ]:


import warnings
warnings.simplefilter(action='ignore',)
warnings.simplefilter(action='ignore',)

import sys
from pathlib import Path

# Add ARCADIA/src to path for imports
# Handle both script and notebook execution
try:
    # In script mode, __file__ is available
    script_dir = Path(__file__).parent
except NameError:
    # In notebook mode, use current working directory
    script_dir = Path.cwd()
arcadia_src = script_dir / "ARCADIA" / "src"
if str(arcadia_src) not in sys.path:
    sys.path.insert(0, str(arcadia_src))

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import seaborn as sns

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_style("ticks")

from arcadia.utils.args import find_checkpoint_from_experiment_name
from arcadia.analysis.post_hoc_utils import load_checkpoint_data
from arcadia.training.metrics import calc_dist
import os
import glob

# Change to ARCADIA directory for MLflow tracking
# Handle both script and notebook execution
try:
    # In script mode, __file__ is available
    script_dir = Path(__file__).parent
except NameError:
    # In notebook mode, use current working directory
    script_dir = Path.cwd()
arcadia_dir = script_dir / "ARCADIA"
original_cwd = os.getcwd()
os.chdir(arcadia_dir)

# Load cite_seq data from latest MLflow checkpoint
checkpoint_path, _, _ = find_checkpoint_from_experiment_name("cite_seq")
synthetic_rna, synthetic_protein = load_checkpoint_data(Path(checkpoint_path))

# Restore original directory
os.chdir(original_cwd)

print('synthetic_protein')
print(synthetic_protein)
print('synthetic_rna')
print(synthetic_rna)


# In[ ]:


# Change to ARCADIA directory for MLflow tracking
os.chdir(arcadia_dir)

# Load tonsil data from latest MLflow checkpoint
checkpoint_path, _, _ = find_checkpoint_from_experiment_name("tonsil")
tonsil_rna, tonsil_protein = load_checkpoint_data(Path(checkpoint_path))

# Restore original directory
os.chdir(original_cwd)

print('tonsil_protein')
print(tonsil_protein)
print('tonsil_rna')
print(tonsil_rna)


# In[ ]:


# Extract latent representations from loaded checkpoint data
tonsil_protein_scVI = tonsil_protein.obsm['X_scVI'].copy()
tonsil_rna_scVI = tonsil_rna.obsm['X_scVI'].copy()

print(tonsil_protein_scVI.shape)
print(tonsil_rna_scVI.shape)


# In[ ]:


# Extract CN assignments from loaded checkpoint data (already in obs)
tonsil_protein_CN_assignments = pd.DataFrame({'CN': tonsil_protein.obs['CN']}, index=tonsil_protein.obs_names)
tonsil_rna_CN_assignments = pd.DataFrame({'CN': tonsil_rna.obs['CN']}, index=tonsil_rna.obs_names)

synthetic_protein_CN_assignments = pd.DataFrame({'CN': synthetic_protein.obs['CN']}, index=synthetic_protein.obs_names)
synthetic_rna_CN_assignments = pd.DataFrame({'CN': synthetic_rna.obs['CN']}, index=synthetic_rna.obs_names)


# In[ ]:


sc.settings.figdir = 'fig_khh'
sc.set_figure_params(figsize = (6,6), dpi_save = 300, format = 'pdf', transparent = True)


# ## dummy dot plots

# In[ ]:


np.random.seed(42)  # reproducible

# Cell and gene labels
cells = [f"Cell_{i:02d}" for i in range(1, 21)]
genes = ["Gene A", "Gene B", "Gene C", "Gene D", "Gene E", "Gene F"]

# Initialize dataframe
expr = pd.DataFrame(index=cells, columns=genes, dtype=float)

# First 10 cells: high in first three genes, low in last three
expr.iloc[:10, 0:3] = np.random.uniform(0.8, 1.0, size=(10, 3))
expr.iloc[:10, 3:6] = np.random.uniform(0.1, 0.3, size=(10, 3))

# Second 10 cells: low in first two genes, high in last four
expr.iloc[10:, 0:2] = np.random.uniform(0.1, 0.3, size=(10, 2))
expr.iloc[10:, 2:6] = np.random.uniform(0.8, 1.0, size=(10, 4))

# Mapping dataframe
groups = pd.DataFrame(
    {"Group": ["CN_1"] * 10 + ["CN_2"] * 10},
    index=cells
)

# Inject sparsity
expr.iloc[[11,13,15,17],[1]] = 0
expr.iloc[[11,13,15,17],[2]] = 0
expr.iloc[[1,3,5,7],[3]] = 0
expr.iloc[[1,3,5,7],[4]] = 0

dummy_ad = ad.AnnData(expr, obs=groups)


# In[ ]:


dummy_dict = {'DE in CN_1': ['Gene A', 'Gene B', 'Gene C'], 
            'DE in CN_2': ['Gene D', 'Gene E','Gene F']}
dummy_list = ['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E','Gene F']
sc.pl.dotplot(dummy_ad, groupby='Group', var_names=dummy_list, figsize=(4, 1), cmap='Blues', save='_dummy_dotplot_blues.pdf')
sc.pl.dotplot(dummy_ad, groupby='Group', var_names=dummy_list, figsize=(4, 1), cmap='Greens', save='_dummy_dotplot_greens.pdf')


# In[ ]:


# create dummy heatmap legend bar
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 1) Define green→white→blue colormap
cmap_gwb = LinearSegmentedColormap.from_list(
    "gwb", ["#1b9e77", "#ffffff", "#377eb8"]  # green, white, blue
)

# 2) Example data centered at 0
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
Z = np.outer(x, y)

# 3) Normalize so white sits at 0
norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0.0, vmax=Z.max())

sns.heatmap(Z, cmap=cmap_gwb, norm=norm, cbar=True)
plt.savefig('fig_khh/dummy_heatmap_legend_bar.pdf')


# ## main figure plots

# In[ ]:


synthetic_protein.obs['spatial_grid_index'] = synthetic_protein.obs['spatial_grid_index'].map({i: f'CN_{i}' for i in [0,1,2,3]})


# In[ ]:


# order is ['B cells', 'CD4 T', 'CD8 T', 'T cells', 'cDCs']
synthetic_protein.obs['major_cell_types'] = synthetic_protein.obs['major_cell_types'].replace('T cells', 'GD/NK T')
synthetic_rna.obs['major_cell_types'] = synthetic_rna.obs['major_cell_types'].replace('T cells', 'GD/NK T')


# In[ ]:


order = ['Mature B','Transitional B','Ifit3-high B','MZ B',] + \
        ['CD4 T','Ifit3-high CD4 T','Activated CD4 T',] + \
        ['CD8 T','CD122+ CD8 T','Ifit3-high CD8 T',] + \
        ['NKT','GD T',] + \
        ['cDC1s','cDC2s','Migratory DCs']
synthetic_protein.obs['minor_cell_types'] = pd.Categorical(synthetic_protein.obs['minor_cell_types'], categories=order)


# In[ ]:


# order is ['B-CD22-CD40', 'B-Ki67', 'Plasma', 'CD4 T', 'CD8 T', 'DC']
tonsil_protein.obs['cell_types'] = pd.Categorical(tonsil_protein.obs['cell_types'], 
                    categories=['B-CD22-CD40', 'B-Ki67', 'Plasma', 'CD4 T', 'CD8 T', 'DC'])


# In[ ]:


synthetic_palette = { 
    'Protein': '#a17bb9', # light purple
    'RNA': '#a0522d', # orangish-brownish
    
    'B cells': '#2ecc71', # green
    'CD4 T': '#e67e22', # orange
    'CD8 T': '#3498db', # blue
    'GD/NK T': '#9b59b6', # purple # note that "T cells" was renamed to this category
    'cDCs': '#e74c3c', # red

    # family color is green 
    'Mature B': '#2ecc71', # bright green
    'Transitional B': '#1abc9c', # sea green
    'Ifit3-high B': '#66cc00', # lime green
    'MZ B': '#006400', # dark forest green

    # family color is orange 
    'CD4 T': '#e67e22', # orange
    'Ifit3-high CD4 T': '#f39c12', # light orange
    'Activated CD4 T': '#d35400', # dark orange

    # family color is blue 
    'CD8 T': '#0066cc', # medium blue
    'CD122+ CD8 T': '#99ccff', # light blue
    'Ifit3-high CD8 T': '#003366', # dark blue

    # family color is purple 
    'NKT': '#bb8fce', # light purple
    'GD T': '#663399', # dark purple

    # family color is red 
    'cDC1s': '#ff0000', # bright red
    'cDC2s': '#8b0000', # dark red
    'Migratory DCs': '#ff6666', # light red

    'NA': '#808080', # gray

    # for CNs, avoid blue, orange, green, and red
    'CN_0': '#edc948', # tableau yellow
    'CN_1': '#b07aa1', # tableau purple  
    'CN_2': '#ff9da7', # tableau pink
    'CN_3': '#9c755f', # tableau brown
}

tonsil_palette = {
    'CODEX': '#a17bb9', # light purple
    'scRNA-seq': '#a0522d', # orangish-brownish

    # B cell family color is green 
    'B-CD22-CD40': '#996633', # light brown
    'B-Ki67': '#8B0000', # dark red
    'Plasma': '#58d68d', # light green
    'CD4 T': '#e67e22', # orange
    'CD8 T': '#3498db', # blue
    'DC': '#9b59b6', # purple

    # for CNs, avoid blue, orange, purple, and brown
    'CN_0': '#a0cbe8', # tableau 20 light blue
    'CN_1': '#ffb37d', # tableau 20 light orange  
    'CN_2': '#59a14f', # tableu green
    'CN_3': '#d62728', # tableau red
    'CN_4': '#d4a6c8', # tableau 20 light purple
    'CN_5': '#d7b5a6', # tableau 20 light brown
    'CN_6': '#e377c2', # tableau pink
    'CN_7': '#7f7f7f', # tableau gray
    'CN_8': '#bcbd22', # tableau yellow
    'CN_9': '#17becf', # tableau cyan
    'CN_10': '#9467bd', # tableau purple (additional CN if needed)
}


# In[ ]:


sc.pl.embedding(synthetic_protein, 'spatial', color=['spatial_grid_index','major_cell_types','minor_cell_types'], 
                    palette = synthetic_palette, s=30, wspace=0.25, save='_semisynthetic_protein.pdf')


# In[ ]:


sc.pl.embedding(synthetic_rna, 'original_umap', color=['major_cell_types','minor_cell_types'], 
                    palette = synthetic_palette, s=10, save='_semisynthetic_rna.pdf')


# In[ ]:


sns.histplot(synthetic_protein.obs, x='spatial_grid_index', hue='major_cell_types', palette=synthetic_palette, multiple='stack', legend=False, edgecolor='none')
plt.title('CN distribution of semi-synthetic protein cells')
plt.savefig('fig_khh/CN_distribution_major_ct_semisynthetic_protein.pdf')


# In[ ]:


with plt.rc_context({'figure.figsize': (3, 3)}):
    sns.histplot(synthetic_protein.obs, x='spatial_grid_index', hue='minor_cell_types', palette=synthetic_palette, multiple='stack', legend=False, edgecolor='none')
    plt.title('CN Placement of Cell Subtypes')
    plt.xticks(fontsize=10)
    plt.xlabel(None)
    plt.savefig('fig_khh/CN_distribution_minor_ct_semisynthetic_protein.pdf')


# In[ ]:


with plt.rc_context({'figure.figsize': (6, 3)}):
    sns.histplot(synthetic_protein.obs, x='minor_cell_types', hue='CN', palette=synthetic_palette, multiple='stack', legend=False, edgecolor='none')
    plt.title('CN Placement of Cell Subtypes')
    plt.xticks(rotation=90, fontsize=10)
    plt.xlabel(None)
    plt.savefig('fig_khh/CN_distribution_minor_ct_semisynthetic_protein_ct_axis.pdf')


# In[ ]:


# Ensure all CN values in data have colors in palette
all_tonsil_cns = set(tonsil_protein.obs['CN'].unique()) | set(tonsil_rna.obs['CN'].unique())
missing_cns = all_tonsil_cns - set(tonsil_palette.keys())
if missing_cns:
    # Generate colors for missing CNs using matplotlib colormap
    import matplotlib.cm as cm
    n_missing = len(missing_cns)
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_missing))
    for i, cn in enumerate(sorted(missing_cns)):
        tonsil_palette[cn] = matplotlib.colors.to_hex(colors[i])

sc.pl.embedding(tonsil_protein, 'spatial', color=['CN','cell_types'], 
                    palette = tonsil_palette, s=10, save='_tonsil_protein.pdf')


# In[ ]:


with plt.rc_context({'figure.figsize': (3, 4)}):
    sns.histplot(tonsil_protein.obs, x='CN', hue='major_cell_types', palette=tonsil_palette, multiple='stack', legend=False, edgecolor='none')
    plt.xticks(rotation=90)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('Cell Type Counts (CODEX)')
    plt.savefig('fig_khh/CN_distribution_major_ct_tonsil_protein.pdf')


# In[ ]:


sc.pl.embedding(tonsil_rna, 'original_umap', color=['cell_types','CN'], 
                    palette = tonsil_palette, s=10, save='_tonsil_rna.pdf')


# ### visualize ARCADIA entangled Z embeddings - semi-synthetic

# In[ ]:


synthetic_protein.obs['CN_csv'] = synthetic_protein_CN_assignments.loc[synthetic_protein.obs_names, 'CN']
synthetic_protein.obs['CN_csv'] = synthetic_protein.obs['CN_csv'].replace({
    'CN_0': 'CN_0',
    'CN_1': 'CN_3',
    'CN_2': 'CN_2',
    'CN_3': 'CN_1',
})
sc.pl.embedding(synthetic_protein, 'spatial', color=['spatial_grid_index','CN_csv'], s=30)


# In[ ]:


synthetic_rna.obs['CN'] = synthetic_rna_CN_assignments.loc[synthetic_rna.obs_names, 'CN']
synthetic_rna.obs['CN'] = synthetic_rna.obs['CN'].replace({
    'CN_0': 'CN_0',
    'CN_1': 'CN_3',
    'CN_2': 'CN_2',
    'CN_3': 'CN_1',
})
sc.pl.embedding(synthetic_rna, 'original_umap', color=['CN'], 
                    palette = synthetic_palette, s=10, save='_synthetic_rna_CN.pdf')


# In[ ]:


adata_protein_df = synthetic_protein.obs.copy()
adata_protein_df['CN'] = adata_protein_df['spatial_grid_index']
for z in range(synthetic_protein.obsm['X_scVI'].shape[1]):
    adata_protein_df[f'z{z}'] = synthetic_protein.obsm['X_scVI'][:, z]
adata_protein_df['modality'] = 'Protein'

adata_rna_df = synthetic_rna.obs.copy()
z_keys = []
for z in range(synthetic_rna.obsm['X_scVI'].shape[1]):
    adata_rna_df[f'z{z}'] = synthetic_rna.obsm['X_scVI'][:, z]
    z_keys.append(f'z{z}')
adata_rna_df['modality'] = 'RNA'

adata_merged_df = pd.concat([adata_protein_df, adata_rna_df], axis=0)
synthetic_merged = ad.AnnData(adata_merged_df[z_keys], obs=adata_merged_df.copy())
synthetic_merged.obsm['X_scVI'] = synthetic_merged.X.copy()
synthetic_merged


# In[ ]:


sc.pp.neighbors(synthetic_merged, use_rep='X_scVI')
sc.tl.umap(synthetic_merged, min_dist=0.3)


# In[ ]:


sc.pl.umap(synthetic_merged, color=['modality','major_cell_types','minor_cell_types','CN'], 
           palette=synthetic_palette, s=10, wspace=0.2, save='_synthetic_merged.pdf')


# ### make confusion matrices

# In[ ]:


# Helper function to compute confusion matrix from adata objects
def compute_confusion_matrix(rna_adata, prot_adata, normalize='columns', latent_key=None):
    """
    Compute confusion matrix for cell type matching between RNA and protein modalities.
    
    Args:   
        rna_adata: AnnData object with RNA data (must have latent representation and cell_types)
        prot_adata: AnnData object with protein data (must have latent representation and cell_types)
        normalize: 'columns' to normalize by columns (percentage), None for raw counts
        latent_key: Key in obsm containing latent representation (e.g., 'X_scVI', 'latent', None to use X)
    
    Returns:
        pd.DataFrame: Confusion matrix (normalized percentages if normalize='columns')
    """
    # Extract latent representations
    if latent_key is None:
        # Try to find latent representation
        if 'X_scVI' in rna_adata.obsm:
            latent_key = 'X_scVI'
        elif 'latent' in rna_adata.obsm:
            latent_key = 'latent'
        else:
            # Use X directly (assuming it's already the latent representation)
            latent_key = None
    
    # Create AnnData objects with latent in X
    if latent_key:
        rna_latent = ad.AnnData(rna_adata.obsm[latent_key], obs=rna_adata.obs.copy())
        prot_latent = ad.AnnData(prot_adata.obsm[latent_key], obs=prot_adata.obs.copy())
    else:
        rna_latent = rna_adata.copy()
        prot_latent = prot_adata.copy()
    
    # Compute nearest neighbors and predicted cell types
    nn_celltypes = calc_dist(rna_latent, prot_latent, label_key="cell_types")
    true_celltypes = rna_latent.obs["cell_types"]
    
    # Create confusion matrix
    cm = pd.crosstab(
        true_celltypes.values,
        nn_celltypes.values,
        rownames=["True"],
        colnames=["Predicted"],
        margins=False,
    )
    
    if normalize == 'columns':
        # Normalize by columns (predicted) - percentage of each predicted class
        col_sums = cm.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        cm_percent = (cm / col_sums) * 100
        return cm_percent
    else:
        return cm


# In[ ]:


# Compute ARCADIA cite_seq confusion matrix
print("Computing ARCADIA cite_seq confusion matrix...")
cm_arcadia_cite_seq = compute_confusion_matrix(synthetic_rna, synthetic_protein)

# Plot all cell types confusion matrix (if matrix format)
if cm_arcadia_cite_seq.shape[1] > 1:
    with plt.rc_context({'figure.figsize': (4, 4)}):
        sns.heatmap(cm_arcadia_cite_seq, annot=cm_arcadia_cite_seq.round(1).astype(str) + "%", fmt='', cmap='Blues', cbar=False, square=True)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('CN Assignment Accuracy (All Cell Types)')
        plt.savefig('fig_khh/CN_confusion_matrix_all.pdf')
        plt.clf()

# Compute B cells only confusion matrix
bcells_mask = synthetic_rna.obs['cell_types'].str.contains('B', case=False, na=False).values
if bcells_mask.sum() > 0:
    cm_bcells = compute_confusion_matrix(synthetic_rna[bcells_mask], synthetic_protein[bcells_mask])
    f1_score = 'XX%'
    if cm_bcells.shape[1] > 1:
        with plt.rc_context({'figure.figsize': (4, 4)}):
            sns.heatmap(cm_bcells, annot=cm_bcells.round(1).astype(str) + "%", fmt='', cmap='Blues', cbar=False, square=True)
            plt.xlabel('Predicted')
            plt.ylabel('Ground Truth')
            plt.title('B Cell CN Assignment, F1 Score: ' + f1_score)
            plt.savefig('fig_khh/CN_confusion_matrix_bcells.pdf')
            plt.clf()


# In[ ]:


from matplotlib.colors import to_rgb


# In[ ]:


def ct_cf_plot(cm, method_str, f1_score, save_path):
    # Check if confusion matrix has correct shape
    if cm.shape[1] == 1:
        print(f"Confusion matrix for {method_str} is not in matrix format (shape: {cm.shape})")
        return
    if cm.shape[0] != 5 or cm.shape[1] != 5:
        raise ValueError(f"Confusion matrix for {method_str} has unexpected shape {cm.shape}, expected (5, 5)")
    cm.columns = ['B cells','CD4 T','CD8 T','GD/NK T','cDCs']
    cm.index = ['B cells','CD4 T','CD8 T','GD/NK T','cDCs']
    row_colors = [to_rgb(synthetic_palette[x]) for x in cm.index]
    col_colors = [to_rgb(synthetic_palette[x]) for x in cm.columns]
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[0.12, 4.0],
        height_ratios=[4.0, 0.12],
        wspace=0.02, hspace=0.02
    )
    ax_left   = fig.add_subplot(gs[:, 0])
    ax_hm     = fig.add_subplot(gs[0, 1])
    # ax_bottom = fig.add_subplot(gs[1, 1])

    sns.heatmap(cm, annot=cm.round(1).astype(str) + "%",
                fmt='', cmap='Blues', cbar=False, square=True,
                xticklabels=False, yticklabels=False, annot_kws={"fontsize": 12}, ax=ax_hm)

    x0, x1 = ax_hm.get_xlim()
    y0, y1 = ax_hm.get_ylim()

    ax_left.imshow(
        np.array(row_colors)[:, None, :],
        extent=[0, 1, 4.8, y1],  # y exactly matches heatmap
        aspect='auto',
        interpolation='nearest',
        origin='upper'
    )
    ax_left.set_xlim(0, 1); ax_left.set_ylim(y0, y1); ax_left.set_axis_off()

    # ax_bottom.imshow(
    #     np.array([col_colors]),
    #     extent=[.05, x1, 0, 1],  # x exactly matches heatmap
    #     aspect='auto',
    #     interpolation='nearest',
    #     origin='lower'
    # )
    # ax_bottom.set_xlim(x0, x1); ax_bottom.set_ylim(0, 1); ax_bottom.set_axis_off()

    ax_hm.set_title(f'{method_str}, F1 Score: {f1_score}', pad=8)

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


# In[ ]:

# Plot ARCADIA cite_seq confusion matrix (already computed above)
ct_cf_plot(cm_arcadia_cite_seq, 'ARCADIA', '94.1%', 'fig_khh/synthetic_confusion_matrix_arcadia.pdf')


# In[ ]:


# MaxFuse confusion matrix will be computed after loading MaxFuse adata objects (see below)


# In[ ]:


# scMODAL confusion matrix will be computed after loading scMODAL adata objects (see below)


# In[ ]:


def ct_cf_plot_tonsil(cm, method_str, f1_score, save_path):
    # Check if confusion matrix has correct shape
    if cm.shape[1] == 1:
        print(f"Confusion matrix for {method_str} (tonsil) is not in matrix format (shape: {cm.shape})")
        return
    if cm.shape[0] != 6 or cm.shape[1] != 6:
        raise ValueError(f"Confusion matrix for {method_str} (tonsil) has unexpected shape {cm.shape}, expected (6, 6)")
    cm.columns = ['B-CD22-CD40', 'B-Ki67', 'CD4 T', 'CD8 T', 'DC', 'Plasma']
    cm.index = ['B-CD22-CD40', 'B-Ki67', 'CD4 T', 'CD8 T', 'DC', 'Plasma']

    row_colors = [to_rgb(tonsil_palette[x]) for x in cm.index]
    col_colors = [to_rgb(tonsil_palette[x]) for x in cm.columns]

    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[0.12, 4.0],
        height_ratios=[4.0, 0.12],
        wspace=0.02, hspace=0.02
    )
    ax_left   = fig.add_subplot(gs[:, 0])
    ax_hm     = fig.add_subplot(gs[0, 1])

    sns.heatmap(cm, annot=cm.round(1).astype(str) + "%",
                fmt='', cmap='Blues', cbar=False, square=True,
                xticklabels=False, yticklabels=False, annot_kws={"fontsize": 10}, ax=ax_hm)

    x0, x1 = ax_hm.get_xlim()
    y0, y1 = ax_hm.get_ylim()

    ax_left.imshow(
        np.array(row_colors)[:, None, :],
        extent=[0, 1, 5.77, y1],  # y exactly matches heatmap
        aspect='auto',
        interpolation='nearest',
        origin='upper'
    )
    ax_left.set_xlim(0, 1); ax_left.set_ylim(y0, y1); ax_left.set_axis_off()

    ax_hm.set_title(f'{method_str}, F1 Score: {f1_score}', pad=8)

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


# In[ ]:


# Compute ARCADIA tonsil confusion matrix
print("Computing ARCADIA tonsil confusion matrix...")
cm_arcadia_tonsil = compute_confusion_matrix(tonsil_rna, tonsil_protein)
ct_cf_plot_tonsil(cm_arcadia_tonsil, 'ARCADIA', 'XX%', 'fig_khh/tonsil_confusion_matrix_arcadia.pdf')


# MaxFuse tonsil confusion matrix will be computed after loading MaxFuse tonsil adata objects (see below)


# scMODAL tonsil confusion matrix will be computed after loading scMODAL tonsil adata objects (see below)


# ### visualize other integration embeddings - semi-synthetic

# In[ ]:


synthetic_merged_copy = synthetic_merged.copy()
sc.pl.umap(synthetic_merged_copy, color='minor_cell_types', palette=synthetic_palette, s=15,
            save='_synthetic_merged_minor_cell_types.pdf')
sc.pl.umap(synthetic_merged_copy, color='matched_archetype_weight', cmap='viridis', s=15, vmin=0.3, vmax=0.9,
            mask_obs=(synthetic_merged_copy.obs['modality'] == 'RNA'),
            save='_synthetic_merged_matched_archetype_weight_RNA.pdf')
synthetic_merged_copy = synthetic_merged.copy()
sc.pl.umap(synthetic_merged_copy, color='matched_archetype_weight', cmap='viridis', s=15, vmin=0.3, vmax=0.9,
            mask_obs=(synthetic_merged_copy.obs['modality'] == 'Protein'),
            save='_synthetic_merged_matched_archetype_weight_Protein.pdf')
# synthetic_merged.obs.columns


# In[ ]:


random_indices = np.random.permutation(list(range(synthetic_merged_copy.shape[0])))
sc.pl.umap(synthetic_merged_copy[random_indices], color='modality', palette=synthetic_palette, s=15,
            save='_synthetic_merged_modality.pdf')


# In[ ]:


sc.pl.umap(synthetic_merged_copy[random_indices], color='CN', palette=synthetic_palette, s=15,
            save='_synthetic_merged_CN.pdf')


# In[ ]:


synthetic_merged.obs_names = synthetic_merged.obs_names + synthetic_merged.obs['modality'].astype(str)


# In[ ]:


# Load scMODAL adata objects and extract UMAP from obsm['X_umap']
scmodal_rna = None
scmodal_protein = None
scmodal_rna_files = glob.glob('model_comparison/outputs/scmodal_cite_seq/*rna*.h5ad')
scmodal_protein_files = glob.glob('model_comparison/outputs/scmodal_cite_seq/*protein*.h5ad') + \
                        glob.glob('model_comparison/outputs/scmodal_cite_seq/*prot*.h5ad')

if not scmodal_rna_files:
    raise FileNotFoundError("scMODAL RNA output files not found in 'model_comparison/outputs/scmodal_cite_seq/'")
if not scmodal_protein_files:
    raise FileNotFoundError("scMODAL protein output files not found in 'model_comparison/outputs/scmodal_cite_seq/'")

# Load the most recent files
scmodal_rna_file = max(scmodal_rna_files, key=os.path.getmtime)
scmodal_protein_file = max(scmodal_protein_files, key=os.path.getmtime)
scmodal_rna = sc.read_h5ad(scmodal_rna_file)
scmodal_protein = sc.read_h5ad(scmodal_protein_file)

# Check if UMAP exists in obsm, if not compute it from latent representation
if 'X_umap' not in scmodal_rna.obsm:
    if 'latent' in scmodal_rna.obsm:
        print("Computing UMAP for scMODAL RNA from latent representation...")
        scmodal_rna.obsm['X_latent'] = scmodal_rna.obsm['latent']
        sc.pp.neighbors(scmodal_rna, use_rep='X_latent')
        sc.tl.umap(scmodal_rna)
    else:
        raise KeyError(f"scMODAL RNA adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(scmodal_rna.obsm.keys())}")
if 'X_umap' not in scmodal_protein.obsm:
    if 'latent' in scmodal_protein.obsm:
        print("Computing UMAP for scMODAL protein from latent representation...")
        scmodal_protein.obsm['X_latent'] = scmodal_protein.obsm['latent']
        sc.pp.neighbors(scmodal_protein, use_rep='X_latent')
        sc.tl.umap(scmodal_protein)
    else:
        raise KeyError(f"scMODAL protein adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(scmodal_protein.obsm.keys())}")

# Create merged scMODAL adata with UMAP
scmodal_rna.obs['modality'] = 'RNA'
scmodal_protein.obs['modality'] = 'Protein'
scmodal_merged = sc.concat([scmodal_rna, scmodal_protein], join='outer', label='modality', keys=['RNA', 'Protein'])
scmodal_merged.obs_names = scmodal_merged.obs_names + scmodal_merged.obs['modality'].astype(str)

# Match with synthetic_merged by obs_names
synthetic_merged_other_embeddings = synthetic_merged.copy()
common_names = synthetic_merged_other_embeddings.obs_names[synthetic_merged_other_embeddings.obs_names.isin(scmodal_merged.obs_names)]
synthetic_merged_other_embeddings = synthetic_merged_other_embeddings[synthetic_merged_other_embeddings.obs_names.isin(common_names)]
scmodal_merged_subset = scmodal_merged[scmodal_merged.obs_names.isin(common_names)]

# Align by obs_names
scmodal_merged_subset = scmodal_merged_subset[synthetic_merged_other_embeddings.obs_names]
synthetic_merged_other_embeddings.obsm['X_scmodal'] = scmodal_merged_subset.obsm['X_umap']

random_indices = np.random.permutation(list(range(synthetic_merged_other_embeddings.shape[0])))
sc.pl.embedding(synthetic_merged_other_embeddings[random_indices], 'scmodal', color=['modality','major_cell_types','minor_cell_types','CN'], 
                palette=synthetic_palette, s=10, save='_synthetic_merged_scmodal.pdf')

# Compute scMODAL cite_seq confusion matrix
print("Computing scMODAL cite_seq confusion matrix...")
cm_scmodal_cite_seq = compute_confusion_matrix(scmodal_rna, scmodal_protein)
ct_cf_plot(cm_scmodal_cite_seq, 'scModal', 'XX%', 'fig_khh/synthetic_confusion_matrix_scmodal.pdf')


# In[ ]:


# Load MaxFuse adata objects and extract UMAP from obsm['X_umap']
maxfuse_rna_files = glob.glob('model_comparison/outputs/maxfuse_cite_seq/*rna*.h5ad')
maxfuse_protein_files = glob.glob('model_comparison/outputs/maxfuse_cite_seq/*protein*.h5ad') + \
                       glob.glob('model_comparison/outputs/maxfuse_cite_seq/*prot*.h5ad')

if not maxfuse_rna_files:
    raise FileNotFoundError("MaxFuse RNA output files not found in 'model_comparison/outputs/maxfuse_cite_seq/'")
if not maxfuse_protein_files:
    raise FileNotFoundError("MaxFuse protein output files not found in 'model_comparison/outputs/maxfuse_cite_seq/'")
if synthetic_merged_other_embeddings is None:
    raise ValueError("synthetic_merged_other_embeddings is None - scMODAL data must be loaded first")

# Load the most recent files
maxfuse_rna_file = max(maxfuse_rna_files, key=os.path.getmtime)
maxfuse_protein_file = max(maxfuse_protein_files, key=os.path.getmtime)
maxfuse_rna = sc.read_h5ad(maxfuse_rna_file)
maxfuse_protein = sc.read_h5ad(maxfuse_protein_file)

# Check if UMAP exists in obsm, if not compute it from latent representation
if 'X_umap' not in maxfuse_rna.obsm:
    if 'latent' in maxfuse_rna.obsm:
        print("Computing UMAP for MaxFuse RNA from latent representation...")
        maxfuse_rna.obsm['X_latent'] = maxfuse_rna.obsm['latent']
        sc.pp.neighbors(maxfuse_rna, use_rep='X_latent')
        sc.tl.umap(maxfuse_rna)
    else:
        raise KeyError(f"MaxFuse RNA adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(maxfuse_rna.obsm.keys())}")
if 'X_umap' not in maxfuse_protein.obsm:
    if 'latent' in maxfuse_protein.obsm:
        print("Computing UMAP for MaxFuse protein from latent representation...")
        maxfuse_protein.obsm['X_latent'] = maxfuse_protein.obsm['latent']
        sc.pp.neighbors(maxfuse_protein, use_rep='X_latent')
        sc.tl.umap(maxfuse_protein)
    else:
        raise KeyError(f"MaxFuse protein adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(maxfuse_protein.obsm.keys())}")

# Create merged MaxFuse adata with UMAP
maxfuse_rna.obs['modality'] = 'RNA'
maxfuse_protein.obs['modality'] = 'Protein'
maxfuse_merged = sc.concat([maxfuse_rna, maxfuse_protein], join='outer', label='modality', keys=['RNA', 'Protein'])
maxfuse_merged.obs_names = maxfuse_merged.obs_names + maxfuse_merged.obs['modality'].astype(str)

# Match with synthetic_merged_other_embeddings by obs_names
common_names = synthetic_merged_other_embeddings.obs_names[synthetic_merged_other_embeddings.obs_names.isin(maxfuse_merged.obs_names)]
maxfuse_merged_subset = maxfuse_merged[maxfuse_merged.obs_names.isin(common_names)]

# Align by obs_names
maxfuse_merged_subset = maxfuse_merged_subset[synthetic_merged_other_embeddings.obs_names]
synthetic_merged_other_embeddings.obsm['X_maxfuse'] = maxfuse_merged_subset.obsm['X_umap']

sc.pl.embedding(synthetic_merged_other_embeddings[random_indices], 'maxfuse', color=['modality','major_cell_types','minor_cell_types','CN'], 
                palette=synthetic_palette, s=10, save='_synthetic_merged_maxfuse.pdf')

# Compute MaxFuse cite_seq confusion matrix
print("Computing MaxFuse cite_seq confusion matrix...")
cm_maxfuse_cite_seq = compute_confusion_matrix(maxfuse_rna, maxfuse_protein)
ct_cf_plot(cm_maxfuse_cite_seq, 'MaxFuse', 'XX%', 'fig_khh/synthetic_confusion_matrix_maxfuse.pdf')


# ### visualize entangled Z embedding - tonsil data

# In[ ]:


rs = np.random.RandomState(42)
unique_labels = adata_protein_df.index.unique()
label_pick = pd.Series(rs.rand(len(unique_labels)) < 0.20, index=unique_labels)
mask = adata_protein_df.index.to_series().map(label_pick).to_numpy()


# In[ ]:


print(tonsil_protein_scVI.shape)
print(tonsil_rna_scVI.shape)


# In[ ]:


adata_protein_df = tonsil_protein.obs.copy()
adata_protein_df['modality'] = 'CODEX'

adata_rna_df = tonsil_rna.obs.copy()
z_keys = []
for z in range(60):
    z_keys.append(f'z{z}')
adata_rna_df['modality'] = 'scRNA-seq'

adata_protein_df[z_keys] = tonsil_protein_scVI
adata_rna_df[z_keys] = tonsil_rna_scVI

adata_merged_df = pd.concat([adata_protein_df, adata_rna_df], axis=0)
tonsil_merged = ad.AnnData(adata_merged_df[z_keys], obs=adata_merged_df.copy())
tonsil_merged.obsm['X_scVI'] = tonsil_merged.X.copy()
print(tonsil_merged.shape)

adata_merged_df_subsampled = pd.concat([adata_protein_df.loc[mask,:], adata_rna_df], axis=0)
tonsil_merged_subsampled = ad.AnnData(adata_merged_df_subsampled[z_keys], obs=adata_merged_df_subsampled.copy())
tonsil_merged_subsampled.obsm['X_scVI'] = tonsil_merged_subsampled.X.copy()
print(tonsil_merged_subsampled.shape)


# In[ ]:


print(tonsil_merged.obs.value_counts(['modality']))
print(tonsil_merged_subsampled.obs.value_counts(['modality']))


# In[ ]:


sc.pp.neighbors(tonsil_merged, use_rep='X_scVI')
sc.tl.umap(tonsil_merged, min_dist=0.3)
sc.pl.umap(tonsil_merged, color=['modality','cell_types'])


# In[ ]:


sc.pp.neighbors(tonsil_merged_subsampled, use_rep='X_scVI')
sc.tl.umap(tonsil_merged_subsampled, min_dist=0.3)
sc.pl.umap(tonsil_merged_subsampled, color=['modality','cell_types'])


# In[ ]:


tonsil_CN_assignments = pd.concat([tonsil_protein_CN_assignments, tonsil_rna_CN_assignments], axis=0)
tonsil_CN_assignments.index = tonsil_CN_assignments.index.astype(str)


# In[ ]:


tonsil_merged_subsampled.obs['CN'] = tonsil_CN_assignments.loc[tonsil_merged_subsampled.obs_names.values, 'CN']


# In[ ]:


sc.pl.umap(tonsil_merged_subsampled, color=['modality','cell_types','CN'], 
           palette=tonsil_palette, s=10, wspace=0.2, save='_tonsil_merged_subsampled.pdf')


# In[ ]:


sc.pl.umap(tonsil_merged_subsampled[tonsil_merged_subsampled.obs['modality'] == 'scRNA-seq'], color=['cell_types','CN'],
           palette=tonsil_palette, s=10, legend_loc=None)


# In[ ]:


tonsil_merged_subsampled_copy = tonsil_merged_subsampled.copy() # prevent NA overwriting
sc.pl.umap(tonsil_merged_subsampled_copy, color=['cell_types','CN'], mask_obs=(tonsil_merged_subsampled_copy.obs['modality'] == 'scRNA-seq'),
           palette=tonsil_palette, s=10, legend_loc=None, save='_tonsil_merged_subsampled_RNA_only.pdf')
tonsil_merged_subsampled_copy = tonsil_merged_subsampled.copy() # prevent NA overwriting
sc.pl.umap(tonsil_merged_subsampled_copy, color=['cell_types','CN'], mask_obs=(tonsil_merged_subsampled_copy.obs['modality'] == 'CODEX'),
           palette=tonsil_palette, s=10, legend_loc=None, save='_tonsil_merged_subsampled_prot_only.pdf')


# ### visualize other integration embeddings - tonsil

# In[ ]:


tonsil_merged_subsampled_other_embeddings = tonsil_merged_subsampled.copy()


# In[ ]:


# Load scMODAL adata objects for tonsil and extract UMAP from obsm['X_umap']
tonsil_scmodal_rna_files = glob.glob('model_comparison/outputs/scmodal_tonsil/*rna*.h5ad')
tonsil_scmodal_protein_files = glob.glob('model_comparison/outputs/scmodal_tonsil/*protein*.h5ad') + \
                               glob.glob('model_comparison/outputs/scmodal_tonsil/*prot*.h5ad')

if not tonsil_scmodal_rna_files:
    raise FileNotFoundError("scMODAL RNA output files not found in 'model_comparison/outputs/scmodal_tonsil/'")
if not tonsil_scmodal_protein_files:
    raise FileNotFoundError("scMODAL protein output files not found in 'model_comparison/outputs/scmodal_tonsil/'")

tonsil_scmodal_rna_file = max(tonsil_scmodal_rna_files, key=os.path.getmtime)
tonsil_scmodal_protein_file = max(tonsil_scmodal_protein_files, key=os.path.getmtime)
tonsil_scmodal_rna = sc.read_h5ad(tonsil_scmodal_rna_file)
tonsil_scmodal_protein = sc.read_h5ad(tonsil_scmodal_protein_file)

# Check if UMAP exists in obsm, if not compute it from latent representation
if 'X_umap' not in tonsil_scmodal_rna.obsm:
    if 'latent' in tonsil_scmodal_rna.obsm:
        print("Computing UMAP for tonsil scMODAL RNA from latent representation...")
        tonsil_scmodal_rna.obsm['X_latent'] = tonsil_scmodal_rna.obsm['latent']
        sc.pp.neighbors(tonsil_scmodal_rna, use_rep='X_latent')
        sc.tl.umap(tonsil_scmodal_rna)
    else:
        raise KeyError(f"Tonsil scMODAL RNA adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(tonsil_scmodal_rna.obsm.keys())}")
if 'X_umap' not in tonsil_scmodal_protein.obsm:
    if 'latent' in tonsil_scmodal_protein.obsm:
        print("Computing UMAP for tonsil scMODAL protein from latent representation...")
        tonsil_scmodal_protein.obsm['X_latent'] = tonsil_scmodal_protein.obsm['latent']
        sc.pp.neighbors(tonsil_scmodal_protein, use_rep='X_latent')
        sc.tl.umap(tonsil_scmodal_protein)
    else:
        raise KeyError(f"Tonsil scMODAL protein adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(tonsil_scmodal_protein.obsm.keys())}")

tonsil_scmodal_rna.obs['modality'] = 'scRNA-seq'
tonsil_scmodal_protein.obs['modality'] = 'CODEX'
tonsil_scmodal_adata = sc.concat([tonsil_scmodal_protein, tonsil_scmodal_rna], join='outer', label='modality', keys=['CODEX', 'scRNA-seq'])
tonsil_scmodal_adata.obsm['X_scmodal'] = np.vstack([
    tonsil_scmodal_protein.obsm['X_umap'],
    tonsil_scmodal_rna.obsm['X_umap']
])

# subsample protein 20%
tonsil_scmodal_adata_subsampled = []
tonsil_scmodal_adata_subsampled.append(tonsil_scmodal_adata[tonsil_scmodal_adata.obs['modality'] == 'CODEX'].copy())
tonsil_scmodal_adata_subsampled.append(tonsil_scmodal_adata[tonsil_scmodal_adata.obs['modality'] == 'scRNA-seq'].copy())
tonsil_scmodal_adata_subsampled[0] = tonsil_scmodal_adata_subsampled[0][mask]
tonsil_scmodal_adata_subsampled = sc.concat(tonsil_scmodal_adata_subsampled)

random_indices = np.random.permutation(list(range(tonsil_scmodal_adata_subsampled.shape[0])))

sc.pl.embedding(tonsil_scmodal_adata_subsampled[random_indices], 'scmodal', color=['modality','cell_types','CN'],
                palette=tonsil_palette, s=10, save='_tonsil_merged_scmodal.pdf')
tonsil_scmodal_adata_subsampled_copy = tonsil_scmodal_adata_subsampled.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_scmodal_adata_subsampled_copy, 'scmodal', color=['CN'], mask_obs=(tonsil_scmodal_adata_subsampled_copy.obs['modality'] == 'CODEX'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_scmodal_CN_codex.pdf', show=False)
tonsil_scmodal_adata_subsampled_copy = tonsil_scmodal_adata_subsampled.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_scmodal_adata_subsampled_copy, 'scmodal', color=['CN'], mask_obs=(tonsil_scmodal_adata_subsampled_copy.obs['modality'] == 'scRNA-seq'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_scmodal_CN_rna.pdf', show=False)

# Compute scMODAL tonsil confusion matrix
print("Computing scMODAL tonsil confusion matrix...")
cm_scmodal_tonsil = compute_confusion_matrix(tonsil_scmodal_rna, tonsil_scmodal_protein)
ct_cf_plot_tonsil(cm_scmodal_tonsil, 'scModal', 'XX%', 'fig_khh/tonsil_confusion_matrix_scmodal.pdf')


# In[ ]:


# Load MaxFuse adata objects for tonsil and extract UMAP from obsm['X_umap']
tonsil_maxfuse_rna_files = glob.glob('model_comparison/outputs/maxfuse_tonsil/*rna*.h5ad')
tonsil_maxfuse_protein_files = glob.glob('model_comparison/outputs/maxfuse_tonsil/*protein*.h5ad') + \
                               glob.glob('model_comparison/outputs/maxfuse_tonsil/*prot*.h5ad')

if not tonsil_maxfuse_rna_files:
    raise FileNotFoundError("MaxFuse RNA output files not found in 'model_comparison/outputs/maxfuse_tonsil/'")
if not tonsil_maxfuse_protein_files:
    raise FileNotFoundError("MaxFuse protein output files not found in 'model_comparison/outputs/maxfuse_tonsil/'")

tonsil_maxfuse_rna_file = max(tonsil_maxfuse_rna_files, key=os.path.getmtime)
tonsil_maxfuse_protein_file = max(tonsil_maxfuse_protein_files, key=os.path.getmtime)
tonsil_maxfuse_rna = sc.read_h5ad(tonsil_maxfuse_rna_file)
tonsil_maxfuse_protein = sc.read_h5ad(tonsil_maxfuse_protein_file)

# Check if UMAP exists in obsm, if not compute it from latent representation
if 'X_umap' not in tonsil_maxfuse_rna.obsm:
    if 'latent' in tonsil_maxfuse_rna.obsm:
        print("Computing UMAP for tonsil MaxFuse RNA from latent representation...")
        tonsil_maxfuse_rna.obsm['X_latent'] = tonsil_maxfuse_rna.obsm['latent']
        sc.pp.neighbors(tonsil_maxfuse_rna, use_rep='X_latent')
        sc.tl.umap(tonsil_maxfuse_rna)
    else:
        raise KeyError(f"Tonsil MaxFuse RNA adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(tonsil_maxfuse_rna.obsm.keys())}")
if 'X_umap' not in tonsil_maxfuse_protein.obsm:
    if 'latent' in tonsil_maxfuse_protein.obsm:
        print("Computing UMAP for tonsil MaxFuse protein from latent representation...")
        tonsil_maxfuse_protein.obsm['X_latent'] = tonsil_maxfuse_protein.obsm['latent']
        sc.pp.neighbors(tonsil_maxfuse_protein, use_rep='X_latent')
        sc.tl.umap(tonsil_maxfuse_protein)
    else:
        raise KeyError(f"Tonsil MaxFuse protein adata object doesn't have 'X_umap' or 'latent' in obsm. Available keys: {list(tonsil_maxfuse_protein.obsm.keys())}")

tonsil_maxfuse_rna.obs['modality'] = 'scRNA-seq'
tonsil_maxfuse_protein.obs['modality'] = 'CODEX'
tonsil_maxfuse_adata = sc.concat([tonsil_maxfuse_protein, tonsil_maxfuse_rna], join='outer', label='modality', keys=['CODEX', 'scRNA-seq'])
tonsil_maxfuse_adata.obsm['X_maxfuse'] = np.vstack([
    tonsil_maxfuse_protein.obsm['X_umap'],
    tonsil_maxfuse_rna.obsm['X_umap']
])

# subsample protein 20%
tonsil_maxfuse_adata_subsampled = []
tonsil_maxfuse_adata_subsampled.append(tonsil_maxfuse_adata[tonsil_maxfuse_adata.obs['modality'] == 'CODEX'].copy())
tonsil_maxfuse_adata_subsampled.append(tonsil_maxfuse_adata[tonsil_maxfuse_adata.obs['modality'] == 'scRNA-seq'].copy())
tonsil_maxfuse_adata_subsampled[0] = tonsil_maxfuse_adata_subsampled[0][mask]
tonsil_maxfuse_adata_subsampled = sc.concat(tonsil_maxfuse_adata_subsampled)

random_indices = np.random.permutation(list(range(tonsil_maxfuse_adata_subsampled.shape[0])))

sc.pl.embedding(tonsil_maxfuse_adata_subsampled[random_indices], 'maxfuse', color=['modality','cell_types','CN'],
                palette=tonsil_palette, s=10, save='_tonsil_merged_maxfuse.pdf')
tonsil_maxfuse_adata_subsampled_copy = tonsil_maxfuse_adata_subsampled.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_maxfuse_adata_subsampled_copy, 'maxfuse', color=['CN'], mask_obs=(tonsil_maxfuse_adata_subsampled_copy.obs['modality'] == 'CODEX'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_maxfuse_CN_codex.pdf', show=False)
tonsil_maxfuse_adata_subsampled_copy = tonsil_maxfuse_adata_subsampled.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_maxfuse_adata_subsampled_copy, 'maxfuse', color=['CN'], mask_obs=(tonsil_maxfuse_adata_subsampled_copy.obs['modality'] == 'scRNA-seq'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_maxfuse_CN_rna.pdf', show=False)

# Compute MaxFuse tonsil confusion matrix
print("Computing MaxFuse tonsil confusion matrix...")
cm_maxfuse_tonsil = compute_confusion_matrix(tonsil_maxfuse_rna, tonsil_maxfuse_protein)
ct_cf_plot_tonsil(cm_maxfuse_tonsil, 'MaxFuse', 'XX%', 'fig_khh/tonsil_confusion_matrix_maxfuse.pdf')


# In[ ]:


random_indices = np.random.permutation(list(range(tonsil_merged_subsampled_other_embeddings.shape[0])))
sc.pl.embedding(tonsil_merged_subsampled_other_embeddings[random_indices], 'umap', color=['modality','major_cell_types'], 
                palette=tonsil_palette, s=10, save='_tonsil_merged_arcadia.pdf')
tonsil_merged_subsampled_other_embeddings_copy = tonsil_merged_subsampled_other_embeddings.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_merged_subsampled_other_embeddings_copy, 'umap', color=['CN'], mask_obs=(tonsil_merged_subsampled_other_embeddings_copy.obs['modality'] == 'CODEX'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_arcadia_CN_codex.pdf')
tonsil_merged_subsampled_other_embeddings_copy = tonsil_merged_subsampled_other_embeddings.copy() # prevent NA overwriting
sc.pl.embedding(tonsil_merged_subsampled_other_embeddings_copy, 'umap', color=['CN'], mask_obs=(tonsil_merged_subsampled_other_embeddings_copy.obs['modality'] == 'scRNA-seq'),
                palette=tonsil_palette, s=10, save='_tonsil_merged_arcadia_CN_rna.pdf')


# ## counterfactual spatial neighborhood DEG analysis - tonsil dataset

# In[ ]:


tonsil_rna.X = tonsil_rna.layers['counts'].copy()
sc.pp.normalize_total(tonsil_rna)
sc.pp.log1p(tonsil_rna)


# In[ ]:


tonsil_rna.var['pct_cells'] = tonsil_rna.var['n_cells'] / tonsil_rna.shape[0]
gene_mask = tonsil_rna.var[tonsil_rna.var['pct_cells'] > 0.05].index


# ### B-Ki67

# In[ ]:


# B-Ki67: CN_3, CN_7, CN_9 (germinal centers)
bcells = tonsil_rna[tonsil_rna.obs['cell_types'] == 'B-Ki67']
sc.tl.rank_genes_groups(bcells, groupby='CN', method='wilcoxon')
deg_df = sc.get.rank_genes_groups_df(bcells, group=None)
deg_df = deg_df[deg_df['group'].isin(['CN_3', 'CN_7', 'CN_9'])].sort_values('logfoldchanges', ascending=False)

deg_df_filtered = deg_df[deg_df['names'].isin(gene_mask)]
deg_df_filtered = deg_df_filtered[deg_df_filtered['pvals_adj'] < 0.05]
deg_df_filtered.to_csv('B_Ki67_deg_df.csv')


# In[ ]:


keep_cn = ['CN_0', 'CN_1', 'CN_2', 'CN_3', 'CN_6','CN_7', 'CN_9'] # do not examine CN4, CN5, and CN8 due to low counts


# In[ ]:


df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_3') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_3:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_3') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_3:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_7') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_7:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_7') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_7:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_9') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_9:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_9') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_9:')
print(df.iloc[:100]['names'].tolist())


# In[ ]:


# just choose top 100 upregulated genes for interpretation
CN_3 = ['PIF1', 'PLK1', 'TOP2A', 'CENPF', 'CENPE', 'ASPM', 'UBE2C', 'DEPDC1', 'AURKA', 'HMMR', 'SGO2', 'KIF14', 'DLGAP5', 'CDC20', 'TPX2', 'CCNB2', 'GTSE1', 'CDK1', 'NUF2', 'CDKN3', 'UBE2S', 'CCNB1', 'TUBB4B', 'AURKB', 'MKI67', 'NUSAP1', 'AICDA', 'HIST1H4C', 'BIRC5', 'LPP', 'CCNA2', 'MYBL1', 'BACH2', 'KIF11', 'HIST1H1B', 'NCAPG', 'H1FX', 'HMGB2', 'KIF20B', 'PTTG1', 'NEIL1', 'HMCES', 'SMC4', 'CD84', 'KNL1', 'TMEM131L', 'TCL1A', 'HIST1H1C', 'TBC1D4', 'DAAM1', 'AC025569.1', 'SPN', 'TUBA1B', 'FAM107B', 'SEC14L1', 'SYNE2', 'HIST1H1E', 'HIST1H1D', 'STMN1', 'LRMP', 'BRWD1', 'CXCR4', 'IKZF2', 'KLHL6', 'TMPO', 'MIS18BP1', 'DDX3Y', 'MZB1', 'BCL6', 'LNPEP', 'H2AFV', 'CD38', 'UBE2J1', 'TUBB', 'LBR', 'CCDC144A', 'CCDC88A', 'SUGCT', 'NUCKS1', 'ANP32E', 'BCL7A', 'HMGN2', 'GGA2', 'UCP2', 'RPS4Y1', 'PAG1', 'HMGB1', 'MME', 'RASSF6', 'VPREB3', 'PEG10', 'MALAT1', 'VIM', 'SEL1L3', 'HIST1H2BG', 'ZBTB20', 'MARCKSL1', 'H3F3A', 'SMC2', 'ZNF106']
CN_7 = ['IRF4', 'PCNA', 'HSP90AB1', 'PRDX4', 'FAM111B', 'BATF', 'IL2RB', 'RRM2', 'MCM4', 'CLSPN', 'CYTOR', 'HELLS', 'DUSP2', 'CTSC', 'TNFRSF13B', 'ATAD2', 'SUB1', 'MPEG1', 'GPR183', 'DUT', 'LDHA', 'MCM3', 'HELB', 'KLF2', 'FKBP5', 'DHFR', 'TYMS', 'MIR155HG', 'BHLHE41', 'MCM7', 'BCL2A1', 'XBP1', 'PREX1', 'CD44', 'JUNB', 'FKBP11', 'S100A6', 'TK1', 'HSP90B1', 'PCLAF', 'CALR', 'HSPA5', 'PASK', 'PRDX1', 'DGKH', 'ZEB2', 'FCRL5', 'BCL9L', 'GABPB1-AS1', 'YBX1', 'RAN', 'SEC11C', 'HMGA1', 'SSR4', 'BRCA1', 'ODC1', 'MYBL2', 'CD83', 'PKM', 'GSTP1', 'IER2', 'SKAP1', 'LYST', 'ARAP2', 'CD69', 'SELL', 'SRGN', 'COTL1', 'HLA-DRB5', 'H2AFZ', 'NABP1', 'FOSB', 'HLA-DQB1', 'HLA-A', 'DERL3', 'DUSP1', 'XIST', 'DDIT4', 'MBP', 'JUN', 'RGS1', 'GAPDH', 'NAP1L1', 'NEAT1', 'PARP14', 'PFN1', 'BRCA2', 'PPIA', 'SLC25A5', 'HLA-DQA1', 'DNMT1', 'KIAA1551', 'LCP1', 'MACF1', 'RNF213', 'C12ORF75', 'PLEK', 'SMC2', 'VMP1', 'SPIB']
CN_9 = ['HIST1H1B', 'FAM111B', 'TK1', 'DHFR', 'RRM2', 'DUT', 'CLSPN', 'LMO2', 'TYMS', 'BIRC5', 'MCM3', 'H2AFZ', 'DEK', 'PCLAF', 'TUBB', 'LDHA', 'HMGB1', 'HMGA1', 'ODC1', 'MYBL2', 'DNMT1', 'RAN', 'YBX1', 'GAPDH', 'PKM', 'PFN1']


# In[ ]:


# visualize top 50 in supplemental
dp = sc.pl.dotplot(bcells[bcells.obs['CN'].isin(keep_cn)], CN_3[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_bcells_CN_3.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(bcells[bcells.obs['CN'].isin(keep_cn)], CN_7[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_bcells_CN_7.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(bcells[bcells.obs['CN'].isin(keep_cn)], CN_9[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_bcells_CN_9.pdf', bbox_inches='tight')


# In[ ]:


# just choose a subset of upregulated genes for each CN of interest
bcell_degs = { 
    'CN_3 B-Ki67 DEGs' : [ # high in somatic hypermutation, B cell activation / BCR signaling, GC and DZ patterns, and proliferation
              'BCL6', 'AICDA', 'MYBL1', 'BACH2', 'ZBTB20', 'BRWD1', 'BCL7A',
              'CD38', 'CD84', 'MME', 'LRMP', 'TCL1A', 'KLHL6', 'PAG1', 'CXCR4', 'IKZF2',
              'MKI67', 'BIRC5'],
    'CN_7 B-Ki67 DEGs' : [ # high in plasma cell differentiation, activation and signaling, metabolic remodeling, and immune presentation
              'IRF4', 'XBP1', 'FKBP11', 'CALR', 'HSPA5', 'HSP90B1', 'SSR4', 'PRDX4', 'PARP14',
              'CD83', 'CD69', 'TNFRSF13B', 'IL2RB', 'FCRL5', 'PREX1', 'GPR183', 'CD44', 'SELL', 'RGS1', 'SRGN', 'LCP1', 'PLEK',
              'GAPDH', 'PKM', 'LDHA', 'ODC1', 'SLC25A5', 'VMP1',
              'HLA-A', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB5', 'CTSC', 'DERL3'],
    'CN_9 B-Ki67 DEGs' : [ # high in cell cycle progression / survival and metabolic reprogramming
              'MYBL2', 'BIRC5',
              'GAPDH', 'PKM', 'LDHA', 'ODC1'],
}
bcell_degs_selected = {
    'CN_3 B-Ki67 DEGs' : [ # high in somatic hypermutation, B cell activation / BCR signaling, GC and DZ patterns, and proliferation
              'BCL6', 'AICDA', 'MYBL1', 'BACH2', 'ZBTB20', 'BRWD1', 'BCL7A',
              'CD38', 'CD84', 'MME', 'LRMP', 'TCL1A', 'KLHL6', 'PAG1', 'CXCR4',
              'MKI67', 'BIRC5'],
    'CN_7 B-Ki67 DEGs' : [ # high in plasma cell differentiation, activation and signaling, metabolic remodeling, and immune presentation
              'CD83', 'FCRL5', 
              'LCP1', 'PLEK',
              'GAPDH', 'PKM', 'LDHA', 'ODC1', 'SLC25A5',
              'HLA-A', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB5'],
    'CN_9 B-Ki67 DEGs' : [ # high in cell cycle progression / survival and metabolic reprogramming
              'MYBL2', 'BIRC5',
              'GAPDH', 'PKM', 'LDHA', 'ODC1'],
}
bcell_degs_selected_interpretation = {
    'CN_3 B-Ki67 DEGs' : [ # high in somatic hypermutation, B cell activation / BCR signaling, and GC and DZ proliferation
              'BCL6', 'AICDA', 'MYBL1', 'BACH2', 'ZBTB20', 'BRWD1', 'BCL7A',
              'CD84', 'MME', 'LRMP', 'KLHL6', 'CXCR4',
              'MKI67', 'BIRC5'],
    'CN_7 and CN_9 B-Ki67 DEGs' : [ # high in plasma cell differentiation, activation and signaling, survival and metabolic remodeling, and immune presentation
              'CD83', 'FCRL5', 
              'LCP1', 'PLEK',
              'GAPDH', 'LDHA', 'ODC1', 
              'HLA-A', 'HLA-DQA1', 'HLA-DQB1'],
    'CN_9' : [ # high in cell cycle progression and other programs from CN_7
              'MYBL2', 'BIRC5',],
}

dp = sc.pl.dotplot(bcells[bcells.obs['CN'].isin(['CN_3','CN_7','CN_9'])], 
                   bcell_degs_selected_interpretation, 
                   groupby='CN', 
                   standard_scale='var', 
                   show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")

plt.savefig('fig_khh/dotplot_tonsil_rna_bcells_CN_379.pdf', bbox_inches='tight')


# ### CD8 T

# In[ ]:


# CD8 T: CN_0, CN_1, CN_5, CN_8 (B CD22 CD40 neighboring cells)
cd8t = tonsil_rna[tonsil_rna.obs['cell_types'] == 'CD8 T']
sc.tl.rank_genes_groups(cd8t, groupby='CN', method='wilcoxon', groups=['CN_0','CN_1','CN_5','CN_8'])
deg_df = sc.get.rank_genes_groups_df(cd8t, group=None)
deg_df = deg_df[deg_df['group'].isin(['CN_0', 'CN_1', 'CN_5', 'CN_8'])].sort_values('logfoldchanges', ascending=False)

deg_df_filtered = deg_df[deg_df['names'].isin(gene_mask)]
deg_df_filtered = deg_df_filtered[deg_df_filtered['pvals_adj'] < 0.05]
deg_df_filtered.to_csv('CD8_T_deg_df.csv')


# In[ ]:


keep_cn = ['CN_0', 'CN_1', 'CN_4', 'CN_5', 'CN_8'] # do not examine CN2, CN3, CN6, CN7, and CN9 due to low counts


# In[ ]:


df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_0') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_0:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_0') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_0:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_1') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_1:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_1') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_1:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_5') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_5:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_5') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_5:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_8') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_8:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_8') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_8:')
print(df.iloc[:100]['names'].tolist())


# In[ ]:


# just choose top 100 upregulated genes for interpretation
CN_0 = ['TNFRSF4', 'TNFRSF18', 'KLRB1', 'TXK', 'SRGAP3', 'GNAQ', 'LAIR1', 'HOPX', 'LYN', 'BHLHE40', 'TNFRSF25', 'FOS', 'NOL4L', 'CD7', 'ID2', 'PLAC8', 'AC044849.1', 'LTB', 'IFITM2', 'IL7R', 'DUSP1', 'BIRC3', 'CTSW', 'DDIT4', 'AKT3', 'DUT', 'TSC22D3', 'IKZF2', 'SELL', 'SATB1', 'TCF7', 'GABPB1-AS1', 'XIST', 'IER2', 'FOSB', 'ZFP36L1', 'HMGN1', 'HSP90AB1', 'MACF1', 'NAP1L1', 'H3F3A']
CN_8 = ['CCL4', 'DTHD1', 'GZMA', 'CCL5', 'GRAP2', 'SH2D1A', 'MIAT', 'GZMK', 'GZMM', 'CYTOR', 'ITGA1', 'COTL1', 'CST7', 'CD3D', 'NKG7', 'BCL11B', 'ITM2A', 'ATXN1', 'PYHIN1', 'TRAT1', 'A2M-AS1', 'CD6', 'EOMES', 'TIGIT', 'KLRG1', 'TC2N', 'SLAMF7', 'LAPTM5', 'IKZF3', 'CD84', 'ARAP2', 'TRAC', 'RARRES3', 'TRBC2', 'SYNE2', 'CD3G', 'FYN', 'PDCD4', 'RNF213', 'LCP1', 'PTPRC', 'HLA-A']
print(len(CN_0))
print(len(CN_8))


# In[ ]:


# visualize top 50 in supplemental
dp = sc.pl.dotplot(cd8t[cd8t.obs['CN'].isin(keep_cn)], CN_0[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd8t_CN_0.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(cd8t[cd8t.obs['CN'].isin(keep_cn)], CN_8[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd8t_CN_8.pdf', bbox_inches='tight')


# In[ ]:


# just choose a subset of upregulated genes for each CN of interest
cd8t_degs = { 
    'CN_0 CD8 T cell DEGs' : [
               'TNFRSF4', 'TNFRSF18', 'KLRB1', 'TXK', 'HOPX', 'BHLHE40', 'TNFRSF25', 'FOS', 'CD7', 'ID2', 'LTB',
               'IFITM2', 'IL7R', 'CTSW', 'DDIT4', 'SELL', 'SATB1', 'TCF7', 'IER2', 'FOSB', 'ZFP36L1', ],
    'CN_8 CD8 T cell DEGs' : [
               'CCL4', 'GZMA', 'CCL5', 'GRAP2', 'SH2D1A', 'GZMK', 'GZMM', 'ITGA1', 'COTL1', 'CST7', 'NKG7', 
               'BCL11B', 'ITM2A', 'PYHIN1', 'TRAT1', 'CD6', 'EOMES', 'TIGIT', 'KLRG1', 'SLAMF7', 'CD84',
               'TRAC', 'RARRES3', 'TRBC2', 'FYN', 'PDCD4', 'PTPRC', 'HLA-A']
}
cd8t_degs_selected = {
    
}

dp = sc.pl.dotplot(cd8t[cd8t.obs['CN'].isin(keep_cn)], 
                   cd8t_degs, 
                   groupby='CN', 
                   standard_scale='var', 
                   show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")

plt.savefig('fig_khh/dotplot_tonsil_rna_cd8t_CN_08.pdf', bbox_inches='tight')


# ### CD4 T

# In[ ]:


# CD4 T: CN_0, CN_1, CN_5, CN_8 (B CD22 CD40 neighboring cells)
cd4t = tonsil_rna[tonsil_rna.obs['cell_types'] == 'CD4 T']
sc.tl.rank_genes_groups(cd4t, groupby='CN', method='wilcoxon', groups=['CN_0', 'CN_1', 'CN_5', 'CN_8'])
deg_df = sc.get.rank_genes_groups_df(cd4t, group=None)
deg_df = deg_df[deg_df['group'].isin(['CN_0', 'CN_1', 'CN_5', 'CN_8'])].sort_values('logfoldchanges', ascending=False)

deg_df_filtered = deg_df[deg_df['names'].isin(gene_mask)]
deg_df_filtered = deg_df_filtered[deg_df_filtered['pvals_adj'] < 0.05]
deg_df_filtered.to_csv('CD4_T_deg_df.csv')


# In[ ]:


keep_cn = ['CN_0', 'CN_1', 'CN_4', 'CN_5', 'CN_8'] # do not examine CN2, CN3, CN6, CN7, and CN9 due to low counts


# In[ ]:


df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_0') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_0:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_0') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_0:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_1') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_1:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_1') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_1:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_5') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_5:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_5') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_5:')
print(df.iloc[:100]['names'].tolist())

df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_8') & (deg_df_filtered['logfoldchanges'] > 0)].sort_values('logfoldchanges', ascending=False)
print('upregulated genes in CN_8:')
print(df.iloc[:100]['names'].tolist())
df = deg_df_filtered[(deg_df_filtered['group'] == 'CN_8') & (deg_df_filtered['logfoldchanges'] < 0)].sort_values('logfoldchanges', ascending=True)
print('downregulated genes in CN_8:')
print(df.iloc[:100]['names'].tolist())


# In[ ]:


# just choose top 100 upregulated genes for interpretation
CN_0 = ['EGR2', 'ASCL2', 'ITGB8', 'LAG3', 'ZEB2', 'DRAIC', 'AC022239.1', 'KIAA1324', 'LINC01480', 'TNFRSF18', 'BCL6', 'KSR2', 'IGHM', 'IER5L', 'PDCD1', 'PTMS', 'DUSP6', 'SRGN', 'CTLA4', 'EGR1', 'FGFR1', 'CD200', 'PKM', 'MYO6', 'TNFRSF4', 'ICA1', 'POU2AF1', 'SPATS2L', 'MCTP1', 'TENT5C', 'TNFRSF1B', 'DUSP2', 'PRDX4', 'ODC1', 'BATF', 'GBP2', 'TIGIT', 'TSPAN5', 'UCP2', 'MAF', 'TOX2', 'CORO1B', 'DTHD1', 'PRDX1', 'KIAA1671', 'ARHGAP10', 'ITM2A', 'SH2D1A', 'PASK', 'MTUS1', 'EZR', 'TBC1D4', 'IL6ST', 'XIST', 'AC004585.1', 'PGGHG', 'ACTN1', 'ICOS', 'HMCES', 'LYST', 'GAPDH', 'LDHA', 'SEC11C', 'CALR', 'LRMP', 'IL2RB', 'CD38', 'H2AFV', 'COTL1', 'SLC25A5', 'POU2F2', 'VMP1', 'CD69', 'PARP1', 'H3F3A', 'CYTOR', 'HNRNPLL', 'BTLA', 'SIRPG', 'SUB1', 'ISG20', 'FBLN7', 'ANP32E', 'HERPUD1', 'METAP2', 'PHACTR2', 'ARPC3', 'FAM107B', 'BCL2', 'FKBP5', 'PFN1', 'RAN', 'CHI3L2', 'IKZF3', 'HSP90AB1', 'SPN', 'CEMIP2', 'IPCEF1', 'CDK5R1', 'HCST']
CN_1 = ['TENT5C', 'DRAIC', 'PDCD1', 'IGHM', 'DUSP6', 'XIST', 'ICA1', 'TOX2', 'TNFRSF1B', 'BCL6', 'MAF', 'TIGIT', 'CTLA4', 'CORO1B', 'AC004585.1', 'ISG20', 'PKM', 'SH2D1A', 'TBC1D4', 'UCP2', 'IL2RB', 'MIS18BP1', 'SRGN', 'IKZF3', 'GAPDH', 'COTL1', 'PHACTR2', 'H2AFZ', 'EZR', 'H3F3A', 'HLA-A']
CN_5 = ['CCR7', 'LINC00861', 'CAMK4', 'TXNIP', 'VIM', 'HELB', 'RPS4Y1', 'IL7R', 'SLFN5', 'GIMAP7', 'AAK1', 'ARL4C', 'SARAF', 'GIMAP4', 'TRAF3IP3', 'ZFP36L2', 'FYB1', 'MALAT1']
CN_8 = ['IL7R', 'TXNIP', 'SCML4', 'VIM', 'LINC00861', 'SORL1', 'HELB', 'GIMAP5', 'AHNAK', 'ARL4C', 'ZFP36L2', 'GPR183', 'AAK1', 'CAMK4', 'DYRK2', 'CCR7', 'TSC22D3', 'GIMAP1', 'GIMAP7', 'TRAF3IP3', 'MALAT1', 'GIMAP4', 'FYB1', 'BCL11B']
print(len(CN_0))
print(len(CN_1))
print(len(CN_5))
print(len(CN_8))


# In[ ]:


# visualize top 50 in supplemental
dp = sc.pl.dotplot(cd4t[cd4t.obs['CN'].isin(keep_cn)], CN_0[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd4t_CN_0.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(cd4t[cd4t.obs['CN'].isin(keep_cn)], CN_1[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd4t_CN_1.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(cd4t[cd4t.obs['CN'].isin(keep_cn)], CN_5[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd4t_CN_5.pdf', bbox_inches='tight')

dp = sc.pl.dotplot(cd4t[cd4t.obs['CN'].isin(keep_cn)], CN_8[:50], groupby='CN', standard_scale='var', show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")
plt.savefig('fig_khh/dotplot_tonsil_rna_cd4t_CN_8.pdf', bbox_inches='tight')


# In[ ]:


# just choose a subset of upregulated genes for each CN of interest
cd4t_degs = { 
    'CN_0 CD4 T cell DEGs' : [
        'EGR2', 'ASCL2', 'ITGB8', 'LAG3', 'ZEB2', 'TNFRSF18', 'BCL6', 'PDCD1', 'SRGN', 'CTLA4',
        'EGR1', 'CD200', 'PKM', 'TNFRSF4', 'TNFRSF1B', 'BATF', 'GBP2', 'TIGIT', 'MAF', 'TOX2', 'ITM2A', 'SH2D1A'],
    'CN_1 CD4 T cell DEGs' : [
        'PDCD1', 'TOX2', 'TNFRSF1B', 'BCL6', 'MAF', 'TIGIT', 'CTLA4', 'ISG20', 'GAPDH', 'HLA-A'],
    'CN_5 CD4 T cell DEGs' : [
        'CCR7', 'CAMK4', 'IL7R', 'SLFN5', 'GIMAP7', 'GIMAP4', 'TRAF3IP3'],
    'CN_8 CD4 T cell DEGs' : [
        'IL7R', 'GIMAP5', 'ZFP36L2', 'GPR183', 'CCR7', 'GIMAP1', 'GIMAP7', 'TRAF3IP3', 'MALAT1', 'GIMAP4', 'FYB1', 'BCL11B'],
}
cd4t_degs_selected = {
    
}

dp = sc.pl.dotplot(cd4t[cd4t.obs['CN'].isin(keep_cn)], 
                   cd4t_degs, 
                   groupby='CN', 
                   standard_scale='var', 
                   show=False)
ax = dp["mainplot_ax"]
for l in ax.get_yticklabels():
    l.set_color(tonsil_palette[l.get_text()])
    l.set_fontweight("bold")

plt.savefig('fig_khh/dotplot_tonsil_rna_cd4t_CN_0158.pdf', bbox_inches='tight')

