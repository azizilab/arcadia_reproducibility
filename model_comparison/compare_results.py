#!/usr/bin/env python
# coding: utf-8
"""
Compare ARCADIA results against baseline models (MaxFuse, scMODAL, etc.).

This script compares model performance metrics between ARCADIA and other baseline models
using checkpoints from MLflow experiments. Results are appended to a CSV file for tracking.

Usage Examples:
    # Using experiment name (recommended):
    python compare_results.py --experiment_name "cite_seq" --other_model_name "maxfuse"

    # Using checkpoint path:
    python compare_results.py --checkpoint_path "/path/to/mlruns/.../epoch_0499" --other_model_name "maxfuse"

    # Interactive/notebook use (will use defaults):
    # Just run the script without arguments

Note:
    Dataset name is automatically inferred from the ARCADIA checkpoint metadata.

Arguments:
    --experiment_name: MLflow experiment name (e.g., "cite_seq", "tonsil")
    --checkpoint_path: Direct path to checkpoint folder (alternative to experiment_name)
    --other_model_name: Baseline model name (default: "maxfuse")
    --experiment_id: MLflow experiment ID (auto-inferred if using experiment_name)
    --run_id: MLflow run ID (auto-inferred if using experiment_name)

Output:
    - Prints comparison metrics to console
    - Appends results to 'metrics/results_comparison_{dataset}_{other_model}.csv'
    - Each row contains: timestamp, experiment info, model name, metrics

Note:
    Results are automatically appended to the same CSV file, making it easy to
    track model performance across multiple experiments and runs.
"""
# In[1]:


import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
from anndata import AnnData

FILENAME = "compare_results.py"

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")


def here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# Set up paths - model_comparison is at project root level
THIS_DIR = here()
PROJECT_ROOT = THIS_DIR.parent
ARCADIA_ROOT = PROJECT_ROOT / "ARCADIA"
ARCADIA_SRC = ARCADIA_ROOT / "src"

# Update sys.path to include ARCADIA/src for imports
sys.path.insert(0, str(ARCADIA_SRC))
sys.path.insert(0, str(ARCADIA_ROOT))

# Change to ARCADIA directory for MLflow tracking
os.chdir(str(ARCADIA_ROOT))

import anndata as ad
import numpy as np
import scanpy as sc

import matplotlib as mpl
from arcadia.analysis.comparison_utils import (
    align_data,
    calculate_single_model_metrics,
    merge_model_results,
    save_comparison_results,
)
from arcadia.analysis.post_hoc_utils import assign_rna_cn_from_protein, load_checkpoint_data
from arcadia.archetypes.generation import add_matched_archetype_weight
from arcadia.data_utils.loading import load_adata_latest
from arcadia.plotting.post_hoc import (
    plot_individual_and_combined_umaps,
    plot_morans_i,
    plot_umap_latent_arcadia,
    plot_umap_per_cell_type,
)
from arcadia.training import metrics as mtrc
from arcadia.utils.args import find_checkpoint_from_experiment_name, parse_compare_arguments

# Load config_ if exists
config_path = ARCADIA_ROOT / "CODEX_RNA_seq" / "config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

# Parse command line arguments
args, unknown = parse_compare_arguments()

mpl.rcParams.update(
    {
        "savefig.format": "pdf",
        "figure.figsize": (6, 4),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
    }
)
sc.set_figure_params(
    scanpy=True, fontsize=10, dpi=100, dpi_save=300, vector_friendly=False, format="pdf"
)

start_time = datetime.now()
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")


sc.settings.set_figure_params(dpi=50, facecolor="white", fontsize=10)

base_path = "CODEX_RNA_seq/data"


# In[2]:


# load arcadia data
if args.checkpoint_path is not None:
    # Use explicitly provided checkpoint path (highest priority)
    checkpoint_path = args.checkpoint_path
    experiment_id_from_path = None
    run_id_from_path = None
elif args.experiment_name is not None:
    # Find checkpoint from experiment name
    print(f"Finding checkpoint from experiment: {args.experiment_name}")
    checkpoint_path, experiment_id_from_path, run_id_from_path = (
        find_checkpoint_from_experiment_name(args.experiment_name)
    )
    print(f"Found checkpoint: {checkpoint_path}")
else:
    # Default checkpoint paths for interactive use
    # checkpoint_path = "/home/barroz/projects/ARCADIA/mlruns/549776225454865377/00136e5d52e04cbab4142f754bf201c5/artifacts/checkpoints/epoch_0499"  # cite-seq
    # checkpoint_path = "/home/barroz/projects/ARCADIA/mlruns/549776225454865377/03ff9e2ecacc4de4abf465276ddbe712/artifacts/checkpoints/epoch_0499"  # cite-seq
    checkpoint_path = "/home/barroz/projects/ARCADIA/mlruns/159590843374850282/223886ff922c4a89bf58e8c435029775/artifacts/checkpoints/epoch_0499"  # tonsil
    # checkpoint_path = "/home/barroz/projects/ARCADIA/mlruns/549776225454865377/5532ccd46166450795b0deb3254f0fc0/artifacts/checkpoints/epoch_0499"  # cite-seq
    experiment_id_from_path = None
    run_id_from_path = None

# Update experiment_id and run_id if found from experiment name
if experiment_id_from_path:
    args.experiment_id = experiment_id_from_path
if run_id_from_path:
    args.run_id = run_id_from_path

config_path = Path(checkpoint_path).parent.parent / "model_config.json"
checkpoint_folder = Path(checkpoint_path)

# Print checkpoint age
checkpoint_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_folder))
time_diff = datetime.now() - checkpoint_time

if time_diff.days > 0:
    print(
        f"Checkpoint was created {time_diff.days} days, {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
    )
elif time_diff.seconds > 3600:
    print(
        f"Checkpoint was created {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
    )
else:
    print(f"Checkpoint was created {time_diff.seconds} seconds ago")

adata_rna_arcadia, adata_prot_arcadia = load_checkpoint_data(checkpoint_folder)

# Get dataset_name from ARCADIA checkpoint
dataset_name = adata_rna_arcadia.uns.get("dataset_name", "unknown")
if args.other_model_name is None:
    # other_model_name = "maxfuse"
    other_model_name = "scmodal"
else:
    other_model_name = args.other_model_name


if dataset_name == "tonsil":
    adata_rna_arcadia.uns["dataset_name"] = "tonsil"
    adata_prot_arcadia.uns["dataset_name"] = "tonsil"
    dataset_name = "tonsil"
print(f"Dataset name from checkpoint: {dataset_name}")
print(f"Comparing against baseline model: {other_model_name}")

# %%
# load other data

adata_rna_other = load_adata_latest(
    str(PROJECT_ROOT / f"model_comparison/outputs/{other_model_name}_{dataset_name}"),
    "rna",
    dataset_name=dataset_name,
    exact_step=7,
    index_from_end=0,
    return_path=False,
)
adata_prot_other = load_adata_latest(
    str(PROJECT_ROOT / f"model_comparison/outputs/{other_model_name}_{dataset_name}"),
    "protein",
    dataset_name=dataset_name,
    exact_step=7,
    index_from_end=0,
    return_path=False,
)
if dataset_name == "tonsil":
    adata_rna_other.uns["dataset_name"] = "tonsil"
    adata_prot_other.uns["dataset_name"] = "tonsil"

print(adata_rna_other.uns["dataset_name"])

# temp todo only for tonsil dataset - will not be needed for future trainings
# as the _0 file was updated to add proper index
if dataset_name == "tonsil":
    adata_rna_other.obs.index = range(adata_rna_other.n_obs)
    adata_rna_other.obs.index = adata_rna_other.obs.index.astype(str)

# In[3]:


# %%
adata_rna_arcadia = add_matched_archetype_weight(adata_rna_arcadia)
adata_prot_arcadia = add_matched_archetype_weight(adata_prot_arcadia)

adata_rna_arcadia, adata_rna_other, mutual_genes = align_data(
    adata_rna_arcadia, adata_rna_other, "rna", other_model_name
)
adata_prot_arcadia, adata_prot_other, mutual_prot = align_data(
    adata_prot_arcadia, adata_prot_other, "protein", other_model_name
)

if num_rna_cells < 12000 or num_protein_cells < 12000:
    sc.pp.subsample(adata_rna_arcadia, n_obs=(min(num_rna_cells, adata_rna_arcadia.n_obs)))
    sc.pp.subsample(adata_prot_arcadia, n_obs=(min(num_protein_cells, adata_prot_arcadia.n_obs)))
    sc.pp.subsample(adata_rna_other, n_obs=(min(num_rna_cells, adata_rna_other.n_obs)))
    sc.pp.subsample(adata_prot_other, n_obs=(min(num_protein_cells, adata_prot_other.n_obs)))
# todo set to False for future trainings
select_most_common_cell_type = False
if select_most_common_cell_type:
    cell_type_counts = adata_rna_arcadia.obs["cell_types"].value_counts()
    most_common_cell_type = cell_type_counts.index[0]
    adata_rna_arcadia = adata_rna_arcadia[
        adata_rna_arcadia.obs["cell_types"] == most_common_cell_type
    ].copy()
    adata_prot_arcadia = adata_prot_arcadia[
        adata_prot_arcadia.obs["cell_types"] == most_common_cell_type
    ].copy()
adata_rna_arcadia.obsm["latent"] = adata_rna_arcadia.obsm["X_scVI"]
adata_prot_arcadia.obsm["latent"] = adata_prot_arcadia.obsm["X_scVI"]
adata_rna_arcadia.obsm.pop("X_scVI")
adata_prot_arcadia.obsm.pop("X_scVI")


# add CN and matched_archetype_weight and filter genes from adata_rna_arcadia to match adata_rna_other
latent_arcadia_rna = adata_rna_arcadia.obsm["latent"]
latent_arcadia_prot = adata_prot_arcadia.obsm["latent"]
latent_other_rna = adata_rna_other.obsm["latent"]
latent_other_prot = adata_prot_other.obsm["latent"]
adata_latent_arcadia_rna = AnnData(latent_arcadia_rna, obs=adata_rna_arcadia.obs)
adata_latent_arcadia_prot = AnnData(latent_arcadia_prot, obs=adata_prot_arcadia.obs)
adata_latent_other_rna = AnnData(latent_other_rna, obs=adata_rna_other.obs)
adata_latent_other_prot = AnnData(latent_other_prot, obs=adata_prot_other.obs)
sc.pp.neighbors(adata_latent_arcadia_rna, n_neighbors=15)
sc.pp.neighbors(adata_latent_arcadia_prot, n_neighbors=15)
sc.pp.neighbors(adata_latent_other_rna, n_neighbors=15)
sc.pp.neighbors(adata_latent_other_prot, n_neighbors=15)

# %%
adata_latent_arcadia_rna.obs["modality"] = "RNA"
adata_latent_arcadia_prot.obs["modality"] = "Protein"
adata_latent_other_rna.obs["modality"] = "RNA"
adata_latent_other_prot.obs["modality"] = "Protein"

# Add pair_id for metrics that require matched pairs (FOSKNN, FOSCTTM, pair_distance)
# Use the obs index as pair_id since RNA and protein have matching indices
adata_latent_arcadia_rna.obs["pair_id"] = adata_latent_arcadia_rna.obs.index
adata_latent_arcadia_prot.obs["pair_id"] = adata_latent_arcadia_prot.obs.index
adata_latent_other_rna.obs["pair_id"] = adata_latent_other_rna.obs.index
adata_latent_other_prot.obs["pair_id"] = adata_latent_other_prot.obs.index

# Create combined latent data for metrics that need both modalities
combined_latent_arcadia = ad.concat(
    [adata_latent_arcadia_rna, adata_latent_arcadia_prot],
    label="modality_concat",
    keys=["RNA", "Protein"],
)
combined_latent_other = ad.concat(
    [adata_latent_other_rna, adata_latent_other_prot],
    label="modality_concat",
    keys=["RNA", "Protein"],
)

# Compute neighbors for iLISI calculation
sc.pp.neighbors(combined_latent_arcadia, use_rep="X")
sc.pp.neighbors(combined_latent_other, use_rep="X")

# In[7]:


# %%
# Precompute cross-modal distance matrices for efficiency
print("Precomputing cross-modal distance matrices...")
current_time = time()

# Arcadia distances
rna_X_arcadia = (
    adata_latent_arcadia_rna.X.toarray()
    if hasattr(adata_latent_arcadia_rna.X, "toarray")
    else adata_latent_arcadia_rna.X
)
prot_X_arcadia = (
    adata_latent_arcadia_prot.X.toarray()
    if hasattr(adata_latent_arcadia_prot.X, "toarray")
    else adata_latent_arcadia_prot.X
)

# Other model distances
rna_X_other = (
    adata_latent_other_rna.X.toarray()
    if hasattr(adata_latent_other_rna.X, "toarray")
    else adata_latent_other_rna.X
)
prot_X_other = (
    adata_latent_other_prot.X.toarray()
    if hasattr(adata_latent_other_prot.X, "toarray")
    else adata_latent_other_prot.X
)


# Combined latent distances (for FOSKNN and FOSCTTM)
combined_X_arcadia = (
    combined_latent_arcadia.X.toarray()
    if hasattr(combined_latent_arcadia.X, "toarray")
    else combined_latent_arcadia.X
)

# %%
# cross_modal_distances_arcadia = cdist(rna_X_arcadia, prot_X_arcadia, metric="euclidean")
# cross_modal_distances_other = cdist(rna_X_other, prot_X_other, metric="euclidean")
# combined_distances_arcadia = cdist(combined_X_arcadia, combined_X_arcadia, metric="euclidean")
cross_modal_distances_arcadia = None
cross_modal_distances_other = None
combined_distances_arcadia = None

combined_X_other = (
    combined_latent_other.X.toarray()
    if hasattr(combined_latent_other.X, "toarray")
    else combined_latent_other.X
)
# combined_distances_other = cdist(combined_X_other, combined_X_other, metric="euclidean")
combined_distances_other = None

print(f"Distance matrix computation took {time() - current_time:.2f} seconds")
# %%
# Get metrics functions once for reuse
metrics_funcs_two_modalities = {
    "combined_latent_silhouette_f1": mtrc.compute_silhouette_f1,
    "cross_modality_cell_type_accuracy": mtrc.matching_accuracy,
    "cross_modality_cn_accuracy": mtrc.cross_modality_cn_accuracy,
    "f1_score": mtrc.f1_score_calc,
    "cn_f1_score": mtrc.f1_score_calc,
    # "ari_score": mtrc.ari_score_calc,
}

metric_funcs_one_modality = {
    "cn_kbet_within_cell_types": mtrc.kbet_within_cell_types,
    "calculate_cell_type_silhouette": mtrc.calculate_cell_type_silhouette,
    "morans_i": mtrc.morans_i,
}

metric_funcs_combined_modalities = {
    "ari_f1": mtrc.compute_ari_f1,
    "cn_ilisi_within_cell_types": mtrc.calculate_cn_ilisi_within_cell_types,
    "silhouette_score": mtrc.silhouette_score_calc,
    "calculate_iLISI": mtrc.calculate_iLISI,
    "cn_kbet_within_cell_types": mtrc.kbet_within_cell_types,
    "modality_kbet_mixing_score": mtrc.modality_kbet_mixing_score,
    "pair_distance": mtrc.pair_distance,
    "mixing_metric": mtrc.mixing_metric,
    "morans_i": mtrc.morans_i,  # Moran's I on combined data
}

# Metric parameters (base kwargs without distance matrices)
metrics_kwargs = {
    "cn_kbet_within_cell_types": {"label_key": "CN", "group_key": "cell_types", "rep_key": "X"},
    "modality_kbet_mixing_score": {"label_key": "modality", "group_key": None, "rep_key": "X"},
    "calculate_cell_type_silhouette": {"celltype_key": "cell_types", "use_rep": "X"},
    "calculate_iLISI": {"batch_key": "modality", "plot_flag": plot_flag},
    "cn_f1_score": {"label_key": "CN"},
    "cross_modality_cn_accuracy": {"k": 3, "global_step": 0 if plot_flag else None},
    "morans_i": {"score_key": "matched_archetype_weight", "use_rep": "X", "n_neighbors": 15},
    "pair_distance": {
        "modality_key": "modality",
        "pair_key": "pair_id",
        "rep_key": "X",
        "print_flag": True,
    },
    "foscttm": {
        "modality_key": "modality",
        "pair_key": "pair_id",
        "rep_key": "X",
        "print_flag": True,
    },
    "fosknn": {"modality_key": "modality", "pair_key": "pair_id", "rep_key": "X", "k": 3},
    "mixing_metric": {
        "modality_key": "modality",
        "rep_key": "X",
        "k_neighborhood": 300,
        "neighbor_ref": 5,
    },
    "cross_modality_cell_type_accuracy": {"plot_flag": False},
}
# Create metrics directory if it doesn't exist
metrics_dir = PROJECT_ROOT / "metrics"
metrics_dir.mkdir(exist_ok=True)
# %%
# mtrc.kbet_within_cell_types(combined_latent_arcadia, group_key=None,label_key="modality")
# 0.53 somewhat mixed
# mtrc.kbet_within_cell_types(combined_latent_arcadia, group_key=None,label_key="cell_types")
# 0.9992914262577184 not mixed at all
# thi means the higher the values the less mixed the data is.
# # Calculate metrics for ARCADIA
print("\n" + "=" * 80)
print("CALCULATING METRICS FOR ARCADIA")
print("=" * 80)
mtrc.mixing_metric_parallel(combined_latent_arcadia, group_key="modality", rep_key="X")
if dataset_name == "tonsil":
    assign_rna_cn_from_protein(adata_latent_arcadia_rna, adata_latent_arcadia_prot, latent_key="X")
    assign_rna_cn_from_protein(adata_latent_other_rna, adata_latent_other_prot, latent_key="X")

if plot_flag:
    plot_individual_and_combined_umaps(adata_rna_arcadia, adata_prot_arcadia, "arcadia")
    plot_individual_and_combined_umaps(adata_rna_other, adata_prot_other, other_model_name)

# %%

results_df_arcadia = calculate_single_model_metrics(
    adata_latent_arcadia_rna,
    adata_latent_arcadia_prot,
    combined_latent_arcadia,
    model_name="arcadia",
    combined_distances=combined_distances_arcadia,
    cross_modal_distances=cross_modal_distances_arcadia,
    metrics_funcs_two_modalities=metrics_funcs_two_modalities,
    metric_funcs_one_modality=metric_funcs_one_modality,
    metric_funcs_combined_modalities=metric_funcs_combined_modalities,
    metrics_kwargs=metrics_kwargs,
    plot_flag=plot_flag,
)

# Calculate metrics for other model
print("\n" + "=" * 80)
print(f"CALCULATING METRICS FOR {other_model_name.upper()}")
print("=" * 80)
results_df_other = pd.DataFrame({})
results_df_other = calculate_single_model_metrics(
    adata_latent_other_rna,
    adata_latent_other_prot,
    combined_latent_other,
    model_name=other_model_name,
    combined_distances=combined_distances_other,
    cross_modal_distances=cross_modal_distances_other,
    metrics_funcs_two_modalities=metrics_funcs_two_modalities,
    metric_funcs_one_modality=metric_funcs_one_modality,
    metric_funcs_combined_modalities=metric_funcs_combined_modalities,
    metrics_kwargs=metrics_kwargs,
    plot_flag=plot_flag,
)
results_df_other.to_csv(
    metrics_dir / f"results_comparison_{other_model_name}_{dataset_name}_{timestamp_str}.csv",
    index=False,
)
print(
    f'saved results_df_other to {metrics_dir / f"results_comparison_{other_model_name}_{dataset_name}_{timestamp_str}.csv"}'
)
# %%
# Merge results and create comparison DataFrame
print("\n" + "=" * 80)
print("MERGING RESULTS AND CREATING COMPARISON")
print("=" * 80)
results_pivot = merge_model_results(
    results_df_arcadia, results_df_other, other_model_name=other_model_name
)

# Add metadata columns
results_pivot["timestamp"] = timestamp_str
if args.experiment_id:
    results_pivot["experiment_id"] = args.experiment_id
if args.run_id:
    results_pivot["run_id"] = args.run_id
results_pivot["dataset_name"] = dataset_name
results_pivot["other_model_name"] = other_model_name
results_pivot["checkpoint_path"] = str(checkpoint_path)
results_pivot["n_rna_cells"] = adata_latent_arcadia_rna.n_obs
results_pivot["n_protein_cells"] = adata_latent_arcadia_prot.n_obs

# Reorder columns to put metadata first
cols = [
    "timestamp",
    "experiment_id",
    "run_id",
    "dataset_name",
    "other_model_name",
    "checkpoint_path",
    "n_rna_cells",
    "n_protein_cells",
    "metric_name",
] + [
    c
    for c in results_pivot.columns
    if c
    not in [
        "timestamp",
        "experiment_id",
        "run_id",
        "dataset_name",
        "other_model_name",
        "checkpoint_path",
        "n_rna_cells",
        "n_protein_cells",
        "metric_name",
    ]
]
cols = [c for c in cols if c in results_pivot.columns]
results_pivot = results_pivot[cols]


# Save results (without timestamp so results accumulate in same file)
output_file = metrics_dir / f"results_comparison_{dataset_name}_{other_model_name}.csv"
save_comparison_results(results_pivot, output_file=str(output_file))

print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print(results_pivot.to_string(index=False))
print("=" * 80)


# %%

if plot_flag:
    sc.pp.neighbors(adata_latent_arcadia_rna, n_neighbors=15)
    sc.tl.umap(adata_latent_arcadia_rna)
    sc.pp.neighbors(adata_latent_arcadia_prot, n_neighbors=15)
    sc.tl.umap(adata_latent_arcadia_prot)
    plot_umap_latent_arcadia(adata_latent_arcadia_rna)
    plot_umap_latent_arcadia(adata_latent_arcadia_prot)
    # %%
    sc.pp.neighbors(adata_latent_other_rna, n_neighbors=15)
    sc.tl.umap(adata_latent_other_rna)
    sc.pp.neighbors(adata_latent_other_prot, n_neighbors=15)
    sc.tl.umap(adata_latent_other_prot)
    plot_umap_latent_arcadia(adata_latent_other_rna)
    plot_umap_latent_arcadia(adata_latent_other_prot)
    # %%

    plot_umap_per_cell_type(adata_latent_arcadia_rna, "RNA", "arcadia")
    plot_umap_per_cell_type(adata_latent_arcadia_prot, "Protein", "arcadia")
    plot_umap_per_cell_type(adata_latent_other_rna, "RNA", other_model_name)
    plot_umap_per_cell_type(adata_latent_other_prot, "Protein", other_model_name)
    # %%

    # Create proper adata with both embeddings in obsm
    adata_compare = combined_latent_arcadia.copy()
    adata_compare.obsm["arcadia"] = combined_latent_arcadia.X
    adata_compare.obsm[other_model_name] = combined_latent_other.X

    # Compute and report Moran's I for arcadia vs other model on combined data
    morans_arcadia = mtrc.morans_i(
        adata_compare, score_key="matched_archetype_weight", use_rep="arcadia", n_neighbors=15
    )
    morans_other = mtrc.morans_i(
        adata_compare,
        score_key="matched_archetype_weight",
        use_rep=other_model_name,
        n_neighbors=15,
    )

    print(f"Moran's I (arcadia): {morans_arcadia:.4f}")
    print(f"Moran's I ({other_model_name}): {morans_other:.4f}")
    print(
        f"Winner (higher is better): {'arcadia' if morans_arcadia > morans_other else other_model_name}"
    )

    plot_morans_i(
        adata_compare, "matched_archetype_weight", "arcadia", other_model_name, n_neighbors=15
    )

    # %%

