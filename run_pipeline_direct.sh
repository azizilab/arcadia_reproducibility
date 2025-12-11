#!/usr/bin/env bash
set -e  # Exit on error

# Parse command line arguments
DATASET_NAME="${1:-cite_seq}"  # Default to cite_seq if not provided

# Change to ARCADIA directory
cd "$(dirname "$0")/ARCADIA"

# Save original plot_flag and set to false
CONFIG_FILE="configs/config.json"
ORIGINAL_PLOT_FLAG=$(python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); print(str(d.get('plot_flag', True)).lower())")
python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); d['plot_flag']=False; f=open('${CONFIG_FILE}','w'); json.dump(d, f, indent=2); f.close()"

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate scvi

echo "=========================================="
echo "Running ARCADIA Pipeline (Direct Python Execution)"
echo "Dataset: ${DATASET_NAME}"
echo "=========================================="

# Step 0: Preprocess dataset
echo ""
echo "=== Step 0: Preprocessing Dataset ==="
python scripts/_0_preprocess_${DATASET_NAME}.py

# Step 1: Align datasets
echo ""
echo "=== Step 1: Aligning Datasets ==="
python scripts/_1_align_datasets.py --dataset_name "${DATASET_NAME}"

# Step 2: Spatial integration
echo ""
echo "=== Step 2: Spatial Integration ==="
python scripts/_2_spatial_integrate.py --dataset_name "${DATASET_NAME}"

# Step 3: Generate archetypes
echo ""
echo "=== Step 3: Generating Archetypes ==="
python scripts/_3_generate_archetypes.py --dataset_name "${DATASET_NAME}"

# Step 4: Prepare training data
echo ""
echo "=== Step 4: Preparing Training Data ==="
python scripts/_4_prepare_training.py --dataset_name "${DATASET_NAME}"

# Clean up scales cache if it exists
[ -f scales_cache.json ] && rm scales_cache.json

# Step 5: Train VAE (using hyperparameter search)
echo ""
echo "=== Step 5: Training VAE (Hyperparameter Search) ==="
python scripts/_5_train_vae.py

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

# Restore original plot_flag
python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); d['plot_flag']='${ORIGINAL_PLOT_FLAG}' == 'true'; f=open('${CONFIG_FILE}','w'); json.dump(d, f, indent=2); f.close()"

exit 0

