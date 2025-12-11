#!/usr/bin/env bash
set -e  # Exit on error

# Parse command line arguments
DATASET_NAME="${1:-cite_seq}"  # Default to cite_seq if not provided

# Change to ARCADIA directory
cd "$(dirname "$0")/ARCADIA"

# Run the pipeline script with dataset name
bash run_pipeline.sh "${DATASET_NAME}"

exit 0
