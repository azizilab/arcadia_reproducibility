#!/usr/bin/env bash
# Comprehensive Test Suite for ARCADIA Reproducibility
# Tests all execution methods mentioned in README.md

# Ensure script is run with bash
if [ -z "$BASH_VERSION" ]; then
    echo "ERROR: This script must be run with bash, not sh or other shells." >&2
    echo "Please run: bash $0" >&2
    exit 1
fi

set -eo pipefail  # Removed -u to allow unbound variables (needed for conda activation)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ARCADIA_DIR="$REPO_ROOT/ARCADIA"
MODEL_COMPARISON_DIR="$REPO_ROOT/model_comparison"
TEST_RESULTS_DIR="$REPO_ROOT/test_results"

# Test results tracking
declare -a TEST_NAMES
declare -a TEST_STATUSES
declare -a TEST_LOGS

# Initialize test results directory
setup_logging() {
    mkdir -p "$TEST_RESULTS_DIR"
    echo "Test results will be saved to: $TEST_RESULTS_DIR"
}

# Source conda
source_conda() {
    set +e
    set +u
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
    fi
    set -e
}

# Check prerequisites
check_prerequisites() {
    local missing=0
    
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Conda
    source_conda
    if command -v conda &> /dev/null || [ -f /opt/conda/bin/conda ] || [ -f "$HOME/miniconda3/bin/conda" ] || [ -f "$HOME/anaconda3/bin/conda" ]; then
        echo -e "${GREEN}✓ Conda found${NC}"
    else
        echo -e "${YELLOW}⚠ Conda not found (required for conda/env tests)${NC}"
        missing=$((missing + 1))
    fi
    
    # Check ARCADIA directory
    if [ ! -d "$ARCADIA_DIR" ]; then
        echo -e "${RED}✗ ARCADIA directory not found: $ARCADIA_DIR${NC}"
        missing=$((missing + 1))
    else
        echo -e "${GREEN}✓ ARCADIA directory found${NC}"
    fi
    
    # Check required scripts
    if [ ! -f "$REPO_ROOT/run_pipeline_direct.sh" ]; then
        echo -e "${RED}✗ run_pipeline_direct.sh not found${NC}"
        missing=$((missing + 1))
    else
        echo -e "${GREEN}✓ run_pipeline_direct.sh found${NC}"
    fi
    
    if [ ! -f "$REPO_ROOT/run_pipeline_notebooks.sh" ]; then
        echo -e "${RED}✗ run_pipeline_notebooks.sh not found${NC}"
        missing=$((missing + 1))
    else
        echo -e "${GREEN}✓ run_pipeline_notebooks.sh found${NC}"
    fi
    
    # Check model comparison scripts
    if [ ! -f "$MODEL_COMPARISON_DIR/model_scmodal_dataset_cite_seq.py" ]; then
        echo -e "${YELLOW}⚠ scMODAL comparison scripts not found${NC}"
    else
        echo -e "${GREEN}✓ scMODAL comparison scripts found${NC}"
    fi
    
    if [ ! -f "$MODEL_COMPARISON_DIR/model_maxfuse_dataset_cite_seq.py" ]; then
        echo -e "${YELLOW}⚠ MaxFuse comparison scripts not found${NC}"
    else
        echo -e "${GREEN}✓ MaxFuse comparison scripts found${NC}"
    fi
    
    echo ""
    return $missing
}

# Run a test and capture results
run_test() {
    local test_name="$1"
    local test_func="$2"
    # Replace spaces with underscores in log file name
    local test_name_safe=$(echo "$test_name" | tr ' ' '_')
    local log_file="$TEST_RESULTS_DIR/test_${test_name_safe}.log"
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $test_name${NC}"
    echo -e "${BLUE}Log: $log_file${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Record test name and log file
    TEST_NAMES+=("$test_name")
    TEST_LOGS+=("$log_file")
    
    # Run test with error handling (don't exit on failure)
    set +e
    {
        echo "Test started: $(date)"
        echo "Test: $test_name"
        echo "========================================"
        $test_func
        local exit_code=$?
        echo "========================================"
        echo "Test finished: $(date)"
        echo "Exit code: $exit_code"
    } > "$log_file" 2>&1
    local exit_code=$?
    set -e
    
    # Record status
    if [ $exit_code -eq 0 ]; then
        TEST_STATUSES+=("PASS")
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
    else
        TEST_STATUSES+=("FAIL")
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        echo -e "${YELLOW}   Check log: $log_file${NC}"
    fi
    
    return $exit_code
}

# Test 1: Docker execution
test_docker_execution() {
    echo "Testing ARCADIA Docker execution..."
    
    cd "$ARCADIA_DIR" || return 1
    
    # Build Docker image if needed (run_docker.sh will check and skip if exists)
    cd environments/docker || return 1
    bash run_docker.sh test || return 1
    cd "$ARCADIA_DIR" || return 1
    
    # Run pipeline inside Docker for both datasets
    cd "$ARCADIA_DIR/environments/docker" || return 1
    
    echo "Running Docker pipeline for cite_seq..."
    bash run_docker.sh pipeline cite_seq || return 1
    
    echo "Running Docker pipeline for tonsil..."
    bash run_docker.sh pipeline tonsil || return 1
    
    echo "Docker execution test completed successfully"
}

# Test 2: Requirements.txt installation
test_requirements_installation() {
    echo "Testing requirements.txt installation..."
    
    # Check if requirements.txt exists
    if [ ! -f "$ARCADIA_DIR/environments/requirements.txt" ]; then
        echo "ERROR: requirements.txt not found at $ARCADIA_DIR/environments/requirements.txt"
        return 1
    fi
    
    # Create temporary virtual environment
    local venv_dir="$TEST_RESULTS_DIR/venv_requirements"
    if [ -d "$venv_dir" ]; then
        rm -rf "$venv_dir"
    fi
    
    python3 -m venv "$venv_dir" || return 1
    source "$venv_dir/bin/activate" || return 1
    
    # Install requirements
    pip install --upgrade pip || return 1
    
    # Install PyTorch from PyTorch index first (for CUDA versions)
    echo "Installing PyTorch from PyTorch index..."
    pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 || {
        echo "WARNING: Failed to install PyTorch from PyTorch index, trying PyPI..."
        pip install torch==2.2.0 torchaudio==2.2.0 || return 1
    }
    
    echo "Installing remaining requirements..."
    grep -v "^torch==\|^torchaudio==" "$ARCADIA_DIR/environments/requirements.txt" | pip install -r /dev/stdin || return 1
    
    echo "Requirements installed successfully"
    deactivate
    
    echo "Requirements installation test completed"
}

# Test 3: Conda environment YAML
test_conda_environments() {
    echo "Testing conda environment YAML..."
    
    local env_name="scvi"
    source_conda
    
    # Check if scvi environment already exists
    if conda env list | grep -q "^$env_name "; then
        echo "Environment $env_name already exists, using existing environment"
    else
        # Check if environment YAML exists
        if [ ! -f "$ARCADIA_DIR/environments/environment_gpu_cuda12.1.yaml" ]; then
            echo "ERROR: environment_gpu_cuda12.1.yaml not found and scvi environment doesn't exist"
            return 1
        fi
        
        echo "Creating conda environment $env_name from YAML..."
        
        # Create environment - disable error exit temporarily
        set +e
        conda env create -f "$ARCADIA_DIR/environments/environment_gpu_cuda12.1.yaml" > /tmp/conda_create.log 2>&1
        local create_exit=$?
        set -e
        
        if [ $create_exit -ne 0 ]; then
            echo "ERROR: Failed to create conda environment"
            echo "Conda error output:"
            cat /tmp/conda_create.log | grep -v "__vsc_prompt_cmd_original" || true
            rm -f /tmp/conda_create.log
            return 1
        fi
        rm -f /tmp/conda_create.log
        echo "Environment $env_name created successfully"
    fi
    
    set +e
    set +u
    conda activate "$env_name" 2>&1 | grep -v "__vsc_prompt_cmd_original" > /dev/null || true
    set -e
    
    # Just proceed - if activation failed, the next commands will fail anyway
    
    # Run direct pipeline (using conda environment)
    echo "Running direct pipeline with conda environment..."
    cd "$REPO_ROOT" || return 1
    bash run_pipeline_direct.sh cite_seq || return 1
    
    # Deactivate (but don't remove the environment - keep it for future use)
    conda deactivate
    
    echo "Conda environment test completed successfully"
}

# Test 4: Notebook execution
test_notebook_execution() {
    echo "Testing notebook execution..."
    
    source_conda
    
    if conda env list | grep -q "scvi"; then
        conda activate scvi || return 1
    else
        echo "WARNING: scvi environment not found, trying base"
        conda activate base || return 1
    fi
    
    cd "$REPO_ROOT" || return 1
    
    # Run notebook pipeline
    echo "Running notebook pipeline for cite_seq..."
    bash run_pipeline_notebooks.sh cite_seq || return 1
    
    conda deactivate
    
    echo "Notebook execution test completed successfully"
}

# Test 6: scMODAL comparisons
test_scmodal_comparisons() {
    echo "Testing scMODAL comparisons with Docker..."
    
    # Check if scMODAL is cloned
    if [ ! -d "$MODEL_COMPARISON_DIR/scMODAL_main" ]; then
        echo "ERROR: scMODAL not cloned. Please run:"
        echo "  cd model_comparison"
        echo "  git clone https://github.com/gefeiwang/scMODAL.git scMODAL_main"
        return 1
    fi
    
    # Check if Dockerfile exists
    if [ ! -f "$MODEL_COMPARISON_DIR/Dockerfile.scmodal" ]; then
        echo "ERROR: Dockerfile.scmodal not found"
        return 1
    fi
    
    cd "$MODEL_COMPARISON_DIR" || return 1
    
    # Build Docker image
    echo "Building scMODAL Docker image..."
    docker build -f Dockerfile.scmodal -t scmodal:latest . || return 1
    
    # Set up directories and permissions for Docker write access
    echo "Setting up permissions for Docker write access..."
    mkdir -p "$MODEL_COMPARISON_DIR/CITE-seq_PBMC" \
             "$MODEL_COMPARISON_DIR/scMODAL_tonsil" \
             "$MODEL_COMPARISON_DIR/outputs/scmodal_cite_seq" \
             "$MODEL_COMPARISON_DIR/outputs/scmodal_tonsil"
    chmod -R 777 "$MODEL_COMPARISON_DIR/CITE-seq_PBMC" \
                 "$MODEL_COMPARISON_DIR/scMODAL_tonsil" \
                 "$MODEL_COMPARISON_DIR/outputs" \
                 "$MODEL_COMPARISON_DIR" 2>/dev/null || true
    
    # Determine GPU flag
    local gpu_flag=""
    if command -v nvidia-smi &> /dev/null; then
        gpu_flag="--gpus all"
    fi
    
    # Run scMODAL comparisons inside Docker
    echo "Running scMODAL comparison for cite_seq..."
    docker run --rm $gpu_flag \
        -v "$REPO_ROOT:/workspace" \
        -w /workspace/model_comparison \
        scmodal:latest \
        bash -c "python model_scmodal_dataset_cite_seq.py" || return 1
    
    echo "Running scMODAL comparison for tonsil..."
    docker run --rm $gpu_flag \
        -v "$REPO_ROOT:/workspace" \
        -w /workspace/model_comparison \
        scmodal:latest \
        bash -c "python model_scmodal_dataset_tonsil.py" || return 1
    
    echo "scMODAL comparisons test completed successfully"
}

# Test 7: MaxFuse comparisons
test_maxfuse_comparisons() {
    echo "Testing MaxFuse comparisons..."
    
    source_conda
    
    if conda env list | grep -q "maxfuse"; then
        conda activate maxfuse || return 1
    else
        echo "WARNING: maxfuse environment not found. Please create it first."
        echo "  conda env create -f ARCADIA/environments/environment_maxfuse.yaml"
        return 1
    fi
    
    cd "$MODEL_COMPARISON_DIR" || return 1
    
    # Run MaxFuse comparisons
    echo "Running MaxFuse comparison for cite_seq..."
    python model_maxfuse_dataset_cite_seq.py || return 1
    
    echo "Running MaxFuse comparison for tonsil..."
    python model_maxfuse_dataset_tonsil.py || return 1
    
    conda deactivate
    
    echo "MaxFuse comparisons test completed successfully"
}

# Print summary report
print_summary() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Summary Report${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    local passed=0
    local failed=0
    local total=${#TEST_NAMES[@]}
    
    for i in "${!TEST_NAMES[@]}"; do
        local name="${TEST_NAMES[$i]}"
        local status="${TEST_STATUSES[$i]}"
        local log="${TEST_LOGS[$i]}"
        
        if [ "$status" = "PASS" ]; then
            echo -e "${GREEN}✅ PASS: $name${NC}"
            passed=$((passed + 1))
        else
            echo -e "${RED}❌ FAIL: $name${NC}"
            failed=$((failed + 1))
        fi
        echo -e "   Log: $log"
        echo ""
    done
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total: ${passed}/${total} tests passed"
    if [ $failed -gt 0 ]; then
        echo -e "${RED}Failed: $failed tests${NC}"
    fi
    echo -e "${BLUE}========================================${NC}"
}

# Main execution
main() {
    # Disable exit on error to allow tests to continue even if one fails
    set +e
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}ARCADIA Reproducibility Test Suite${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    setup_logging
    check_prerequisites
    
    # Run all tests
    # run_test "Docker Execution" test_docker_execution
    # run_test "Requirements Installation" test_requirements_installation
    # run_test "Conda Environments" test_conda_environments
    # run_test "Notebook Execution" test_notebook_execution
    run_test "scMODAL Comparisons" test_scmodal_comparisons
    # run_test "MaxFuse Comparisons" test_maxfuse_comparisons
    
    print_summary
    
    # Exit with error if any tests failed
    for status in "${TEST_STATUSES[@]}"; do
        if [ "$status" = "FAIL" ]; then
            exit 1
        fi
    done
    exit 0
}

# Run main function
main
