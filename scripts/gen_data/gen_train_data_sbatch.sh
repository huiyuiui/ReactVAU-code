# ==========================================
# 1. Environment Setup
# ==========================================

# Load project configuration and environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/../config/paths.sh"
CONFIG_PATH="${REACTVAU_CONFIG:-${DEFAULT_CONFIG_PATH}}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

reactvau_activate_conda

# Enable real-time output (unbuffered)
export PYTHONUNBUFFERED=1

# Redirect stderr to stdout for unified logging
exec 2>&1

# ==========================================
# 2. Run Preprocessing
# ==========================================
cd "${REACTVAU_ROOT}/vad" || exit 1

# Create an output directory for logs if it doesn't exist
LOG_ROOT="${LOG_ROOT:-${REACTVAU_ROOT}/logs}"
mkdir -p "${LOG_ROOT}"

echo "=========================================="
echo "Generating training data for Grid Image Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "=========================================="

python generate_paligemma_train_data.py

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "End Time: $(date)"
echo "=========================================="
