set -euo pipefail

echo "=== Starting Black-box Model Experiment ==="

echo "Setting GPU environment variables..."

export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_PATH=/tmp/cuda_cache
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false 
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
# === end of your original block ===

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# (Optional) quick sanity checks.
if [ ! -f "main.py" ]; then
  echo "Error: main.py not found in $(pwd)"
  exit 1
fi
if [ ! -f "config/config.yml" ]; then
  echo "Error: config/config.yml not found"
  exit 1
fi
echo "Configuration file found"

# Run your original command.
# Usage examples:
# - Full pipeline:   python main.py --config config/config.yml --phase full --verbose
# - Jailbreak only:  python main.py --config config/config.yml --phase jailbreak --verbose
# - Judge only:      python main.py --config config/config.yml --phase judge --verbose
# - Resume:          python main.py --config config/config.yml --phase resume --verbose
echo "Starting experiment with main.py..."
python main.py --config config/config.yml --phase full --verbose

echo
echo "=== Experiment finished ==="
