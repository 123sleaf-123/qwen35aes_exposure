#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="$SCRIPT_DIR/qwen35vl_env.yml"
ENV_NAME="qwen35vl"

if conda env list | awk '{print $1}' | grep -Ex "$ENV_NAME" >/dev/null 2>&1; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
python - <<'PY'
import importlib
mods = [
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "torchvision",
    "PIL",
]
for name in mods:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK {version}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
PY