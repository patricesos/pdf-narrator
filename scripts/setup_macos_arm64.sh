#!/usr/bin/env bash
# One-shot setup script for arm64 macOS (Apple Silicon)
# Usage:
#   chmod +x scripts/setup_macos_arm64.sh
#   ./scripts/setup_macos_arm64.sh
# This script installs Homebrew packages (tesseract, poppler, openjdk),
# installs Miniforge (if missing), creates a conda env `pdf-narrator` with Python 3.10,
# installs PyTorch via conda, and installs the remaining Python dependencies via pip.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="$REPO_DIR/requirements.txt"
PATCHED_REQ="$REPO_DIR/requirements-conda.txt"
MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="pdf-narrator"
PYTHON_VERSION="3.10"

echo "[setup] Starting macOS arm64 setup for pdf-narrator"

# 1) Basic checks
arch="$(uname -m)"
if [ "$arch" != "arm64" ]; then
  echo "[warning] Detected arch: $arch — this script targets arm64 (Apple Silicon). Continue? (y/N)"
  read -r ans || true
  if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
    echo "Aborting. Run this script on an Apple Silicon mac or modify as needed."
    exit 1
  fi
fi

# 2) Install Homebrew if missing
if ! command -v brew >/dev/null 2>&1; then
  echo "[info] Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  echo "[info] Homebrew installed. Make sure your shell environment is configured as the installer suggests."
fi

# 3) Install system packages
echo "[info] Installing required system packages via brew: tesseract, poppler, openjdk, pkg-config"
brew update
brew install tesseract poppler openjdk pkg-config cmake || true

# 4) Ensure Java is available for Apache Tika
if ! command -v java >/dev/null 2>&1; then
  echo "[info] Setting JAVA_HOME for this session (brew openjdk)
  export JAVA_HOME=$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home"
fi

# 5) Install Miniforge (if missing)
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "[info] Miniforge not found. Installing Miniforge (arm64)..."
  tmp_installer="/tmp/Miniforge3-MacOSX-arm64.sh"
  curl -fsSL -o "$tmp_installer" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
  bash "$tmp_installer" -b -p "$MINIFORGE_DIR"
  rm -f "$tmp_installer"
  echo "[info] Miniforge installed at $MINIFORGE_DIR"
fi

CONDA_BIN="$MINIFORGE_DIR/bin/conda"
if [ ! -x "$CONDA_BIN" ]; then
  echo "[error] conda not found at $CONDA_BIN"
  exit 1
fi

# 6) Create conda env with Python 3.10 (if missing)
if ! "$CONDA_BIN" env list | grep -q "$ENV_NAME"; then
  echo "[info] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=$PYTHON_VERSION
fi

# 7) Install PyTorch (Apple silicon friendly) using conda
echo "[info] Installing PyTorch, torchvision, torchaudio into conda env '$ENV_NAME'"
# Use conda channels (pytorch + conda-forge). For macOS arm64 this will pick compatible builds.
"$CONDA_BIN" install -y -n "$ENV_NAME" -c pytorch -c conda-forge pytorch torchvision torchaudio || true

# 8) Prepare patched requirements: remove torch packages (we installed via conda)
if [ -f "$REQ_FILE" ]; then
  echo "[info] Creating patched requirements file without torch/torchvision/torchaudio: $PATCHED_REQ"
  grep -Ev '^(torch|torchvision|torchaudio)(==|>=|~=|<=)' "$REQ_FILE" > "$PATCHED_REQ" || true
else
  echo "[warning] $REQ_FILE not found — skipping pip install of requirements"
fi

# 9) Install Python packages in the conda env via pip
echo "[info] Upgrading pip and installing Python dependencies into '$ENV_NAME' using pip"
"$CONDA_BIN" run -n "$ENV_NAME" pip install --upgrade pip
if [ -f "$PATCHED_REQ" ]; then
# 7) Install required native image libraries into the conda env (fixes torchvision image.so errors)
echo "[info] Installing native image libs (libjpeg-turbo, libpng) into conda env '$ENV_NAME' via conda-forge"
"$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge libjpeg-turbo libpng || true

# 8) Install PyTorch (Apple silicon friendly) using conda
echo "[info] Installing PyTorch, torchvision, torchaudio into conda env '$ENV_NAME'"
# Use conda channels (pytorch + conda-forge). For macOS arm64 this will pick compatible builds.
"$CONDA_BIN" install -y -n "$ENV_NAME" -c pytorch -c conda-forge pytorch torchvision torchaudio || true
cat <<'EOF'

[done] Setup finished.
Next steps:
  - Activate the environment:
      source "$HOME/miniforge3/bin/activate" && conda activate pdf-narrator
    or
      $CONDA_BIN" run -n pdf-narrator <command>

Notes:
  - If you see issues with binary packages (opencv, soundfile), consider installing them via conda-forge instead:
      conda install -n pdf-narrator -c conda-forge opencv soundfile

  - If you prefer pip-only workflow, remove torch/torchvision/torchaudio entries from requirements.txt and
    run pip install with the official PyTorch CPU wheels as described in the README.


If you want, run the GUI with:
  python main.py
EOF

exit 0
