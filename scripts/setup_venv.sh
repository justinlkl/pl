#!/usr/bin/env bash
# Create and bootstrap a Python virtual environment (.venv) on Unix-like systems.
# Usage:
#   ./scripts/setup_venv.sh         # creates .venv and installs core requirements
#   ./scripts/setup_venv.sh --ensemble  # also installs ensemble extras

set -euo pipefail

ENSEMBLE=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --ensemble) ENSEMBLE=1; shift ;;
    *) shift ;;
  esac
done

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
else
  echo ".venv already exists"
fi

echo "Activating virtualenv and installing requirements..."
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
if [[ "$ENSEMBLE" -eq 1 ]]; then
  echo "Installing ensemble extras (may require build tools)..."
  python -m pip install -r requirements-ensemble.txt
fi

echo "Done. Activate with: source $VENV_DIR/bin/activate"
