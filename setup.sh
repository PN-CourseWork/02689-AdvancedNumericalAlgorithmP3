
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "  Project 3 Environment Setup"
echo "  Location: $PROJECT_ROOT"
echo "================================================================"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------
# 1. Create venv if it doesn't exist
# ---------------------------------------------------------------
if [ -d ".venv" ]; then
    echo "[1/5] Existing .venv found — skipping creation."
else
    echo "[1/5] Creating new uv virtual environment (Python 3.12, seeded)..."
    uv venv --python 3.12 --seed .venv
fi


# ---------------------------------------------------------------
# 3. Install pyproject dependencies via uv sync
# ---------------------------------------------------------------
echo "[3/5] Installing pyproject dependencies via uv sync..."
source .venv/bin/activate
uv sync


# ---------------------------------------------------------------
# 4. Install PETSc + petsc4py using pip inside the venv
# ---------------------------------------------------------------
echo "[4/5] Installing PETSc + petsc4py via pip..."

python -m pip install --upgrade pip wheel
python -m pip install petsc petsc4py


# ---------------------------------------------------------------
# 5. Sanity check PETSc
# ---------------------------------------------------------------
echo "[5/5] Verifying PETSc installation..."

python - <<'EOF'
import petsc4py
from petsc4py import PETSc
petsc4py.init()
print(" petsc4py version:", petsc4py.__version__)
print(" PETSc version:", PETSc.Sys.getVersion())
EOF

echo "================================================================"
echo " Setup complete!"
echo "================================================================"
