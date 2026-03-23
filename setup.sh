#!/usr/bin/env bash
#
# setup.py - Full environment setup for llama.cpp
# 1) Checks for Python
# 2) Uses pyenv to install/activate Python 3.12
# 3) Creates virtual environment
# 4) Installs dependencies
# 5) Runs build_llama.py
#

set -e  # Exit on error

echo "=========================================="
echo "🔧 Llama.cpp Environment Setup Script"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}➜${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 0: Check for Python
echo ""
log_info "Step 0: Checking for Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_info "   Found Python: $PYTHON_VERSION"
else
    log_error "   Python not found!"
    exit 1
fi

# Step 4: Create virtual environment
echo ""
log_info "Step 4: Creating virtual environment..."
if [ -d "venv" ]; then
    log_warn "   Existing 'venv' directory found. Removing..."
    rm -rf venv
fi

python -m venv venv

# Activate the virtual environment
source venv/bin/activate

log_info "   Virtual environment created and activated"

# Step 5: Install dependencies
echo ""
log_info "Step 5: Installing Python dependencies..."
if [ ! -f "requirements.txt" ]; then
    log_error "   requirements.txt not found!"
    exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt

log_info "   Dependencies installed successfully"

# Step 6: Run build_llama.py
echo ""
log_info "Step 6: Running build_llama.py..."
python build_llama.py

log_info ""
log_info "=========================================="
log_info "✅ Setup completed successfully!"
log_info "=========================================="
