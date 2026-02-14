#!/bin/bash
# Setup script for PDNAC-NC Algorithm (Linux/Mac)
# Automatically creates virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "PDNAC-NC Algorithm Setup Script"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python version: $python_version"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ New virtual environment created"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Check if requirements file exists
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ All dependencies installed"
elif [ -f "requirements-minimal.txt" ]; then
    echo "📥 Installing minimal dependencies..."
    pip install -r requirements-minimal.txt
    echo "✓ Minimal dependencies installed"
else
    echo "📥 Installing core dependencies manually..."
    pip install torch>=2.0.0 numpy>=1.21.0
    echo "✓ Core dependencies installed"
fi
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 << EOF
import sys
try:
    import torch
    import numpy as np
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"✓ NumPy {np.__version__} installed")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF
echo ""

# Create activation reminder
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "  python demo_pdnac_nc.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "=========================================="
