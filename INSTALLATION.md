# Installation Guide for PDNAC-NC Algorithm

Complete guide for setting up the environment and running the PDNAC-NC implementation.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [Verification](#verification)
6. [Running the Demo](#running-the-demo)
7. [GPU Support](#gpu-support)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk Space**: 2GB for dependencies

### Recommended
- **Python**: 3.10 or 3.11
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional but faster)

---

## Installation Methods

Choose one of the following methods based on your system:

### Method 1: Using venv (Python Built-in) - RECOMMENDED

#### Windows
```bash
# 1. Open Command Prompt or PowerShell
# 2. Navigate to your project directory
cd path\to\pdnac-nc

# 3. Create virtual environment
python -m venv venv

# 4. Activate virtual environment
venv\Scripts\activate

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# OR install minimal dependencies only
pip install -r requirements-minimal.txt
```

#### Linux/Mac
```bash
# 1. Open terminal
# 2. Navigate to your project directory
cd /path/to/pdnac-nc

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# OR install minimal dependencies only
pip install -r requirements-minimal.txt
```

---

### Method 2: Using Conda/Anaconda

#### All Platforms (Windows/Linux/Mac)
```bash
# 1. Create conda environment with Python 3.10
conda create -n pdnac-nc python=3.10 -y

# 2. Activate environment
conda activate pdnac-nc

# 3. Install PyTorch (choose based on your system)

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# For CUDA 11.8 (NVIDIA GPU):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1 (NVIDIA GPU):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install other dependencies
pip install numpy matplotlib scipy tensorboard tqdm

# OR install from requirements
pip install -r requirements.txt
```

---

### Method 3: Using Poetry (Modern Dependency Management)

```bash
# 1. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Create project
poetry init

# 3. Add dependencies
poetry add torch numpy matplotlib scipy tensorboard tqdm

# 4. Install dependencies
poetry install

# 5. Activate environment
poetry shell
```

---

## Detailed Virtual Environment Setup

### Step-by-Step Guide for Beginners

#### 1. Check Python Installation
```bash
# Check if Python is installed
python --version
# OR
python3 --version

# Should show Python 3.8 or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Mac**: `brew install python3` (requires Homebrew)
- **Linux**: `sudo apt-get install python3 python3-pip` (Ubuntu/Debian)

#### 2. Create Project Directory
```bash
# Create and navigate to project directory
mkdir pdnac-nc-project
cd pdnac-nc-project

# Copy your files here
# - pdnac_nc_algorithm.py
# - demo_pdnac_nc.py
# - requirements.txt
# - README.md
```

#### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv

# Linux/Mac
python3 -m venv venv
```

This creates a `venv` folder containing an isolated Python environment.

#### 4. Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1

# If you get an error about execution policies:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

#### 5. Upgrade pip
```bash
python -m pip install --upgrade pip
```

---

## Installing Dependencies

### Option A: Full Installation (Recommended)
```bash
# Activate your virtual environment first!
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Matplotlib (plotting)
- SciPy (scientific computing)
- TensorBoard (visualization)
- tqdm (progress bars)
- Gymnasium (RL environments)

### Option B: Minimal Installation
```bash
# Only core dependencies
pip install -r requirements-minimal.txt
```

This installs only PyTorch and NumPy (sufficient to run the algorithm).

### Option C: Manual Installation
```bash
# Install one by one
pip install torch>=2.0.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
```

---

## Verification

### 1. Verify Installation
```bash
# Check installed packages
pip list

# Should see:
# torch        2.x.x
# numpy        1.x.x
# ...
```

### 2. Test Python Environment
```python
# Create test file: test_install.py
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Test basic operations
x = torch.randn(3, 3)
y = np.random.randn(3, 3)
print("\nTorch tensor:")
print(x)
print("\nNumPy array:")
print(y)
print("\nInstallation successful! ✓")
```

Run the test:
```bash
python test_install.py
```

### 3. Test PDNAC-NC Import
```python
# Create test file: test_pdnac.py
from pdnac_nc_algorithm import PDNAC_NC, AlgorithmConfig, NeuralCriticNetwork

config = AlgorithmConfig()
print(f"Algorithm configuration: {config}")
print("\nPDNAC-NC modules imported successfully! ✓")
```

Run the test:
```bash
python test_pdnac.py
```

---

## Running the Demo

### Basic Demo
```bash
# Make sure virtual environment is activated
# (venv) should appear in your terminal

python demo_pdnac_nc.py
```

### Expected Output
```
======================================================================
PDNAC-NC ALGORITHM DEMONSTRATION
======================================================================

Paper: 'Global Convergence of Constrained MDPs with Neural Critic
        and General Policy Parameterization'
...
Starting PDNAC-NC training...
Iteration 0/50: λ=0.0000, J_c=-1.2345
Iteration 10/50: λ=0.1234, J_c=-0.5678
...
Training complete!
```

---

## GPU Support

### Installing PyTorch with CUDA

#### Check CUDA Version
```bash
# Linux/Mac
nvidia-smi

# Look for "CUDA Version: X.X"
```

#### Install PyTorch with GPU Support

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only (no GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Using GPU in PDNAC-NC
```python
# In demo_pdnac_nc.py or your code, set:
config = AlgorithmConfig(
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## Deactivating Virtual Environment

When you're done working:
```bash
# Simply type:
deactivate

# This returns you to your system Python
```

---

## Complete Setup Script

### Windows (setup.bat)
```batch
@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete!
echo To activate environment in the future, run: venv\Scripts\activate
pause
```

### Linux/Mac (setup.sh)
```bash
#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate environment in the future, run: source venv/bin/activate"
```

Make executable and run:
```bash
chmod +x setup.sh
./setup.sh
```

---

## Troubleshooting

### Common Issues

#### 1. "python: command not found"
**Solution:**
- Try `python3` instead of `python`
- Install Python from python.org
- Add Python to PATH

#### 2. "pip: command not found"
**Solution:**
```bash
python -m pip install --upgrade pip
# OR
python -m ensurepip --upgrade
```

#### 3. Permission Denied (Mac/Linux)
**Solution:**
```bash
# Don't use sudo with pip in virtual environment
# Make sure virtual environment is activated
source venv/bin/activate
```

#### 4. SSL Certificate Error
**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### 5. PyTorch Installation Fails
**Solution:**
```bash
# Try installing from PyTorch website directly
# Visit: https://pytorch.org/get-started/locally/
# Select your configuration and copy the command
```

#### 6. "venv\Scripts\activate" Not Working (Windows PowerShell)
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

#### 7. Import Error: "No module named 'torch'"
**Solution:**
- Make sure virtual environment is activated
- Reinstall PyTorch: `pip install torch`
- Check: `pip list | grep torch`

#### 8. Out of Memory Error
**Solution:**
```python
# Reduce batch size or network size in config
config = AlgorithmConfig(
    network_width=64,  # Instead of 128
    H=50,              # Instead of 100
)
```

---

## Quick Reference Commands

### Activate Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Packages
```bash
pip install package-name
pip install -r requirements.txt
```

### List Installed Packages
```bash
pip list
pip freeze > requirements.txt  # Save current packages
```

### Update Package
```bash
pip install --upgrade package-name
```

### Uninstall Package
```bash
pip uninstall package-name
```

### Deactivate Environment
```bash
deactivate
```

### Remove Virtual Environment
```bash
# Windows
rmdir /s venv

# Linux/Mac
rm -rf venv
```

---

## Next Steps

After installation:

1. **Run the demo:**
   ```bash
   python demo_pdnac_nc.py
   ```

2. **Read the documentation:**
   - See `README.md` for algorithm details
   - Check function docstrings in `pdnac_nc_algorithm.py`

3. **Customize for your problem:**
   - Create your own environment
   - Adjust hyperparameters in `AlgorithmConfig`
   - Define custom policy networks

4. **Experiment:**
   - Try different network architectures
   - Tune learning rates
   - Add visualization code

---

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Python venv Guide**: https://docs.python.org/3/library/venv.html
- **Conda User Guide**: https://docs.conda.io/
- **pip User Guide**: https://pip.pypa.io/en/stable/user_guide/

---

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify Python version: `python --version`
3. Verify pip version: `pip --version`
4. Check installed packages: `pip list`
5. Try minimal installation first
6. Search error messages online

---

**Happy Training! 🚀**
