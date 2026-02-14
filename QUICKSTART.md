# QUICK START GUIDE

Get up and running with PDNAC-NC in 5 minutes!

---

## 🚀 Super Quick Setup (Copy-Paste)

### Windows
```batch
# 1. Open Command Prompt in the project folder
# 2. Run the setup script
setup.bat

# 3. Run the demo
python demo_pdnac_nc.py
```

### Linux/Mac
```bash
# 1. Open terminal in the project folder
# 2. Make setup script executable
chmod +x setup.sh

# 3. Run the setup script
./setup.sh

# 4. Run the demo
python demo_pdnac_nc.py
```

---

## 📋 Manual Setup (3 Steps)

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Demo
```bash
python demo_pdnac_nc.py
```

---

## ✅ Verify Installation

```python
# Create test.py and run it
import torch
import numpy as np

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("GPU Available:", torch.cuda.is_available())
```

---

## 🎯 What You Get

After running the demo, you should see:
- Training progress with iterations
- Constraint satisfaction metrics
- Final policy performance
- Test episode results

---

## 📦 Files You Need

```
pdnac-nc-project/
├── pdnac_nc_algorithm.py    ← Main algorithm
├── demo_pdnac_nc.py          ← Demo script
├── requirements.txt          ← Dependencies
├── setup.sh                  ← Linux/Mac setup
├── setup.bat                 ← Windows setup
└── README.md                 ← Documentation
```

---

## 🔧 Basic Usage

```python
from pdnac_nc_algorithm import PDNAC_NC, AlgorithmConfig

# 1. Configure
config = AlgorithmConfig(
    K=100,              # Training iterations
    H=50,               # Inner loops
    gamma=0.99,         # Discount factor
    network_width=64    # Neural network size
)

# 2. Create algorithm
algorithm = PDNAC_NC(config, env, policy, features)

# 3. Train
metrics = algorithm.train()
```

---

## 🆘 Problems?

**Virtual environment not activating?**
- Windows: Try `venv\Scripts\activate.bat`
- Mac/Linux: Try `source venv/bin/activate`

**"Module not found" error?**
- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

**Python not found?**
- Install from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add to PATH" during installation

---

## 📚 Next Steps

1. ✅ Run the demo
2. 📖 Read `README.md` for details
3. 🔧 Modify hyperparameters in `AlgorithmConfig`
4. 🎮 Create your own environment
5. 📊 Add visualization code

---

## 💡 Key Hyperparameters

```python
AlgorithmConfig(
    alpha=0.01,        # Policy learning rate
    beta=0.01,         # Constraint learning rate
    gamma=0.99,        # Discount factor (0-1)
    K=1000,            # Training epochs
    H=100,             # Inner iterations
    network_width=128, # Critic size (32-256)
    T_max=1000        # Max trajectory length
)
```

Smaller values → Faster but less accurate
Larger values → Slower but more accurate

---

## 🎓 Learn More

- **Full installation guide**: See `INSTALLATION.md`
- **Algorithm details**: See `README.md`
- **Paper equations**: Check comments in code
- **Examples**: Run `demo_pdnac_nc.py`

---

**That's it! You're ready to go! 🎉**

For detailed instructions, see `INSTALLATION.md`
