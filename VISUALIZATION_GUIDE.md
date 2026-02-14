# 📊 VISUALIZATION GUIDE
## How to Generate Plots Like Your Reference Images

This guide shows you exactly how to reproduce the training curves and comparison plots from your reference images.

---

## 🎯 What You'll Get

After following this guide, you'll generate:

1. **Training Curves** - Individual algorithm performance over time
2. **Comparison Plots** - Multiple algorithms side-by-side (like your reference images)
3. **Confidence Intervals** - Shaded regions showing variance across seeds
4. **Publication-Ready Figures** - High-resolution PNG and PDF files

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install additional dependencies
pip install matplotlib seaborn

# 2. Run single algorithm with visualization
python train_with_visualization.py

# 3. Run multi-algorithm comparison
python compare_algorithms.py
```

**That's it!** Your plots will be in the `results/` folder.

---

## 📁 New Files You Have

1. **train_with_visualization.py** - Enhanced training with comprehensive logging
2. **compare_algorithms.py** - Multi-algorithm comparison (generates plots like your images)

---

## 🎨 Method 1: Single Algorithm Training with Plots

### Basic Usage

```bash
python train_with_visualization.py
```

This will:
- Train PDNAC-NC for 100 iterations
- Run 3 different random seeds
- Generate 6 subplots showing:
  - Cumulative reward
  - Constraint cost
  - Dual variable (λ)
  - Critic loss
  - Policy gradient norm
  - Constraint violation rate

### Output Structure

```
results/
└── pdnac_nc_hopper_seed0/
    ├── plots/
    │   └── training_curves.png    ← Main plot
    ├── checkpoints/
    └── metrics.json               ← Raw data
```

### Customize the Training

Edit `train_with_visualization.py`:

```python
# Change number of iterations
config = AlgorithmConfig(
    K=200,              # More iterations
    H=50,               # More inner loops
    # ... other parameters
)

# Change number of seeds
results = run_experiment_with_visualization(
    # ...
    num_seeds=5,        # Run 5 seeds instead of 3
    # ...
)

# Change environment
env_kwargs = {
    'goal_position': np.array([2.0, 2.0]),  # Different goal
    'cost_threshold': 1.5,                   # Tighter constraint
}
```

---

## 🔬 Method 2: Multi-Algorithm Comparison (Like Your Reference Images)

This generates plots **exactly like your reference images** - comparing multiple algorithms with shaded confidence intervals.

### Basic Usage

```bash
python compare_algorithms.py
```

This will:
- Train 3 variants of PDNAC-NC (different dual step sizes)
- Train 5 baseline algorithms (PG, NPG, TRPO, PPO, PDNAC-NC)
- Generate comparison plots with confidence bands
- Run multiple seeds per algorithm

### Output Structure

```
results/
├── comparison_variants/
│   └── plots/
│       ├── comparison_Hopper-v3.png     ← 3-panel comparison
│       └── single_comparison_Hopper-v3.png  ← Single plot
└── comparison_baselines/
    └── plots/
        ├── comparison_Hopper-v3.png
        └── single_comparison_Hopper-v3.png
```

### The Generated Plots Look Like This:

**Plot 1: Three-Panel Comparison** (similar to your Image 1)
```
[Return Plot] [Cost Plot] [Lambda Plot]
NAC-DD-1 (blue) NAC-DD-3 (orange) NAC-DD-5 (green)
```

**Plot 2: Single-Panel Comparison** (similar to your Image 2)
```
All algorithms on one plot with confidence bands
PG, NPG, TRPO, PPO, PDNAC-NC
```

---

## 🎛️ Customization Options

### Option 1: Compare Different Hyperparameters

Edit `create_algorithm_variants()` in `compare_algorithms.py`:

```python
def create_algorithm_variants():
    configs = {}
    
    # Try different learning rates
    configs['PDNAC-alpha-0.001'] = AlgorithmConfig(
        alpha=0.001,  # Small
        # ...
    )
    
    configs['PDNAC-alpha-0.01'] = AlgorithmConfig(
        alpha=0.01,   # Large
        # ...
    )
    
    # Try different network sizes
    configs['PDNAC-width-32'] = AlgorithmConfig(
        network_width=32,
        # ...
    )
    
    configs['PDNAC-width-128'] = AlgorithmConfig(
        network_width=128,
        # ...
    )
    
    return configs
```

### Option 2: Compare on Different Environments

```python
# Create multiple environments
environments = {
    'Hopper': SimpleConstrainedEnv(goal=[1.0, 1.0], cost_threshold=2.0),
    'Walker': SimpleConstrainedEnv(goal=[2.0, 2.0], cost_threshold=1.5),
    'HalfCheetah': SimpleConstrainedEnv(goal=[1.5, 0.5], cost_threshold=1.8),
}

# Run comparison on each
for env_name, env_kwargs in environments.items():
    results = run_algorithm_comparison(
        # ...
        env_kwargs=env_kwargs,
        # ...
    )
```

### Option 3: Adjust Plot Style

Edit `generate_comparison_plots()` in `compare_algorithms.py`:

```python
# Change colors
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Change plot size
fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Wider

# Change line styles
ax.plot(iterations, rewards, 
        linestyle='--',  # Dashed
        linewidth=3,     # Thicker
        marker='o',      # Add markers
        markersize=4)

# Change confidence band opacity
ax.fill_between(iterations, lower, upper, alpha=0.3)  # More transparent
```

---

## 📊 Understanding the Plots

### Plot Elements

1. **Solid Lines** - Mean performance across seeds
2. **Shaded Regions** - Standard deviation (confidence interval)
3. **X-axis** - "Number of Outer Loops" = training iterations
4. **Y-axis** - Performance metric (Return, Cost, Lambda)

### What to Look For

**Good Training:**
- ✅ Return increases steadily
- ✅ Cost approaches 0 from below (constraint satisfied)
- ✅ Lambda stabilizes at a positive value

**Problems:**
- ❌ Return decreases or oscillates wildly
- ❌ Cost stays very negative (constraint violated)
- ❌ Lambda keeps increasing (constraint can't be satisfied)

---

## 🎯 Exact Reproduction of Your Reference Images

### Reference Image 1: Three Environments Comparison

```python
# Run this code
environments = [
    ('Hopper-v3', {'goal': [1.0, 1.0], 'cost_threshold': 2.0}),
    ('HalfCheetah-v3', {'goal': [2.0, 0.5], 'cost_threshold': 1.5}),
    ('Walker2d-v3', {'goal': [1.5, 1.5], 'cost_threshold': 1.8}),
]

for env_name, env_kwargs in environments:
    env_kwargs['env_name'] = env_name
    
    configs = create_algorithm_variants()  # NAC-DD-1, NAC-DD-3, NAC-DD-5
    
    results = run_algorithm_comparison(
        env_class=SimpleConstrainedEnv,
        env_kwargs=env_kwargs,
        algorithms_config=configs,
        num_seeds=5,
        save_dir=f"results/{env_name}_comparison"
    )
```

### Reference Image 2: Algorithm Comparison

```python
# Compare against baselines
configs = {
    'PDNAC-NC': config_pdnac,
    'PG': config_pg,
    'NPG': config_npg,
    'TRPO': config_trpo,
    'PPO': config_ppo,
}

results = run_algorithm_comparison(
    env_class=SimpleConstrainedEnv,
    env_kwargs={'env_name': 'Hopper-v3', ...},
    algorithms_config=configs,
    num_seeds=5,
    save_dir="results/method_comparison"
)
```

---

## 💾 Saving and Loading Results

### Save Raw Data

All metrics are automatically saved as JSON:

```python
# Located at: results/experiment_name/metrics.json
{
  "training": {
    "iteration": [0, 10, 20, ...],
    "episode_reward": [100, 150, 200, ...],
    "cost_estimate": [-0.5, -0.2, 0.1, ...],
    "lambda": [0.0, 0.1, 0.15, ...]
  }
}
```

### Load and Re-plot

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('results/experiment/metrics.json', 'r') as f:
    data = json.load(f)

# Plot
plt.plot(data['training']['iteration'], 
         data['training']['episode_reward'])
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.savefig('my_custom_plot.png', dpi=300)
```

---

## 🎨 Publication-Quality Figures

### High Resolution

```python
# In compare_algorithms.py, change DPI
plt.savefig('plot.png', dpi=300)  # Good for papers
plt.savefig('plot.png', dpi=600)  # Publication quality
```

### Vector Format (PDF)

```python
plt.savefig('plot.pdf')  # Scalable, best for LaTeX
```

### Custom Font Sizes

```python
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
})
```

---

## 🔧 Troubleshooting

### Issue: "Module matplotlib not found"

```bash
pip install matplotlib seaborn
```

### Issue: Plots don't show up

```python
# Add this at the end of your script
plt.show()

# Or force display
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Issue: Out of memory with many seeds

```python
# Reduce number of seeds
num_seeds=3  # Instead of 5

# Or reduce iterations per seed
config.K = 100  # Instead of 200
```

### Issue: Training takes too long

```python
# Reduce inner iterations
config.H = 10  # Instead of 50

# Reduce trajectory length
config.T_max = 50  # Instead of 200

# Use fewer evaluation episodes
eval_frequency = 20  # Instead of 10
```

---

## 📈 Advanced: Custom Metrics

### Add Your Own Metrics

Edit `EnhancedPDNAC_NC.evaluate_policy()` in `train_with_visualization.py`:

```python
def evaluate_policy(self, num_episodes: int = 10):
    # ... existing code ...
    
    # Add custom metric
    max_speeds = []
    for _ in range(num_episodes):
        # ... collect data ...
        max_speeds.append(max_speed)
    
    return {
        # ... existing metrics ...
        'max_speed': np.mean(max_speeds),
    }
```

Then plot it:

```python
# In plot_training_curves()
if 'max_speed' in self.metrics:
    ax.plot(iterations, self.metrics['max_speed'])
    ax.set_title('Maximum Speed')
```

---

## 📋 Complete Workflow Example

```bash
# 1. Install dependencies
pip install matplotlib seaborn

# 2. Quick test - single algorithm
python train_with_visualization.py

# 3. Full comparison - multiple algorithms
python compare_algorithms.py

# 4. Check results
ls results/comparison_*/plots/

# 5. View plots
# Open: results/comparison_variants/plots/comparison_Hopper-v3.png
# Open: results/comparison_baselines/plots/comparison_Hopper-v3.png
```

**Expected Output:**
```
results/
├── comparison_variants/
│   └── plots/
│       ├── comparison_Hopper-v3.png    ← 3-panel comparison
│       └── single_comparison_Hopper-v3.png
└── comparison_baselines/
    └── plots/
        ├── comparison_Hopper-v3.png    ← Method comparison
        └── single_comparison_Hopper-v3.png
```

---

## ✅ Summary

**To get plots like your reference images:**

1. **Install**: `pip install matplotlib seaborn`
2. **Run**: `python compare_algorithms.py`
3. **View**: Check `results/comparison_*/plots/`

**Files generated:**
- ✅ PNG files (high res, for viewing)
- ✅ PDF files (vector, for papers)
- ✅ JSON files (raw data, for analysis)

**Customization:**
- Change algorithms in `create_algorithm_variants()`
- Adjust plot style in `generate_comparison_plots()`
- Modify evaluation in `EnhancedPDNAC_NC.evaluate_policy()`

---

## 🎓 Tips for Best Results

1. **Run multiple seeds** (3-5) for smooth curves
2. **Train longer** (K=200+) for convergence
3. **Adjust learning rates** for your environment
4. **Save frequently** to avoid losing progress
5. **Use descriptive names** for experiments

---

**You're ready to generate beautiful training plots!** 🎉

Check the generated images in `results/` folder!
