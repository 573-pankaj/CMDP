# PDNAC-NC: Primal-Dual Natural Actor-Critic with Neural Critic

PyTorch implementation of Algorithm 1 from the paper:

**"Global Convergence of Constrained MDPs with Neural Critic and General Policy Parameterization"**

This implementation achieves order-optimal **O(ε⁻²)** sample complexity for constrained Markov Decision Processes (CMDPs) with general policy parameterization and multi-layer neural network critics.

## Overview

### Key Features

- **Constrained RL**: Solves `max J_r(θ)` subject to `J_c(θ) ≥ 0`
- **Neural Critics**: Multi-layer feedforward networks for Q-function approximation
- **NTK Regime**: Projection to maintain Neural Tangent Kernel properties
- **MLMC Estimation**: Multi-Level Monte Carlo for variance reduction
- **Natural Policy Gradient**: Fisher-information-based updates
- **Primal-Dual Method**: Lagrangian approach for constraint handling

### Algorithm Components

1. **Neural Critic Estimation** (Lines 5-16)
   - Minimizes mean-squared projected Bellman error
   - Uses linearized neural network class F_{R,m}
   - Projects parameters to NTK ball S_R

2. **NPG Direction Estimation** (Lines 17-28)
   - Solves `min_ω f_g(θ,ω) = (1/2)ω^T F(θ)ω - ω^T ∇_θ J_g(θ)`
   - Computes natural gradient using Fisher information matrix

3. **Policy and Dual Updates** (Lines 30-33)
   - Primal update: `θ_{k+1} = θ_k + α ω_k`
   - Dual update: `λ_{k+1} = P_{[0,2/δ]}[λ_k - β Ĵ_c^k]`

## Mathematical Formulation

### Problem Setup

**CMDP**: M = (S, A, r, c, P, ρ, γ)
- State space: S
- Action space: A  
- Reward function: r : S × A → [0,1]
- Cost function: c : S × A → [-1,1]
- Transition kernel: P : S × A → Δ(S)
- Initial distribution: ρ ∈ Δ(S)
- Discount factor: γ ∈ [0,1)

**Objective**:
```
max_θ J_r(θ) = E_π_θ [Σ_{t=0}^∞ γ^t r(s_t, a_t)]
subject to J_c(θ) ≥ 0
```

### Neural Critic Architecture

**Network Structure** (Equation 12):
```
x^(l) = (1/√m) σ(W_l x^(l-1))  for l ∈ {1,...,L}
```

**Q-Function** (Equation 13):
```
Q_g(φ_g(s,a); ζ_g) = (1/√m) b_g^T x^(L)
```

where:
- m: network width
- L: network depth  
- σ: smooth activation (GELU, ELU, Sigmoid)
- b_g: fixed random output weights

**NTK Projection** (Equation 14):
```
S_R := {ζ : ||ζ - ζ_0||_2 ≤ R}
```

### MLMC Estimator

**Trajectory Length** (Equation 22):
```
P ~ Geom(1/2)
ℓ = (2^P - 1)·𝟙(2^P ≤ T_max) + 1
```

**MLMC Gradient**:
```
v_g^MLMC = v_g^0 + 2^P(v_g^P - v_g^{P-1})  if 2^P ≤ T_max
```

## Implementation

### Core Classes

#### 1. `AlgorithmConfig`
Configuration dataclass containing all hyperparameters:

```python
config = AlgorithmConfig(
    alpha=0.01,          # Primal step size
    beta=0.01,           # Dual step size  
    gamma_zeta=0.001,    # Critic learning rate
    gamma_omega=0.001,   # NPG learning rate
    gamma=0.99,          # Discount factor
    K=1000,              # Outer iterations
    H=100,               # Inner iterations
    T_max=1000,          # Max trajectory length
    network_width=128,   # Critic width (m)
    network_depth=3,     # Critic depth (L)
    radius_R=10.0,       # NTK projection radius
    delta_slater=0.1     # Slater parameter
)
```

#### 2. `NeuralCriticNetwork`
Multi-layer feedforward network for Q-function approximation.

**Key Methods**:
- `forward()`: Compute Q-values
- `project_to_ntk_ball()`: Project to S_R for NTK regime
- `get_param_vector()`: Get parameters as vector ζ_g
- `set_param_vector()`: Set parameters from vector

**Features**:
- Smooth activation functions (GELU, ELU, Sigmoid)
- Fixed random output layer
- Stores initial parameters for NTK projection

#### 3. `PDNAC_NC`
Main algorithm class implementing Algorithm 1.

**Key Methods**:

```python
# Initialize algorithm
algorithm = PDNAC_NC(
    config=config,
    env=environment,
    policy_network=policy,
    feature_extractor=feature_fn
)

# Train
metrics = algorithm.train()
```

**Internal Methods**:
- `update_critic()`: Neural critic estimation (Lines 5-16)
- `update_npg_direction()`: NPG direction (Lines 17-28)
- `compute_mlmc_critic_gradient()`: MLMC estimator (Eq. 22)
- `estimate_cost_value()`: Cost estimation (Line 31)

## Usage

### Basic Example

```python
import torch
from pdnac_nc_algorithm import PDNAC_NC, AlgorithmConfig, SimplePolicy

# 1. Define environment
class MyEnv:
    def reset(self): ...
    def step(self, action): ...  # Returns (state, reward, cost, done, info)

env = MyEnv()

# 2. Create policy network
policy = SimplePolicy(state_dim=4, action_dim=2)

# 3. Define feature extractor
def feature_extractor(state, action):
    # Map (s,a) to features φ(s,a)
    return torch.cat([state, action], dim=-1)

# 4. Configure algorithm  
config = AlgorithmConfig(K=1000, H=100)

# 5. Initialize and train
algorithm = PDNAC_NC(config, env, policy, feature_extractor)
metrics = algorithm.train()
```

### Running the Demo

```bash
python demo_pdnac_nc.py
```

The demo includes:
- Simple 2D constrained navigation environment
- Complete training loop
- Performance evaluation
- Metric visualization

## Key Implementation Details

### 1. MLMC Estimation

The Multi-Level Monte Carlo estimator reduces variance by combining trajectories of different lengths:

```python
def sample_geometric_trajectory_length(self):
    P = np.random.geometric(p=0.5) - 1
    length = (2**P - 1) * int(2**P <= T_max) + 1
    return length, P
```

### 2. NTK Projection

Maintains network in Neural Tangent Kernel regime:

```python
def project_to_ntk_ball(self, radius):
    current = self.get_param_vector()
    initial = self.initial_params_vector()
    diff = current - initial
    
    if torch.norm(diff) > radius:
        diff = diff * (radius / torch.norm(diff))
        self.set_param_vector(initial + diff)
```

### 3. Temporal Difference Learning

Computes advantage estimates using TD error:

```python
# TD error (Equation 21)
δ = g(s,a) + γ·Q(φ(s',a'); ζ) - Q(φ(s,a); ζ)

# Advantage (Equation 26)
A(s,a) = δ
```

### 4. Natural Policy Gradient

Uses Fisher information matrix for natural gradient:

```python
# Fisher matrix sample (Equation 24)
F̂ = ∇_θ log π(a|s) ⊗ ∇_θ log π(a|s)

# NPG gradient (Equation 27)
∇_ω f_g = F̂·ω - A·∇_θ log π(a|s)
```

## Theoretical Guarantees

### Sample Complexity
**O(ε⁻²)** to achieve ε-optimal policy (up to approximation errors)

### Convergence Rate  
**O(1/√T)** for both optimality gap and constraint violation

### Assumptions

1. **Slater Condition** (Assumption 3.1): ∃ θ̄ such that J_c(θ̄) ≥ δ
2. **Smooth Activation** (Assumption 3.2): σ is L₁-Lipschitz and L₂-smooth
3. **Bounded Score** (Assumption 5.1): ||∇_θ log π_θ(a|s)|| ≤ G₁
4. **Fisher Non-degeneracy** (Assumption 5.2): F(θ) ⪰ μI_d
5. **Neural Critic Approximation** (Assumption 5.4): Bounded ε_app
6. **Feature Covariance** (Assumption 5.5): Positive definite covariance

## Comparison with Related Work

| Method | Sample Complexity | Constraints | Critic | Policy |
|--------|------------------|-------------|--------|--------|
| Wang et al. 2019 | O(ε⁻⁴) | ✗ | Neural | Tabular |
| Bai et al. 2023 | O(ε⁻⁴) | ✓ | Linear | General |
| Gaur et al. 2024 | O(ε⁻³) | ✗ | Neural | General |
| Ganesh et al. 2025 | O(ε⁻²) | ✗ | Neural | General |
| **This work** | **O(ε⁻²)** | **✓** | **Neural** | **General** |

## Files

- `pdnac_nc_algorithm.py`: Main implementation
- `demo_pdnac_nc.py`: Demonstration with simple environment
- `README.md`: This documentation

## Requirements

```
torch >= 1.9.0
numpy >= 1.19.0
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{pdnac-nc-2025,
  title={Global Convergence of Constrained MDPs with Neural Critic and General Policy Parameterization},
  author={Anonymous Authors},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## License

This implementation is provided for research and educational purposes.

## Key Equations Reference

### Algorithm Updates

**Critic Update** (Line 15):
```
ζ_{g,h+1}^k = Π_R(ζ_{g,h}^k - γ_ζ v_g^MLMC(θ_k, ζ_{g,h}^k))
```

**NPG Update** (Line 27):
```
ω_{g,h+1}^k = ω_{g,h}^k - γ_ω u_g^MLMC(θ_k, ω_{g,h}^k, ζ_g^k)
```

**Policy Update** (Line 32):
```
θ_{k+1} = θ_k + α(ω_r^k + λ_k ω_c^k)
```

**Dual Update** (Line 33):
```
λ_{k+1} = P_{[0,2/δ]}[λ_k - β Ĵ_c^k]
```

### Gradient Estimators

**Policy Gradient** (Equation 8):
```
∇_θ J_g(θ) = (1/(1-γ)) E[A_g^{π_θ}(s,a) ∇_θ log π_θ(a|s)]
```

**Natural Policy Gradient** (Equation 9):
```
ω*_{g,θ} = F(θ)⁻¹ ∇_θ J_g(θ)
```

**Fisher Information** (Equation 10):
```
F(θ) = E[∇_θ log π_θ(a|s) ⊗ ∇_θ log π_θ(a|s)]
```

## Troubleshooting

### Common Issues

1. **NaN in critic updates**: Reduce `gamma_zeta` or `radius_R`
2. **Constraint violation**: Increase `beta` or adjust `delta_slater`
3. **Slow convergence**: Increase `H` (inner iterations) or adjust step sizes
4. **Memory issues**: Reduce `network_width` or `T_max`

### Hyperparameter Tuning

**Step Sizes**:
- Start with `alpha = beta = 0.01`
- Critic: `gamma_zeta = 2·log(T) / (λ₀(1-γ)H)`
- NPG: `gamma_omega = 2·log(T) / (λ₀(1-γ)H)`

**Network Size**:
- Width: Larger m → better approximation but slower
- Depth: L = 2-3 typically sufficient
- Radius: R = O(log T) for theory

**Iterations**:
- Inner loops H: 50-200 for good convergence
- Outer loops K: Depends on problem complexity
- T_max: Balance bias-variance (500-2000)

## Extensions

Possible extensions of this implementation:

1. **Multi-constraint**: Extend to J_c₁(θ) ≥ 0, ..., J_cₘ(θ) ≥ 0
2. **Adaptive step sizes**: Learning rate schedules
3. **Experience replay**: Store and reuse transitions
4. **Parallel rollouts**: Multiple environments
5. **Trust regions**: Add KL constraints on policy updates

## Contact

For questions or issues, please refer to the original paper or open an issue in the repository.
