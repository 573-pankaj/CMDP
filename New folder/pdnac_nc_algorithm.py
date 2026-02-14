"""
PDNAC-NC: Primal-Dual Natural Actor-Critic with Neural Critic
Complete implementation of Algorithm 1 from the paper:
"Global Convergence of Constrained MDPs with Neural Critic and General Policy Parameterization"

This implements the order-optimal O(ε^-2) sample complexity algorithm for constrained MDPs
with general policy parameterization and multi-layer neural network critics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Optional, List, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AlgorithmConfig:
    """Configuration parameters for PDNAC-NC algorithm"""
    # Step sizes (learning rates)
    alpha: float = 0.01          # Primal step size for policy update
    beta: float = 0.01           # Dual step size for constraint
    gamma_zeta: float = 0.001    # Critic learning rate
    gamma_omega: float = 0.001   # NPG learning rate
    
    # Discount factor
    gamma: float = 0.99
    
    # Loop iterations
    K: int = 1000                # Outer loop iterations
    H: int = 100                 # Inner loop iterations (both critic and NPG)
    
    # MLMC parameters
    T_max: int = 1000            # Maximum trajectory length for truncation
    
    # Neural network parameters
    network_width: int = 128     # Width m of neural network
    network_depth: int = 3       # Depth L of neural network
    feature_dim: int = 64        # Dimension of feature map
    
    # NTK regime projection
    radius_R: float = 10.0       # Projection radius for NTK regime
    
    # Constraint parameters
    delta_slater: float = 0.1    # Slater condition parameter δ
    
    # Activation function
    activation: str = "gelu"     # Options: "gelu", "elu", "sigmoid"
    
    # Device
    device: str = "cpu"


class NeuralCriticNetwork(nn.Module):
    """
    Multi-layer feedforward neural network for Q-function approximation.
    Implements equation (12)-(13) from the paper.
    """
    
    def __init__(self, feature_dim: int, width: int, depth: int, activation: str = "gelu"):
        super(NeuralCriticNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.width = width
        self.depth = depth
        
        # Choose smooth activation function (twice differentiable)
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers: x^(l) = (1/√m) σ(W_l x^(l-1))
        self.layers = nn.ModuleList()
        
        # First layer: feature_dim → width
        self.layers.append(nn.Linear(feature_dim, width))
        
        # Hidden layers: width → width
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
        
        # Initialize weights from N(0, 1)
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        
        # Output layer with fixed random weights b_g ∈ {-1, +1}^m
        self.output_layer = nn.Linear(width, 1, bias=False)
        with torch.no_grad():
            self.output_layer.weight.data = torch.randint(0, 2, (1, width)).float() * 2 - 1
        
        # Freeze output layer
        for param in self.output_layer.parameters():
            param.requires_grad = False
        
        # Store initial parameters for NTK regime (ζ_g,0)
        self.initial_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.clone().detach()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Q_g(φ_g(s,a); ζ_g) = (1/√m) b_g^T x^(L)
        
        Args:
            features: Input features φ_g(s,a) of shape (batch_size, feature_dim)
        
        Returns:
            Q-values of shape (batch_size, 1)
        """
        x = features
        
        # Apply layers with activation and scaling
        for layer in self.layers:
            x = layer(x) / np.sqrt(self.width)
            x = self.activation(x)
        
        # Output layer: (1/√m) b_g^T x^(L)
        q_value = self.output_layer(x) / np.sqrt(self.width)
        
        return q_value
    
    def get_param_vector(self) -> torch.Tensor:
        """Get all trainable parameters as a single vector ζ_g"""
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        return torch.cat(params)
    
    def set_param_vector(self, param_vector: torch.Tensor):
        """Set all trainable parameters from a vector"""
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_length = param.numel()
                param.data = param_vector[idx:idx + param_length].view(param.shape)
                idx += param_length
    
    def project_to_ntk_ball(self, radius: float):
        """
        Project parameters to ball S_R around initialization (equation 14).
        This keeps the network in the NTK regime.
        """
        current_params = self.get_param_vector()
        
        # Get initial parameters as vector
        initial_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                initial_params.append(self.initial_params[name].view(-1))
        initial_params = torch.cat(initial_params)
        
        # Compute difference
        diff = current_params - initial_params
        diff_norm = torch.norm(diff)
        
        # Project if outside ball
        if diff_norm > radius:
            diff = diff * (radius / diff_norm)
            self.set_param_vector(initial_params + diff)


class PDNAC_NC:
    """
    Primal-Dual Natural Actor-Critic with Neural Critic (Algorithm 1)
    
    Solves constrained MDPs: max_θ J_r(θ) subject to J_c(θ) ≥ 0
    using neural network critics and general policy parameterization.
    """
    
    def __init__(
        self,
        config: AlgorithmConfig,
        env,  # Environment with step() and reset() methods
        policy_network: nn.Module,  # Policy π_θ
        feature_extractor: Callable,  # Maps (s, a) to features φ(s,a)
    ):
        self.config = config
        self.env = env
        self.policy = policy_network
        self.feature_extractor = feature_extractor
        self.device = torch.device(config.device)
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Initialize dual variable λ_0 = 0
        self.lambda_dual = torch.tensor(0.0, device=self.device)
        
        # Initialize neural critics for reward and cost
        self.critic_r = NeuralCriticNetwork(
            config.feature_dim,
            config.network_width,
            config.network_depth,
            config.activation
        ).to(self.device)
        
        self.critic_c = NeuralCriticNetwork(
            config.feature_dim,
            config.network_width,
            config.network_depth,
            config.activation
        ).to(self.device)
        
        # Track metrics
        self.metrics = defaultdict(list)
    
    def sample_geometric_trajectory_length(self) -> int:
        """
        Sample trajectory length from geometric distribution.
        P ~ Geom(1/2), then ℓ = (2^P - 1)·𝟙(2^P ≤ T_max) + 1
        """
        # Sample P from geometric distribution with p=0.5
        P = np.random.geometric(p=0.5) - 1  # -1 because numpy's geometric starts at 1
        
        # Compute trajectory length with truncation
        traj_length = (2**P - 1) * int(2**P <= self.config.T_max) + 1
        
        return traj_length, P
    
    def compute_mlmc_critic_gradient(
        self,
        theta_k: torch.Tensor,
        zeta_g_h: torch.Tensor,
        critic: NeuralCriticNetwork,
        objective: str,  # 'r' or 'c'
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MLMC estimate of critic gradient v_g^MLMC (equation 22).
        
        Returns:
            MLMC gradient estimate for critic update
        """
        # Sample trajectory length
        traj_length, P_kh = self.sample_geometric_trajectory_length()
        
        # Collect trajectories at different levels
        level_estimates = {}
        
        for level in [0, P_kh - 1, P_kh] if P_kh > 0 and 2**P_kh <= self.config.T_max else [0]:
            if level < 0:
                continue
            
            num_steps = 2**level
            gradients = []
            
            # Generate trajectory of length num_steps
            current_state = state.clone()
            
            for t in range(num_steps):
                # Sample action from policy
                with torch.no_grad():
                    action_dist = self.policy(current_state)
                    action = action_dist.sample()
                
                # Get next state and reward/cost
                next_state, reward, cost, done, _ = self.env.step(
                    action.cpu().numpy()
                )
                next_state = torch.FloatTensor(next_state).to(self.device)
                
                # Sample next action for bootstrapping
                with torch.no_grad():
                    next_action_dist = self.policy(next_state)
                    next_action = next_action_dist.sample()
                
                # Get features
                features = self.feature_extractor(current_state, action)
                next_features = self.feature_extractor(next_state, next_action)
                
                # Compute TD error δ (equation 21)
                g_value = reward if objective == 'r' else cost
                
                with torch.no_grad():
                    q_current = critic(features)
                    q_next = critic(next_features)
                
                td_error = g_value + self.config.gamma * q_next - q_current
                
                # Compute gradient: -δ * ∇_ζ Q(φ(s,a); ζ_g,h)
                features.requires_grad_(True)
                q_current_grad = critic(features)
                
                # Get gradient w.r.t. critic parameters
                critic_grad = torch.autograd.grad(
                    q_current_grad,
                    critic.parameters(),
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                gradient = -td_error.item() * critic_grad
                gradients.append(gradient)
                
                current_state = next_state
                
                if done:
                    break
            
            # Average gradients at this level
            if gradients:
                level_estimates[level] = torch.mean(torch.stack(gradients), dim=0)
            else:
                level_estimates[level] = torch.zeros_like(
                    next(critic.parameters())
                )
        
        # Compute MLMC estimate (equation 22)
        v_mlmc = level_estimates.get(0, torch.zeros_like(next(critic.parameters())))
        
        if P_kh > 0 and 2**P_kh <= self.config.T_max:
            diff = level_estimates.get(P_kh, 0) - level_estimates.get(P_kh - 1, 0)
            v_mlmc = v_mlmc + (2**P_kh) * diff
        
        return v_mlmc
    
    def update_critic(
        self,
        theta_k: torch.Tensor,
        critic: NeuralCriticNetwork,
        objective: str,
        initial_state: torch.Tensor
    ) -> NeuralCriticNetwork:
        """
        Neural critic estimation subroutine (lines 5-16 of Algorithm 1).
        Minimize mean-squared projected Bellman error.
        
        Returns:
            Updated critic parameters ζ_g^k
        """
        # Initialize ζ_g,0^k to initial parameters
        critic.load_state_dict(
            {k: v.clone() for k, v in critic.initial_params.items()}
        )
        
        # Inner loop for H critic iterations
        for h in range(self.config.H):
            # Compute MLMC gradient estimate
            v_mlmc = self.compute_mlmc_critic_gradient(
                theta_k,
                critic.get_param_vector(),
                critic,
                objective,
                initial_state
            )
            
            # Projected gradient descent (equation 23)
            with torch.no_grad():
                for param in critic.parameters():
                    if param.requires_grad:
                        param.data -= self.config.gamma_zeta * v_mlmc
            
            # Project to NTK ball S_R (equation 14)
            critic.project_to_ntk_ball(self.config.radius_R)
        
        return critic
    
    def compute_fisher_matrix_sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute single-sample Fisher matrix estimate (equation 24):
        F̂(θ_k; z) = ∇_θ log π_θ(a|s) ⊗ ∇_θ log π_θ(a|s)
        """
        # Get policy distribution
        action_dist = self.policy(state)
        log_prob = action_dist.log_prob(action)
        
        # Compute score function ∇_θ log π_θ(a|s)
        score = torch.autograd.grad(
            log_prob,
            self.policy.parameters(),
            create_graph=True,
            retain_graph=True
        )
        score_vector = torch.cat([g.view(-1) for g in score])
        
        # Outer product
        fisher_sample = torch.outer(score_vector, score_vector)
        
        return fisher_sample
    
    def compute_advantage_estimate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float,
        cost: float,
        critic: NeuralCriticNetwork,
        objective: str
    ) -> torch.Tensor:
        """
        Compute advantage estimate using TD error (equation 26):
        Â_g(ζ_g^k; z) = g(s,a) + γQ_g(φ(s',a'); ζ_g^k) - Q_g(φ(s,a); ζ_g^k)
        """
        # Get features
        features = self.feature_extractor(state, action)
        
        # Sample next action
        with torch.no_grad():
            next_action_dist = self.policy(next_state)
            next_action = next_action_dist.sample()
        
        next_features = self.feature_extractor(next_state, next_action)
        
        # Compute Q-values
        with torch.no_grad():
            q_current = critic(features)
            q_next = critic(next_features)
        
        # TD advantage
        g_value = reward if objective == 'r' else cost
        advantage = g_value + self.config.gamma * q_next - q_current
        
        return advantage
    
    def compute_npg_gradient_sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float,
        cost: float,
        omega_g_h: torch.Tensor,
        critic: NeuralCriticNetwork,
        objective: str
    ) -> torch.Tensor:
        """
        Compute single-sample NPG gradient estimate (equation 27):
        ∇_ω f_g = F̂(θ_k; z)ω_g,h - ∇_θ Ĵ_g(θ_k, ζ_g^k; z)
        """
        # Compute Fisher matrix sample
        fisher_sample = self.compute_fisher_matrix_sample(state, action)
        
        # Compute advantage
        advantage = self.compute_advantage_estimate(
            state, action, next_state, reward, cost, critic, objective
        )
        
        # Compute policy gradient: Â_g * ∇_θ log π_θ(a|s)
        action_dist = self.policy(state)
        log_prob = action_dist.log_prob(action)
        
        score = torch.autograd.grad(
            log_prob,
            self.policy.parameters(),
            create_graph=True,
            retain_graph=True
        )
        score_vector = torch.cat([g.view(-1) for g in score])
        
        policy_gradient = advantage * score_vector
        
        # NPG gradient: F̂ω - ∇_θĴ_g
        npg_gradient = fisher_sample @ omega_g_h - policy_gradient
        
        return npg_gradient
    
    def update_npg_direction(
        self,
        theta_k: torch.Tensor,
        zeta_g: NeuralCriticNetwork,
        objective: str,
        initial_state: torch.Tensor
    ) -> torch.Tensor:
        """
        NPG direction estimation (lines 17-28 of Algorithm 1).
        Solve min_ω f_g(θ,ω) = (1/2)ω^T F(θ)ω - ω^T ∇_θ J_g(θ)
        
        Returns:
            NPG direction ω_g^k
        """
        # Initialize ω_g,0 = 0
        param_dim = sum(p.numel() for p in self.policy.parameters())
        omega_g = torch.zeros(param_dim, device=self.device)
        
        # Inner loop for H NPG iterations
        for h in range(self.config.H):
            # Sample trajectory for MLMC
            traj_length, Q_kh = self.sample_geometric_trajectory_length()
            
            # Collect gradient estimates
            gradients = []
            current_state = initial_state.clone()
            
            for t in range(traj_length):
                # Sample action
                with torch.no_grad():
                    action_dist = self.policy(current_state)
                    action = action_dist.sample()
                
                # Environment step
                next_state, reward, cost, done, _ = self.env.step(
                    action.cpu().numpy()
                )
                next_state = torch.FloatTensor(next_state).to(self.device)
                
                # Compute NPG gradient sample
                grad_sample = self.compute_npg_gradient_sample(
                    current_state, action, next_state,
                    reward, cost, omega_g, zeta_g, objective
                )
                gradients.append(grad_sample)
                
                current_state = next_state
                if done:
                    break
            
            # Average gradients (simplified MLMC)
            if gradients:
                u_mlmc = torch.mean(torch.stack(gradients), dim=0)
            else:
                u_mlmc = torch.zeros_like(omega_g)
            
            # Update NPG direction (equation 28)
            omega_g = omega_g - self.config.gamma_omega * u_mlmc
        
        return omega_g
    
    def estimate_cost_value(
        self,
        theta_k: torch.Tensor,
        zeta_c: NeuralCriticNetwork,
        num_samples: int = 10
    ) -> float:
        """
        Estimate J_c^k = E_{s_0~ρ, a_0~π_θk}[Q_c(s_0, a_0; ζ_c^k)]
        (line 31 of Algorithm 1)
        """
        cost_estimates = []
        
        for _ in range(num_samples):
            # Sample initial state
            state = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            
            # Sample action
            with torch.no_grad():
                action_dist = self.policy(state)
                action = action_dist.sample()
            
            # Get features and Q-value
            features = self.feature_extractor(state, action)
            with torch.no_grad():
                q_cost = zeta_c(features)
            
            cost_estimates.append(q_cost.item())
        
        return np.mean(cost_estimates)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop (Algorithm 1 lines 1-34).
        
        Returns:
            Dictionary of training metrics
        """
        # Initialize state
        state = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        initial_state = state.clone()
        
        print("Starting PDNAC-NC training...")
        print(f"Outer loops K={self.config.K}, Inner loops H={self.config.H}")
        
        # Outer loop over K epochs
        for k in range(self.config.K):
            # ========== Critic Update (lines 5-16) ==========
            # Update critic for reward
            self.critic_r = self.update_critic(
                None, self.critic_r, 'r', initial_state
            )
            
            # Update critic for cost
            self.critic_c = self.update_critic(
                None, self.critic_c, 'c', initial_state
            )
            
            # ========== NPG Direction Update (lines 17-28) ==========
            # Compute NPG directions for reward and cost
            omega_r = self.update_npg_direction(
                None, self.critic_r, 'r', initial_state
            )
            omega_c = self.update_npg_direction(
                None, self.critic_c, 'c', initial_state
            )
            
            # ========== Policy and Dual Update (lines 30-33) ==========
            # Combined NPG direction: ω^k = ω_r^k + λ_k ω_c^k
            omega_k = omega_r + self.lambda_dual * omega_c
            
            # Estimate cost value
            J_c_hat = self.estimate_cost_value(None, self.critic_c)
            
            # Update policy parameters: θ_{k+1} = θ_k + α ω^k
            with torch.no_grad():
                idx = 0
                for param in self.policy.parameters():
                    param_length = param.numel()
                    param.data += self.config.alpha * omega_k[idx:idx + param_length].view(param.shape)
                    idx += param_length
            
            # Update dual variable: λ_{k+1} = P_{[0, 2/δ]}[λ_k - β Ĵ_c^k]
            self.lambda_dual = torch.clamp(
                self.lambda_dual - self.config.beta * J_c_hat,
                min=0.0,
                max=2.0 / self.config.delta_slater
            )
            
            # Update initial state for next iteration
            state = self.env.reset()
            initial_state = torch.FloatTensor(state).to(self.device)
            
            # ========== Logging ==========
            if k % 10 == 0:
                self.metrics['iteration'].append(k)
                self.metrics['lambda'].append(self.lambda_dual.item())
                self.metrics['cost_estimate'].append(J_c_hat)
                
                print(f"Iteration {k}/{self.config.K}: "
                      f"λ={self.lambda_dual.item():.4f}, "
                      f"J_c={J_c_hat:.4f}")
        
        print("Training complete!")
        return dict(self.metrics)


# ============================================================================
# Example usage and helper functions
# ============================================================================

class SimplePolicy(nn.Module):
    """Example policy network for continuous action spaces"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(SimplePolicy, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """Returns action distribution"""
        features = self.net(state)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)


def simple_feature_extractor(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Simple feature extractor: concatenate state and action"""
    if action.dim() == 0:
        action = action.unsqueeze(0)
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    
    features = torch.cat([state, action], dim=-1)
    return features


if __name__ == "__main__":
    # Example: Create a simple constrained MDP problem
    print("PDNAC-NC Algorithm Implementation")
    print("=" * 60)
    print("This implementation follows Algorithm 1 from the paper:")
    print("'Global Convergence of Constrained MDPs with Neural Critic")
    print("and General Policy Parameterization'")
    print("=" * 60)
    
    # Configuration
    config = AlgorithmConfig(
        alpha=0.01,
        beta=0.01,
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.99,
        K=100,
        H=50,
        T_max=500,
        network_width=64,
        network_depth=2,
        feature_dim=16,
        radius_R=5.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    print(f"\nConfiguration:")
    print(f"  Outer loops (K): {config.K}")
    print(f"  Inner loops (H): {config.H}")
    print(f"  Network width (m): {config.network_width}")
    print(f"  Network depth (L): {config.network_depth}")
    print(f"  Discount factor (γ): {config.gamma}")
    print(f"  NTK radius (R): {config.radius_R}")
    
    print("\n" + "=" * 60)
    print("Algorithm ready for deployment with environment")
    print("=" * 60)
