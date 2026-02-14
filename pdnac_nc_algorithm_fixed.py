"""
PDNAC-NC: Primal-Dual Natural Actor-Critic with Neural Critic (FIXED VERSION)
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
    alpha: float = 0.01          
    beta: float = 0.01           
    gamma_zeta: float = 0.001    
    gamma_omega: float = 0.001   
    gamma: float = 0.99
    K: int = 1000                
    H: int = 100                 
    T_max: int = 1000            
    network_width: int = 128     
    network_depth: int = 3       
    feature_dim: int = 64        
    radius_R: float = 10.0       
    delta_slater: float = 0.1    
    activation: str = "gelu"     
    device: str = "cpu"


class NeuralCriticNetwork(nn.Module):
    def __init__(self, feature_dim: int, width: int, depth: int, activation: str = "gelu"):
        super(NeuralCriticNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.width = width
        self.depth = depth
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_dim, width))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
        
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        
        self.output_layer = nn.Linear(width, 1, bias=False)
        with torch.no_grad():
            self.output_layer.weight.data = torch.randint(0, 2, (1, width)).float() * 2 - 1
        
        for param in self.output_layer.parameters():
            param.requires_grad = False
        
        self.initial_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.clone().detach()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for layer in self.layers:
            x = layer(x) / np.sqrt(self.width)
            x = self.activation(x)
        q_value = self.output_layer(x) / np.sqrt(self.width)
        return q_value
    
    def get_param_vector(self) -> torch.Tensor:
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        return torch.cat(params) if params else torch.tensor([])
    
    def set_param_vector(self, param_vector: torch.Tensor):
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_length = param.numel()
                param.data = param_vector[idx:idx + param_length].view(param.shape)
                idx += param_length
    
    def project_to_ntk_ball(self, radius: float):
        current_params = self.get_param_vector()
        initial_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                initial_params.append(self.initial_params[name].view(-1))
        if not initial_params: return
        initial_params = torch.cat(initial_params)
        diff = current_params - initial_params
        diff_norm = torch.norm(diff)
        if diff_norm > radius:
            diff = diff * (radius / diff_norm)
            self.set_param_vector(initial_params + diff)


class PDNAC_NC:
    def __init__(self, config: AlgorithmConfig, env, policy_network: nn.Module, feature_extractor: Callable):
        self.config = config
        self.env = env
        self.policy = policy_network
        self.feature_extractor = feature_extractor
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.lambda_dual = torch.tensor(0.0, device=self.device)
        self.critic_r = NeuralCriticNetwork(config.feature_dim, config.network_width, config.network_depth, config.activation).to(self.device)
        self.critic_c = NeuralCriticNetwork(config.feature_dim, config.network_width, config.network_depth, config.activation).to(self.device)
        self.metrics = defaultdict(list)
    
    def sample_geometric_trajectory_length(self) -> Tuple[int, int]:
        P = np.random.geometric(p=0.5) - 1
        traj_length = (2**P - 1) * int(2**P <= self.config.T_max) + 1
        return traj_length, P
    
    def compute_mlmc_critic_gradient(self, critic: NeuralCriticNetwork, objective: str, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        traj_length, P_kh = self.sample_geometric_trajectory_length()
        level_estimates = {}
        levels_to_compute = [0]
        if P_kh > 0 and 2**P_kh <= self.config.T_max:
            if P_kh > 0: levels_to_compute.append(P_kh - 1)
            levels_to_compute.append(P_kh)
        
        for level in levels_to_compute:
            num_steps = 2**level
            gradients = {name: [] for name, p in critic.named_parameters() if p.requires_grad}
            current_state = state.clone()
            for t in range(num_steps):
                with torch.no_grad():
                    action_dist = self.policy(current_state)
                    action = action_dist.sample()
                next_state, reward, cost, done, _ = self.env.step(action.cpu().numpy())
                next_state = torch.FloatTensor(next_state).to(self.device)
                with torch.no_grad():
                    next_action_dist = self.policy(next_state)
                    next_action = next_action_dist.sample()
                features = self.feature_extractor(current_state, action)
                next_features = self.feature_extractor(next_state, next_action)
                g_value = reward if objective == 'r' else cost
                with torch.no_grad():
                    q_current_val = critic(features)
                    q_next = critic(next_features)
                td_error = g_value + self.config.gamma * q_next - q_current_val
                critic.zero_grad()
                q_current = critic(features)
                q_current.backward()
                for name, param in critic.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad = -td_error.item() * param.grad.clone().detach()
                        gradients[name].append(grad)
                current_state = next_state
                if done: break
            level_est = {}
            for name in gradients:
                if gradients[name]:
                    level_est[name] = torch.mean(torch.stack(gradients[name]), dim=0)
                else:
                    for pname, param in critic.named_parameters():
                        if pname == name and param.requires_grad:
                            level_est[name] = torch.zeros_like(param)
                            break
            level_estimates[level] = level_est
        
        v_mlmc = {}
        if 0 in level_estimates:
            v_mlmc = {k: v.clone() for k, v in level_estimates[0].items()}
        else:
            for name, param in critic.named_parameters():
                if param.requires_grad: v_mlmc[name] = torch.zeros_like(param)
        
        if P_kh > 0 and 2**P_kh <= self.config.T_max and P_kh in level_estimates:
            for name in v_mlmc:
                level_P = level_estimates.get(P_kh, {}).get(name, torch.zeros_like(v_mlmc[name]))
                level_P_minus_1 = level_estimates.get(P_kh - 1, {}).get(name, torch.zeros_like(v_mlmc[name]))
                v_mlmc[name] = v_mlmc[name] + (2**P_kh) * (level_P - level_P_minus_1)
        return v_mlmc
    
    def update_critic(self, critic: NeuralCriticNetwork, objective: str, initial_state: torch.Tensor) -> NeuralCriticNetwork:
        with torch.no_grad():
            for name, param in critic.named_parameters():
                if param.requires_grad and name in critic.initial_params:
                    param.data.copy_(critic.initial_params[name])
        for h in range(self.config.H):
            v_mlmc_dict = self.compute_mlmc_critic_gradient(critic, objective, initial_state)
            with torch.no_grad():
                for name, param in critic.named_parameters():
                    if param.requires_grad and name in v_mlmc_dict:
                        param.data -= self.config.gamma_zeta * v_mlmc_dict[name]
            critic.project_to_ntk_ball(self.config.radius_R)
        return critic
    
    def compute_fisher_matrix_sample(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_dist = self.policy(state)
        log_prob = action_dist.log_prob(action).sum()
        self.policy.zero_grad()
        log_prob.backward(create_graph=True, retain_graph=True)
        score = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        score_vector = torch.cat(score)
        return torch.outer(score_vector, score_vector)
    
    def compute_advantage_estimate(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: float, cost: float, critic: NeuralCriticNetwork, objective: str) -> torch.Tensor:
        features = self.feature_extractor(state, action)
        with torch.no_grad():
            next_action_dist = self.policy(next_state)
            next_action = next_action_dist.sample()
            next_features = self.feature_extractor(next_state, next_action)
            q_current = critic(features)
            q_next = critic(next_features)
        g_value = reward if objective == 'r' else cost
        return g_value + self.config.gamma * q_next - q_current
    
    def compute_npg_gradient_sample(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: float, cost: float, omega_g_h: torch.Tensor, critic: NeuralCriticNetwork, objective: str) -> torch.Tensor:
        fisher_sample = self.compute_fisher_matrix_sample(state, action)
        advantage = self.compute_advantage_estimate(state, action, next_state, reward, cost, critic, objective)
        action_dist = self.policy(state)
        log_prob = action_dist.log_prob(action).sum()
        self.policy.zero_grad()
        log_prob.backward(create_graph=True, retain_graph=True)
        score = [p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None]
        score_vector = torch.cat(score)
        return fisher_sample @ omega_g_h - (advantage.item() * score_vector)
    
    def update_npg_direction(self, zeta_g: NeuralCriticNetwork, objective: str, initial_state: torch.Tensor) -> torch.Tensor:
        param_dim = sum(p.numel() for p in self.policy.parameters())
        omega_g = torch.zeros(param_dim, device=self.device)
        for h in range(self.config.H):
            traj_length, Q_kh = self.sample_geometric_trajectory_length()
            gradients = []
            current_state = initial_state.clone()
            for t in range(traj_length):
                with torch.no_grad():
                    action_dist = self.policy(current_state)
                    action = action_dist.sample()
                next_state, reward, cost, done, _ = self.env.step(action.cpu().numpy())
                next_state = torch.FloatTensor(next_state).to(self.device)
                try:
                    grad_sample = self.compute_npg_gradient_sample(current_state, action, next_state, reward, cost, omega_g, zeta_g, objective)
                    gradients.append(grad_sample)
                except: pass
                current_state = next_state
                if done: break
            u_mlmc = torch.mean(torch.stack(gradients), dim=0) if gradients else torch.zeros_like(omega_g)
            omega_g = omega_g - self.config.gamma_omega * u_mlmc
        return omega_g
    
    def estimate_cost_value(self, zeta_c: NeuralCriticNetwork, num_samples: int = 10) -> float:
        cost_estimates = []
        for _ in range(num_samples):
            state = torch.FloatTensor(self.env.reset()).to(self.device)
            with torch.no_grad():
                action = self.policy(state).sample()
                q_cost = zeta_c(self.feature_extractor(state, action))
            cost_estimates.append(q_cost.item())
        return np.mean(cost_estimates)
    
    def train(self) -> Dict[str, List[float]]:
        initial_state = torch.FloatTensor(self.env.reset()).to(self.device)
        for k in range(self.config.K):
            self.critic_r = self.update_critic(self.critic_r, 'r', initial_state)
            self.critic_c = self.update_critic(self.critic_c, 'c', initial_state)
            omega_r = self.update_npg_direction(self.critic_r, 'r', initial_state)
            omega_c = self.update_npg_direction(self.critic_c, 'c', initial_state)
            omega_k = omega_r + self.lambda_dual * omega_c
            J_c_hat = self.estimate_cost_value(self.critic_c)
            with torch.no_grad():
                idx = 0
                for param in self.policy.parameters():
                    param_length = param.numel()
                    param.data += self.config.alpha * omega_k[idx:idx + param_length].view(param.shape)
                    idx += param_length
            self.lambda_dual = torch.clamp(self.lambda_dual - self.config.beta * J_c_hat, min=0.0, max=2.0 / self.config.delta_slater)
            initial_state = torch.FloatTensor(self.env.reset()).to(self.device)
            if k % 10 == 0:
                self.metrics['iteration'].append(k)
                self.metrics['lambda'].append(self.lambda_dual.item())
                self.metrics['cost_estimate'].append(J_c_hat)
        return dict(self.metrics)

class SimplePolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(SimplePolicy, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, state):
        if state.dim() == 1: state = state.unsqueeze(0)
        features = self.net(state)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        return torch.distributions.Normal(mean, std)

def simple_feature_extractor(state: torch.Tensor, action: torch.Tensor, feature_dim: int = 16) -> torch.Tensor:
    """FIXED: Feature extractor that matches the expected feature_dim"""
    if action.dim() == 0: action = action.unsqueeze(0)
    if state.dim() == 1: state = state.unsqueeze(0)
    if action.dim() == 1: action = action.unsqueeze(0)
    
    combined = torch.cat([state, action], dim=-1) # Dim 3
    # Project to the expected feature_dim (e.g., 16) using a fixed pseudo-random projection
    # or simple padding. Here we pad with zeros to reach feature_dim.
    if combined.shape[-1] < feature_dim:
        padding = torch.zeros(*combined.shape[:-1], feature_dim - combined.shape[-1]).to(combined.device)
        return torch.cat([combined, padding], dim=-1)
    return combined[..., :feature_dim]