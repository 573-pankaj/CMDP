"""
Demonstration of PDNAC-NC Algorithm
Includes a simple constrained environment and complete training example
"""

import numpy as np
import torch
import torch.nn as nn
from pdnac_nc_algorithm_fixed import (
    PDNAC_NC, AlgorithmConfig, SimplePolicy, simple_feature_extractor
)


class SimpleConstrainedEnv:
    """
    Simple constrained MDP environment for demonstration.
    
    State space: 2D continuous
    Action space: 1D continuous
    Reward: Negative distance to goal
    Cost: Constraint violation (penalize going too far from origin)
    """
    
    def __init__(self, goal_position=np.array([1.0, 1.0]), cost_threshold=2.0):
        self.goal = goal_position
        self.cost_threshold = cost_threshold
        self.state_dim = 2
        self.action_dim = 1
        self.max_steps = 100
        
        self.state = None
        self.steps = 0
    
    def reset(self):
        """Reset environment to initial state"""
        # Start near origin with some randomness
        self.state = np.random.randn(self.state_dim) * 0.1
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        """
        Take action in environment
        
        Returns:
            next_state, reward, cost, done, info
        """
        # Update state based on action
        action = np.clip(action, -1, 1)
        
        # Simple dynamics: state changes based on action
        direction = np.array([np.cos(self.steps * 0.1), np.sin(self.steps * 0.1)])
        self.state = self.state + 0.1 * action[0] * direction
        
        # Compute reward: negative distance to goal
        distance_to_goal = np.linalg.norm(self.state - self.goal)
        reward = -distance_to_goal
        
        # Compute cost: penalize if too far from origin (constraint)
        distance_from_origin = np.linalg.norm(self.state)
        cost = distance_from_origin - self.cost_threshold  # Negative cost is good
        
        # Episode terminates after max steps or if very far
        self.steps += 1
        done = (self.steps >= self.max_steps) or (distance_from_origin > 5.0)
        
        info = {
            'distance_to_goal': distance_to_goal,
            'distance_from_origin': distance_from_origin
        }
        
        return self.state.copy(), reward, cost, done, info
    
    def render(self):
        """Optional: visualize state"""
        print(f"State: {self.state}, Distance to goal: {np.linalg.norm(self.state - self.goal):.3f}")


def create_feature_extractor(state_dim, action_dim, feature_dim):
    """
    Create a learnable feature extractor φ_g: S × A → ℝ^n
    For simplicity, we use a simple neural network
    """
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 32),
                nn.ReLU(),
                nn.Linear(32, feature_dim),
                nn.Tanh()
            )
        
        def forward(self, state, action):
            # Ensure proper dimensions
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 0:
                action = action.unsqueeze(0).unsqueeze(0)
            elif action.dim() == 1:
                action = action.unsqueeze(0)
            
            # Concatenate state and action
            sa = torch.cat([state, action], dim=-1)
            features = self.net(sa)
            
            # Normalize to have ||φ(s,a)|| ≤ 1
            features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
            
            return features
    
    return FeatureExtractor()


def run_demo():
    """Run complete demonstration of PDNAC-NC algorithm"""
    
    print("=" * 70)
    print("PDNAC-NC ALGORITHM DEMONSTRATION")
    print("=" * 70)
    print("\nPaper: 'Global Convergence of Constrained MDPs with Neural Critic")
    print("        and General Policy Parameterization'")
    print("\nThis demo shows the algorithm solving a simple constrained MDP:")
    print("  - Objective: Navigate to goal position")
    print("  - Constraint: Stay within cost threshold from origin")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Environment parameters
    STATE_DIM = 2
    ACTION_DIM = 1
    FEATURE_DIM = 16
    
    # Create environment
    env = SimpleConstrainedEnv(
        goal_position=np.array([1.0, 1.0]),
        cost_threshold=2.0
    )
    
    print(f"\nEnvironment:")
    print(f"  State dimension: {STATE_DIM}")
    print(f"  Action dimension: {ACTION_DIM}")
    print(f"  Goal position: {env.goal}")
    print(f"  Cost threshold: {env.cost_threshold}")
    
    # Create policy network
    policy = SimplePolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=32
    )
    
    print(f"\nPolicy Network:")
    print(f"  Type: Gaussian policy")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Create feature extractor
    feature_net = create_feature_extractor(STATE_DIM, ACTION_DIM, FEATURE_DIM)
    
    def feature_extractor(state, action):
        return feature_net(state, action)
    
    # Algorithm configuration
    config = AlgorithmConfig(
        alpha=0.005,           # Primal step size
        beta=0.01,             # Dual step size
        gamma_zeta=0.001,      # Critic learning rate
        gamma_omega=0.001,     # NPG learning rate
        gamma=0.95,            # Discount factor
        K=50,                  # Outer loops (reduced for demo)
        H=20,                  # Inner loops (reduced for demo)
        T_max=100,             # Max trajectory length
        network_width=32,      # Neural critic width
        network_depth=2,       # Neural critic depth
        feature_dim=FEATURE_DIM,
        radius_R=3.0,          # NTK projection radius
        delta_slater=0.1,      # Slater parameter
        activation="gelu",
        device="cpu"
    )
    
    print(f"\nAlgorithm Configuration:")
    print(f"  Outer iterations (K): {config.K}")
    print(f"  Inner iterations (H): {config.H}")
    print(f"  Critic width (m): {config.network_width}")
    print(f"  Critic depth (L): {config.network_depth}")
    print(f"  Discount (γ): {config.gamma}")
    print(f"  Primal step (α): {config.alpha}")
    print(f"  Dual step (β): {config.beta}")
    print(f"  NTK radius (R): {config.radius_R}")
    
    # Create PDNAC-NC algorithm
    algorithm = PDNAC_NC(
        config=config,
        env=env,
        policy_network=policy,
        feature_extractor=feature_extractor
    )
    
    print(f"\nNeural Critics:")
    print(f"  Reward critic parameters: {sum(p.numel() for p in algorithm.critic_r.parameters())}")
    print(f"  Cost critic parameters: {sum(p.numel() for p in algorithm.critic_c.parameters())}")
    
    # Train algorithm
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    try:
        metrics = algorithm.train()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        if metrics:
            print(f"\nFinal Metrics:")
            print(f"  Final λ: {metrics['lambda'][-1]:.4f}")
            print(f"  Final cost estimate: {metrics['cost_estimate'][-1]:.4f}")
            print(f"  Constraint satisfied: {metrics['cost_estimate'][-1] >= 0}")
        
        # Test learned policy
        print("\n" + "=" * 70)
        print("TESTING LEARNED POLICY")
        print("=" * 70 + "\n")
        
        test_episodes = 5
        total_rewards = []
        total_costs = []
        
        for ep in range(test_episodes):
            state = env.reset()
            episode_reward = 0
            episode_cost = 0
            done = False
            steps = 0
            
            print(f"Episode {ep + 1}:")
            
            while not done and steps < 50:
                state_tensor = torch.FloatTensor(state)
                
                with torch.no_grad():
                    action_dist = policy(state_tensor)
                    action = action_dist.mean  # Use mean action for testing
                
                state, reward, cost, done, info = env.step(action.numpy())
                
                episode_reward += reward
                episode_cost += cost
                steps += 1
            
            total_rewards.append(episode_reward)
            total_costs.append(episode_cost)
            
            print(f"  Total reward: {episode_reward:.3f}")
            print(f"  Total cost: {episode_cost:.3f}")
            print(f"  Steps: {steps}")
        
        print(f"\nAverage Performance:")
        print(f"  Avg reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"  Avg cost: {np.mean(total_costs):.3f} ± {np.std(total_costs):.3f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
