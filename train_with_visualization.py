"""
Enhanced Training Script with Visualization and Logging
Generates plots similar to the reference images showing training progress
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import os
from datetime import datetime
import json

# Import the fixed algorithm
from pdnac_nc_algorithm_fixed import (
    PDNAC_NC, AlgorithmConfig, SimplePolicy, simple_feature_extractor
)


class TrainingLogger:
    """Enhanced logger for tracking and visualizing training metrics"""
    
    def __init__(self, save_dir: str = "results", experiment_name: str = None):
        self.save_dir = save_dir
        if experiment_name is None:
            experiment_name = f"pdnac_nc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.exp_dir = os.path.join(save_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
    def log_iteration(self, iteration: int, metrics: Dict):
        """Log metrics for a single iteration"""
        self.metrics['iteration'].append(iteration)
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def log_episode(self, episode: int, reward: float, cost: float, 
                    steps: int, constraint_satisfied: bool):
        """Log evaluation episode metrics"""
        self.episode_metrics['episode'].append(episode)
        self.episode_metrics['reward'].append(reward)
        self.episode_metrics['cost'].append(cost)
        self.episode_metrics['steps'].append(steps)
        self.episode_metrics['constraint_satisfied'].append(constraint_satisfied)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_dict = {
            'training': dict(self.metrics),
            'evaluation': dict(self.episode_metrics)
        }
        
        with open(os.path.join(self.exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def plot_training_curves(self, show: bool = True, save: bool = True):
        """Generate comprehensive training plots"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'PDNAC-NC Training Progress - {self.experiment_name}', 
                     fontsize=16, fontweight='bold')
        
        iterations = self.metrics['iteration']
        
        # 1. Return/Reward over time
        if 'episode_reward' in self.metrics:
            ax = axes[0, 0]
            ax.plot(iterations, self.metrics['episode_reward'], 
                   label='Episode Reward', color='blue', alpha=0.7)
            ax.fill_between(iterations, 
                           np.array(self.metrics['episode_reward']) - 
                           np.array(self.metrics.get('reward_std', [0]*len(iterations))),
                           np.array(self.metrics['episode_reward']) + 
                           np.array(self.metrics.get('reward_std', [0]*len(iterations))),
                           alpha=0.3, color='blue')
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Return')
            ax.set_title('Cumulative Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Constraint Cost over time
        if 'cost_estimate' in self.metrics:
            ax = axes[0, 1]
            ax.plot(iterations, self.metrics['cost_estimate'], 
                   label='Cost Estimate', color='red', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', label='Constraint Threshold')
            ax.fill_between(iterations, 
                           np.array(self.metrics['cost_estimate']) - 
                           np.array(self.metrics.get('cost_std', [0]*len(iterations))),
                           np.array(self.metrics['cost_estimate']) + 
                           np.array(self.metrics.get('cost_std', [0]*len(iterations))),
                           alpha=0.3, color='red')
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Cost Value')
            ax.set_title('Constraint Cost (J_c)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Dual Variable (Lambda) over time
        if 'lambda' in self.metrics:
            ax = axes[0, 2]
            ax.plot(iterations, self.metrics['lambda'], 
                   label='λ (Dual Variable)', color='green', alpha=0.7)
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Lambda')
            ax.set_title('Dual Variable Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Critic Loss
        if 'critic_loss_r' in self.metrics:
            ax = axes[1, 0]
            ax.plot(iterations, self.metrics['critic_loss_r'], 
                   label='Reward Critic', color='blue', alpha=0.7)
            if 'critic_loss_c' in self.metrics:
                ax.plot(iterations, self.metrics['critic_loss_c'], 
                       label='Cost Critic', color='red', alpha=0.7)
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Critic Loss')
            ax.set_title('Critic Learning Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # 5. Policy Gradient Norm
        if 'policy_grad_norm' in self.metrics:
            ax = axes[1, 1]
            ax.plot(iterations, self.metrics['policy_grad_norm'], 
                   label='Gradient Norm', color='purple', alpha=0.7)
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Policy Gradient Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Constraint Violation Rate
        if 'constraint_violation' in self.metrics:
            ax = axes[1, 2]
            violations = np.array(self.metrics['constraint_violation'])
            ax.plot(iterations, violations, 
                   label='Violation Rate', color='orange', alpha=0.7)
            ax.set_xlabel('Number of Outer Loops')
            ax.set_ylabel('Violation Rate')
            ax.set_title('Constraint Violation Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.exp_dir, 'plots', 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved training curves to {self.exp_dir}/plots/training_curves.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(self, comparison_data: Dict[str, Dict], 
                       metric: str = 'episode_reward',
                       show: bool = True, save: bool = True):
        """
        Generate comparison plots similar to reference images
        
        Args:
            comparison_data: Dict of {algorithm_name: metrics_dict}
            metric: Metric to plot ('episode_reward', 'cost_estimate', etc.)
        """
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (algo_name, metrics) in enumerate(comparison_data.items()):
            iterations = metrics.get('iteration', [])
            values = metrics.get(metric, [])
            
            if len(values) > 0:
                color = colors[idx % len(colors)]
                
                # Plot mean
                plt.plot(iterations, values, label=algo_name, 
                        color=color, alpha=0.8, linewidth=2)
                
                # Plot confidence interval if std available
                if f'{metric}_std' in metrics:
                    std = np.array(metrics[f'{metric}_std'])
                    mean = np.array(values)
                    plt.fill_between(iterations, mean - std, mean + std,
                                   alpha=0.2, color=color)
        
        plt.xlabel('Number of Outer Loops', fontsize=12)
        plt.ylabel('Return' if 'reward' in metric else metric.replace('_', ' ').title(), 
                  fontsize=12)
        plt.title(f'Algorithm Comparison - {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.exp_dir, 'plots', f'comparison_{metric}.png'),
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {self.exp_dir}/plots/comparison_{metric}.png")
        
        if show:
            plt.show()
        else:
            plt.close()


class EnhancedPDNAC_NC(PDNAC_NC):
    """Enhanced PDNAC-NC with detailed logging"""
    
    def __init__(self, config, env, policy_network, feature_extractor, logger=None):
        super().__init__(config, env, policy_network, feature_extractor)
        self.logger = logger
        self.eval_env = env  # Keep reference for evaluation
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy performance"""
        rewards = []
        costs = []
        steps_list = []
        violations = 0
        
        for _ in range(num_episodes):
            state = self.eval_env.reset()
            state = torch.FloatTensor(state).to(self.device)
            
            episode_reward = 0
            episode_cost = 0
            steps = 0
            done = False
            
            while not done and steps < 200:
                with torch.no_grad():
                    action_dist = self.policy(state)
                    action = action_dist.mean  # Use mean for evaluation
                
                next_state, reward, cost, done, _ = self.eval_env.step(
                    action.cpu().numpy()
                )
                
                episode_reward += reward
                episode_cost += cost
                steps += 1
                
                state = torch.FloatTensor(next_state).to(self.device)
            
            rewards.append(episode_reward)
            costs.append(episode_cost)
            steps_list.append(steps)
            
            if episode_cost < 0:
                violations += 1
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'mean_steps': np.mean(steps_list),
            'violation_rate': violations / num_episodes
        }
    
    def train_with_logging(self, eval_frequency: int = 10) -> Dict[str, List[float]]:
        """Enhanced training with comprehensive logging"""
        # Initialize state
        state = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        initial_state = state.clone()
        
        print("=" * 70)
        print("Starting PDNAC-NC training with enhanced logging...")
        print(f"Outer loops K={self.config.K}, Inner loops H={self.config.H}")
        print(f"Evaluation every {eval_frequency} iterations")
        print("=" * 70 + "\n")
        
        # Outer loop over K epochs
        for k in range(self.config.K):
            # ========== Critic Update ==========
            self.critic_r = self.update_critic(
                self.critic_r, 'r', initial_state
            )
            self.critic_c = self.update_critic(
                self.critic_c, 'c', initial_state
            )
            
            # ========== NPG Direction Update ==========
            omega_r = self.update_npg_direction(
                self.critic_r, 'r', initial_state
            )
            omega_c = self.update_npg_direction(
                self.critic_c, 'c', initial_state
            )
            
            # ========== Policy and Dual Update ==========
            omega_k = omega_r + self.lambda_dual * omega_c
            J_c_hat = self.estimate_cost_value(self.critic_c)
            
            # Compute gradient norm
            grad_norm = torch.norm(omega_k).item()
            
            # Update policy
            with torch.no_grad():
                idx = 0
                for param in self.policy.parameters():
                    param_length = param.numel()
                    update = self.config.alpha * omega_k[idx:idx + param_length].view(param.shape)
                    param.data += update
                    idx += param_length
            
            # Update dual variable
            self.lambda_dual = torch.clamp(
                self.lambda_dual - self.config.beta * J_c_hat,
                min=0.0,
                max=2.0 / self.config.delta_slater
            )
            
            # ========== Evaluation ==========
            if k % eval_frequency == 0:
                eval_metrics = self.evaluate_policy(num_episodes=5)
                
                # Log to logger
                if self.logger:
                    self.logger.log_iteration(k, {
                        'episode_reward': eval_metrics['mean_reward'],
                        'reward_std': eval_metrics['std_reward'],
                        'cost_estimate': eval_metrics['mean_cost'],
                        'cost_std': eval_metrics['std_cost'],
                        'lambda': self.lambda_dual.item(),
                        'policy_grad_norm': grad_norm,
                        'constraint_violation': eval_metrics['violation_rate']
                    })
                
                # Console output
                print(f"Iteration {k}/{self.config.K}:")
                print(f"  Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Cost: {eval_metrics['mean_cost']:.2f} ± {eval_metrics['std_cost']:.2f}")
                print(f"  Lambda: {self.lambda_dual.item():.4f}")
                print(f"  Grad Norm: {grad_norm:.4f}")
                print(f"  Violations: {eval_metrics['violation_rate']*100:.1f}%")
                print()
            
            # Store basic metrics every iteration
            self.metrics['iteration'].append(k)
            self.metrics['lambda'].append(self.lambda_dual.item())
            self.metrics['cost_estimate'].append(J_c_hat)
            
            # Update state
            state = self.env.reset()
            initial_state = torch.FloatTensor(state).to(self.device)
        
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        
        return dict(self.metrics)


def run_experiment_with_visualization(
    env_class,
    env_kwargs: Dict,
    config: AlgorithmConfig,
    num_seeds: int = 3,
    experiment_name: str = "pdnac_nc_experiment"
):
    """
    Run complete experiment with multiple seeds and generate plots
    """
    print("=" * 70)
    print("PDNAC-NC EXPERIMENT WITH VISUALIZATION")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Seeds: {num_seeds}")
    print(f"Iterations: {config.K}")
    print("=" * 70 + "\n")
    
    # Storage for all seeds
    all_seeds_metrics = []
    
    for seed in range(num_seeds):
        print(f"\n{'='*70}")
        print(f"Running Seed {seed+1}/{num_seeds}")
        print(f"{'='*70}\n")
        
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env = env_class(**env_kwargs)
        
        # Create policy
        policy = SimplePolicy(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=32
        )
        
        # Feature extractor
        def feature_extractor(state, action):
            return simple_feature_extractor(state, action)
        
        # Create logger
        logger = TrainingLogger(
            save_dir="results",
            experiment_name=f"{experiment_name}_seed{seed}"
        )
        
        # Create algorithm
        algorithm = EnhancedPDNAC_NC(
            config=config,
            env=env,
            policy_network=policy,
            feature_extractor=feature_extractor,
            logger=logger
        )
        
        # Train
        metrics = algorithm.train_with_logging(eval_frequency=10)
        all_seeds_metrics.append(metrics)
        
        # Save metrics
        logger.save_metrics()
        
        # Plot individual seed
        logger.plot_training_curves(show=False, save=True)
    
    # Aggregate metrics across seeds
    print(f"\n{'='*70}")
    print("Aggregating Results Across Seeds")
    print(f"{'='*70}\n")
    
    aggregated = aggregate_metrics(all_seeds_metrics)
    
    # Create master logger for aggregated results
    master_logger = TrainingLogger(
        save_dir="results",
        experiment_name=f"{experiment_name}_aggregated"
    )
    master_logger.metrics = aggregated
    
    # Plot aggregated results
    master_logger.plot_training_curves(show=True, save=True)
    master_logger.save_metrics()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: results/{experiment_name}_*/")
    print(f"Plots saved to: results/{experiment_name}_*/plots/")
    
    return aggregated


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across multiple runs"""
    aggregated = defaultdict(list)
    
    # Find common iterations
    all_iterations = [m.get('iteration', []) for m in all_metrics]
    common_iterations = sorted(set.intersection(*[set(it) for it in all_iterations if it]))
    
    aggregated['iteration'] = common_iterations
    
    # Aggregate each metric
    metric_keys = set()
    for m in all_metrics:
        metric_keys.update(m.keys())
    metric_keys.discard('iteration')
    
    for key in metric_keys:
        values_per_iter = []
        
        for iteration in common_iterations:
            values = []
            for metrics in all_metrics:
                if 'iteration' in metrics and iteration in metrics['iteration']:
                    idx = metrics['iteration'].index(iteration)
                    if idx < len(metrics.get(key, [])):
                        values.append(metrics[key][idx])
            
            if values:
                values_per_iter.append(np.mean(values))
                aggregated[f'{key}_std'].append(np.std(values))
            else:
                values_per_iter.append(0)
                aggregated[f'{key}_std'].append(0)
        
        aggregated[key] = values_per_iter
    
    return dict(aggregated)


if __name__ == "__main__":
    # Import environment
    import sys
    sys.path.append('.')
    from demo_pdnac_nc import SimpleConstrainedEnv
    
    # Configure experiment
    config = AlgorithmConfig(
        alpha=0.005,
        beta=0.01,
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=100,              # Number of outer iterations
        H=20,               # Inner iterations
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    # Environment configuration
    env_kwargs = {
        'goal_position': np.array([1.0, 1.0]),
        'cost_threshold': 2.0
    }
    
    # Run experiment
    results = run_experiment_with_visualization(
        env_class=SimpleConstrainedEnv,
        env_kwargs=env_kwargs,
        config=config,
        num_seeds=3,
        experiment_name="pdnac_nc_hopper"
    )
    
    print("\n✓ All visualizations generated successfully!")
    print("Check the 'results/' folder for plots and metrics.")
