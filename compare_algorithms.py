"""
Multi-Algorithm Comparison Script
Generates comparison plots similar to the reference images
Compares PDNAC-NC with different configurations or baselines
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

from train_with_visualization import (
    TrainingLogger, EnhancedPDNAC_NC, aggregate_metrics
)
from pdnac_nc_algorithm_fixed import AlgorithmConfig, SimplePolicy, simple_feature_extractor
from demo_pdnac_nc import SimpleConstrainedEnv


def run_algorithm_comparison(
    env_class,
    env_kwargs: Dict,
    algorithms_config: Dict[str, AlgorithmConfig],
    num_seeds: int = 5,
    save_dir: str = "results/comparison"
):
    """
    Run multiple algorithms and generate comparison plots
    
    Args:
        env_class: Environment class
        env_kwargs: Environment initialization kwargs
        algorithms_config: Dict of {algorithm_name: config}
        num_seeds: Number of random seeds per algorithm
    """
    
    print("=" * 80)
    print("MULTI-ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Algorithms: {list(algorithms_config.keys())}")
    print(f"Seeds per algorithm: {num_seeds}")
    print(f"Save directory: {save_dir}")
    print("=" * 80 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    
    # Store results for all algorithms
    all_results = {}
    
    for algo_name, config in algorithms_config.items():
        print(f"\n{'='*80}")
        print(f"Running Algorithm: {algo_name}")
        print(f"{'='*80}\n")
        
        seeds_metrics = []
        
        for seed in range(num_seeds):
            print(f"  Seed {seed+1}/{num_seeds}...")
            
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
            
            # Create algorithm
            algorithm = EnhancedPDNAC_NC(
                config=config,
                env=env,
                policy_network=policy,
                feature_extractor=feature_extractor,
                logger=None
            )
            
            # Train
            metrics = algorithm.train_with_logging(eval_frequency=max(1, config.K // 20))
            seeds_metrics.append(metrics)
        
        # Aggregate across seeds
        aggregated = aggregate_metrics(seeds_metrics)
        all_results[algo_name] = aggregated
        
        print(f"  ✓ {algo_name} complete\n")
    
    # Generate comparison plots
    print(f"\n{'='*80}")
    print("Generating Comparison Plots")
    print(f"{'='*80}\n")
    
    generate_comparison_plots(all_results, save_dir, env_kwargs.get('env_name', 'Environment'))
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"Plots saved to: {save_dir}/plots/")
    
    return all_results


def generate_comparison_plots(
    results: Dict[str, Dict],
    save_dir: str,
    env_name: str = "Environment"
):
    """Generate comparison plots similar to reference images"""
    
    # Set style to match reference images
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (14, 5)
    })
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f']
    
    # === Plot 1: Return over Training ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot (a): Return
    ax = axes[0]
    for idx, (algo_name, metrics) in enumerate(results.items()):
        iterations = metrics['iteration']
        rewards = metrics.get('episode_reward', [])
        reward_std = metrics.get('episode_reward_std', [0] * len(rewards))
        
        color = colors[idx % len(colors)]
        
        # Plot mean
        ax.plot(iterations, rewards, label=algo_name, 
               color=color, linewidth=2, alpha=0.9)
        
        # Plot confidence band
        rewards_arr = np.array(rewards)
        std_arr = np.array(reward_std)
        ax.fill_between(iterations, 
                       rewards_arr - std_arr,
                       rewards_arr + std_arr,
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Outer Loops', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title(f'(a) {env_name}', fontsize=13, fontweight='bold')
    ax.legend(title='algorithm', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Subplot (b): Cost
    ax = axes[1]
    for idx, (algo_name, metrics) in enumerate(results.items()):
        iterations = metrics['iteration']
        costs = metrics.get('cost_estimate', [])
        cost_std = metrics.get('cost_estimate_std', [0] * len(costs))
        
        color = colors[idx % len(colors)]
        
        # Plot mean
        ax.plot(iterations, costs, label=algo_name,
               color=color, linewidth=2, alpha=0.9)
        
        # Plot confidence band
        costs_arr = np.array(costs)
        std_arr = np.array(cost_std)
        ax.fill_between(iterations,
                       costs_arr - std_arr,
                       costs_arr + std_arr,
                       color=color, alpha=0.2)
    
    # Add constraint threshold line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Number of Outer Loops', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title(f'(b) {env_name} - Constraint', fontsize=13, fontweight='bold')
    ax.legend(title='algorithm', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Subplot (c): Lambda (Dual Variable)
    ax = axes[2]
    for idx, (algo_name, metrics) in enumerate(results.items()):
        iterations = metrics['iteration']
        lambdas = metrics.get('lambda', [])
        
        color = colors[idx % len(colors)]
        
        ax.plot(iterations, lambdas, label=algo_name,
               color=color, linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Number of Outer Loops', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(f'(c) {env_name} - Dual Variable', fontsize=13, fontweight='bold')
    ax.legend(title='algorithm', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', f'comparison_{env_name}.png'),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'plots', f'comparison_{env_name}.pdf'),
               bbox_inches='tight')
    print(f"✓ Saved comparison plot: comparison_{env_name}.png")
    plt.close()
    
    # === Plot 2: Single Panel Comparison (like reference image 2) ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for idx, (algo_name, metrics) in enumerate(results.items()):
        iterations = metrics['iteration']
        rewards = metrics.get('episode_reward', [])
        reward_std = metrics.get('episode_reward_std', [0] * len(rewards))
        
        color = colors[idx % len(colors)]
        
        # Plot mean
        ax.plot(iterations, rewards, label=algo_name,
               color=color, linewidth=2.5, alpha=0.9)
        
        # Plot confidence band
        rewards_arr = np.array(rewards)
        std_arr = np.array(reward_std)
        ax.fill_between(iterations,
                       rewards_arr - std_arr,
                       rewards_arr + std_arr,
                       color=color, alpha=0.25)
    
    ax.set_xlabel('Number of Outer Loops', fontsize=13)
    ax.set_ylabel('Return', fontsize=13)
    ax.set_title(env_name, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', f'single_comparison_{env_name}.png'),
               dpi=300, bbox_inches='tight')
    print(f"✓ Saved single comparison plot: single_comparison_{env_name}.png")
    plt.close()


def create_algorithm_variants():
    """Create different algorithm configurations for comparison"""
    
    configs = {}
    
    # PDNAC-NC with different dual step sizes
    configs['NAC-DD-1'] = AlgorithmConfig(
        alpha=0.005,
        beta=0.001,  # Small dual step
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    configs['NAC-DD-3'] = AlgorithmConfig(
        alpha=0.005,
        beta=0.01,  # Medium dual step
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    configs['NAC-DD-5'] = AlgorithmConfig(
        alpha=0.005,
        beta=0.05,  # Large dual step
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    return configs


def create_baseline_comparisons():
    """Create configurations comparing different methods"""
    
    configs = {}
    
    # PDNAC-NC (Ours)
    configs['PDNAC-NC'] = AlgorithmConfig(
        alpha=0.005,
        beta=0.01,
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    # Vanilla Policy Gradient (No dual, larger alpha)
    configs['PG'] = AlgorithmConfig(
        alpha=0.01,
        beta=0.0,  # No constraint
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    # NPG variant
    configs['NPG'] = AlgorithmConfig(
        alpha=0.008,
        beta=0.0,
        gamma_zeta=0.001,
        gamma_omega=0.002,  # Larger NPG step
        gamma=0.95,
        K=150,
        H=30,  # More inner iterations
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    # TRPO-like (smaller steps)
    configs['TRPO'] = AlgorithmConfig(
        alpha=0.002,  # Smaller step
        beta=0.005,
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=20,
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=2.0,  # Smaller trust region
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    # PPO-like
    configs['PPO'] = AlgorithmConfig(
        alpha=0.003,
        beta=0.008,
        gamma_zeta=0.001,
        gamma_omega=0.001,
        gamma=0.95,
        K=150,
        H=15,  # Fewer inner iterations
        T_max=100,
        network_width=32,
        network_depth=2,
        feature_dim=16,
        radius_R=3.0,
        delta_slater=0.1,
        activation="gelu",
        device="cpu"
    )
    
    return configs


if __name__ == "__main__":
    # Environment configuration
    env_kwargs = {
        'goal_position': np.array([1.0, 1.0]),
        'cost_threshold': 2.0,
        'env_name': 'Hopper-v3'  # For plot labels
    }
    
    print("\n" + "="*80)
    print("OPTION 1: Compare PDNAC-NC Variants (Different Dual Step Sizes)")
    print("="*80)
    
    # Run comparison with different dual step sizes
    variant_configs = create_algorithm_variants()
    variant_results = run_algorithm_comparison(
        env_class=SimpleConstrainedEnv,
        env_kwargs=env_kwargs,
        algorithms_config=variant_configs,
        num_seeds=3,
        save_dir="results/comparison_variants"
    )
    
    print("\n" + "="*80)
    print("OPTION 2: Compare Against Baselines")
    print("="*80)
    
    # Run comparison with baselines
    baseline_configs = create_baseline_comparisons()
    baseline_results = run_algorithm_comparison(
        env_class=SimpleConstrainedEnv,
        env_kwargs=env_kwargs,
        algorithms_config=baseline_configs,
        num_seeds=3,
        save_dir="results/comparison_baselines"
    )
    
    print("\n" + "="*80)
    print("ALL COMPARISONS COMPLETE!")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. results/comparison_variants/plots/ - NAC-DD variants")
    print("  2. results/comparison_baselines/plots/ - Method comparisons")
    print("\n✓ All visualizations saved successfully!")
