#!/usr/bin/env python3
"""
Simplified grid-world experiment script without visualization.

This script tests the consolidated agent comparison functionality.
"""

import sys
import os
import numpy as np
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.environment import GridEnvironment
from tinycrops_hall_of_mirrors.grid_world.agents import (
    FlatAgent, FractalAgent, FractalAttentionAgent
)


# Predefined experiment configurations
EXPERIMENTS = {
    'flat_vs_fractal_shaped': {
        'description': 'Compare flat agent vs fractal agent with shaped rewards',
        'agents': [
            ('Flat Agent', 'flat', {}),
            ('Fractal Agent (Shaped)', 'fractal', {'reward_shaping': 'shaped'})
        ]
    },
    
    'flat_vs_fractal_sparse': {
        'description': 'Compare flat agent vs fractal agent with sparse rewards',
        'agents': [
            ('Flat Agent', 'flat', {}),
            ('Fractal Agent (Sparse)', 'fractal', {'reward_shaping': 'sparse'})
        ]
    },
    
    'fractal_vs_attention': {
        'description': 'Compare fractal agent vs fractal agent with attention',
        'agents': [
            ('Fractal Agent', 'fractal', {'reward_shaping': 'shaped'}),
            ('Fractal Attention Agent', 'attention', {'reward_shaping': 'shaped'})
        ]
    },
    
    'all_agents_shaped': {
        'description': 'Compare all agent types with shaped rewards',
        'agents': [
            ('Flat Agent', 'flat', {}),
            ('Fractal Agent', 'fractal', {'reward_shaping': 'shaped'}),
            ('Fractal Attention Agent', 'attention', {'reward_shaping': 'shaped'})
        ]
    }
}


def create_agent(agent_type, env, **kwargs):
    """Create an agent of the specified type."""
    if agent_type == 'flat':
        return FlatAgent(env, **kwargs)
    elif agent_type == 'fractal':
        return FractalAgent(env, **kwargs)
    elif agent_type == 'attention':
        return FractalAttentionAgent(env, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def compare_agents_summary(logs, labels, training_times=None):
    """
    Generate a summary comparison of multiple agents.
    
    Args:
        logs: List of training logs
        labels: List of agent labels
        training_times: List of training times (optional)
    """
    print("\n" + "="*70)
    print("AGENT PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    metrics = {}
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        metrics[label] = {
            'Final Performance (steps)': log[-1],
            'Best Performance (steps)': min(log),
            'Average Performance (last 50 episodes)': np.mean(log[-50:]),
            'Standard Deviation (last 50 episodes)': np.std(log[-50:]),
            'Total Steps Taken': np.sum(log),
        }
        
        if training_times:
            metrics[label]['Training Time (seconds)'] = training_times[i]
        
        # Episodes to reach good performance (< 100 steps for 10 consecutive episodes)
        window = 10
        threshold = 100
        episodes_to_good = None
        
        for j in range(len(log) - window + 1):
            if np.mean(log[j:j+window]) < threshold:
                episodes_to_good = j
                break
                
        metrics[label]['Episodes to Good Performance'] = episodes_to_good
    
    # Print table
    metric_names = list(next(iter(metrics.values())).keys())
    metric_width = max(len(name) for name in metric_names) + 2
    value_width = 20
    
    # Header
    header = f"{'Metric':<{metric_width}}"
    for label in labels:
        header += f" | {label:<{value_width}}"
    print(header)
    print("-" * len(header))
    
    # Metrics
    for metric in metric_names:
        row = f"{metric:<{metric_width}}"
        for label in labels:
            value = metrics[label][metric]
            if value is None:
                value_str = "N/A"
            elif isinstance(value, (int, np.integer)):
                value_str = str(value)
            else:
                value_str = f"{value:.2f}"
            row += f" | {value_str:<{value_width}}"
        print(row)
    
    print("\n" + "="*70)


def run_single_experiment(experiment_config, episodes=50, horizon=200, 
                         env_size=20, seed=0, save_data=True):
    """
    Run a single experiment with the specified configuration.
    
    Args:
        experiment_config: Experiment configuration dict
        episodes: Number of training episodes
        horizon: Maximum steps per episode
        env_size: Grid environment size
        seed: Random seed
        save_data: Whether to save training logs
    
    Returns:
        results: Dict containing logs, agents, and timing info
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_config['description'].upper()}")
    print(f"{'='*70}")
    
    # Create environment
    env = GridEnvironment(size=env_size, seed=seed)
    print(f"Environment: {env_size}x{env_size} grid with {len(env.obstacles)} obstacles")
    
    # Initialize results
    results = {
        'logs': [],
        'labels': [],
        'agents': [],
        'training_times': [],
        'config': experiment_config
    }
    
    # Train each agent
    for label, agent_type, agent_kwargs in experiment_config['agents']:
        print(f"\nTraining {label}...")
        
        # Create and train agent
        agent = create_agent(agent_type, env, **agent_kwargs)
        log, training_time = agent.train(episodes=episodes, horizon=horizon)
        
        # Store results
        results['logs'].append(log)
        results['labels'].append(label)
        results['agents'].append(agent)
        results['training_times'].append(training_time)
        
        print(f"  Final performance: {log[-1]} steps")
        print(f"  Best performance: {min(log)} steps")
        print(f"  Training time: {training_time:.2f} seconds")
        
        # Save individual log if requested
        if save_data:
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            filename = label.lower().replace(' ', '_').replace('(', '').replace(')', '') + '_log.npy'
            np.save(data_dir / filename, np.array(log))
    
    # Print summary comparison
    compare_agents_summary(results['logs'], results['labels'], results['training_times'])
    
    return results


def run_experiment_by_name(experiment_name, **kwargs):
    """Run a predefined experiment by name."""
    if experiment_name not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment_name}'")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return None
    
    return run_single_experiment(EXPERIMENTS[experiment_name], **kwargs)


def list_experiments():
    """List all available predefined experiments."""
    print("Available Experiments:")
    print("=" * 50)
    for name, config in EXPERIMENTS.items():
        print(f"{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Agents: {[agent[0] for agent in config['agents']]}")
        print()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Run grid-world RL experiments (no viz)')
    
    parser.add_argument('experiment', nargs='?', default='fractal_vs_attention',
                       help='Experiment name to run (default: fractal_vs_attention)')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (default: 50)')
    parser.add_argument('--horizon', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    parser.add_argument('--env-size', type=int, default=20,
                       help='Grid environment size (default: 20)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--no-save-data', action='store_true',
                       help='Don\'t save training logs')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Run the experiment
    results = run_experiment_by_name(
        args.experiment,
        episodes=args.episodes,
        horizon=args.horizon,
        env_size=args.env_size,
        seed=args.seed,
        save_data=not args.no_save_data
    )
    
    if results:
        print(f"\nExperiment '{args.experiment}' completed successfully!")


if __name__ == "__main__":
    main() 