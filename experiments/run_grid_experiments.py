#!/usr/bin/env python3
"""
Consolidated grid-world experiment script.

This script replaces the individual comparison scripts and provides
a unified interface for running and comparing different grid-world agents.
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
from tinycrops_hall_of_mirrors.grid_world.visualization import (
    plot_learning_curve, plot_q_values, plot_hierarchical_q_values,
    animate_agent_path, plot_attention_evolution, compare_agents_summary
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
    
    'fractal_reward_comparison': {
        'description': 'Compare fractal agents with different reward shaping',
        'agents': [
            ('Fractal Agent (Shaped)', 'fractal', {'reward_shaping': 'shaped'}),
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
    },
    
    'all_agents_sparse': {
        'description': 'Compare all agent types with sparse rewards',
        'agents': [
            ('Flat Agent', 'flat', {}),
            ('Fractal Agent', 'fractal', {'reward_shaping': 'sparse'}),
            ('Fractal Attention Agent', 'attention', {'reward_shaping': 'sparse'})
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


def run_single_experiment(experiment_config, episodes=600, horizon=500, 
                         env_size=20, seed=0, save_data=True, show_plots=True,
                         save_plots=False):
    """
    Run a single experiment with the specified configuration.
    
    Args:
        experiment_config: Experiment configuration dict
        episodes: Number of training episodes
        horizon: Maximum steps per episode
        env_size: Grid environment size
        seed: Random seed
        save_data: Whether to save training logs
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
    
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
    
    # Generate comparison plots
    if show_plots or save_plots:
        print(f"\nGenerating comparison plots...")
        
        # Learning curves
        save_path = 'data/learning_curves.png' if save_plots else None
        plot_learning_curve(results['logs'], results['labels'], 
                           title=experiment_config['description'], 
                           save_path=save_path)
        
        # Q-value visualizations for each agent
        for i, (agent, label) in enumerate(zip(results['agents'], results['labels'])):
            save_path = f'data/{label.lower().replace(" ", "_")}_qvalues.png' if save_plots else None
            
            if hasattr(agent, 'Q_super'):
                # Hierarchical agent
                plot_hierarchical_q_values(agent, env, title=f"{label} Q-Values", 
                                          save_path=save_path)
            else:
                # Flat agent
                plot_q_values(agent, env, title=f"{label} Q-Values", 
                            save_path=save_path)
            
            # Animate agent path
            if show_plots:
                save_path = f'data/{label.lower().replace(" ", "_")}_path.gif' if save_plots else None
                animate_agent_path(agent, env, title=f"{label} Path", 
                                 save_path=save_path, 
                                 show_hierarchical=hasattr(agent, 'Q_super'))
        
        # Attention evolution for attention agents
        for agent, label in zip(results['agents'], results['labels']):
            if hasattr(agent, 'attention_history') and agent.attention_history:
                save_path = f'data/{label.lower().replace(" ", "_")}_attention.png' if save_plots else None
                plot_attention_evolution(agent.attention_history, 
                                       title=f"{label} Attention Evolution",
                                       save_path=save_path)
    
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
    parser = argparse.ArgumentParser(description='Run grid-world RL experiments')
    
    parser.add_argument('experiment', nargs='?', default='all_agents_shaped',
                       help='Experiment name to run (default: all_agents_shaped)')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    parser.add_argument('--episodes', type=int, default=600,
                       help='Number of training episodes (default: 600)')
    parser.add_argument('--horizon', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--env-size', type=int, default=20,
                       help='Grid environment size (default: 20)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--no-save-data', action='store_true',
                       help='Don\'t save training logs')
    parser.add_argument('--no-show-plots', action='store_true',
                       help='Don\'t display plots')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    
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
        save_data=not args.no_save_data,
        show_plots=not args.no_show_plots,
        save_plots=args.save_plots
    )
    
    if results:
        print(f"\nExperiment '{args.experiment}' completed successfully!")
        if args.save_plots:
            print("Plots saved to 'data/' directory")


if __name__ == "__main__":
    main() 