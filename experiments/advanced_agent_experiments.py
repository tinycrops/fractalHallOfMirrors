#!/usr/bin/env python3
"""
Advanced agent experiments testing novel approaches and innovations.

This script tests cutting-edge agent implementations including:
- Adaptive hierarchical structures
- Curiosity-driven exploration
- Multi-head attention mechanisms
- Meta-learning capabilities
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
from tinycrops_hall_of_mirrors.grid_world.advanced_agents import (
    AdaptiveFractalAgent, CuriosityDrivenAgent, 
    MultiHeadAttentionAgent, MetaLearningAgent
)


# Advanced experiment configurations
ADVANCED_EXPERIMENTS = {
    'adaptive_vs_static': {
        'description': 'Compare adaptive hierarchy vs static fractal agent',
        'agents': [
            ('Fractal Agent (Static)', 'fractal', {'reward_shaping': 'shaped'}),
            ('Adaptive Fractal Agent', 'adaptive', {'reward_shaping': 'shaped', 'min_block_size': 3, 'max_block_size': 8})
        ]
    },
    
    'curiosity_exploration': {
        'description': 'Compare standard vs curiosity-driven exploration',
        'agents': [
            ('Fractal Attention Agent', 'attention', {'reward_shaping': 'shaped'}),
            ('Curiosity-Driven Agent', 'curiosity', {'reward_shaping': 'shaped', 'curiosity_weight': 0.1})
        ]
    },
    
    'attention_mechanisms': {
        'description': 'Compare single vs multi-head attention',
        'agents': [
            ('Single Attention Agent', 'attention', {'reward_shaping': 'shaped'}),
            ('Multi-Head Attention Agent', 'multihead', {'reward_shaping': 'shaped', 'num_heads': 3})
        ]
    },
    
    'meta_learning': {
        'description': 'Test meta-learning adaptation capabilities',
        'agents': [
            ('Fractal Attention Agent', 'attention', {'reward_shaping': 'shaped'}),
            ('Meta-Learning Agent', 'meta', {'reward_shaping': 'shaped', 'strategy_memory_size': 50})
        ]
    },
    
    'novel_approaches_showcase': {
        'description': 'Showcase all novel agent approaches',
        'agents': [
            ('Fractal Agent (Baseline)', 'fractal', {'reward_shaping': 'shaped'}),
            ('Adaptive Hierarchy', 'adaptive', {'reward_shaping': 'shaped'}),
            ('Curiosity-Driven', 'curiosity', {'reward_shaping': 'shaped', 'curiosity_weight': 0.15}),
            ('Multi-Head Attention', 'multihead', {'reward_shaping': 'shaped', 'num_heads': 3}),
            ('Meta-Learning', 'meta', {'reward_shaping': 'shaped'})
        ]
    },
    
    'exploration_study': {
        'description': 'Study different exploration mechanisms',
        'agents': [
            ('ε-greedy (Standard)', 'attention', {'reward_shaping': 'sparse'}),
            ('Curiosity-Driven', 'curiosity', {'reward_shaping': 'sparse', 'curiosity_weight': 0.2}),
            ('Adaptive + Curiosity', 'adaptive_curiosity', {'reward_shaping': 'sparse'})
        ]
    }
}


def create_agent(agent_type, env, **kwargs):
    """Create an agent of the specified type including advanced agents."""
    if agent_type == 'flat':
        return FlatAgent(env, **kwargs)
    elif agent_type == 'fractal':
        return FractalAgent(env, **kwargs)
    elif agent_type == 'attention':
        return FractalAttentionAgent(env, **kwargs)
    elif agent_type == 'adaptive':
        return AdaptiveFractalAgent(env, **kwargs)
    elif agent_type == 'curiosity':
        return CuriosityDrivenAgent(env, **kwargs)
    elif agent_type == 'multihead':
        return MultiHeadAttentionAgent(env, **kwargs)
    elif agent_type == 'meta':
        return MetaLearningAgent(env, **kwargs)
    elif agent_type == 'adaptive_curiosity':
        # Hybrid agent combining adaptive hierarchy with curiosity
        return CuriosityDrivenAdaptiveAgent(env, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


class CuriosityDrivenAdaptiveAgent(AdaptiveFractalAgent, CuriosityDrivenAgent):
    """
    Novel Hybrid: Combines adaptive hierarchy with curiosity-driven exploration.
    
    This demonstrates how novel approaches can be combined for even better performance.
    """
    
    def __init__(self, env, **kwargs):
        # Initialize both parent classes
        AdaptiveFractalAgent.__init__(self, env, **kwargs)
        
        # Add curiosity-specific attributes
        self.curiosity_weight = kwargs.get('curiosity_weight', 0.1)
        self.prediction_lr = kwargs.get('prediction_lr', 0.01)
        
        # Curiosity tracking
        from collections import defaultdict
        self.state_prediction_errors = defaultdict(list)
        self.state_visit_counts = defaultdict(int)
        self.intrinsic_rewards = []
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Use curiosity method from CuriosityDrivenAgent."""
        return CuriosityDrivenAgent.compute_intrinsic_reward(self, state, action, next_state)
    
    def _predict_next_state(self, state, action):
        """Use prediction method from CuriosityDrivenAgent."""
        return CuriosityDrivenAgent._predict_next_state(self, state, action)
    
    def train(self, episodes=600, horizon=500):
        """Combine adaptive hierarchy with curiosity-driven training."""
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training Adaptive+Curiosity Agent"):
            pos = self.env.reset()
            done = False
            primitive_steps = 0
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                # Use adaptive hierarchy for goal generation
                s_super = self.idx_super(pos)
                a_super = choose_action(self.Q_super, s_super, epsilon)
                
                target_super_block = np.add(divmod(s_super, self.env.size // self.block_macro), 
                                          self.env.actions[a_super])
                target_super_block = np.clip(target_super_block, 0, 
                                           self.env.size // self.block_macro - 1)
                super_goal = tuple(target_super_block * self.block_macro + self.block_macro // 2)
                
                s_mac = self.idx_macro(pos)
                a_mac = choose_action(self.Q_macro, s_mac, epsilon)
                
                target_block = np.add(divmod(s_mac, self.env.size // self.block_micro), 
                                    self.env.actions[a_mac])
                target_block = np.clip(target_block, 0, 
                                     self.env.size // self.block_micro - 1)
                macro_goal = tuple(target_block * self.block_micro + self.block_micro // 2)
                
                # Execute with curiosity-driven exploration
                for _ in range(self.block_micro * self.block_micro):
                    if done:
                        break
                        
                    s_mic = self.idx_micro(pos)
                    a_mic = choose_action(self.Q_micro, s_mic, epsilon)
                    
                    nxt, extrinsic_reward, done = self.env.step(pos, a_mic)
                    primitive_steps += 1
                    
                    # Add intrinsic reward
                    intrinsic_reward = self.compute_intrinsic_reward(s_mic, a_mic, self.idx_micro(nxt))
                    total_reward = extrinsic_reward + intrinsic_reward
                    
                    s2_mic = self.idx_micro(nxt)
                    self.micro_buffer.append((s_mic, a_mic, total_reward, s2_mic, done))
                    
                    pos = nxt
                    
                    if done or pos == macro_goal:
                        break
                
                # Standard hierarchical updates
                if done:
                    r_mac = 10
                elif pos == macro_goal:
                    r_mac = 5
                elif pos == super_goal:
                    r_mac = 3
                else:
                    r_mac = -1 + self._compute_shaped_reward(pos, macro_goal, 'macro')
                
                s2_mac = self.idx_macro(pos)
                self.macro_buffer.append((s_mac, a_mac, r_mac, s2_mac, done))
                
                if done:
                    r_super = 10
                elif pos == super_goal:
                    r_super = 5
                else:
                    r_super = -1 + self._compute_shaped_reward(pos, super_goal, 'super')
                
                s2_super = self.idx_super(pos)
                self.super_buffer.append((s_super, a_super, r_super, s2_super, done))
                
                self.update_hierarchical_q_tables()
            
            log.append(primitive_steps)
            self.performance_history.append(primitive_steps)
            
            # Periodic adaptation
            if ep % 25 == 0 and ep > 0:
                self.adapt_hierarchy()
            
            # Additional batch updates
            for _ in range(5):
                self.update_hierarchical_q_tables()
                
        training_time = time.time() - start_time
        
        print(f"  Intrinsic reward avg: {np.mean(self.intrinsic_rewards):.4f}")
        print(f"  States explored: {len(self.state_visit_counts)}")
        
        return log, training_time


def compare_agents_summary(logs, labels, training_times=None):
    """Enhanced comparison with advanced metrics."""
    print("\n" + "="*80)
    print("ADVANCED AGENT PERFORMANCE COMPARISON")
    print("="*80)
    
    metrics = {}
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        # Standard metrics
        metrics[label] = {
            'Final Performance': log[-1],
            'Best Performance': min(log),
            'Mean Performance': np.mean(log),
            'Std Performance': np.std(log),
            'Total Steps': np.sum(log),
            'Sample Efficiency': np.sum(log) / len(log),
        }
        
        if training_times:
            metrics[label]['Training Time'] = training_times[i]
        
        # Advanced metrics
        
        # Learning rate (improvement over episodes)
        if len(log) > 20:
            early_perf = np.mean(log[:10])
            late_perf = np.mean(log[-10:])
            learning_rate = (early_perf - late_perf) / early_perf if early_perf > 0 else 0
            metrics[label]['Learning Rate'] = learning_rate
        
        # Stability (inverse of variance in later episodes)
        if len(log) > 30:
            late_stability = 1.0 / (1.0 + np.var(log[-20:]))
            metrics[label]['Stability'] = late_stability
        
        # Convergence episodes (when performance stabilizes)
        convergence_ep = None
        if len(log) > 40:
            target_perf = min(log) * 1.2  # Within 20% of best
            for j in range(20, len(log) - 10):
                if np.mean(log[j:j+10]) <= target_perf:
                    convergence_ep = j
                    break
        metrics[label]['Convergence Episode'] = convergence_ep
    
    # Print enhanced table
    metric_names = list(next(iter(metrics.values())).keys())
    
    print(f"{'Metric':<25}", end="")
    for label in labels:
        print(f" | {label:<18}", end="")
    print()
    print("-" * (25 + len(labels) * 21))
    
    for metric in metric_names:
        print(f"{metric:<25}", end="")
        for label in labels:
            value = metrics[label][metric]
            if value is None:
                value_str = "N/A"
            elif isinstance(value, (int, np.integer)):
                value_str = str(value)
            elif isinstance(value, float):
                if metric in ['Learning Rate', 'Stability']:
                    value_str = f"{value:.3f}"
                else:
                    value_str = f"{value:.1f}"
            else:
                value_str = str(value)
            print(f" | {value_str:<18}", end="")
        print()
    
    print("\n" + "="*80)
    
    # Performance ranking
    print("\nPERFORMANCE RANKING:")
    print("-" * 30)
    
    # Rank by sample efficiency (lower is better)
    ranking = sorted(labels, key=lambda l: metrics[l]['Sample Efficiency'])
    
    for i, label in enumerate(ranking):
        efficiency = metrics[label]['Sample Efficiency']
        learning_rate = metrics[label].get('Learning Rate', 0)
        print(f"{i+1}. {label}: {efficiency:.1f} avg steps/episode "
              f"(learning rate: {learning_rate:.3f})")


def run_single_experiment(experiment_config, episodes=50, horizon=200, 
                         env_size=20, seed=0, save_data=True):
    """Run advanced experiment with enhanced analysis."""
    print(f"\n{'='*80}")
    print(f"ADVANCED EXPERIMENT: {experiment_config['description'].upper()}")
    print(f"{'='*80}")
    
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
        
        try:
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
            print(f"  Average performance: {np.mean(log):.1f} steps")
            print(f"  Training time: {training_time:.2f} seconds")
            
            # Agent-specific insights
            if hasattr(agent, 'intrinsic_rewards') and agent.intrinsic_rewards:
                print(f"  Avg intrinsic reward: {np.mean(agent.intrinsic_rewards):.4f}")
                print(f"  States explored: {len(agent.state_visit_counts)}")
            
            if hasattr(agent, 'strategy_library'):
                print(f"  Strategy library size: {len(agent.strategy_library)}")
                
            if hasattr(agent, 'adaptation_episode'):
                print(f"  Hierarchy adaptations: {agent.adaptation_episode}")
            
            # Save individual log if requested
            if save_data:
                data_dir = Path('data')
                data_dir.mkdir(exist_ok=True)
                filename = label.lower().replace(' ', '_').replace('(', '').replace(')', '') + '_log.npy'
                np.save(data_dir / filename, np.array(log))
                
        except Exception as e:
            print(f"  ERROR training {label}: {e}")
            import traceback
            traceback.print_exc()
    
    # Enhanced comparison analysis
    if len(results['logs']) > 1:
        compare_agents_summary(results['logs'], results['labels'], results['training_times'])
    
    return results


def run_experiment_by_name(experiment_name, **kwargs):
    """Run a predefined advanced experiment by name."""
    if experiment_name not in ADVANCED_EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment_name}'")
        print(f"Available experiments: {list(ADVANCED_EXPERIMENTS.keys())}")
        return None
    
    return run_single_experiment(ADVANCED_EXPERIMENTS[experiment_name], **kwargs)


def list_experiments():
    """List all available advanced experiments."""
    print("Advanced Agent Experiments:")
    print("=" * 60)
    for name, config in ADVANCED_EXPERIMENTS.items():
        print(f"{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Agents: {[agent[0] for agent in config['agents']]}")
        print()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Run advanced grid-world RL experiments')
    
    parser.add_argument('experiment', nargs='?', default='novel_approaches_showcase',
                       help='Experiment name to run (default: novel_approaches_showcase)')
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
        print(f"\nAdvanced experiment '{args.experiment}' completed successfully!")
        print("\nKey Innovations Demonstrated:")
        print("- Adaptive hierarchical structures")
        print("- Curiosity-driven exploration")
        print("- Multi-head attention mechanisms")
        print("- Meta-learning strategy adaptation")
        print("- Novel hybrid approaches")


if __name__ == "__main__":
    main() 