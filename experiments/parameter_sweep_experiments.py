#!/usr/bin/env python3
"""
Advanced Parameter Sweep and Ablation Study Framework.

This script provides rigorous analysis of novel agent hyperparameters and components:
- Parameter sweeps for curiosity weight, adaptation rates, attention heads
- Ablation studies for multi-head attention, meta-learning components
- Statistical significance testing and confidence intervals
- Advanced visualization of parameter sensitivity
"""

import sys
import os
import numpy as np
import argparse
import time
from pathlib import Path
from itertools import product
from collections import defaultdict
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.environment import GridEnvironment
from tinycrops_hall_of_mirrors.grid_world.advanced_agents import (
    AdaptiveFractalAgent, CuriosityDrivenAgent, 
    MultiHeadAttentionAgent, MetaLearningAgent
)


class ParameterSweepRunner:
    """Advanced parameter sweep framework with statistical analysis."""
    
    def __init__(self, base_config=None):
        self.base_config = base_config or {
            'episodes': 100,
            'horizon': 200,
            'env_size': 20,
            'num_seeds': 5,
            'save_data': True
        }
        self.results = {}
        
    def run_curiosity_sweep(self):
        """Parameter sweep for CuriosityDrivenAgent."""
        print("="*80)
        print("CURIOSITY-DRIVEN AGENT PARAMETER SWEEP")
        print("="*80)
        
        curiosity_weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        prediction_lrs = [0.01, 0.02, 0.05]
        
        sweep_results = defaultdict(list)
        
        for curiosity_weight, prediction_lr in product(curiosity_weights, prediction_lrs):
            print(f"\nTesting curiosity_weight={curiosity_weight}, prediction_lr={prediction_lr}")
            
            config_results = []
            for seed in range(self.base_config['num_seeds']):
                env = GridEnvironment(size=self.base_config['env_size'], seed=seed)
                agent = CuriosityDrivenAgent(
                    env, 
                    curiosity_weight=curiosity_weight,
                    prediction_lr=prediction_lr,
                    reward_shaping='shaped'
                )
                
                log, training_time = agent.train(
                    episodes=self.base_config['episodes'],
                    horizon=self.base_config['horizon']
                )
                
                # Collect detailed metrics
                metrics = {
                    'final_performance': log[-1],
                    'best_performance': min(log),
                    'mean_performance': np.mean(log),
                    'convergence_episode': self._find_convergence_episode(log),
                    'learning_rate': self._compute_learning_rate(log),
                    'states_explored': len(agent.state_visit_counts),
                    'avg_intrinsic_reward': np.mean(agent.intrinsic_rewards) if agent.intrinsic_rewards else 0,
                    'training_time': training_time
                }
                config_results.append(metrics)
                
                print(f"  Seed {seed}: {metrics['mean_performance']:.1f} steps, "
                      f"{metrics['states_explored']} states explored")
            
            # Statistical analysis
            key = f"cw_{curiosity_weight}_lr_{prediction_lr}"
            sweep_results[key] = self._compute_statistics(config_results)
            
        # Analyze results
        self._analyze_curiosity_sweep(sweep_results, curiosity_weights, prediction_lrs)
        return sweep_results
    
    def run_adaptive_sweep(self):
        """Parameter sweep for AdaptiveFractalAgent."""
        print("="*80)
        print("ADAPTIVE FRACTAL AGENT PARAMETER SWEEP")
        print("="*80)
        
        min_block_sizes = [2, 3, 4]
        max_block_sizes = [6, 8, 10]
        adaptation_rates = [0.05, 0.1, 0.15, 0.2]
        
        sweep_results = defaultdict(list)
        
        for min_size, max_size, adapt_rate in product(min_block_sizes, max_block_sizes, adaptation_rates):
            if min_size >= max_size:
                continue
                
            print(f"\nTesting min_block={min_size}, max_block={max_size}, adapt_rate={adapt_rate}")
            
            config_results = []
            for seed in range(self.base_config['num_seeds']):
                env = GridEnvironment(size=self.base_config['env_size'], seed=seed)
                agent = AdaptiveFractalAgent(
                    env,
                    min_block_size=min_size,
                    max_block_size=max_size,
                    adaptation_rate=adapt_rate,
                    reward_shaping='shaped'
                )
                
                log, training_time = agent.train(
                    episodes=self.base_config['episodes'],
                    horizon=self.base_config['horizon']
                )
                
                metrics = {
                    'final_performance': log[-1],
                    'best_performance': min(log),
                    'mean_performance': np.mean(log),
                    'convergence_episode': self._find_convergence_episode(log),
                    'learning_rate': self._compute_learning_rate(log),
                    'final_block_micro': agent.block_micro,
                    'final_block_macro': agent.block_macro,
                    'adaptation_count': len(agent.performance_history),
                    'training_time': training_time
                }
                config_results.append(metrics)
                
                print(f"  Seed {seed}: {metrics['mean_performance']:.1f} steps, "
                      f"final blocks: {metrics['final_block_micro']}/{metrics['final_block_macro']}")
            
            key = f"min_{min_size}_max_{max_size}_rate_{adapt_rate}"
            sweep_results[key] = self._compute_statistics(config_results)
            
        self._analyze_adaptive_sweep(sweep_results, min_block_sizes, max_block_sizes, adaptation_rates)
        return sweep_results
    
    def run_multihead_ablation(self):
        """Ablation study for MultiHeadAttentionAgent."""
        print("="*80)
        print("MULTI-HEAD ATTENTION ABLATION STUDY")
        print("="*80)
        
        # Test different numbers of heads
        num_heads_configs = [1, 2, 3, 4, 5]
        ablation_results = {}
        
        for num_heads in num_heads_configs:
            print(f"\nTesting {num_heads} attention heads...")
            
            config_results = []
            for seed in range(self.base_config['num_seeds']):
                env = GridEnvironment(size=self.base_config['env_size'], seed=seed)
                agent = MultiHeadAttentionAgent(
                    env,
                    num_heads=num_heads,
                    reward_shaping='shaped'
                )
                
                log, training_time = agent.train(
                    episodes=self.base_config['episodes'],
                    horizon=self.base_config['horizon']
                )
                
                # Analyze attention patterns
                attention_diversity = self._compute_attention_diversity(agent)
                
                metrics = {
                    'final_performance': log[-1],
                    'best_performance': min(log),
                    'mean_performance': np.mean(log),
                    'convergence_episode': self._find_convergence_episode(log),
                    'learning_rate': self._compute_learning_rate(log),
                    'attention_diversity': attention_diversity,
                    'training_time': training_time
                }
                config_results.append(metrics)
                
                print(f"  Seed {seed}: {metrics['mean_performance']:.1f} steps, "
                      f"attention diversity: {attention_diversity:.3f}")
            
            ablation_results[f"{num_heads}_heads"] = self._compute_statistics(config_results)
        
        # Analyze optimal number of heads
        self._analyze_multihead_ablation(ablation_results, num_heads_configs)
        return ablation_results
    
    def run_meta_learning_ablation(self):
        """Ablation study for MetaLearningAgent components."""
        print("="*80)
        print("META-LEARNING AGENT ABLATION STUDY")
        print("="*80)
        
        # Test different strategy memory sizes and similarity thresholds
        memory_sizes = [10, 25, 50, 100]
        
        ablation_results = {}
        
        for memory_size in memory_sizes:
            print(f"\nTesting strategy memory size {memory_size}...")
            
            config_results = []
            for seed in range(self.base_config['num_seeds']):
                env = GridEnvironment(size=self.base_config['env_size'], seed=seed)
                agent = MetaLearningAgent(
                    env,
                    strategy_memory_size=memory_size,
                    reward_shaping='shaped'
                )
                
                log, training_time = agent.train(
                    episodes=self.base_config['episodes'],
                    horizon=self.base_config['horizon']
                )
                
                metrics = {
                    'final_performance': log[-1],
                    'best_performance': min(log),
                    'mean_performance': np.mean(log),
                    'convergence_episode': self._find_convergence_episode(log),
                    'learning_rate': self._compute_learning_rate(log),
                    'strategies_learned': len(agent.strategy_library),
                    'env_obstacle_density': agent.env_characteristics.get('obstacle_density', 0),
                    'training_time': training_time
                }
                config_results.append(metrics)
                
                print(f"  Seed {seed}: {metrics['mean_performance']:.1f} steps, "
                      f"{metrics['strategies_learned']} strategies learned")
            
            ablation_results[f"memory_{memory_size}"] = self._compute_statistics(config_results)
        
        self._analyze_meta_learning_ablation(ablation_results, memory_sizes)
        return ablation_results
    
    def _find_convergence_episode(self, log, threshold=0.2):
        """Find when agent converges (performance stabilizes)."""
        if len(log) < 20:
            return None
            
        best_performance = min(log)
        target_performance = best_performance * (1 + threshold)
        
        for i in range(10, len(log) - 10):
            if np.mean(log[i:i+10]) <= target_performance:
                return i
        return None
    
    def _compute_learning_rate(self, log):
        """Compute learning improvement rate."""
        if len(log) < 20:
            return 0
            
        early_performance = np.mean(log[:10])
        late_performance = np.mean(log[-10:])
        
        if early_performance <= 0:
            return 0
            
        return (early_performance - late_performance) / early_performance
    
    def _compute_attention_diversity(self, agent):
        """Compute diversity of attention head usage."""
        if not hasattr(agent, 'attention_head_history') or not agent.attention_head_history:
            return 0
            
        # Compute entropy across attention heads
        attention_matrix = np.array(agent.attention_head_history)
        if attention_matrix.size == 0:
            return 0
            
        # Average attention weights across time
        avg_attention = np.mean(attention_matrix, axis=0)
        
        # Compute diversity (entropy) across heads and levels
        diversity = 0
        for head_idx in range(avg_attention.shape[0]):
            head_weights = avg_attention[head_idx]
            # Normalize to probabilities
            head_weights = head_weights / (np.sum(head_weights) + 1e-8)
            # Compute entropy
            entropy = -np.sum(head_weights * np.log(head_weights + 1e-8))
            diversity += entropy
            
        return diversity / avg_attention.shape[0]  # Average across heads
    
    def _compute_statistics(self, results_list):
        """Compute statistical measures across seeds."""
        metrics = defaultdict(list)
        
        for result in results_list:
            for key, value in result.items():
                if value is not None:
                    metrics[key].append(value)
        
        stats = {}
        for key, values in metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'confidence_interval': self._compute_confidence_interval(values)
                }
            
        return stats
    
    def _compute_confidence_interval(self, values, confidence=0.95):
        """Compute confidence interval for the mean."""
        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std_err = np.std(values) / np.sqrt(n)
        
        # Use t-distribution for small samples
        from scipy import stats as scipy_stats
        try:
            t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_val * std_err
        except ImportError:
            # Fallback to normal approximation
            z_val = 1.96  # 95% confidence
            margin = z_val * std_err
            
        return (mean - margin, mean + margin)
    
    def _analyze_curiosity_sweep(self, results, curiosity_weights, prediction_lrs):
        """Analyze curiosity parameter sweep results."""
        print("\n" + "="*80)
        print("CURIOSITY PARAMETER SWEEP ANALYSIS")
        print("="*80)
        
        # Find best configuration
        best_config = None
        best_performance = float('inf')
        
        print(f"{'Curiosity Weight':<15} {'Prediction LR':<12} {'Mean Perf':<12} {'CI Lower':<10} {'CI Upper':<10} {'States':<8}")
        print("-" * 75)
        
        for cw in curiosity_weights:
            for lr in prediction_lrs:
                key = f"cw_{cw}_lr_{lr}"
                if key in results:
                    stats = results[key]
                    mean_perf = stats['mean_performance']['mean']
                    ci_lower, ci_upper = stats['mean_performance']['confidence_interval']
                    states_explored = stats['states_explored']['mean']
                    
                    print(f"{cw:<15.2f} {lr:<12.3f} {mean_perf:<12.1f} {ci_lower:<10.1f} {ci_upper:<10.1f} {states_explored:<8.0f}")
                    
                    if mean_perf < best_performance:
                        best_performance = mean_perf
                        best_config = (cw, lr)
        
        if best_config:
            print(f"\n🏆 BEST CONFIGURATION: curiosity_weight={best_config[0]}, prediction_lr={best_config[1]}")
            print(f"   Performance: {best_performance:.1f} ± {results[f'cw_{best_config[0]}_lr_{best_config[1]}']['mean_performance']['std']:.1f} steps")
    
    def _analyze_adaptive_sweep(self, results, min_sizes, max_sizes, adapt_rates):
        """Analyze adaptive parameter sweep results."""
        print("\n" + "="*80)
        print("ADAPTIVE HIERARCHY PARAMETER SWEEP ANALYSIS")
        print("="*80)
        
        best_config = None
        best_performance = float('inf')
        
        print(f"{'Min Block':<10} {'Max Block':<10} {'Adapt Rate':<12} {'Mean Perf':<12} {'Final Micro':<12} {'Final Macro':<12}")
        print("-" * 80)
        
        for min_size in min_sizes:
            for max_size in max_sizes:
                for adapt_rate in adapt_rates:
                    if min_size >= max_size:
                        continue
                        
                    key = f"min_{min_size}_max_{max_size}_rate_{adapt_rate}"
                    if key in results:
                        stats = results[key]
                        mean_perf = stats['mean_performance']['mean']
                        final_micro = stats['final_block_micro']['mean']
                        final_macro = stats['final_block_macro']['mean']
                        
                        print(f"{min_size:<10} {max_size:<10} {adapt_rate:<12.2f} {mean_perf:<12.1f} {final_micro:<12.1f} {final_macro:<12.1f}")
                        
                        if mean_perf < best_performance:
                            best_performance = mean_perf
                            best_config = (min_size, max_size, adapt_rate)
        
        if best_config:
            print(f"\n🏆 BEST CONFIGURATION: min_block={best_config[0]}, max_block={best_config[1]}, adapt_rate={best_config[2]}")
            print(f"   Performance: {best_performance:.1f} steps")
    
    def _analyze_multihead_ablation(self, results, num_heads_configs):
        """Analyze multi-head attention ablation results."""
        print("\n" + "="*80)
        print("MULTI-HEAD ATTENTION ABLATION ANALYSIS")
        print("="*80)
        
        print(f"{'Num Heads':<10} {'Mean Perf':<12} {'Std':<8} {'Diversity':<12} {'Convergence':<12}")
        print("-" * 60)
        
        best_heads = None
        best_performance = float('inf')
        
        for num_heads in num_heads_configs:
            key = f"{num_heads}_heads"
            if key in results:
                stats = results[key]
                mean_perf = stats['mean_performance']['mean']
                std_perf = stats['mean_performance']['std']
                diversity = stats['attention_diversity']['mean']
                convergence = stats['convergence_episode']['mean'] if 'convergence_episode' in stats else 'N/A'
                
                print(f"{num_heads:<10} {mean_perf:<12.1f} {std_perf:<8.1f} {diversity:<12.3f} {convergence}")
                
                if mean_perf < best_performance:
                    best_performance = mean_perf
                    best_heads = num_heads
        
        if best_heads:
            print(f"\n🏆 OPTIMAL NUMBER OF HEADS: {best_heads}")
            print(f"   Performance: {best_performance:.1f} steps")
            print(f"   Attention diversity: {results[f'{best_heads}_heads']['attention_diversity']['mean']:.3f}")
    
    def _analyze_meta_learning_ablation(self, results, memory_sizes):
        """Analyze meta-learning ablation results."""
        print("\n" + "="*80)
        print("META-LEARNING ABLATION ANALYSIS")
        print("="*80)
        
        print(f"{'Memory Size':<12} {'Mean Perf':<12} {'Strategies':<12} {'Learning Rate':<14}")
        print("-" * 55)
        
        best_memory = None
        best_performance = float('inf')
        
        for memory_size in memory_sizes:
            key = f"memory_{memory_size}"
            if key in results:
                stats = results[key]
                mean_perf = stats['mean_performance']['mean']
                strategies = stats['strategies_learned']['mean']
                learning_rate = stats['learning_rate']['mean']
                
                print(f"{memory_size:<12} {mean_perf:<12.1f} {strategies:<12.1f} {learning_rate:<14.3f}")
                
                if mean_perf < best_performance:
                    best_performance = mean_perf
                    best_memory = memory_size
        
        if best_memory:
            print(f"\n🏆 OPTIMAL MEMORY SIZE: {best_memory}")
            print(f"   Performance: {best_performance:.1f} steps")
            print(f"   Strategies learned: {results[f'memory_{best_memory}']['strategies_learned']['mean']:.1f}")
    
    def save_results(self, filename=None):
        """Save all sweep results to JSON."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"data/parameter_sweep_results_{timestamp}.json"
        
        # Convert numpy types to regular Python types for JSON serialization
        json_results = {}
        for study_name, study_results in self.results.items():
            json_results[study_name] = {}
            for config_name, config_stats in study_results.items():
                json_results[study_name][config_name] = {}
                for metric_name, metric_stats in config_stats.items():
                    if isinstance(metric_stats, dict):
                        json_results[study_name][config_name][metric_name] = {
                            k: float(v) if isinstance(v, (np.integer, np.floating)) else 
                               [float(x) for x in v] if isinstance(v, (list, tuple)) else v
                            for k, v in metric_stats.items()
                        }
                    else:
                        json_results[study_name][config_name][metric_name] = metric_stats
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n📊 Results saved to {filename}")


def main():
    """Main function for parameter sweep experiments."""
    parser = argparse.ArgumentParser(description='Run parameter sweeps and ablation studies')
    
    parser.add_argument('study', nargs='?', default='all',
                       choices=['all', 'curiosity', 'adaptive', 'multihead', 'meta'],
                       help='Which study to run (default: all)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Episodes per configuration (default: 100)')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds (default: 5)')
    parser.add_argument('--env-size', type=int, default=20,
                       help='Environment size (default: 20)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Configure the sweep runner
    config = {
        'episodes': args.episodes,
        'horizon': 200,
        'env_size': args.env_size,
        'num_seeds': args.seeds,
        'save_data': True
    }
    
    runner = ParameterSweepRunner(config)
    
    print("🧪 ADVANCED PARAMETER SWEEP AND ABLATION FRAMEWORK")
    print("="*80)
    print(f"Configuration: {args.episodes} episodes × {args.seeds} seeds per setting")
    print(f"Environment: {args.env_size}×{args.env_size} grid")
    print("="*80)
    
    # Run requested studies
    if args.study in ['all', 'curiosity']:
        runner.results['curiosity_sweep'] = runner.run_curiosity_sweep()
        
    if args.study in ['all', 'adaptive']:
        runner.results['adaptive_sweep'] = runner.run_adaptive_sweep()
        
    if args.study in ['all', 'multihead']:
        runner.results['multihead_ablation'] = runner.run_multihead_ablation()
        
    if args.study in ['all', 'meta']:
        runner.results['meta_learning_ablation'] = runner.run_meta_learning_ablation()
    
    # Save results if requested
    if args.save_results:
        runner.save_results()
    
    print("\n🎯 PARAMETER SWEEP COMPLETE!")
    print("Key insights can be found in the analysis sections above.")


if __name__ == "__main__":
    main() 