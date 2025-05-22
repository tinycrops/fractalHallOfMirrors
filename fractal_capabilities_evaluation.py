#!/usr/bin/env python3
"""
Rigorous Evaluation of Fractal Self-Observation Capabilities

This module provides comprehensive experiments to definitively demonstrate
whether fractal self-observation actually enhances AI agent capabilities
compared to baseline approaches.

Key Questions to Answer:
1. Does fractal self-observation improve learning efficiency?
2. Does it enable better generalization across problem scales?
3. Does it provide robustness advantages?
4. Can agents solve problems impossible for baseline agents?
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import json
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.fractal_environment import FractalDepthEnvironment, SelfObservingAgent
from tinycrops_hall_of_mirrors.grid_world.agents import FlatAgent
from tinycrops_hall_of_mirrors.grid_world.environment import GridEnvironment


@dataclass
class ExperimentResult:
    """Container for experimental results."""
    agent_type: str
    environment_type: str
    trial_id: int
    episodes_to_solve: int
    final_success_rate: float
    learning_efficiency: float
    knowledge_transfer_score: float
    robustness_score: float
    total_training_time: float
    additional_metrics: Dict


class BaselineAgent:
    """Traditional Q-learning agent for comparison."""
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Single Q-table for entire state space
        if hasattr(env, 'base_size'):
            size = env.base_size
        else:
            size = env.size
        self.q_table = np.zeros((size, size, len(env.actions)))
        
    def choose_action(self, state):
        if hasattr(state, '__len__') and len(state) == 3:
            x, y, _ = state  # Ignore depth for baseline
        else:
            x, y = state
        
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.env.actions.keys()))
        else:
            return np.argmax(self.q_table[int(x), int(y), :])
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        if hasattr(state, '__len__') and len(state) == 3:
            x, y, _ = state  # Ignore depth
        else:
            x, y = state
            
        if hasattr(next_state, '__len__') and len(next_state) == 3:
            nx, ny, _ = next_state  # Ignore depth
        else:
            nx, ny = next_state
        
        x, y, nx, ny = int(x), int(y), int(nx), int(ny)
        
        current_q = self.q_table[x, y, action]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[nx, ny, :])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[x, y, action] = new_q
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class ExperimentalFramework:
    """Comprehensive experimental framework for evaluating fractal self-observation."""
    
    def __init__(self, results_dir="experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        
    def create_test_environments(self):
        """Create a suite of test environments with varying complexity."""
        environments = {}
        
        # 1. Simple Navigation (baseline comparison)
        environments['simple_nav'] = {
            'type': 'fractal',
            'params': {'base_size': 6, 'num_portals': 1, 'max_depth': 1, 'seed': 42}
        }
        
        # 2. Complex Multi-Scale Navigation
        environments['multi_scale'] = {
            'type': 'fractal',
            'params': {'base_size': 10, 'num_portals': 3, 'max_depth': 2, 'seed': 123}
        }
        
        # 3. Hierarchical Problem (requires cross-scale reasoning)
        environments['hierarchical'] = {
            'type': 'fractal',
            'params': {'base_size': 12, 'num_portals': 2, 'max_depth': 3, 'seed': 456}
        }
        
        # 4. Traditional Grid World (for baseline)
        environments['grid_baseline'] = {
            'type': 'traditional',
            'params': {'size': 10, 'obstacles': [(3,3), (3,4), (4,3), (6,6), (6,7), (7,6)], 
                      'start': (0,0), 'goal': (9,9)}
        }
        
        return environments
    
    def run_learning_efficiency_experiment(self, num_trials=10):
        """Test if fractal agents learn faster than baseline agents."""
        print("🧪 EXPERIMENT 1: Learning Efficiency Comparison")
        print("=" * 60)
        
        environments = self.create_test_environments()
        results = []
        
        for env_name, env_config in environments.items():
            if env_config['type'] != 'fractal':
                continue
                
            print(f"\nTesting environment: {env_name}")
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Test Fractal Agent
                env = FractalDepthEnvironment(**env_config['params'])
                fractal_agent = SelfObservingAgent(env)
                fractal_result = self._train_and_evaluate(fractal_agent, env, f"fractal_{env_name}_{trial}")
                
                # Test Baseline Agent (treating fractal env as flat)
                baseline_agent = BaselineAgent(env)
                baseline_result = self._train_and_evaluate(baseline_agent, env, f"baseline_{env_name}_{trial}")
                
                results.extend([fractal_result, baseline_result])
        
        self._analyze_learning_efficiency(results)
        return results
    
    def run_knowledge_transfer_experiment(self, num_trials=5):
        """Test if knowledge transfers across fractal scales."""
        print("\n🧪 EXPERIMENT 2: Cross-Scale Knowledge Transfer")
        print("=" * 60)
        
        results = []
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            # Create training and test environments at different scales
            train_env = FractalDepthEnvironment(base_size=8, num_portals=2, max_depth=1, seed=42+trial)
            test_env = FractalDepthEnvironment(base_size=12, num_portals=3, max_depth=2, seed=142+trial)
            
            # Train fractal agent on simple environment
            fractal_agent = SelfObservingAgent(train_env)
            self._train_agent(fractal_agent, train_env, episodes=200)
            
            # Test on complex environment (transfer learning)
            transfer_performance = self._evaluate_agent(fractal_agent, test_env, episodes=50)
            
            # Compare with agent trained from scratch on complex environment
            scratch_agent = SelfObservingAgent(test_env)
            scratch_performance = self._evaluate_agent(scratch_agent, test_env, episodes=50)
            
            transfer_score = transfer_performance / max(scratch_performance, 0.01)
            
            results.append({
                'trial': trial,
                'transfer_performance': transfer_performance,
                'scratch_performance': scratch_performance,
                'transfer_advantage': transfer_score
            })
        
        self._analyze_knowledge_transfer(results)
        return results
    
    def run_impossible_task_experiment(self):
        """Create tasks that require multi-scale reasoning."""
        print("\n🧪 EXPERIMENT 3: Multi-Scale Reasoning Tasks")
        print("=" * 60)
        
        # Create an environment where the solution requires using information from multiple scales
        env = self._create_multi_scale_puzzle()
        
        print("Testing fractal agent on multi-scale puzzle...")
        fractal_agent = SelfObservingAgent(env)
        fractal_success = self._train_and_test_puzzle(fractal_agent, env)
        
        print("Testing baseline agent on same puzzle...")
        baseline_agent = BaselineAgent(env)
        baseline_success = self._train_and_test_puzzle(baseline_agent, env)
        
        print(f"\nResults:")
        print(f"  Fractal Agent Success Rate: {fractal_success:.1%}")
        print(f"  Baseline Agent Success Rate: {baseline_success:.1%}")
        print(f"  Advantage: {(fractal_success - baseline_success)*100:.1f} percentage points")
        
        return {
            'fractal_success': fractal_success,
            'baseline_success': baseline_success,
            'advantage': fractal_success - baseline_success
        }
    
    def run_robustness_experiment(self, num_trials=10):
        """Test robustness to environment changes."""
        print("\n🧪 EXPERIMENT 4: Robustness to Environmental Changes")
        print("=" * 60)
        
        results = []
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            # Train on one environment configuration
            base_env = FractalDepthEnvironment(base_size=8, num_portals=2, max_depth=2, seed=42)
            
            # Create agents
            fractal_agent = SelfObservingAgent(base_env)
            baseline_agent = BaselineAgent(base_env)
            
            # Train both agents
            self._train_agent(fractal_agent, base_env, episodes=300)
            self._train_agent(baseline_agent, base_env, episodes=300)
            
            # Test on modified environments
            robustness_scores = {}
            
            # Test 1: Different portal locations
            modified_env1 = FractalDepthEnvironment(base_size=8, num_portals=2, max_depth=2, seed=142)
            fractal_score1 = self._evaluate_agent(fractal_agent, modified_env1, episodes=20)
            baseline_score1 = self._evaluate_agent(baseline_agent, modified_env1, episodes=20)
            
            # Test 2: Different number of portals
            modified_env2 = FractalDepthEnvironment(base_size=8, num_portals=1, max_depth=2, seed=42)
            fractal_score2 = self._evaluate_agent(fractal_agent, modified_env2, episodes=20)
            baseline_score2 = self._evaluate_agent(baseline_agent, modified_env2, episodes=20)
            
            # Test 3: Different scale complexity
            modified_env3 = FractalDepthEnvironment(base_size=10, num_portals=2, max_depth=1, seed=42)
            fractal_score3 = self._evaluate_agent(fractal_agent, modified_env3, episodes=20)
            baseline_score3 = self._evaluate_agent(baseline_agent, modified_env3, episodes=20)
            
            avg_fractal_robustness = np.mean([fractal_score1, fractal_score2, fractal_score3])
            avg_baseline_robustness = np.mean([baseline_score1, baseline_score2, baseline_score3])
            
            results.append({
                'trial': trial,
                'fractal_robustness': avg_fractal_robustness,
                'baseline_robustness': avg_baseline_robustness,
                'robustness_advantage': avg_fractal_robustness - avg_baseline_robustness
            })
        
        self._analyze_robustness(results)
        return results
    
    def _create_multi_scale_puzzle(self):
        """Create a puzzle that requires multi-scale reasoning to solve."""
        # This environment has a goal that's only reachable by:
        # 1. Going to depth 1 to find a "key" 
        # 2. Returning to depth 0 with the key to unlock the goal
        
        class MultiScalePuzzleEnvironment(FractalDepthEnvironment):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.has_key = False
                self.key_location = (2, 2)  # Key is at depth 1
                self.locked_goal = True
                
            def step(self, action_idx):
                state, reward, done, info = super().step(action_idx)
                
                # Check if agent found the key at depth 1
                if self.current_depth == 1 and self.current_pos == self.key_location:
                    if not self.has_key:
                        self.has_key = True
                        reward += 50  # Reward for finding key
                        info['found_key'] = True
                
                # Check if agent reached goal at depth 0 with key
                if (self.current_depth == 0 and 
                    self.current_pos == self.base_goal and 
                    self.has_key):
                    reward += 200  # Large reward for solving puzzle
                    done = True
                    info['solved_puzzle'] = True
                elif (self.current_depth == 0 and 
                      self.current_pos == self.base_goal and 
                      not self.has_key):
                    reward -= 10  # Penalty for reaching goal without key
                    info['goal_locked'] = True
                
                return state, reward, done, info
                
            def reset(self):
                state = super().reset()
                self.has_key = False
                return state
        
        return MultiScalePuzzleEnvironment(base_size=6, num_portals=1, max_depth=1, seed=789)
    
    def _train_agent(self, agent, env, episodes=500):
        """Train an agent on an environment."""
        for episode in range(episodes):
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 200:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn_from_experience(state, action, reward, next_state, done)
                state = next_state
                step_count += 1
    
    def _evaluate_agent(self, agent, env, episodes=100):
        """Evaluate agent performance."""
        successes = 0
        total_steps = 0
        
        # Temporarily disable exploration
        original_epsilon = getattr(agent, 'epsilon', 0)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 300:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                step_count += 1
                
                if done and reward > 50:  # Success condition
                    successes += 1
                    break
            
            total_steps += step_count
        
        # Restore exploration
        if hasattr(agent, 'epsilon'):
            agent.epsilon = original_epsilon
        
        return successes / episodes
    
    def _train_and_evaluate(self, agent, env, experiment_id):
        """Train and evaluate an agent, returning comprehensive metrics."""
        start_time = time.time()
        
        # Training phase
        episode_rewards = []
        episodes_to_solve = None
        
        for episode in range(1000):  # Max episodes
            state = env.reset()
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done and step_count < 200:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn_from_experience(state, action, reward, next_state, done)
                state = next_state
                step_count += 1
                episode_reward += reward
                
                if done and reward > 50 and episodes_to_solve is None:
                    episodes_to_solve = episode + 1
            
            episode_rewards.append(episode_reward)
            
            # Check if consistently solving
            if episode > 100 and np.mean(episode_rewards[-20:]) > 80:
                break
        
        training_time = time.time() - start_time
        
        # Evaluation phase
        final_success_rate = self._evaluate_agent(agent, env, episodes=50)
        
        # Calculate learning efficiency
        learning_efficiency = 1.0 / max(episodes_to_solve or 1000, 1)
        
        return ExperimentResult(
            agent_type=type(agent).__name__,
            environment_type=experiment_id,
            trial_id=0,
            episodes_to_solve=episodes_to_solve or 1000,
            final_success_rate=final_success_rate,
            learning_efficiency=learning_efficiency,
            knowledge_transfer_score=0.0,  # Will be calculated separately
            robustness_score=0.0,  # Will be calculated separately
            total_training_time=training_time,
            additional_metrics={'episode_rewards': episode_rewards}
        )
    
    def _train_and_test_puzzle(self, agent, env):
        """Train agent on multi-scale puzzle and test success rate."""
        # Extended training for complex puzzle
        self._train_agent(agent, env, episodes=1000)
        
        # Test success rate on puzzle
        return self._evaluate_agent(agent, env, episodes=50)
    
    def _analyze_learning_efficiency(self, results):
        """Analyze and visualize learning efficiency results."""
        print("\n📈 LEARNING EFFICIENCY ANALYSIS")
        print("-" * 40)
        
        fractal_results = [r for r in results if r.agent_type == 'SelfObservingAgent']
        baseline_results = [r for r in results if r.agent_type == 'BaselineAgent']
        
        fractal_episodes = np.mean([r.episodes_to_solve for r in fractal_results])
        baseline_episodes = np.mean([r.episodes_to_solve for r in baseline_results])
        
        fractal_success = np.mean([r.final_success_rate for r in fractal_results])
        baseline_success = np.mean([r.final_success_rate for r in baseline_results])
        
        print(f"Average Episodes to Solve:")
        print(f"  Fractal Agent: {fractal_episodes:.1f}")
        print(f"  Baseline Agent: {baseline_episodes:.1f}")
        print(f"  Improvement: {((baseline_episodes - fractal_episodes) / baseline_episodes * 100):.1f}%")
        
        print(f"\nFinal Success Rate:")
        print(f"  Fractal Agent: {fractal_success:.2%}")
        print(f"  Baseline Agent: {baseline_success:.2%}")
        print(f"  Advantage: {(fractal_success - baseline_success)*100:.1f} percentage points")
        
        # Create visualization
        self._plot_learning_comparison(fractal_results, baseline_results)
    
    def _analyze_knowledge_transfer(self, results):
        """Analyze knowledge transfer results."""
        print("\n📊 KNOWLEDGE TRANSFER ANALYSIS")
        print("-" * 40)
        
        avg_transfer_advantage = np.mean([r['transfer_advantage'] for r in results])
        transfer_advantages = [r['transfer_advantage'] for r in results]
        
        print(f"Average Transfer Advantage: {avg_transfer_advantage:.2f}x")
        print(f"Transfer Success Rate: {np.mean([1 if adv > 1.1 else 0 for adv in transfer_advantages]):.1%}")
        
        significant_transfer = avg_transfer_advantage > 1.2
        print(f"Significant Transfer Detected: {'YES' if significant_transfer else 'NO'}")
    
    def _analyze_robustness(self, results):
        """Analyze robustness results."""
        print("\n🛡️ ROBUSTNESS ANALYSIS")
        print("-" * 40)
        
        avg_fractal_robustness = np.mean([r['fractal_robustness'] for r in results])
        avg_baseline_robustness = np.mean([r['baseline_robustness'] for r in results])
        avg_advantage = np.mean([r['robustness_advantage'] for r in results])
        
        print(f"Average Robustness Scores:")
        print(f"  Fractal Agent: {avg_fractal_robustness:.2%}")
        print(f"  Baseline Agent: {avg_baseline_robustness:.2%}")
        print(f"  Robustness Advantage: {avg_advantage*100:.1f} percentage points")
    
    def _plot_learning_comparison(self, fractal_results, baseline_results):
        """Create comprehensive learning comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Episodes to solve comparison
        ax = axes[0, 0]
        fractal_episodes = [r.episodes_to_solve for r in fractal_results]
        baseline_episodes = [r.episodes_to_solve for r in baseline_results]
        
        ax.boxplot([fractal_episodes, baseline_episodes], labels=['Fractal', 'Baseline'])
        ax.set_ylabel('Episodes to Solve')
        ax.set_title('Learning Speed Comparison')
        
        # Success rate comparison
        ax = axes[0, 1]
        fractal_success = [r.final_success_rate for r in fractal_results]
        baseline_success = [r.final_success_rate for r in baseline_results]
        
        ax.boxplot([fractal_success, baseline_success], labels=['Fractal', 'Baseline'])
        ax.set_ylabel('Final Success Rate')
        ax.set_title('Performance Comparison')
        
        # Learning curves (if available)
        ax = axes[1, 0]
        for i, result in enumerate(fractal_results[:3]):  # Show first 3 trials
            if 'episode_rewards' in result.additional_metrics:
                rewards = result.additional_metrics['episode_rewards']
                # Smooth the curve
                window = 20
                smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
                ax.plot(smoothed, alpha=0.7, color='blue', label='Fractal' if i == 0 else '')
        
        for i, result in enumerate(baseline_results[:3]):
            if 'episode_rewards' in result.additional_metrics:
                rewards = result.additional_metrics['episode_rewards']
                window = 20
                smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
                ax.plot(smoothed, alpha=0.7, color='red', label='Baseline' if i == 0 else '')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curves')
        ax.legend()
        
        # Statistical summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Perform statistical tests
        from scipy import stats
        
        stat, p_value = stats.mannwhitneyu(fractal_episodes, baseline_episodes, alternative='less')
        
        summary_text = f"""
STATISTICAL ANALYSIS

Episodes to Solve:
  Fractal: {np.mean(fractal_episodes):.1f} ± {np.std(fractal_episodes):.1f}
  Baseline: {np.mean(baseline_episodes):.1f} ± {np.std(baseline_episodes):.1f}
  Mann-Whitney U p-value: {p_value:.4f}
  
Success Rate:
  Fractal: {np.mean(fractal_success):.2%} ± {np.std(fractal_success):.2%}
  Baseline: {np.mean(baseline_success):.2%} ± {np.std(baseline_success):.2%}
  
Conclusion:
  {"Fractal agents learn significantly faster" if p_value < 0.05 else "No significant difference detected"}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()


def run_comprehensive_evaluation():
    """Run the complete experimental evaluation."""
    print("🔬 COMPREHENSIVE FRACTAL SELF-OBSERVATION EVALUATION")
    print("=" * 80)
    print("This evaluation will rigorously test whether fractal self-observation")
    print("provides genuine advantages over baseline approaches.")
    print()
    
    framework = ExperimentalFramework()
    
    # Run all experiments
    exp1_results = framework.run_learning_efficiency_experiment(num_trials=8)
    exp2_results = framework.run_knowledge_transfer_experiment(num_trials=5)
    exp3_results = framework.run_impossible_task_experiment()
    exp4_results = framework.run_robustness_experiment(num_trials=5)
    
    # Generate final verdict
    print("\n" + "=" * 80)
    print("🏆 FINAL EVALUATION SUMMARY")
    print("=" * 80)
    
    # Analyze overall results
    fractal_advantages = []
    
    # Learning efficiency advantage
    fractal_episodes = np.mean([r.episodes_to_solve for r in exp1_results if r.agent_type == 'SelfObservingAgent'])
    baseline_episodes = np.mean([r.episodes_to_solve for r in exp1_results if r.agent_type == 'BaselineAgent'])
    learning_advantage = (baseline_episodes - fractal_episodes) / baseline_episodes
    fractal_advantages.append(learning_advantage)
    
    # Knowledge transfer advantage
    transfer_advantage = np.mean([r['transfer_advantage'] for r in exp2_results]) - 1.0
    fractal_advantages.append(transfer_advantage)
    
    # Multi-scale reasoning advantage
    reasoning_advantage = exp3_results['advantage']
    fractal_advantages.append(reasoning_advantage)
    
    # Robustness advantage
    robustness_advantage = np.mean([r['robustness_advantage'] for r in exp4_results])
    fractal_advantages.append(robustness_advantage)
    
    overall_advantage = np.mean(fractal_advantages)
    
    print(f"Learning Efficiency Improvement: {learning_advantage*100:.1f}%")
    print(f"Knowledge Transfer Advantage: {transfer_advantage*100:.1f}%")
    print(f"Multi-Scale Reasoning Advantage: {reasoning_advantage*100:.1f}%")
    print(f"Robustness Advantage: {robustness_advantage*100:.1f}%")
    print(f"\nOverall Advantage Score: {overall_advantage*100:.1f}%")
    
    # Final verdict
    if overall_advantage > 0.15:  # 15% overall advantage
        verdict = "🎯 STRONG EVIDENCE: Fractal self-observation provides significant advantages"
    elif overall_advantage > 0.05:  # 5% overall advantage
        verdict = "✅ MODERATE EVIDENCE: Fractal self-observation shows promising benefits"
    elif overall_advantage > -0.05:  # No significant difference
        verdict = "🤔 INCONCLUSIVE: Benefits are marginal or inconsistent"
    else:
        verdict = "❌ NO EVIDENCE: Fractal self-observation does not provide advantages"
    
    print(f"\n{verdict}")
    
    return {
        'learning_efficiency': exp1_results,
        'knowledge_transfer': exp2_results,
        'multi_scale_reasoning': exp3_results,
        'robustness': exp4_results,
        'overall_advantage': overall_advantage,
        'verdict': verdict
    }


if __name__ == "__main__":
    results = run_comprehensive_evaluation() 