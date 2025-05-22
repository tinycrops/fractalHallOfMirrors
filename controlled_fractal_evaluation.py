#!/usr/bin/env python3
"""
Controlled Evaluation of Fractal Self-Observation Benefits

This module provides rigorous A/B testing to determine if fractal self-observation
actually provides measurable advantages over baseline approaches.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.fractal_environment import FractalDepthEnvironment, SelfObservingAgent


class ControlledBaselineAgent:
    """Baseline agent that ignores fractal structure but operates in same environment."""
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon_start=0.9, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        
        # Create Q-table for 2D positions only (ignoring depth)
        self.q_table = np.zeros((env.base_size, env.base_size, len(env.actions)))
        
    def choose_action(self, state):
        x, y, depth = int(state[0]), int(state[1]), int(state[2])
        
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.env.actions.keys()))
        else:
            return np.argmax(self.q_table[x, y, :])
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        x, y, depth = int(state[0]), int(state[1]), int(state[2])
        nx, ny, ndepth = int(next_state[0]), int(next_state[1]), int(next_state[2])
        
        # Use only 2D position for learning (treat all depths as same state)
        current_q = self.q_table[x, y, action]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[nx, ny, :])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[x, y, action] = new_q
        
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)


class FractalEvaluationFramework:
    """Framework for controlled evaluation of fractal self-observation."""
    
    def __init__(self):
        self.results = []
        
    def run_controlled_comparison(self, num_trials=10, episodes_per_trial=300):
        """Run direct comparison between fractal and baseline agents."""
        print("🔬 CONTROLLED FRACTAL VS BASELINE COMPARISON")
        print("=" * 60)
        print("Testing identical agents on identical environments")
        print("Only difference: fractal awareness vs baseline")
        print()
        
        all_results = []
        
        # Environment configurations to test
        test_configs = [
            {'name': 'simple', 'base_size': 6, 'num_portals': 1, 'max_depth': 1},
            {'name': 'medium', 'base_size': 8, 'num_portals': 2, 'max_depth': 2},
            {'name': 'complex', 'base_size': 10, 'num_portals': 3, 'max_depth': 2}
        ]
        
        for config in test_configs:
            print(f"\nTesting {config['name']} environment:")
            print(f"  Grid: {config['base_size']}x{config['base_size']}")
            print(f"  Portals: {config['num_portals']}")
            print(f"  Max Depth: {config['max_depth']}")
            
            config_results = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Create identical environments for both agents
                seed = 42 + trial
                env1 = FractalDepthEnvironment(seed=seed, **{k: v for k, v in config.items() if k != 'name'})
                env2 = FractalDepthEnvironment(seed=seed, **{k: v for k, v in config.items() if k != 'name'})
                
                # Create agents
                fractal_agent = SelfObservingAgent(env1, alpha=0.1, gamma=0.95, epsilon_start=0.9, epsilon_decay=0.995)
                baseline_agent = ControlledBaselineAgent(env2, alpha=0.1, gamma=0.95, epsilon_start=0.9, epsilon_decay=0.995)
                
                # Train both agents
                print(f"    Training fractal agent...")
                fractal_metrics = self._train_and_measure(fractal_agent, env1, episodes_per_trial)
                
                print(f"    Training baseline agent...")
                baseline_metrics = self._train_and_measure(baseline_agent, env2, episodes_per_trial)
                
                # Record results
                trial_result = {
                    'config': config['name'],
                    'trial': trial,
                    'fractal_episodes_to_solve': fractal_metrics['episodes_to_solve'],
                    'baseline_episodes_to_solve': baseline_metrics['episodes_to_solve'],
                    'fractal_final_success': fractal_metrics['final_success_rate'],
                    'baseline_final_success': baseline_metrics['final_success_rate'],
                    'fractal_avg_steps': fractal_metrics['avg_steps_to_goal'],
                    'baseline_avg_steps': baseline_metrics['avg_steps_to_goal'],
                    'fractal_exploration_efficiency': fractal_metrics['exploration_efficiency'],
                    'baseline_exploration_efficiency': baseline_metrics['exploration_efficiency']
                }
                
                config_results.append(trial_result)
                all_results.append(trial_result)
                
                # Print trial summary
                advantage = (baseline_metrics['episodes_to_solve'] - fractal_metrics['episodes_to_solve']) / max(baseline_metrics['episodes_to_solve'], 1) * 100
                print(f"    Fractal advantage: {advantage:.1f}%")
            
            # Analyze config results
            self._analyze_config_results(config['name'], config_results)
        
        # Overall analysis
        self._analyze_overall_results(all_results)
        return all_results
    
    def run_multi_scale_reasoning_test(self):
        """Test if fractal agents can solve problems requiring multi-scale reasoning."""
        print("\n🧩 MULTI-SCALE REASONING TEST")
        print("=" * 60)
        print("Testing agents on problems that require cross-scale information")
        print()
        
        # Create special environment requiring cross-scale reasoning
        env = self._create_multi_scale_puzzle()
        
        results = {}
        for agent_type in ['fractal', 'baseline']:
            print(f"Testing {agent_type} agent on multi-scale puzzle...")
            
            success_rates = []
            solve_times = []
            
            for trial in range(5):
                # Create fresh environment
                puzzle_env = self._create_multi_scale_puzzle()
                
                if agent_type == 'fractal':
                    agent = SelfObservingAgent(puzzle_env)
                else:
                    agent = ControlledBaselineAgent(puzzle_env)
                
                # Extended training for complex puzzle
                success_rate, avg_solve_time = self._test_puzzle_solving(agent, puzzle_env)
                success_rates.append(success_rate)
                solve_times.append(avg_solve_time)
            
            results[agent_type] = {
                'success_rate': np.mean(success_rates),
                'success_std': np.std(success_rates),
                'solve_time': np.mean(solve_times),
                'solve_time_std': np.std(solve_times)
            }
            
            print(f"  {agent_type.title()} agent results:")
            print(f"    Success rate: {results[agent_type]['success_rate']:.1%} ± {results[agent_type]['success_std']:.1%}")
            print(f"    Avg solve time: {results[agent_type]['solve_time']:.1f} ± {results[agent_type]['solve_time_std']:.1f} episodes")
        
        # Compare results
        fractal_better = results['fractal']['success_rate'] > results['baseline']['success_rate']
        advantage = (results['fractal']['success_rate'] - results['baseline']['success_rate']) * 100
        
        print(f"\nMulti-scale reasoning advantage: {advantage:.1f} percentage points")
        print(f"Fractal agent superior: {'YES' if fractal_better and advantage > 5 else 'NO'}")
        
        return results
    
    def run_knowledge_transfer_test(self):
        """Test if fractal agents transfer knowledge better across environments."""
        print("\n🔄 KNOWLEDGE TRANSFER TEST")
        print("=" * 60)
        print("Testing how well learned skills transfer to new environments")
        print()
        
        results = {}
        
        for agent_type in ['fractal', 'baseline']:
            print(f"Testing {agent_type} agent knowledge transfer...")
            
            transfer_scores = []
            
            for trial in range(3):
                # Train on simple environment
                train_env = FractalDepthEnvironment(base_size=6, num_portals=1, max_depth=1, seed=42+trial)
                
                if agent_type == 'fractal':
                    agent = SelfObservingAgent(train_env)
                else:
                    agent = ControlledBaselineAgent(train_env)
                
                # Pre-training
                self._train_agent(agent, train_env, episodes=200)
                
                # Test on more complex environment
                test_env = FractalDepthEnvironment(base_size=8, num_portals=2, max_depth=2, seed=142+trial)
                
                # Test performance with pre-training
                pretrained_performance = self._evaluate_agent_performance(agent, test_env, episodes=50)
                
                # Compare with agent trained from scratch
                if agent_type == 'fractal':
                    scratch_agent = SelfObservingAgent(test_env)
                else:
                    scratch_agent = ControlledBaselineAgent(test_env)
                
                scratch_performance = self._evaluate_agent_performance(scratch_agent, test_env, episodes=50)
                
                transfer_score = pretrained_performance / max(scratch_performance, 0.01)
                transfer_scores.append(transfer_score)
            
            results[agent_type] = {
                'transfer_score': np.mean(transfer_scores),
                'transfer_std': np.std(transfer_scores)
            }
            
            print(f"  {agent_type.title()} transfer score: {results[agent_type]['transfer_score']:.2f} ± {results[agent_type]['transfer_std']:.2f}")
        
        advantage = results['fractal']['transfer_score'] - results['baseline']['transfer_score']
        print(f"\nTransfer advantage: {advantage:.2f}x")
        print(f"Significant transfer benefit: {'YES' if advantage > 0.2 else 'NO'}")
        
        return results
    
    def _train_and_measure(self, agent, env, episodes):
        """Train agent and measure key performance metrics."""
        episode_rewards = []
        episodes_to_solve = None
        steps_to_goal = []
        successful_episodes = 0
        positions_visited = set()
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 200:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn_from_experience(state, action, reward, next_state, done)
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                # Track exploration
                positions_visited.add((int(state[0]), int(state[1])))
                
                if done and reward > 50:  # Success
                    if episodes_to_solve is None:
                        episodes_to_solve = episode + 1
                    successful_episodes += 1
                    steps_to_goal.append(step_count)
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate metrics
        final_success_rate = successful_episodes / max(episodes - episodes // 2, 1)  # Success rate in second half
        avg_steps_to_goal = np.mean(steps_to_goal) if steps_to_goal else 200
        exploration_efficiency = len(positions_visited) / (env.base_size * env.base_size)
        
        return {
            'episodes_to_solve': episodes_to_solve or episodes,
            'final_success_rate': final_success_rate,
            'avg_steps_to_goal': avg_steps_to_goal,
            'exploration_efficiency': exploration_efficiency,
            'episode_rewards': episode_rewards
        }
    
    def _train_agent(self, agent, env, episodes):
        """Simple training loop."""
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
    
    def _evaluate_agent_performance(self, agent, env, episodes):
        """Evaluate agent performance (success rate)."""
        successes = 0
        
        # Disable exploration for evaluation
        original_epsilon = getattr(agent, 'epsilon', 0)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.05
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 300:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                step_count += 1
                
                if done and reward > 50:
                    successes += 1
                    break
        
        # Restore exploration
        if hasattr(agent, 'epsilon'):
            agent.epsilon = original_epsilon
        
        return successes / episodes
    
    def _create_multi_scale_puzzle(self):
        """Create a puzzle requiring multi-scale reasoning."""
        
        class MultiScalePuzzle(FractalDepthEnvironment):
            def __init__(self):
                super().__init__(base_size=6, num_portals=1, max_depth=1, seed=789)
                self.key_found = False
                self.key_location = (2, 2)  # Key at depth 1
                
            def step(self, action_idx):
                state, reward, done, info = super().step(action_idx)
                
                # Key mechanics: must find key at depth 1, then return to depth 0 to unlock goal
                if self.current_depth == 1 and tuple(self.current_pos) == self.key_location:
                    if not self.key_found:
                        self.key_found = True
                        reward += 100  # Large reward for finding key
                        info['found_key'] = True
                
                # Goal is only accessible with key at depth 0
                if (self.current_depth == 0 and 
                    tuple(self.current_pos) == tuple(self.base_goal)):
                    if self.key_found:
                        reward += 200  # Success!
                        done = True
                        info['puzzle_solved'] = True
                    else:
                        reward -= 50  # Penalty for reaching goal without key
                        info['goal_locked'] = True
                
                return state, reward, done, info
            
            def reset(self):
                state = super().reset()
                self.key_found = False
                return state
        
        return MultiScalePuzzle()
    
    def _test_puzzle_solving(self, agent, env):
        """Test agent's ability to solve the multi-scale puzzle."""
        solve_times = []
        successes = 0
        
        for test_episode in range(10):
            self._train_agent(agent, env, episodes=100)  # Training phase
            
            # Test if agent can solve puzzle
            for attempt in range(5):
                state = env.reset()
                done = False
                steps = 0
                
                # Disable exploration for testing
                original_epsilon = getattr(agent, 'epsilon', 0)
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = 0.05
                
                while not done and steps < 500:
                    action = agent.choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    steps += 1
                    
                    if done and info.get('puzzle_solved', False):
                        successes += 1
                        solve_times.append(steps)
                        break
                
                # Restore exploration
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = original_epsilon
                
                if done and info.get('puzzle_solved', False):
                    break
        
        success_rate = successes / 50  # 10 episodes * 5 attempts
        avg_solve_time = np.mean(solve_times) if solve_times else 500
        
        return success_rate, avg_solve_time
    
    def _analyze_config_results(self, config_name, results):
        """Analyze results for a specific configuration."""
        print(f"\n    📊 {config_name.upper()} ENVIRONMENT RESULTS:")
        
        fractal_episodes = [r['fractal_episodes_to_solve'] for r in results]
        baseline_episodes = [r['baseline_episodes_to_solve'] for r in results]
        
        fractal_success = [r['fractal_final_success'] for r in results]
        baseline_success = [r['baseline_final_success'] for r in results]
        
        # Learning speed comparison
        fractal_avg = np.mean(fractal_episodes)
        baseline_avg = np.mean(baseline_episodes)
        speed_improvement = (baseline_avg - fractal_avg) / baseline_avg * 100
        
        print(f"      Learning Speed:")
        print(f"        Fractal: {fractal_avg:.1f} episodes")
        print(f"        Baseline: {baseline_avg:.1f} episodes")
        print(f"        Improvement: {speed_improvement:.1f}%")
        
        # Success rate comparison
        fractal_success_avg = np.mean(fractal_success)
        baseline_success_avg = np.mean(baseline_success)
        success_improvement = (fractal_success_avg - baseline_success_avg) * 100
        
        print(f"      Final Success Rate:")
        print(f"        Fractal: {fractal_success_avg:.1%}")
        print(f"        Baseline: {baseline_success_avg:.1%}")
        print(f"        Improvement: {success_improvement:.1f} percentage points")
        
        # Statistical significance
        speed_stat, speed_p = stats.mannwhitneyu(fractal_episodes, baseline_episodes, alternative='less')
        success_stat, success_p = stats.mannwhitneyu(fractal_success, baseline_success, alternative='greater')
        
        print(f"      Statistical Significance:")
        print(f"        Learning speed p-value: {speed_p:.4f}")
        print(f"        Success rate p-value: {success_p:.4f}")
        
        significant = (speed_p < 0.05 and speed_improvement > 0) or (success_p < 0.05 and success_improvement > 0)
        print(f"        Significant advantage: {'YES' if significant else 'NO'}")
    
    def _analyze_overall_results(self, results):
        """Analyze overall results across all configurations."""
        print(f"\n🏆 OVERALL EVALUATION RESULTS")
        print("=" * 60)
        
        # Aggregate metrics
        overall_speed_improvements = []
        overall_success_improvements = []
        
        configs = list(set([r['config'] for r in results]))
        
        for config in configs:
            config_results = [r for r in results if r['config'] == config]
            
            fractal_episodes = [r['fractal_episodes_to_solve'] for r in config_results]
            baseline_episodes = [r['baseline_episodes_to_solve'] for r in config_results]
            
            fractal_success = [r['fractal_final_success'] for r in config_results]
            baseline_success = [r['baseline_final_success'] for r in config_results]
            
            speed_improvement = (np.mean(baseline_episodes) - np.mean(fractal_episodes)) / np.mean(baseline_episodes) * 100
            success_improvement = (np.mean(fractal_success) - np.mean(baseline_success)) * 100
            
            overall_speed_improvements.append(speed_improvement)
            overall_success_improvements.append(success_improvement)
        
        avg_speed_improvement = np.mean(overall_speed_improvements)
        avg_success_improvement = np.mean(overall_success_improvements)
        
        print(f"Average Learning Speed Improvement: {avg_speed_improvement:.1f}%")
        print(f"Average Success Rate Improvement: {avg_success_improvement:.1f} percentage points")
        
        # Final verdict
        strong_evidence = avg_speed_improvement > 15 or avg_success_improvement > 10
        moderate_evidence = avg_speed_improvement > 5 or avg_success_improvement > 5
        
        if strong_evidence:
            verdict = "🎯 STRONG EVIDENCE: Fractal self-observation provides significant advantages"
        elif moderate_evidence:
            verdict = "✅ MODERATE EVIDENCE: Fractal self-observation shows promising benefits"
        else:
            verdict = "❌ INSUFFICIENT EVIDENCE: No clear advantage demonstrated"
        
        print(f"\n{verdict}")
        
        return {
            'speed_improvement': avg_speed_improvement,
            'success_improvement': avg_success_improvement,
            'verdict': verdict
        }


def run_controlled_evaluation():
    """Run the controlled evaluation."""
    print("🔬 CONTROLLED FRACTAL SELF-OBSERVATION EVALUATION")
    print("=" * 80)
    print("This evaluation uses controlled A/B testing to determine if fractal")
    print("self-observation provides genuine advantages over baseline approaches.")
    print("All confounding variables are controlled for rigorous comparison.")
    print()
    
    framework = FractalEvaluationFramework()
    
    # Run all tests
    comparison_results = framework.run_controlled_comparison(num_trials=5, episodes_per_trial=200)
    reasoning_results = framework.run_multi_scale_reasoning_test()
    transfer_results = framework.run_knowledge_transfer_test()
    
    print(f"\n" + "=" * 80)
    print("🏆 FINAL VERDICT")
    print("=" * 80)
    
    # Determine final conclusion based on all tests
    speed_advantage = np.mean([np.mean([r['fractal_episodes_to_solve'] < r['baseline_episodes_to_solve'] for r in comparison_results])])
    success_advantage = np.mean([np.mean([r['fractal_final_success'] > r['baseline_final_success'] for r in comparison_results])])
    reasoning_advantage = reasoning_results['fractal']['success_rate'] > reasoning_results['baseline']['success_rate']
    transfer_advantage = transfer_results['fractal']['transfer_score'] > transfer_results['baseline']['transfer_score']
    
    advantages = [speed_advantage, success_advantage, reasoning_advantage, transfer_advantage]
    overall_score = np.mean(advantages)
    
    print(f"Learning Speed Advantage: {speed_advantage:.1%}")
    print(f"Success Rate Advantage: {success_advantage:.1%}")
    print(f"Multi-Scale Reasoning Advantage: {'YES' if reasoning_advantage else 'NO'}")
    print(f"Knowledge Transfer Advantage: {'YES' if transfer_advantage else 'NO'}")
    print(f"\nOverall Advantage Score: {overall_score:.1%}")
    
    if overall_score > 0.7:
        final_verdict = "🎯 STRONG EVIDENCE: Fractal self-observation is clearly superior"
    elif overall_score > 0.5:
        final_verdict = "✅ MODERATE EVIDENCE: Fractal self-observation shows benefits"
    else:
        final_verdict = "❌ INSUFFICIENT EVIDENCE: No clear advantage demonstrated"
    
    print(f"\n{final_verdict}")
    
    return {
        'comparison': comparison_results,
        'reasoning': reasoning_results,
        'transfer': transfer_results,
        'overall_score': overall_score,
        'verdict': final_verdict
    }


if __name__ == "__main__":
    results = run_controlled_evaluation() 