"""
Advanced visualization utilities for novel grid-world agents.

This module provides specialized visualizations for:
- Multi-head attention mechanisms
- Meta-learning strategy evolution
- Curiosity-driven exploration patterns
- Adaptive hierarchical structure changes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import seaborn as sns
from collections import defaultdict


def plot_multihead_attention_analysis(agent, title="Multi-Head Attention Analysis", save_path=None):
    """
    Visualize multi-head attention patterns and diversity.
    
    Args:
        agent: MultiHeadAttentionAgent with attention history
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not hasattr(agent, 'attention_head_history') or not agent.attention_head_history:
        print("No attention history available for visualization")
        return
    
    attention_matrix = np.array(agent.attention_head_history)
    num_heads, num_levels = attention_matrix.shape[1], attention_matrix.shape[2]
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    # Plot 1: Attention evolution over time for each head
    ax1 = fig.add_subplot(gs[0, :2])
    
    head_names = ['Distance', 'Obstacle', 'Progress']
    level_names = ['Micro', 'Macro', 'Super']
    colors = ['red', 'green', 'blue']
    
    for head_idx in range(num_heads):
        for level_idx in range(num_levels):
            attention_values = attention_matrix[:, head_idx, level_idx]
            ax1.plot(attention_values, 
                    color=colors[level_idx], 
                    alpha=0.7,
                    label=f'{head_names[head_idx] if level_idx == 0 else ""} - {level_names[level_idx]}' if head_idx == 0 else "",
                    linestyle='-' if head_idx == 0 else '--' if head_idx == 1 else ':')
    
    ax1.set_title('Attention Weight Evolution Over Time')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Attention Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average attention weights per head and level
    ax2 = fig.add_subplot(gs[0, 2:])
    
    avg_attention = np.mean(attention_matrix, axis=0)
    
    im = ax2.imshow(avg_attention, cmap='viridis', aspect='auto')
    ax2.set_title('Average Attention Weights')
    ax2.set_xlabel('Hierarchical Level')
    ax2.set_ylabel('Attention Head')
    ax2.set_xticks(range(num_levels))
    ax2.set_xticklabels(level_names)
    ax2.set_yticks(range(num_heads))
    ax2.set_yticklabels(head_names)
    
    # Add text annotations
    for i in range(num_heads):
        for j in range(num_levels):
            ax2.text(j, i, f'{avg_attention[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Attention diversity over time
    ax3 = fig.add_subplot(gs[1, :2])
    
    diversity_over_time = []
    for t in range(len(attention_matrix)):
        # Compute entropy for each head at this timestep
        total_diversity = 0
        for head_idx in range(num_heads):
            head_weights = attention_matrix[t, head_idx]
            head_weights = head_weights / (np.sum(head_weights) + 1e-8)
            entropy = -np.sum(head_weights * np.log(head_weights + 1e-8))
            total_diversity += entropy
        diversity_over_time.append(total_diversity / num_heads)
    
    ax3.plot(diversity_over_time, 'purple', linewidth=2)
    ax3.set_title('Attention Diversity Over Time')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Average Entropy (Diversity)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Head specialization matrix
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Compute how much each head specializes in each level
    specialization = np.zeros((num_heads, num_levels))
    for head_idx in range(num_heads):
        for level_idx in range(num_levels):
            # Standard deviation indicates specialization
            specialization[head_idx, level_idx] = np.std(attention_matrix[:, head_idx, level_idx])
    
    im = ax4.imshow(specialization, cmap='plasma', aspect='auto')
    ax4.set_title('Head Specialization (Std Dev)')
    ax4.set_xlabel('Hierarchical Level')
    ax4.set_ylabel('Attention Head')
    ax4.set_xticks(range(num_levels))
    ax4.set_xticklabels(level_names)
    ax4.set_yticks(range(num_heads))
    ax4.set_yticklabels(head_names)
    
    for i in range(num_heads):
        for j in range(num_levels):
            ax4.text(j, i, f'{specialization[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Attention switching patterns
    ax5 = fig.add_subplot(gs[2, :])
    
    # Compute attention switches (when dominant head changes)
    switches = []
    prev_dominant = None
    
    for t in range(len(attention_matrix)):
        # Find dominant head (highest total attention)
        head_totals = np.sum(attention_matrix[t], axis=1)
        dominant_head = np.argmax(head_totals)
        
        if prev_dominant is not None and dominant_head != prev_dominant:
            switches.append(t)
        prev_dominant = dominant_head
    
    # Plot histogram of switch intervals
    if len(switches) > 1:
        intervals = np.diff(switches)
        ax5.hist(intervals, bins=20, alpha=0.7, color='orange')
        ax5.set_title(f'Attention Head Switch Intervals (Total Switches: {len(switches)})')
        ax5.set_xlabel('Steps Between Switches')
        ax5.set_ylabel('Frequency')
    else:
        ax5.text(0.5, 0.5, 'Insufficient attention switches for analysis', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Attention Head Switch Analysis')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_curiosity_exploration_map(agent, env, title="Curiosity-Driven Exploration", save_path=None):
    """
    Visualize exploration patterns and intrinsic rewards for curiosity-driven agents.
    
    Args:
        agent: CuriosityDrivenAgent with exploration data
        env: Grid environment
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not hasattr(agent, 'state_visit_counts') or not agent.state_visit_counts:
        print("No exploration data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: State visit frequency
    visit_map = np.zeros((env.size, env.size))
    for state_key, count in agent.state_visit_counts.items():
        if isinstance(state_key, tuple) and len(state_key) == 2:
            x, y = state_key
            if 0 <= x < env.size and 0 <= y < env.size:
                visit_map[x, y] = count
    
    im1 = axes[0, 0].imshow(visit_map, cmap='Blues', interpolation='nearest')
    axes[0, 0].set_title('State Visit Frequency')
    axes[0, 0].set_xlabel('Y Position')
    axes[0, 0].set_ylabel('X Position')
    
    # Add obstacles and goal
    for obs in env.obstacles:
        axes[0, 0].plot(obs[1], obs[0], 'ks', markersize=6, alpha=0.7)
    axes[0, 0].plot(env.goal[1], env.goal[0], 'g*', markersize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Exploration efficiency (novelty bonus)
    novelty_map = np.zeros((env.size, env.size))
    for state_key, count in agent.state_visit_counts.items():
        if isinstance(state_key, tuple) and len(state_key) == 2:
            x, y = state_key
            if 0 <= x < env.size and 0 <= y < env.size:
                novelty_bonus = 1.0 / (1 + count)
                novelty_map[x, y] = novelty_bonus
    
    im2 = axes[0, 1].imshow(novelty_map, cmap='Oranges', interpolation='nearest')
    axes[0, 1].set_title('Current Novelty Bonus')
    axes[0, 1].set_xlabel('Y Position')
    axes[0, 1].set_ylabel('X Position')
    
    for obs in env.obstacles:
        axes[0, 1].plot(obs[1], obs[0], 'ks', markersize=6, alpha=0.7)
    axes[0, 1].plot(env.goal[1], env.goal[0], 'g*', markersize=12)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Intrinsic reward over time
    if agent.intrinsic_rewards:
        axes[0, 2].plot(agent.intrinsic_rewards, 'purple', alpha=0.7)
        
        # Add rolling average
        window = 100
        if len(agent.intrinsic_rewards) > window:
            rolling_avg = np.convolve(agent.intrinsic_rewards, 
                                    np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(agent.intrinsic_rewards)), 
                          rolling_avg, 'red', linewidth=2, label=f'Rolling Avg ({window})')
            axes[0, 2].legend()
        
        axes[0, 2].set_title('Intrinsic Reward Over Time')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Intrinsic Reward')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No intrinsic reward data', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # Plot 4: Visit count distribution
    visit_counts = list(agent.state_visit_counts.values())
    axes[1, 0].hist(visit_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Visit Count Distribution')
    axes[1, 0].set_xlabel('Number of Visits')
    axes[1, 0].set_ylabel('Number of States')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add statistics
    axes[1, 0].axvline(np.mean(visit_counts), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(visit_counts):.1f}')
    axes[1, 0].axvline(np.median(visit_counts), color='green', linestyle='--', 
                      label=f'Median: {np.median(visit_counts):.1f}')
    axes[1, 0].legend()
    
    # Plot 5: Exploration progression
    axes[1, 1].plot(range(len(agent.state_visit_counts)), 
                   sorted(agent.state_visit_counts.values(), reverse=True), 
                   'darkgreen', marker='o', markersize=3)
    axes[1, 1].set_title('State Exploration Ranking')
    axes[1, 1].set_xlabel('State Rank')
    axes[1, 1].set_ylabel('Visit Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Exploration coverage metrics
    total_states = env.size * env.size - len(env.obstacles)
    explored_states = len(agent.state_visit_counts)
    coverage = explored_states / total_states
    
    metrics = {
        'Coverage': coverage,
        'Unique States': explored_states,
        'Total Intrinsic': sum(agent.intrinsic_rewards) if agent.intrinsic_rewards else 0,
        'Avg Intrinsic': np.mean(agent.intrinsic_rewards) if agent.intrinsic_rewards else 0
    }
    
    y_pos = range(len(metrics))
    values = list(metrics.values())
    
    bars = axes[1, 2].barh(y_pos, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(list(metrics.keys()))
    axes[1, 2].set_title('Exploration Summary Metrics')
    axes[1, 2].set_xlabel('Value')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        axes[1, 2].text(value + max(values)*0.01, i, f'{value:.3f}', 
                       va='center', fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_adaptive_hierarchy_evolution(agent, title="Adaptive Hierarchy Evolution", save_path=None):
    """
    Visualize how adaptive hierarchical structure changes over time.
    
    Args:
        agent: AdaptiveFractalAgent with adaptation history
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not hasattr(agent, 'performance_history') or len(agent.performance_history) < 10:
        print("Insufficient adaptation history for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    performance_data = list(agent.performance_history)
    
    # Plot 1: Performance over time with adaptation triggers
    axes[0, 0].plot(performance_data, 'blue', alpha=0.7, label='Performance')
    
    # Add rolling variance
    window = 20
    if len(performance_data) > window:
        rolling_var = []
        for i in range(window, len(performance_data)):
            rolling_var.append(np.var(performance_data[i-window:i]))
        
        axes[0, 0].plot(range(window, len(performance_data)), 
                       rolling_var, 'red', alpha=0.5, label=f'Variance (window={window})')
    
    axes[0, 0].set_title('Performance and Variance Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Steps to Goal')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hierarchy configuration over time (simulated)
    # Since we don't track history of block sizes, we'll show current vs theoretical
    current_micro = agent.block_micro
    current_macro = agent.block_macro
    
    # Show range of possible configurations
    possible_configs = []
    for micro in range(agent.min_block_size, agent.max_block_size + 1):
        macro = min(micro * 2, agent.env.size // 2)
        possible_configs.append((micro, macro))
    
    micros, macros = zip(*possible_configs)
    axes[0, 1].scatter(micros, macros, c='lightblue', alpha=0.6, s=100, label='Possible Configs')
    axes[0, 1].scatter([current_micro], [current_macro], c='red', s=200, 
                      marker='*', label='Current Config', zorder=5)
    
    axes[0, 1].set_title('Hierarchical Configuration Space')
    axes[0, 1].set_xlabel('Micro Block Size')
    axes[0, 1].set_ylabel('Macro Block Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Adaptation triggers analysis
    # Simulate when adaptations would have occurred
    adaptations = []
    if len(performance_data) >= 20:
        for i in range(20, len(performance_data), 25):  # Check every 25 episodes
            recent_perf = performance_data[max(0, i-10):i]
            if len(recent_perf) >= 10:
                variance = np.var(recent_perf)
                mean_perf = np.mean(recent_perf)
                
                if variance > mean_perf * 0.5:  # High variance
                    adaptations.append((i, 'decrease', 'High variance detected'))
                elif variance < mean_perf * 0.1:  # Low variance
                    adaptations.append((i, 'increase', 'Low variance detected'))
    
    if adaptations:
        episodes, directions, reasons = zip(*adaptations)
        colors = ['red' if d == 'decrease' else 'green' for d in directions]
        
        axes[0, 2].scatter(episodes, [performance_data[e] for e in episodes], 
                          c=colors, s=100, alpha=0.7)
        
        for i, (ep, direction, reason) in enumerate(adaptations):
            axes[0, 2].annotate(f'{direction}\n({reason})', 
                              (ep, performance_data[ep]),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=8, alpha=0.8)
    
    axes[0, 2].plot(performance_data, 'blue', alpha=0.3)
    axes[0, 2].set_title('Adaptation Trigger Points')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Performance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Performance stability analysis
    stability_window = 10
    stability_scores = []
    
    for i in range(stability_window, len(performance_data)):
        window_data = performance_data[i-stability_window:i]
        # Stability is inverse of coefficient of variation
        cv = np.std(window_data) / (np.mean(window_data) + 1e-8)
        stability = 1.0 / (1.0 + cv)
        stability_scores.append(stability)
    
    if stability_scores:
        axes[1, 0].plot(range(stability_window, len(performance_data)), 
                       stability_scores, 'green', linewidth=2)
        axes[1, 0].set_title('Performance Stability Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add horizontal lines for adaptation thresholds
        axes[1, 0].axhline(0.8, color='red', linestyle='--', alpha=0.5, 
                          label='High Stability (↑ blocks)')
        axes[1, 0].axhline(0.4, color='orange', linestyle='--', alpha=0.5, 
                          label='Low Stability (↓ blocks)')
        axes[1, 0].legend()
    
    # Plot 5: Theoretical adaptation benefits
    # Show how different block sizes might affect different types of environments
    env_types = ['Dense Obstacles', 'Sparse Obstacles', 'Complex Path', 'Simple Path']
    block_sizes = list(range(agent.min_block_size, agent.max_block_size + 1))
    
    # Theoretical performance matrix (smaller blocks better for complex environments)
    perf_matrix = np.zeros((len(env_types), len(block_sizes)))
    for i, env_type in enumerate(env_types):
        for j, block_size in enumerate(block_sizes):
            if 'Dense' in env_type or 'Complex' in env_type:
                # Complex environments benefit from smaller blocks
                perf_matrix[i, j] = 1.0 - (block_size - agent.min_block_size) / (agent.max_block_size - agent.min_block_size)
            else:
                # Simple environments can use larger blocks
                perf_matrix[i, j] = (block_size - agent.min_block_size) / (agent.max_block_size - agent.min_block_size)
    
    im = axes[1, 1].imshow(perf_matrix, cmap='RdYlGn', aspect='auto')
    axes[1, 1].set_title('Theoretical Block Size Effectiveness')
    axes[1, 1].set_xlabel('Block Size')
    axes[1, 1].set_ylabel('Environment Type')
    axes[1, 1].set_xticks(range(len(block_sizes)))
    axes[1, 1].set_xticklabels(block_sizes)
    axes[1, 1].set_yticks(range(len(env_types)))
    axes[1, 1].set_yticklabels(env_types)
    plt.colorbar(im, ax=axes[1, 1])
    
    # Plot 6: Current agent statistics
    stats = {
        'Current Micro': current_micro,
        'Current Macro': current_macro,
        'Adaptation Rate': agent.adaptation_rate,
        'Performance Std': np.std(performance_data),
        'Recent Avg Perf': np.mean(performance_data[-10:]) if len(performance_data) >= 10 else 0
    }
    
    y_pos = range(len(stats))
    values = list(stats.values())
    
    bars = axes[1, 2].barh(y_pos, values, 
                          color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(list(stats.keys()))
    axes[1, 2].set_title('Current Agent Statistics')
    axes[1, 2].set_xlabel('Value')
    
    for i, (bar, value) in enumerate(zip(bars, values)):
        axes[1, 2].text(value + max(values)*0.01, i, f'{value:.2f}', 
                       va='center', fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_meta_learning_strategy_analysis(agent, title="Meta-Learning Strategy Analysis", save_path=None):
    """
    Visualize meta-learning strategy library and adaptation patterns.
    
    Args:
        agent: MetaLearningAgent with strategy library
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not hasattr(agent, 'strategy_library') or not agent.strategy_library:
        print("No strategy library available for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    strategies = agent.strategy_library
    
    # Plot 1: Strategy performance distribution
    performances = []
    for strategy in strategies:
        if strategy.get('performance'):
            performances.extend(strategy['performance'])
    
    if performances:
        axes[0, 0].hist(performances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Strategy Performance Distribution')
        axes[0, 0].set_xlabel('Performance (Steps to Goal)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(performances), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(performances):.1f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Environment characteristics space
    obstacle_densities = []
    connectivities = []
    clusterings = []
    
    for strategy in strategies:
        chars = strategy.get('characteristics', {})
        obstacle_densities.append(chars.get('obstacle_density', 0))
        connectivities.append(chars.get('connectivity', 0))
        clusterings.append(chars.get('obstacle_clustering', 0))
    
    if obstacle_densities and connectivities:
        scatter = axes[0, 1].scatter(obstacle_densities, connectivities, 
                                   c=clusterings, cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_title('Environment Characteristics Space')
        axes[0, 1].set_xlabel('Obstacle Density')
        axes[0, 1].set_ylabel('Connectivity')
        plt.colorbar(scatter, ax=axes[0, 1], label='Clustering')
        
        # Highlight current environment
        if hasattr(agent, 'env_characteristics'):
            current_chars = agent.env_characteristics
            axes[0, 1].scatter([current_chars.get('obstacle_density', 0)], 
                             [current_chars.get('connectivity', 0)],
                             c='red', s=200, marker='*', 
                             label='Current Environment', zorder=5)
            axes[0, 1].legend()
    
    # Plot 3: Strategy parameter correlation
    block_micros = []
    block_macros = []
    learning_rates = []
    
    for strategy in strategies:
        block_micros.append(strategy.get('block_micro', 5))
        block_macros.append(strategy.get('block_macro', 10))
        learning_rates.append(strategy.get('learning_rate', 0.2))
    
    if block_micros and block_macros:
        scatter = axes[0, 2].scatter(block_micros, block_macros, 
                                   c=learning_rates, cmap='plasma', s=100, alpha=0.7)
        axes[0, 2].set_title('Strategy Parameter Relationships')
        axes[0, 2].set_xlabel('Block Micro Size')
        axes[0, 2].set_ylabel('Block Macro Size')
        plt.colorbar(scatter, ax=axes[0, 2], label='Learning Rate')
    
    # Plot 4: Strategy similarity network
    if len(strategies) > 1:
        # Compute pairwise similarities
        similarity_matrix = np.zeros((len(strategies), len(strategies)))
        
        for i, strategy_i in enumerate(strategies):
            for j, strategy_j in enumerate(strategies):
                chars_i = strategy_i.get('characteristics', {})
                chars_j = strategy_j.get('characteristics', {})
                
                # Compute similarity
                total_diff = 0
                common_keys = set(chars_i.keys()) & set(chars_j.keys())
                if common_keys:
                    for key in common_keys:
                        total_diff += abs(chars_i[key] - chars_j[key])
                    similarity = 1.0 - (total_diff / len(common_keys))
                else:
                    similarity = 0
                    
                similarity_matrix[i, j] = similarity
        
        im = axes[1, 0].imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
        axes[1, 0].set_title('Strategy Similarity Matrix')
        axes[1, 0].set_xlabel('Strategy Index')
        axes[1, 0].set_ylabel('Strategy Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Add similarity values as text
        for i in range(len(strategies)):
            for j in range(len(strategies)):
                axes[1, 0].text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                               ha='center', va='center', 
                               color='white' if similarity_matrix[i, j] < 0.5 else 'black')
    
    # Plot 5: Strategy evolution over time
    if len(strategies) > 1:
        # Assume strategies are ordered by discovery time
        discovery_order = range(len(strategies))
        avg_performances = []
        
        for strategy in strategies:
            if strategy.get('performance'):
                avg_performances.append(np.mean(strategy['performance']))
            else:
                avg_performances.append(0)
        
        axes[1, 1].plot(discovery_order, avg_performances, 'bo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Strategy Quality Evolution')
        axes[1, 1].set_xlabel('Strategy Discovery Order')
        axes[1, 1].set_ylabel('Average Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(avg_performances) > 2:
            z = np.polyfit(discovery_order, avg_performances, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(discovery_order, p(discovery_order), 'r--', alpha=0.7, 
                          label=f'Trend (slope: {z[0]:.2f})')
            axes[1, 1].legend()
    
    # Plot 6: Meta-learning statistics
    stats = {
        'Strategy Count': len(strategies),
        'Memory Capacity': agent.strategy_memory_size,
        'Current Obstacle Density': agent.env_characteristics.get('obstacle_density', 0),
        'Avg Strategy Performance': np.mean([np.mean(s.get('performance', [0])) for s in strategies]) if strategies else 0
    }
    
    y_pos = range(len(stats))
    values = list(stats.values())
    
    bars = axes[1, 2].barh(y_pos, values, 
                          color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(list(stats.keys()))
    axes[1, 2].set_title('Meta-Learning Statistics')
    axes[1, 2].set_xlabel('Value')
    
    for i, (bar, value) in enumerate(zip(bars, values)):
        axes[1, 2].text(value + max(values)*0.01, i, f'{value:.3f}', 
                       va='center', fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_novel_agent_comparison_dashboard(agents, labels, logs, title="Novel Agent Comparison Dashboard", save_path=None):
    """
    Create a comprehensive comparison dashboard for multiple novel agents.
    
    Args:
        agents: List of trained agents
        labels: List of agent labels
        logs: List of training logs
        title: Dashboard title
        save_path: Path to save the plot (optional)
    """
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 6, figure=fig)
    
    # Main performance comparison (top row, spans 4 columns)
    ax_main = fig.add_subplot(gs[0, :4])
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        ax_main.plot(log, color=color, alpha=0.3, linewidth=1)
        
        # Rolling average
        window = 30
        if len(log) >= window:
            rolling_avg = np.convolve(log, np.ones(window)/window, mode='valid')
            ax_main.plot(range(window-1, len(log)), rolling_avg, 
                        color=color, linewidth=3, label=label)
    
    ax_main.set_title('Performance Comparison (30-Episode Rolling Average)', fontsize=14)
    ax_main.set_xlabel('Episode')
    ax_main.set_ylabel('Steps to Goal')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Performance statistics (top right)
    ax_stats = fig.add_subplot(gs[0, 4:])
    
    stats_data = []
    for log, label in zip(logs, labels):
        stats_data.append([
            np.mean(log),
            np.std(log),
            min(log),
            np.median(log)
        ])
    
    stats_df = np.array(stats_data)
    metric_names = ['Mean', 'Std', 'Best', 'Median']
    
    x = np.arange(len(labels))
    width = 0.2
    
    for i, metric in enumerate(metric_names):
        offset = (i - 1.5) * width
        ax_stats.bar(x + offset, stats_df[:, i], width, 
                    label=metric, alpha=0.8)
    
    ax_stats.set_title('Performance Statistics')
    ax_stats.set_xlabel('Agent')
    ax_stats.set_ylabel('Steps')
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(labels, rotation=45)
    ax_stats.legend()
    
    # Agent-specific insights (remaining rows)
    insights_plotted = 0
    
    for i, (agent, label) in enumerate(zip(agents, labels)):
        if insights_plotted >= 12:  # Limit to available subplot space
            break
            
        row = 1 + insights_plotted // 6
        col = insights_plotted % 6
        
        if row >= 4:
            break
            
        ax = fig.add_subplot(gs[row, col])
        
        # Curiosity agent insights
        if hasattr(agent, 'intrinsic_rewards') and agent.intrinsic_rewards:
            ax.plot(agent.intrinsic_rewards, 'purple', alpha=0.7)
            ax.set_title(f'{label}: Intrinsic Rewards')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            
        # Multi-head attention insights
        elif hasattr(agent, 'attention_head_history') and agent.attention_head_history:
            attention_matrix = np.array(agent.attention_head_history)
            avg_attention = np.mean(attention_matrix, axis=0)
            
            im = ax.imshow(avg_attention, cmap='viridis', aspect='auto')
            ax.set_title(f'{label}: Avg Attention')
            ax.set_xlabel('Level')
            ax.set_ylabel('Head')
            
        # Adaptive agent insights
        elif hasattr(agent, 'performance_history') and len(agent.performance_history) > 10:
            perf_data = list(agent.performance_history)
            ax.plot(perf_data, 'green', alpha=0.7)
            ax.set_title(f'{label}: Adaptation History')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Performance')
            
        # Meta-learning insights
        elif hasattr(agent, 'strategy_library') and agent.strategy_library:
            strategies = len(agent.strategy_library)
            avg_perf = np.mean([np.mean(s.get('performance', [0])) for s in agent.strategy_library])
            
            ax.bar(['Strategies', 'Avg Performance'], [strategies, avg_perf], 
                   color=['lightblue', 'lightgreen'])
            ax.set_title(f'{label}: Meta-Learning Stats')
            
        # Default: learning curve
        else:
            ax.plot(logs[i], color=colors[i % len(colors)], alpha=0.7)
            ax.set_title(f'{label}: Learning Curve')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
        
        ax.grid(True, alpha=0.3)
        insights_plotted += 1
    
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 