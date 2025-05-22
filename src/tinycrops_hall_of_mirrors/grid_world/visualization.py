"""
Visualization utilities for grid-world agents and experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation


def plot_learning_curve(logs, labels=None, title="Learning Curves", save_path=None):
    """
    Plot learning curves for one or more agents.
    
    Args:
        logs: List of logs (each log is a list of steps per episode) or single log
        labels: List of labels for each log
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Handle single log case
    if not isinstance(logs[0], list):
        logs = [logs]
    
    if labels is None:
        labels = [f"Agent {i+1}" for i in range(len(logs))]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw learning curves with rolling averages
    plt.subplot(2, 1, 1)
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        
        # Raw data
        plt.plot(log, color=color, alpha=0.3, label=f'{label} (raw)')
        
        # Rolling average
        window_size = 30
        if len(log) >= window_size:
            rolling_avg = np.convolve(log, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(log)), rolling_avg, 
                    color=color, linewidth=2, 
                    label=f'{label} (rolling avg, window={window_size})')
    
    plt.title(f'{title}: Steps to Goal vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Log scale with exponential moving averages
    plt.subplot(2, 1, 2)
    alpha_ema = 0.1  # Smoothing factor for exponential moving average
    
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = colors[i % len(colors)]
        
        # Log scale raw
        plt.semilogy(log, color=color, alpha=0.3, label=f'{label} (log scale)')
        
        # Exponential moving average
        ema = [log[0]]
        for j in range(1, len(log)):
            ema.append(alpha_ema * log[j] + (1 - alpha_ema) * ema[-1])
        
        plt.semilogy(ema, color=color, linewidth=2, label=f'{label} (EMA)')
    
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal (log scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_q_values(agent, env, title="Q-Values", save_path=None):
    """
    Plot Q-values as heatmaps for each action.
    
    Args:
        agent: Trained agent with Q-table
        env: Grid environment
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (action_idx, action_delta) in enumerate(env.actions.items()):
        q_values = np.zeros((env.size, env.size))
        
        for x in range(env.size):
            for y in range(env.size):
                pos = (x, y)
                state_idx = agent.get_state_index(pos)
                q_values[x, y] = agent.Q[state_idx, i]
        
        im = axs[i].imshow(q_values, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Action {i}: {action_delta}')
        axs[i].set_xlabel('Y Position')
        axs[i].set_ylabel('X Position')
        
        # Add obstacles
        for obs in env.obstacles:
            axs[i].plot(obs[1], obs[0], 'ks', markersize=8, alpha=0.7)
        
        # Add goal
        axs[i].plot(env.goal[1], env.goal[0], 'g*', markersize=15)
        
        fig.colorbar(im, ax=axs[i])
    
    plt.suptitle(f'{title}: Q-Values by Action', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hierarchical_q_values(fractal_agent, env, title="Hierarchical Q-Values", save_path=None):
    """
    Plot Q-values for hierarchical (fractal) agents at all levels.
    
    Args:
        fractal_agent: Trained fractal agent
        env: Grid environment  
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Micro level Q-values
    for i, (action_idx, action_delta) in enumerate(env.actions.items()):
        ax = plt.subplot(3, 4, i + 1)
        q_values = np.zeros((env.size, env.size))
        
        for x in range(env.size):
            for y in range(env.size):
                pos = (x, y)
                state_idx = fractal_agent.idx_micro(pos)
                q_values[x, y] = fractal_agent.Q_micro[state_idx, i]
        
        im = ax.imshow(q_values, cmap='Reds', interpolation='nearest')
        ax.set_title(f'Micro Action {i}: {action_delta}')
        plt.colorbar(im, ax=ax)
    
    # Macro level Q-values
    for i, (action_idx, action_delta) in enumerate(env.actions.items()):
        ax = plt.subplot(3, 4, i + 5)
        
        # Create macro-level grid
        macro_size = env.size // fractal_agent.block_micro
        q_values = np.zeros((macro_size, macro_size))
        
        for x in range(macro_size):
            for y in range(macro_size):
                macro_state = x * macro_size + y
                if macro_state < fractal_agent.Q_macro.shape[0]:
                    q_values[x, y] = fractal_agent.Q_macro[macro_state, i]
        
        im = ax.imshow(q_values, cmap='Greens', interpolation='nearest')
        ax.set_title(f'Macro Action {i}: {action_delta}')
        plt.colorbar(im, ax=ax)
    
    # Super level Q-values
    for i, (action_idx, action_delta) in enumerate(env.actions.items()):
        ax = plt.subplot(3, 4, i + 9)
        
        # Create super-level grid
        super_size = env.size // fractal_agent.block_macro
        q_values = np.zeros((super_size, super_size))
        
        for x in range(super_size):
            for y in range(super_size):
                super_state = x * super_size + y
                if super_state < fractal_agent.Q_super.shape[0]:
                    q_values[x, y] = fractal_agent.Q_super[super_state, i]
        
        im = ax.imshow(q_values, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Super Action {i}: {action_delta}')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def animate_agent_path(agent, env, title="Agent Path", save_path=None, 
                      show_hierarchical=False):
    """
    Create an animated visualization of an agent's path to the goal.
    
    Args:
        agent: Trained agent
        env: Grid environment
        title: Animation title
        save_path: Path to save animation (optional)
        show_hierarchical: Whether to show hierarchical decomposition for fractal agents
    """
    # Get the path
    path = agent.run_episode(epsilon=0.05)
    
    # Prepare frames
    frames = []
    
    for pos in path:
        # Basic grids
        agent_grid = np.zeros((env.size, env.size))
        obstacle_grid = np.zeros((env.size, env.size))
        goal_grid = np.zeros((env.size, env.size))
        path_grid = np.zeros((env.size, env.size))
        
        # Mark current position
        agent_grid[pos] = 1
        
        # Mark obstacles
        for obs in env.obstacles:
            obstacle_grid[obs] = 1
            
        # Mark goal
        goal_grid[env.goal] = 1
        
        # Mark path history
        path_idx = path.index(pos)
        for p in path[:path_idx + 1]:
            path_grid[p] = 0.5
        
        frame_data = {
            'agent': agent_grid.copy(),
            'obstacles': obstacle_grid.copy(),
            'goal': goal_grid.copy(),
            'path': path_grid.copy()
        }
        
        # Add hierarchical information for fractal agents
        if show_hierarchical and hasattr(agent, 'Q_super'):
            super_grid = np.zeros((env.size, env.size))
            macro_grid = np.zeros((env.size, env.size))
            
            # Mark super block
            spos = agent.idx_super(pos)
            sr, sc = divmod(spos, env.size // agent.block_macro)
            super_grid[sr*agent.block_macro:(sr+1)*agent.block_macro, 
                      sc*agent.block_macro:(sc+1)*agent.block_macro] = 1
            
            # Mark macro block  
            mpos = agent.idx_macro(pos)
            mr, mc = divmod(mpos, env.size // agent.block_micro)
            macro_grid[mr*agent.block_micro:(mr+1)*agent.block_micro, 
                      mc*agent.block_micro:(mc+1)*agent.block_micro] = 1
            
            frame_data['super'] = super_grid.copy()
            frame_data['macro'] = macro_grid.copy()
        
        frames.append(frame_data)
    
    # Create animation
    if show_hierarchical and hasattr(agent, 'Q_super'):
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])
        
        ax_super = fig.add_subplot(gs[0, 0])
        ax_macro = fig.add_subplot(gs[1, 0])
        ax_micro = fig.add_subplot(gs[2, 0])
        ax_combined = fig.add_subplot(gs[:, 1])
        
        # Initialize images
        im_super = ax_super.imshow(frames[0]['super'], cmap='Blues', alpha=0.7)
        im_macro = ax_macro.imshow(frames[0]['macro'], cmap='Greens', alpha=0.7)
        im_micro = ax_micro.imshow(frames[0]['agent'], cmap='Reds', alpha=0.7)
        
        # Add static elements
        for ax, name in [(ax_super, 'Super'), (ax_macro, 'Macro'), (ax_micro, 'Micro')]:
            ax.imshow(frames[0]['obstacles'], cmap='binary', alpha=0.8)
            ax.imshow(frames[0]['goal'], cmap='Oranges', alpha=1.0)
            ax.set_title(f'{name} Level')
            ax.axis('off')
        
        # Combined view
        combined = np.zeros((env.size, env.size, 3))
        combined[:,:,0] = frames[0]['super'] * 0.5
        combined[:,:,1] = frames[0]['macro'] * 0.5
        combined[:,:,2] = frames[0]['agent']
        
        # Add obstacles and goal to combined
        obstacle_mask = frames[0]['obstacles'] > 0
        combined[obstacle_mask] = [0.3, 0.3, 0.3]
        goal_mask = frames[0]['goal'] > 0
        combined[goal_mask] = [1.0, 0.65, 0.0]
        
        im_combined = ax_combined.imshow(combined)
        ax_combined.set_title('Combined View')
        ax_combined.axis('off')
        
        def animate(i):
            # Update individual views
            im_super.set_data(frames[i]['super'])
            im_macro.set_data(frames[i]['macro'])
            im_micro.set_data(frames[i]['agent'])
            
            # Update combined view
            combined = np.zeros((env.size, env.size, 3))
            combined[:,:,0] = frames[i]['super'] * 0.5
            combined[:,:,1] = frames[i]['macro'] * 0.5
            combined[:,:,2] = frames[i]['agent']
            
            obstacle_mask = frames[i]['obstacles'] > 0
            combined[obstacle_mask] = [0.3, 0.3, 0.3]
            goal_mask = frames[i]['goal'] > 0
            combined[goal_mask] = [1.0, 0.65, 0.0]
            path_mask = frames[i]['path'] > 0
            combined[path_mask] = combined[path_mask] + [0.2, 0.2, 0.0]
            
            im_combined.set_array(combined)
            
            return [im_super, im_macro, im_micro, im_combined]
    
    else:
        # Simple single-view animation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im_agent = ax.imshow(frames[0]['agent'], cmap='Reds', alpha=0.7)
        im_obstacles = ax.imshow(frames[0]['obstacles'], cmap='binary', alpha=0.8)
        im_goal = ax.imshow(frames[0]['goal'], cmap='Oranges', alpha=1.0)
        im_path = ax.imshow(frames[0]['path'], cmap='viridis', alpha=0.3)
        
        ax.set_title(title)
        ax.axis('off')
        
        def animate(i):
            im_agent.set_data(frames[i]['agent'])
            im_path.set_data(frames[i]['path'])
            return [im_agent, im_obstacles, im_goal, im_path]
    
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                interval=200, blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
    
    plt.show()
    return ani


def plot_attention_evolution(attention_history, title="Attention Evolution", save_path=None):
    """
    Plot the evolution of attention weights for attention-based agents.
    
    Args:
        attention_history: List of attention weight arrays over time
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not attention_history:
        print("No attention history to plot.")
        return
    
    attention_array = np.array(attention_history)
    
    plt.figure(figsize=(12, 8))
    
    # Plot raw attention weights
    plt.subplot(2, 1, 1)
    plt.plot(attention_array[:, 0], 'r-', alpha=0.7, label='Micro Attention')
    plt.plot(attention_array[:, 1], 'g-', alpha=0.7, label='Macro Attention')
    plt.plot(attention_array[:, 2], 'b-', alpha=0.7, label='Super Attention')
    
    plt.title(f'{title}: Raw Attention Weights')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot smoothed attention weights
    plt.subplot(2, 1, 2)
    window_size = min(50, len(attention_history) // 10)
    
    if len(attention_history) >= window_size:
        for i, label in enumerate(['Micro', 'Macro', 'Super']):
            smoothed = np.convolve(attention_array[:, i], 
                                 np.ones(window_size)/window_size, mode='valid')
            color = ['r', 'g', 'b'][i]
            plt.plot(range(window_size-1, len(attention_history)), smoothed, 
                    f'{color}-', linewidth=2, label=f'{label} (smoothed)')
    
    plt.title(f'{title}: Smoothed Attention Weights (window={window_size})')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


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