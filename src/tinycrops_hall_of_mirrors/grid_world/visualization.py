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


def animate_agent_step_by_step(agent, env, title="Agent Path with Decision Analysis", 
                              save_path=None, show_policy=True, show_values=False,
                              show_hierarchical=True, show_attention=False,
                              show_curiosity=False, show_optimal_path=False,
                              interval=500):
    """
    Create an enhanced animated visualization showing agent's decision-making process.
    
    Args:
        agent: Trained agent
        env: Grid environment
        title: Animation title
        save_path: Path to save animation (optional)
        show_policy: Show policy arrows at each state
        show_values: Show state values as heatmap
        show_hierarchical: Show hierarchical goals for fractal agents
        show_attention: Show attention weights for attention agents
        show_curiosity: Show curiosity/novelty bonus for curiosity agents
        show_optimal_path: Show optimal path computed by A*
        interval: Animation frame interval in ms
    """
    # Get the agent's path
    path = []
    hierarchical_goals = []
    attention_weights = []
    curiosity_values = []
    
    # Run episode and collect data
    pos = env.start
    step = 0
    
    while pos != env.goal and step < 200:
        path.append(pos)
        
        # Collect hierarchical goals if applicable
        if hasattr(agent, 'idx_super') and hasattr(agent, 'idx_macro'):
            super_goal = agent.idx_super(pos)
            macro_goal = agent.idx_macro(pos)
            hierarchical_goals.append((super_goal, macro_goal))
        
        # Collect attention weights if applicable  
        if hasattr(agent, 'attention_weights'):
            attention_weights.append(agent.attention_weights.copy())
            
        # Collect curiosity values if applicable
        if hasattr(agent, 'get_novelty_bonus'):
            state_idx = agent.get_state_index(pos)
            curiosity = agent.get_novelty_bonus(state_idx)
            curiosity_values.append(curiosity)
        
        # Get action
        state_idx = agent.get_state_index(pos)
        action = agent.choose_action(state_idx, epsilon=0.05)
        
        # Take action
        next_pos, _, done = env.step(pos, action)
        pos = next_pos
        step += 1
    
    path.append(pos)  # Add final position
    
    # Compute optimal path if requested
    optimal_path = None
    if show_optimal_path:
        optimal_path = compute_optimal_path(env)
    
    # Prepare figure
    if show_hierarchical and hasattr(agent, 'Q_super'):
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
        
        ax_main = fig.add_subplot(gs[0, :2])
        ax_info = fig.add_subplot(gs[0, 2])
        ax_attention = fig.add_subplot(gs[1, 0]) if show_attention else None
        ax_levels = fig.add_subplot(gs[1, 1])
        ax_curiosity = fig.add_subplot(gs[1, 2]) if show_curiosity else None
    else:
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        ax_main = fig.add_subplot(gs[0, :])
        ax_info = fig.add_subplot(gs[1, 0])
        ax_attention = None  # No attention panel in simple layout
        ax_levels = None     # No levels panel in simple layout
        ax_curiosity = fig.add_subplot(gs[1, 1]) if show_curiosity else None
    
    # Initialize main display
    ax_main.set_title(title, fontsize=16)
    ax_main.set_xlim(-0.5, env.size - 0.5)
    ax_main.set_ylim(-0.5, env.size - 0.5)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    
    # Create base map
    base_map = np.zeros((env.size, env.size, 3))
    
    # Mark obstacles
    for obs in env.obstacles:
        base_map[obs[0], obs[1]] = [0.3, 0.3, 0.3]
    
    # Mark goal
    base_map[env.goal[0], env.goal[1]] = [1.0, 0.65, 0.0]
    
    # Show optimal path if requested
    if optimal_path:
        for pos in optimal_path[1:-1]:  # Exclude start and goal
            base_map[pos[0], pos[1]] = [0.8, 0.8, 1.0]
    
    im_main = ax_main.imshow(base_map, interpolation='nearest')
    
    # Initialize policy arrows
    policy_arrows = []
    if show_policy:
        for x in range(env.size):
            for y in range(env.size):
                if (x, y) not in env.obstacles and (x, y) != env.goal:
                    arrow = ax_main.arrow(y, x, 0, 0, head_width=0.2, 
                                        head_length=0.1, fc='blue', ec='blue', 
                                        alpha=0.5)
                    policy_arrows.append(((x, y), arrow))
    
    # Initialize value heatmap
    value_overlay = None
    if show_values:
        value_data = np.zeros((env.size, env.size))
        value_overlay = ax_main.imshow(value_data, cmap='YlOrRd', alpha=0.5, 
                                     interpolation='nearest')
    
    # Initialize hierarchical goal markers
    super_goal_marker = None
    macro_goal_marker = None
    if show_hierarchical and hierarchical_goals:
        super_goal_marker = ax_main.plot([], [], 'bs', markersize=20, 
                                       alpha=0.5, label='Super Goal')[0]
        macro_goal_marker = ax_main.plot([], [], 'gs', markersize=15, 
                                       alpha=0.5, label='Macro Goal')[0]
    
    # Initialize agent marker
    agent_marker = ax_main.plot([], [], 'ro', markersize=12, label='Agent')[0]
    
    # Initialize path trail
    path_line = ax_main.plot([], [], 'r-', linewidth=2, alpha=0.7)[0]
    
    # Info panel setup
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, 
                           fontsize=12, verticalalignment='top', 
                           fontfamily='monospace')
    
    # Attention panel setup
    attention_bars = []
    if ax_attention is not None and hasattr(agent, 'attention_weights'):
        ax_attention.set_title('Attention Weights')
        ax_attention.set_ylim(0, 1)
        ax_attention.set_xticks([0, 1, 2])
        ax_attention.set_xticklabels(['Micro', 'Macro', 'Super'])
        attention_bars = ax_attention.bar([0, 1, 2], [0, 0, 0], 
                                        color=['red', 'green', 'blue'])
    
    # Hierarchical levels panel
    if ax_levels is not None and hasattr(agent, 'block_micro'):
        ax_levels.set_title('Hierarchical Block Sizes')
        ax_levels.axis('off')
        levels_text = ax_levels.text(0.1, 0.5, 
                                   f'Micro Block: {agent.block_micro}x{agent.block_micro}\n'
                                   f'Macro Block: {agent.block_macro}x{agent.block_macro}',
                                   transform=ax_levels.transAxes, fontsize=12)
    
    # Curiosity panel setup
    if ax_curiosity is not None and hasattr(agent, 'visit_counts'):
        ax_curiosity.set_title('Curiosity/Visit Heatmap')
        visit_data = np.zeros((env.size, env.size))
        for x in range(env.size):
            for y in range(env.size):
                state_idx = agent.get_state_index((x, y))
                if state_idx < len(agent.visit_counts):
                    visit_data[x, y] = agent.visit_counts[state_idx]
        
        curiosity_heatmap = ax_curiosity.imshow(visit_data, cmap='viridis', 
                                               interpolation='nearest')
        plt.colorbar(curiosity_heatmap, ax=ax_curiosity)
    
    def animate(frame):
        if frame >= len(path):
            return []
        
        current_pos = path[frame]
        
        # Update agent position
        agent_marker.set_data([current_pos[1]], [current_pos[0]])
        
        # Update path trail
        if frame > 0:
            path_y = [p[1] for p in path[:frame+1]]
            path_x = [p[0] for p in path[:frame+1]]
            path_line.set_data(path_y, path_x)
        
        # Update policy arrows
        if show_policy:
            for (x, y), arrow in policy_arrows:
                state_idx = agent.get_state_index((x, y))
                q_values = agent.Q[state_idx] if hasattr(agent, 'Q') else agent.Q_micro[state_idx]
                best_action = np.argmax(q_values)
                
                # Get action direction
                action_delta = list(env.actions.values())[best_action]
                arrow.set_data(dx=action_delta[1]*0.3, dy=action_delta[0]*0.3)
                
                # Color based on Q-value confidence
                max_q = q_values[best_action]
                arrow.set_alpha(min(1.0, max(0.1, (max_q + 10) / 20)))
        
        # Update value heatmap
        if show_values and value_overlay:
            value_data = np.zeros((env.size, env.size))
            for x in range(env.size):
                for y in range(env.size):
                    if (x, y) not in env.obstacles:
                        state_idx = agent.get_state_index((x, y))
                        q_values = agent.Q[state_idx] if hasattr(agent, 'Q') else agent.Q_micro[state_idx]
                        value_data[x, y] = np.max(q_values)
            
            value_overlay.set_data(value_data)
            value_overlay.set_clim(vmin=value_data.min(), vmax=value_data.max())
        
        # Update hierarchical goals
        if show_hierarchical and frame < len(hierarchical_goals):
            super_idx, macro_idx = hierarchical_goals[frame]
            
            if hasattr(agent, 'block_macro'):
                # Convert super index to position
                super_size = env.size // agent.block_macro
                super_y = super_idx % super_size
                super_x = super_idx // super_size
                super_center_y = super_y * agent.block_macro + agent.block_macro // 2
                super_center_x = super_x * agent.block_macro + agent.block_macro // 2
                super_goal_marker.set_data([super_center_y], [super_center_x])
            
            if hasattr(agent, 'block_micro'):
                # Convert macro index to position
                macro_size = env.size // agent.block_micro
                macro_y = macro_idx % macro_size
                macro_x = macro_idx // macro_size
                macro_center_y = macro_y * agent.block_micro + agent.block_micro // 2
                macro_center_x = macro_x * agent.block_micro + agent.block_micro // 2
                macro_goal_marker.set_data([macro_center_y], [macro_center_x])
        
        # Update attention weights
        if ax_attention is not None and frame < len(attention_weights):
            weights = attention_weights[frame]
            for i, bar in enumerate(attention_bars):
                bar.set_height(weights[i])
        
        # Update info panel
        info_lines = [
            f"Step: {frame}/{len(path)-1}",
            f"Position: {current_pos}",
            f"Distance to Goal: {abs(current_pos[0]-env.goal[0]) + abs(current_pos[1]-env.goal[1])}"
        ]
        
        if hasattr(agent, 'Q'):
            state_idx = agent.get_state_index(current_pos)
            q_values = agent.Q[state_idx] if hasattr(agent, 'Q') else agent.Q_micro[state_idx]
            best_action = np.argmax(q_values)
            info_lines.extend([
                f"Best Action: {list(env.actions.keys())[best_action]}",
                f"Q-value: {q_values[best_action]:.3f}"
            ])
        
        if frame < len(curiosity_values):
            info_lines.append(f"Curiosity Bonus: {curiosity_values[frame]:.3f}")
        
        info_text.set_text('\n'.join(info_lines))
        
        return [agent_marker, path_line, info_text] + \
               ([super_goal_marker, macro_goal_marker] if show_hierarchical and hierarchical_goals else []) + \
               (list(attention_bars) if ax_attention is not None and attention_weights else [])
    
    ani = animation.FuncAnimation(fig, animate, frames=len(path)+10, 
                                interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=1000//interval)
    
    plt.show()
    return ani


def compute_optimal_path(env):
    """
    Compute optimal path using A* algorithm.
    
    Args:
        env: Grid environment
        
    Returns:
        List of positions representing optimal path
    """
    from heapq import heappush, heappop
    
    def heuristic(pos):
        return abs(pos[0] - env.goal[0]) + abs(pos[1] - env.goal[1])
    
    start = env.start
    goal = env.goal
    
    # Priority queue: (f_score, g_score, position, path)
    pq = [(heuristic(start), 0, start, [start])]
    visited = set()
    
    while pq:
        f_score, g_score, current, path = heappop(pq)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check all neighbors
        for action_delta in env.actions.values():
            next_pos = (current[0] + action_delta[0], current[1] + action_delta[1])
            
            # Check if valid position
            if (0 <= next_pos[0] < env.size and 
                0 <= next_pos[1] < env.size and
                next_pos not in env.obstacles and
                next_pos not in visited):
                
                new_g_score = g_score + 1
                new_f_score = new_g_score + heuristic(next_pos)
                heappush(pq, (new_f_score, new_g_score, next_pos, path + [next_pos]))
    
    return []  # No path found


def visualize_adaptive_hierarchy(agent, env, title="Adaptive Hierarchy Visualization"):
    """
    Visualize how adaptive agents change their hierarchical structure.
    
    Args:
        agent: Adaptive fractal agent
        env: Grid environment
        title: Plot title
    """
    if not hasattr(agent, 'hierarchy_history'):
        print("Agent does not have hierarchy adaptation history")
        return
    
    history = agent.hierarchy_history
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data
    episodes = [h['episode'] for h in history]
    micro_sizes = [h['block_micro'] for h in history]
    macro_sizes = [h['block_macro'] for h in history]
    performances = [h['performance'] for h in history]
    
    # Plot 1: Block sizes over time
    ax1.plot(episodes, micro_sizes, 'r-', marker='o', label='Micro Block Size')
    ax1.plot(episodes, macro_sizes, 'b-', marker='s', label='Macro Block Size')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Block Size')
    ax1.set_title('Hierarchical Block Sizes Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance correlation
    ax2.scatter(micro_sizes, performances, alpha=0.6, label='Micro vs Performance')
    ax2.set_xlabel('Micro Block Size')
    ax2.set_ylabel('Episode Performance (steps)')
    ax2.set_title('Block Size vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Adaptation triggers
    ax3.plot(episodes, performances, 'g-', alpha=0.7, label='Performance')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps to Goal')
    ax3.set_title('Performance with Adaptation Points')
    
    # Mark adaptation points
    for i, h in enumerate(history):
        if i > 0 and (h['block_micro'] != history[i-1]['block_micro'] or 
                     h['block_macro'] != history[i-1]['block_macro']):
            ax3.axvline(x=h['episode'], color='red', alpha=0.5, linestyle='--')
            ax3.text(h['episode'], performances[i], 
                    f"Δ({h['block_micro']},{h['block_macro']})", 
                    rotation=90, fontsize=8)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_meta_learning_strategies(agent, env, title="Meta-Learning Strategy Analysis"):
    """
    Visualize meta-learning agent's strategy selection and performance.
    
    Args:
        agent: Meta-learning agent
        env: Grid environment
        title: Plot title
    """
    if not hasattr(agent, 'strategy_history'):
        print("Agent does not have strategy selection history")
        return
    
    history = agent.strategy_history
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Strategy usage over time
    episodes = list(range(len(history)))
    strategies = [h['strategy'] for h in history]
    unique_strategies = list(set(strategies))
    strategy_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
    
    for i, strategy in enumerate(unique_strategies):
        strategy_episodes = [ep for ep, s in enumerate(strategies) if s == strategy]
        ax1.scatter(strategy_episodes, [i] * len(strategy_episodes), 
                   color=strategy_colors[i], s=50, alpha=0.7, label=strategy)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Strategy')
    ax1.set_yticks(range(len(unique_strategies)))
    ax1.set_yticklabels(unique_strategies)
    ax1.set_title('Strategy Selection Over Time')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot 2: Strategy performance distribution
    strategy_performances = {}
    for h in history:
        strategy = h['strategy']
        performance = h.get('performance', 0)
        if strategy not in strategy_performances:
            strategy_performances[strategy] = []
        strategy_performances[strategy].append(performance)
    
    box_data = [strategy_performances[s] for s in unique_strategies]
    bp = ax2.boxplot(box_data, labels=unique_strategies, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], strategy_colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Performance (steps)')
    ax2.set_title('Strategy Performance Distribution')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Environment fingerprint clustering
    if 'env_features' in history[0]:
        features = np.array([h['env_features'] for h in history])
        
        # Simple 2D projection for visualization
        if features.shape[1] > 2:
            # Use PCA-like projection (just first 2 features for simplicity)
            features_2d = features[:, :2]
        else:
            features_2d = features
        
        for i, strategy in enumerate(unique_strategies):
            strategy_mask = np.array(strategies) == strategy
            ax3.scatter(features_2d[strategy_mask, 0], 
                       features_2d[strategy_mask, 1],
                       color=strategy_colors[i], label=strategy, 
                       alpha=0.6, s=50)
        
        ax3.set_xlabel('Environment Feature 1')
        ax3.set_ylabel('Environment Feature 2')
        ax3.set_title('Strategy Selection by Environment Features')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_multihead_attention_analysis(agent, env, episode_data, 
                                         title="Multi-Head Attention Analysis"):
    """
    Detailed visualization of multi-head attention mechanisms.
    
    Args:
        agent: Multi-head attention agent
        env: Grid environment
        episode_data: Dictionary with 'positions', 'attention_heads', 'head_weights'
        title: Plot title
    """
    positions = episode_data['positions']
    attention_heads = episode_data['attention_heads']  # Which head was active
    head_weights = episode_data['head_weights']  # Weights for each head
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Main path visualization
    ax_main = fig.add_subplot(gs[:2, :2])
    ax_main.set_title(f'{title}: Agent Path with Active Attention Head')
    
    # Create path visualization with color coding by active head
    head_colors = {'distance': 'blue', 'obstacles': 'red', 'progress': 'green'}
    
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        head = attention_heads[i] if i < len(attention_heads) else 'distance'
        color = head_colors.get(head, 'black')
        
        ax_main.plot([start[1], end[1]], [start[0], end[0]], 
                    color=color, linewidth=3, alpha=0.7)
    
    # Add environment
    for obs in env.obstacles:
        ax_main.plot(obs[1], obs[0], 'ks', markersize=10)
    ax_main.plot(env.goal[1], env.goal[0], 'g*', markersize=20)
    ax_main.plot(env.start[1], env.start[0], 'bo', markersize=12)
    
    ax_main.set_xlim(-0.5, env.size - 0.5)
    ax_main.set_ylim(-0.5, env.size - 0.5)
    ax_main.invert_yaxis()
    ax_main.grid(True, alpha=0.3)
    
    # Legend for heads
    for head, color in head_colors.items():
        ax_main.plot([], [], color=color, linewidth=3, label=f'{head} head')
    ax_main.legend()
    
    # Head activation timeline
    ax_timeline = fig.add_subplot(gs[2, :])
    ax_timeline.set_title('Attention Head Activation Timeline')
    
    steps = list(range(len(attention_heads)))
    head_indices = {'distance': 0, 'obstacles': 1, 'progress': 2}
    head_values = [head_indices.get(h, 0) for h in attention_heads]
    
    ax_timeline.scatter(steps, head_values, c=[head_colors.get(h, 'black') 
                                              for h in attention_heads], s=50)
    ax_timeline.set_yticks([0, 1, 2])
    ax_timeline.set_yticklabels(['Distance', 'Obstacles', 'Progress'])
    ax_timeline.set_xlabel('Step')
    ax_timeline.set_ylabel('Active Head')
    ax_timeline.grid(True, alpha=0.3)
    
    # Head weight evolution
    ax_weights = fig.add_subplot(gs[:2, 2])
    ax_weights.set_title('Head Weight Evolution')
    
    if head_weights:
        weights_array = np.array(head_weights)
        ax_weights.plot(weights_array[:, 0], 'b-', label='Distance')
        ax_weights.plot(weights_array[:, 1], 'r-', label='Obstacles')
        ax_weights.plot(weights_array[:, 2], 'g-', label='Progress')
        ax_weights.set_xlabel('Step')
        ax_weights.set_ylabel('Weight')
        ax_weights.legend()
        ax_weights.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 