"""
Fractal Environment Visualization

Specialized visualization for fractal depth environments showing:
- Multi-scale grid representations
- Portal transitions and depth changes
- Agent's perspective at different fractal levels
- Self-observation capabilities visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque


class FractalEnvironmentVisualizer:
    """
    Visualizer for fractal depth environments with multi-scale representations.
    """
    
    def __init__(self, env, figsize=(16, 12)):
        self.env = env
        self.figsize = figsize
        self.trajectory = deque(maxlen=1000)
        self.depth_history = deque(maxlen=1000)
        self.observation_history = deque(maxlen=1000)
        
        # Color schemes for different depths
        self.depth_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.portal_color = '#FD79A8'
        self.goal_color = '#00B894'
        self.obstacle_color = '#636E72'
        
    def reset_tracking(self):
        """Reset trajectory and history tracking."""
        self.trajectory.clear()
        self.depth_history.clear()
        self.observation_history.clear()
    
    def add_step(self, state, observation=None):
        """Add a step to the trajectory tracking."""
        x, y, depth = state
        self.trajectory.append((x, y, depth))
        self.depth_history.append(depth)
        if observation:
            self.observation_history.append(observation)
    
    def plot_fractal_overview(self, agent_state=None, show_trajectory=True):
        """
        Plot overview of fractal environment with multiple depth representations.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        obstacles, goal, portals = self.env.get_current_layout_elements()
        
        # Plot each depth level
        for depth in range(min(4, self.env.max_depth + 1)):
            ax = axes[depth]
            
            # Draw grid
            for i in range(self.env.base_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
            
            # Draw obstacles
            for obs in obstacles:
                rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                       facecolor=self.obstacle_color, alpha=0.8)
                ax.add_patch(rect)
            
            # Draw goal
            goal_circle = patches.Circle((goal[1], goal[0]), 0.3, 
                                       facecolor=self.goal_color, alpha=0.9)
            ax.add_patch(goal_circle)
            
            # Draw portals
            for portal in portals:
                portal_circle = patches.Circle((portal[1], portal[0]), 0.25,
                                             facecolor=self.portal_color, alpha=0.9)
                ax.add_patch(portal_circle)
            
            # Draw agent if at this depth
            if agent_state and agent_state[2] == depth:
                agent_x, agent_y = agent_state[0], agent_state[1]
                agent_circle = patches.Circle((agent_y, agent_x), 0.4,
                                            facecolor=self.depth_colors[depth % len(self.depth_colors)],
                                            edgecolor='black', linewidth=2)
                ax.add_patch(agent_circle)
            
            # Draw trajectory at this depth
            if show_trajectory and self.trajectory:
                depth_traj = [(x, y) for x, y, d in self.trajectory if d == depth]
                if depth_traj:
                    traj_x = [y for x, y in depth_traj]
                    traj_y = [x for x, y in depth_traj]
                    ax.plot(traj_x, traj_y, color=self.depth_colors[depth % len(self.depth_colors)],
                           alpha=0.6, linewidth=2, marker='o', markersize=3)
            
            ax.set_xlim(-0.5, self.env.base_size - 0.5)
            ax.set_ylim(-0.5, self.env.base_size - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Depth {depth} (Scale: {1/(2**depth):.2f})')
            ax.invert_yaxis()
        
        plt.suptitle('Fractal Environment - Multi-Scale View')
        plt.tight_layout()
        return fig
    
    def plot_depth_exploration_analysis(self):
        """
        Analyze and visualize depth exploration patterns.
        """
        if not self.depth_history:
            print("No depth history to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Depth over time
        axes[0, 0].plot(list(self.depth_history), linewidth=2)
        axes[0, 0].set_title('Depth Exploration Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Depth Level')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Depth distribution
        depth_counts = {}
        for depth in self.depth_history:
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        depths = list(depth_counts.keys())
        counts = list(depth_counts.values())
        colors = [self.depth_colors[d % len(self.depth_colors)] for d in depths]
        
        axes[0, 1].bar(depths, counts, color=colors, alpha=0.8)
        axes[0, 1].set_title('Time Spent at Each Depth')
        axes[0, 1].set_xlabel('Depth Level')
        axes[0, 1].set_ylabel('Number of Steps')
        
        # Depth transition matrix
        max_depth = max(self.depth_history) if self.depth_history else 0
        transition_matrix = np.zeros((max_depth + 1, max_depth + 1))
        
        for i in range(1, len(self.depth_history)):
            from_depth = self.depth_history[i-1]
            to_depth = self.depth_history[i]
            transition_matrix[from_depth, to_depth] += 1
        
        # Normalize by row sums
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
        
        im = axes[1, 0].imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[1, 0].set_title('Depth Transition Probabilities')
        axes[1, 0].set_xlabel('To Depth')
        axes[1, 0].set_ylabel('From Depth')
        
        # Add text annotations
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                if transition_matrix[i, j] > 0.1:
                    axes[1, 0].text(j, i, f'{transition_matrix[i, j]:.2f}',
                                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # Depth complexity over time (rolling window analysis)
        window_size = min(50, len(self.depth_history) // 4)
        if window_size > 5:
            complexity_scores = []
            for i in range(window_size, len(self.depth_history)):
                window = list(self.depth_history)[i-window_size:i]
                unique_depths = len(set(window))
                transitions = sum(1 for j in range(1, len(window)) if window[j] != window[j-1])
                complexity = (unique_depths + transitions) / window_size
                complexity_scores.append(complexity)
            
            axes[1, 1].plot(range(window_size, len(self.depth_history)), complexity_scores, linewidth=2)
            axes[1, 1].set_title(f'Exploration Complexity (Window: {window_size})')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Complexity Score')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor complexity analysis',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_self_observation_map(self, observation_memory):
        """
        Visualize the agent's self-observation patterns across scales.
        """
        if not observation_memory:
            print("No observation memory to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Extract observation data
        positions = [(obs['position'][0], obs['position'][1]) for obs in observation_memory]
        depths = [obs['depth'] for obs in observation_memory]
        scale_factors = [obs['scale_factor'] for obs in observation_memory]
        
        # Position heatmap by depth
        for depth in range(min(4, max(depths) + 1)):
            ax = axes[depth // 2, depth % 2]
            
            # Create position heatmap for this depth
            depth_positions = [pos for pos, d in zip(positions, depths) if d == depth]
            if not depth_positions:
                ax.text(0.5, 0.5, f'No observations\nat depth {depth}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Depth {depth} - No Data')
                continue
            
            heatmap = np.zeros((self.env.base_size, self.env.base_size))
            for x, y in depth_positions:
                if 0 <= x < self.env.base_size and 0 <= y < self.env.base_size:
                    heatmap[x, y] += 1
            
            im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Depth {depth} - Observation Density')
            ax.set_xlabel('Y Position')
            ax.set_ylabel('X Position')
            
            # Overlay environment structure
            obstacles, goal, portals = self.env.get_current_layout_elements()
            for obs in obstacles:
                ax.add_patch(patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                             facecolor='blue', alpha=0.3))
            
            goal_circle = patches.Circle((goal[1], goal[0]), 0.3,
                                       facecolor='green', alpha=0.7)
            ax.add_patch(goal_circle)
            
            for portal in portals:
                portal_circle = patches.Circle((portal[1], portal[0]), 0.25,
                                             facecolor='purple', alpha=0.7)
                ax.add_patch(portal_circle)
            
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Self-Observation Density Maps Across Fractal Depths')
        plt.tight_layout()
        return fig
    
    def create_animated_exploration(self, trajectory_data, save_path=None):
        """
        Create animated visualization of agent exploration through fractal depths.
        
        Args:
            trajectory_data: List of (state, observation) tuples
            save_path: Optional path to save animation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Setup main environment view
        obstacles, goal, portals = self.env.get_current_layout_elements()
        
        def setup_env_plot(ax):
            ax.clear()
            # Draw grid
            for i in range(self.env.base_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
            
            # Draw obstacles
            for obs in obstacles:
                rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                       facecolor=self.obstacle_color, alpha=0.8)
                ax.add_patch(rect)
            
            # Draw goal
            goal_circle = patches.Circle((goal[1], goal[0]), 0.3,
                                       facecolor=self.goal_color, alpha=0.9)
            ax.add_patch(goal_circle)
            
            # Draw portals
            for portal in portals:
                portal_circle = patches.Circle((portal[1], portal[0]), 0.25,
                                             facecolor=self.portal_color, alpha=0.9)
                ax.add_patch(portal_circle)
            
            ax.set_xlim(-0.5, self.env.base_size - 0.5)
            ax.set_ylim(-0.5, self.env.base_size - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
        
        def animate(frame):
            if frame >= len(trajectory_data):
                return
            
            state, observation = trajectory_data[frame]
            x, y, depth = state
            
            # Update main view
            setup_env_plot(ax1)
            
            # Draw agent
            agent_circle = patches.Circle((y, x), 0.4,
                                        facecolor=self.depth_colors[depth % len(self.depth_colors)],
                                        edgecolor='black', linewidth=2)
            ax1.add_patch(agent_circle)
            
            # Draw recent trajectory
            recent_steps = max(1, frame - 20)
            for i in range(recent_steps, frame):
                if i < len(trajectory_data):
                    px, py, pd = trajectory_data[i][0]
                    if pd == depth:  # Only show trajectory at current depth
                        ax1.plot(py, px, 'o', color=self.depth_colors[depth % len(self.depth_colors)],
                               alpha=0.3, markersize=2)
            
            ax1.set_title(f'Agent Position - Depth {depth} (Scale: {1/(2**depth):.2f})')
            
            # Update depth history plot
            ax2.clear()
            depths_so_far = [trajectory_data[i][0][2] for i in range(min(frame + 1, len(trajectory_data)))]
            ax2.plot(depths_so_far, linewidth=2, color='blue')
            ax2.set_title('Depth Exploration Over Time')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Depth Level')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, len(trajectory_data))
            ax2.set_ylim(0, max(self.env.max_depth + 0.5, max(depths_so_far) + 0.5) if depths_so_far else 1)
        
        anim = FuncAnimation(fig, animate, frames=len(trajectory_data), interval=200, repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to {save_path}")
        
        plt.tight_layout()
        return anim
    
    def plot_knowledge_transfer_analysis(self, transfer_data):
        """
        Visualize knowledge transfer patterns across fractal scales.
        
        Args:
            transfer_data: Dict with Q-tables or performance data across depths
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        if 'q_tables' in transfer_data:
            q_tables = transfer_data['q_tables']
            
            # Q-value similarity heatmap across depths
            similarities = np.zeros((len(q_tables), len(q_tables)))
            for i in range(len(q_tables)):
                for j in range(len(q_tables)):
                    if i != j:
                        # Calculate correlation between Q-tables
                        corr = np.corrcoef(q_tables[i].flatten(), q_tables[j].flatten())[0, 1]
                        similarities[i, j] = corr if not np.isnan(corr) else 0
                    else:
                        similarities[i, j] = 1.0
            
            im1 = axes[0, 0].imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0, 0].set_title('Q-Value Similarity Across Depths')
            axes[0, 0].set_xlabel('Depth')
            axes[0, 0].set_ylabel('Depth')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Add correlation values as text
            for i in range(len(q_tables)):
                for j in range(len(q_tables)):
                    if abs(similarities[i, j]) > 0.3:
                        axes[0, 0].text(j, i, f'{similarities[i, j]:.2f}',
                                       ha='center', va='center',
                                       color='white' if abs(similarities[i, j]) > 0.7 else 'black')
        
        if 'performance_by_depth' in transfer_data:
            performance = transfer_data['performance_by_depth']
            depths = list(performance.keys())
            values = list(performance.values())
            
            axes[0, 1].bar(depths, values, color=[self.depth_colors[d % len(self.depth_colors)] for d in depths])
            axes[0, 1].set_title('Performance by Depth')
            axes[0, 1].set_xlabel('Depth')
            axes[0, 1].set_ylabel('Success Rate')
        
        if 'transfer_learning_curves' in transfer_data:
            curves = transfer_data['transfer_learning_curves']
            for depth, curve in curves.items():
                axes[1, 0].plot(curve, label=f'Depth {depth}',
                              color=self.depth_colors[depth % len(self.depth_colors)])
            
            axes[1, 0].set_title('Transfer Learning Curves')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'zero_shot_performance' in transfer_data:
            zero_shot = transfer_data['zero_shot_performance']
            baseline = transfer_data.get('baseline_performance', {})
            
            depths = list(zero_shot.keys())
            zero_shot_values = [zero_shot[d] for d in depths]
            baseline_values = [baseline.get(d, 0) for d in depths]
            
            x = np.arange(len(depths))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7)
            axes[1, 1].bar(x + width/2, zero_shot_values, width, label='Zero-shot Transfer', alpha=0.7)
            
            axes[1, 1].set_title('Zero-shot Transfer Performance')
            axes[1, 1].set_xlabel('Target Depth')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(depths)
            axes[1, 1].legend()
        
        plt.suptitle('Knowledge Transfer Analysis Across Fractal Scales')
        plt.tight_layout()
        return fig


def demonstrate_fractal_visualization():
    """
    Demonstrate the fractal visualization capabilities.
    """
    # This would be called from the experiment script to show visualizations
    from .fractal_environment import FractalDepthEnvironment, SelfObservingAgent
    
    print("Creating demonstration of fractal visualization...")
    
    # Create environment
    env = FractalDepthEnvironment(base_size=10, num_portals=2, max_depth=2, seed=42)
    visualizer = FractalEnvironmentVisualizer(env)
    
    # Create some sample trajectory data
    agent = SelfObservingAgent(env)
    trajectory_data = []
    
    state = env.reset()
    for _ in range(100):
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        observation = env.get_observation_perspective()
        
        trajectory_data.append((state, observation))
        visualizer.add_step(state, observation)
        
        state = next_state
        if done:
            state = env.reset()
    
    # Generate visualizations
    print("Generating fractal overview...")
    fig1 = visualizer.plot_fractal_overview(agent_state=state, show_trajectory=True)
    plt.show()
    
    print("Generating depth exploration analysis...")
    fig2 = visualizer.plot_depth_exploration_analysis()
    plt.show()
    
    print("Generating self-observation map...")
    fig3 = visualizer.plot_self_observation_map(list(visualizer.observation_history))
    plt.show()
    
    print("Fractal visualization demonstration complete!")


if __name__ == "__main__":
    demonstrate_fractal_visualization() 