#!/usr/bin/env python3
"""
Educational Fractal Self-Observation Visualization (Fixed Version)

This module creates detailed, educational visualizations that demonstrate
how an AI agent can observe itself from different fractal scales and
develop enhanced awareness through multi-perspective learning.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque, defaultdict
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.fractal_environment import FractalDepthEnvironment, SelfObservingAgent
from tinycrops_hall_of_mirrors.grid_world.fractal_visualization import FractalEnvironmentVisualizer


class EducationalFractalVisualizer:
    """
    Educational visualization system for fractal self-observation.
    
    This class creates intuitive, step-by-step visualizations that clearly
    demonstrate how agents navigate fractal environments and develop
    self-observation capabilities.
    """
    
    def __init__(self, env, figsize=(16, 12)):
        self.env = env
        self.figsize = figsize
        self.step_count = 0
        
        # Enhanced tracking for educational purposes
        self.full_trajectory = []  # Complete step-by-step history
        self.depth_transitions = []  # Track when agent changes depths
        self.learning_moments = []  # Track when agent learns something new
        self.self_observation_events = []  # Track self-observation instances
        self.decision_history = []  # Track decision-making process
        
        # Color schemes optimized for education
        self.depth_colors = {
            0: '#E74C3C',  # Red - Base reality
            1: '#3498DB',  # Blue - First fractal level
            2: '#2ECC71',  # Green - Second fractal level
            3: '#F39C12',  # Orange - Third fractal level
            4: '#9B59B6'   # Purple - Fourth fractal level
        }
        
        self.element_colors = {
            'agent': '#2C3E50',
            'goal': '#27AE60',
            'portal': '#E91E63',
            'obstacle': '#7F8C8D',
            'path': '#3498DB',
            'self_view': '#FF6B6B'
        }
        
    def record_step(self, state, action, next_state, reward, info, q_values=None, observations=None):
        """Record a complete step for educational analysis."""
        self.step_count += 1
        
        step_data = {
            'step': self.step_count,
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'info': info,
            'q_values': q_values.copy() if q_values is not None else None,
            'observations': observations.copy() if observations is not None else None,
            'timestamp': time.time()
        }
        
        self.full_trajectory.append(step_data)
        
        # Track depth transitions
        if state[2] != next_state[2]:
            transition_type = 'zoom_in' if next_state[2] > state[2] else 'zoom_out'
            self.depth_transitions.append({
                'step': self.step_count,
                'from_depth': state[2],
                'to_depth': next_state[2],
                'type': transition_type,
                'position': (state[0], state[1])
            })
            
        # Track self-observation events (when agent uses multi-scale information)
        if observations and len(observations.get('parent_positions', [])) > 0:
            self.self_observation_events.append({
                'step': self.step_count,
                'current_depth': state[2],
                'observed_scales': len(observations['parent_positions']) + 1,
                'scale_factor': observations.get('scale_factor', 1.0)
            })
    
    def create_educational_overview(self, current_step=None, show_explanations=True):
        """Create a comprehensive educational overview of the fractal system."""
        
        fig = plt.figure(figsize=(20, 16))
        
        # Main title and explanation
        fig.suptitle(
            'AI Agent with Fractal Self-Observation Capabilities\n'
            'Watch how the agent learns to navigate between different scales of reality',
            fontsize=16, fontweight='bold'
        )
        
        # Create subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Multi-scale environment view (top row)
        for depth in range(min(4, self.env.max_depth + 1)):
            ax = fig.add_subplot(gs[0, depth])
            self._draw_scale_view(ax, depth, current_step)
            
        # 2. Agent's perspective view (second row, left)
        ax_perspective = fig.add_subplot(gs[1, :2])
        self._draw_agent_perspective(ax_perspective, current_step)
        
        # 3. Self-observation analysis (second row, right)
        ax_self_obs = fig.add_subplot(gs[1, 2:])
        self._draw_self_observation_analysis(ax_self_obs, current_step)
        
        # 4. Learning progression (third row)
        ax_learning = fig.add_subplot(gs[2, :])
        self._draw_learning_progression(ax_learning, current_step)
        
        # 5. Decision making process (bottom row, left)
        ax_decisions = fig.add_subplot(gs[3, :2])
        self._draw_decision_process(ax_decisions, current_step)
        
        # 6. Concept explanation (bottom row, right)
        ax_explanation = fig.add_subplot(gs[3, 2:])
        if show_explanations:
            self._draw_concept_explanation(ax_explanation)
        
        # Add navigation information
        if current_step is not None and self.full_trajectory:
            step_data = self.full_trajectory[min(current_step, len(self.full_trajectory) - 1)]
            info_text = (
                f"Step: {step_data['step']} | "
                f"Position: {step_data['state'][:2]} | "
                f"Depth: {step_data['state'][2]} | "
                f"Action: {step_data['action']} | "
                f"Reward: {step_data['reward']:.2f}"
            )
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        return fig
    
    def _draw_scale_view(self, ax, depth, current_step):
        """Draw the environment at a specific fractal depth."""
        obstacles, goal, portals = self.env.get_current_layout_elements()
        
        # Draw grid
        for i in range(self.env.base_size + 1):
            ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
        
        # Draw obstacles
        for obs in obstacles:
            rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                   facecolor=self.element_colors['obstacle'], alpha=0.8)
            ax.add_patch(rect)
        
        # Draw goal
        goal_circle = patches.Circle((goal[1], goal[0]), 0.3,
                                   facecolor=self.element_colors['goal'], alpha=0.9)
        ax.add_patch(goal_circle)
        
        # Draw portals
        for portal in portals:
            portal_circle = patches.Circle((portal[1], portal[0]), 0.25,
                                         facecolor=self.element_colors['portal'], alpha=0.9)
            ax.add_patch(portal_circle)
        
        # Draw agent if at this depth
        current_agent_state = None
        if current_step is not None and self.full_trajectory:
            step_data = self.full_trajectory[min(current_step, len(self.full_trajectory) - 1)]
            current_agent_state = step_data['state']
        
        if current_agent_state and current_agent_state[2] == depth:
            agent_x, agent_y = current_agent_state[0], current_agent_state[1]
            agent_circle = patches.Circle((agent_y, agent_x), 0.4,
                                        facecolor=self.depth_colors[depth],
                                        edgecolor='black', linewidth=3)
            ax.add_patch(agent_circle)
            
            # Add glow effect to show current position
            glow_circle = patches.Circle((agent_y, agent_x), 0.6,
                                       facecolor=self.depth_colors[depth],
                                       alpha=0.3)
            ax.add_patch(glow_circle)
        
        # Draw trajectory at this depth
        if self.full_trajectory:
            depth_positions = [(step['state'][0], step['state'][1]) 
                             for step in self.full_trajectory 
                             if step['state'][2] == depth]
            
            if len(depth_positions) > 1:
                traj_x = [pos[1] for pos in depth_positions]
                traj_y = [pos[0] for pos in depth_positions]
                ax.plot(traj_x, traj_y, color=self.depth_colors[depth],
                       alpha=0.6, linewidth=2, linestyle='--')
        
        # Styling
        ax.set_xlim(-0.5, self.env.base_size - 0.5)
        ax.set_ylim(-0.5, self.env.base_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Scale {depth}\n(Zoom: {1/(2**depth):.2f}x)', 
                    fontweight='bold', color=self.depth_colors[depth])
        ax.invert_yaxis()
        
        # Add scale indicator
        scale_text = f"Reality Level {depth}"
        if depth == 0:
            scale_text += "\n(Base Reality)"
        elif depth == 1:
            scale_text += "\n(First Fractal)"
        else:
            scale_text += f"\n(Fractal Level {depth})"
            
        ax.text(0.02, 0.98, scale_text, transform=ax.transAxes, 
               va='top', ha='left', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    def _draw_agent_perspective(self, ax, current_step):
        """Draw what the agent currently sees and understands."""
        if not self.full_trajectory or current_step is None:
            ax.text(0.5, 0.5, 'No agent data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title("Agent's Current Perspective")
            return
        
        step_data = self.full_trajectory[min(current_step, len(self.full_trajectory) - 1)]
        state = step_data['state']
        observations = step_data.get('observations', {})
        
        ax.clear()
        
        # Create a visual representation of what the agent "sees"
        current_depth = state[2]
        
        # Draw the agent's current view
        view_radius = 3
        agent_x, agent_y = state[0], state[1]
        
        # Create viewing area
        view_area = patches.Circle((agent_y, agent_x), view_radius, 
                                 facecolor=self.depth_colors[current_depth], 
                                 alpha=0.2, linewidth=2,
                                 edgecolor=self.depth_colors[current_depth])
        ax.add_patch(view_area)
        
        # Draw environment elements within view
        obstacles, goal, portals = self.env.get_current_layout_elements()
        
        for obs in obstacles:
            if abs(obs[0] - agent_x) <= view_radius and abs(obs[1] - agent_y) <= view_radius:
                rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                       facecolor=self.element_colors['obstacle'], alpha=0.8)
                ax.add_patch(rect)
        
        # Draw goal if visible
        if abs(goal[0] - agent_x) <= view_radius and abs(goal[1] - agent_y) <= view_radius:
            goal_circle = patches.Circle((goal[1], goal[0]), 0.3,
                                       facecolor=self.element_colors['goal'], alpha=0.9)
            ax.add_patch(goal_circle)
        
        # Draw portals if visible
        for portal in portals:
            if abs(portal[0] - agent_x) <= view_radius and abs(portal[1] - agent_y) <= view_radius:
                portal_circle = patches.Circle((portal[1], portal[0]), 0.25,
                                             facecolor=self.element_colors['portal'], alpha=0.9)
                ax.add_patch(portal_circle)
        
        # Draw the agent
        agent_circle = patches.Circle((agent_y, agent_x), 0.4,
                                    facecolor=self.element_colors['agent'],
                                    edgecolor='white', linewidth=2)
        ax.add_patch(agent_circle)
        
        # Add sight lines to important elements
        for portal in portals:
            if abs(portal[0] - agent_x) <= view_radius and abs(portal[1] - agent_y) <= view_radius:
                ax.plot([agent_y, portal[1]], [agent_x, portal[0]], 
                       color=self.element_colors['portal'], alpha=0.5, linestyle=':', linewidth=2)
        
        # Add perspective information text
        perspective_info = [
            f"Current Depth: {current_depth}",
            f"Position: ({agent_x}, {agent_y})",
            f"Scale Factor: {observations.get('scale_factor', 1.0):.2f}",
            f"Can Go Deeper: {observations.get('can_zoom_deeper', False)}"
        ]
        
        if observations.get('parent_positions'):
            perspective_info.append(f"Parent Positions: {len(observations['parent_positions'])}")
        
        info_text = '\n'.join(perspective_info)
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
               va='top', ha='right', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
        
        ax.set_xlim(agent_y - view_radius - 1, agent_y + view_radius + 1)
        ax.set_ylim(agent_x - view_radius - 1, agent_x + view_radius + 1)
        ax.set_aspect('equal')
        ax.set_title("Agent's Current Perspective & Awareness", fontweight='bold')
        ax.invert_yaxis()
    
    def _draw_self_observation_analysis(self, ax, current_step):
        """Visualize the agent's self-observation capabilities."""
        if not self.self_observation_events:
            ax.text(0.5, 0.5, 'Building self-observation\ncapabilities...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Self-Observation Development")
            return
        
        # Create a timeline of self-observation events
        steps = [event['step'] for event in self.self_observation_events]
        scales = [event['observed_scales'] for event in self.self_observation_events]
        depths = [event['current_depth'] for event in self.self_observation_events]
        
        # Create scatter plot showing self-observation complexity over time
        colors = [self.depth_colors[depth] for depth in depths]
        scatter = ax.scatter(steps, scales, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add trend line
        if len(steps) > 2:
            z = np.polyfit(steps, scales, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "r--", alpha=0.8, linewidth=2, label='Learning Trend')
        
        ax.set_xlabel('Learning Step')
        ax.set_ylabel('Scales Observed Simultaneously')
        ax.set_title('Self-Observation Capability Development', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add current step indicator
        if current_step is not None and current_step < len(steps):
            ax.axvline(x=current_step, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add legend
        depth_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=10, alpha=0.7)
                        for depth, color in self.depth_colors.items() if depth <= self.env.max_depth]
        depth_labels = [f'Depth {depth}' for depth in range(self.env.max_depth + 1)]
        ax.legend(depth_patches, depth_labels, loc='upper left', title='Current Depth')
        
        # Add insight text
        if self.self_observation_events:
            latest_event = self.self_observation_events[-1]
            max_scales = max(scales)
            insight_text = (
                f"Max scales observed: {max_scales}\n"
                f"Self-observation events: {len(self.self_observation_events)}\n"
                f"Latest depth: {latest_event['current_depth']}"
            )
            ax.text(0.98, 0.02, insight_text, transform=ax.transAxes,
                   va='bottom', ha='right', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))
    
    def _draw_learning_progression(self, ax, current_step):
        """Show the agent's learning progression over time."""
        if not self.full_trajectory:
            ax.text(0.5, 0.5, 'Learning in progress...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title("Learning Progression")
            return
        
        # Extract learning metrics
        steps = [step['step'] for step in self.full_trajectory]
        rewards = [step['reward'] for step in self.full_trajectory]
        depths = [step['state'][2] for step in self.full_trajectory]
        
        # Calculate cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        # Plot cumulative reward
        ax.plot(steps, cumulative_rewards, color='green', linewidth=2, label='Cumulative Reward')
        ax.set_ylabel('Cumulative Reward', color='green')
        ax.tick_params(axis='y', labelcolor='green')
        
        # Plot depth exploration as bars
        depth_colors_list = [self.depth_colors[d] for d in depths]
        ax2.bar(steps, depths, alpha=0.6, color=depth_colors_list, width=1.0, label='Exploration Depth')
        ax2.set_ylabel('Exploration Depth', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Mark depth transitions
        for transition in self.depth_transitions:
            ax.axvline(x=transition['step'], color='red', alpha=0.5, linestyle=':', linewidth=1)
        
        # Mark current step
        if current_step is not None and current_step < len(steps):
            ax.axvline(x=current_step, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Learning Step')
        ax.set_title('Learning Progression: Rewards and Fractal Exploration', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if len(self.depth_transitions) > 0:
            stats_text = (
                f"Depth transitions: {len(self.depth_transitions)}\n"
                f"Max depth reached: {max(depths)}\n"
                f"Current reward: {cumulative_rewards[-1]:.1f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
    
    def _draw_decision_process(self, ax, current_step):
        """Visualize the agent's decision-making process."""
        if not self.full_trajectory or current_step is None:
            ax.text(0.5, 0.5, 'Decision analysis\ncoming soon...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Decision Making Process")
            return
        
        step_data = self.full_trajectory[min(current_step, len(self.full_trajectory) - 1)]
        q_values = step_data.get('q_values')
        
        if q_values is None:
            ax.text(0.5, 0.5, 'Q-values not available\nfor this step', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Decision Making Process")
            return
        
        # Create bar chart of Q-values for current state
        actions = list(self.env.actions.keys())
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        q_vals = [q_values.get(action, 0) for action in actions]
        chosen_action = step_data['action']
        
        # Color bars - highlight chosen action
        colors = ['gold' if action == chosen_action else 'lightblue' for action in actions]
        
        bars = ax.bar(action_names, q_vals, color=colors, alpha=0.8, edgecolor='black')
        
        # Highlight the chosen action
        if chosen_action in actions:
            bars[chosen_action].set_height(q_vals[chosen_action])
            bars[chosen_action].set_color('orange')
            bars[chosen_action].set_edgecolor('red')
            bars[chosen_action].set_linewidth(3)
        
        ax.set_ylabel('Q-Value (Expected Future Reward)')
        ax.set_title(f'Decision Making at Step {step_data["step"]}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add decision info
        decision_info = (
            f"State: {step_data['state']}\n"
            f"Chosen Action: {action_names[chosen_action] if chosen_action < len(action_names) else 'UNKNOWN'}\n"
            f"Expected Reward: {q_vals[chosen_action]:.3f}\n"
            f"Actual Reward: {step_data['reward']:.3f}"
        )
        ax.text(0.98, 0.98, decision_info, transform=ax.transAxes,
               va='top', ha='right', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.9))
    
    def _draw_concept_explanation(self, ax):
        """Draw an explanation of the fractal self-observation concept."""
        ax.axis('off')
        
        explanation_text = """
FRACTAL SELF-OBSERVATION CONCEPT

🌀 What is Fractal Self-Observation?
The agent can enter "portals" that lead to scaled
versions of the same environment, allowing it to
observe itself from different perspectives.

🧠 How does this enhance AI capabilities?
• Multi-scale awareness and reasoning
• Cross-scale knowledge transfer
• Enhanced spatial understanding
• Improved problem-solving strategies

🎯 Key Benefits:
• Agents develop better spatial intuition
• Learning transfers between scales
• More robust navigation strategies
• Enhanced environmental understanding

📊 What to watch for:
• Depth transitions (red lines)
• Multi-scale trajectory patterns
• Self-observation events
• Cross-scale learning improvements
        """
        
        ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
               va='top', ha='left', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95))
        
        ax.set_title("Understanding Fractal Self-Observation", fontweight='bold', fontsize=14)


def run_educational_demonstration():
    """Run a complete educational demonstration of fractal self-observation."""
    print("🌀 EDUCATIONAL FRACTAL SELF-OBSERVATION DEMONSTRATION")
    print("=" * 70)
    print("This demonstration shows how an AI agent develops self-observation")
    print("capabilities by navigating between fractal scales of reality.")
    print()
    
    # Create environment
    env = FractalDepthEnvironment(base_size=8, num_portals=2, max_depth=2, seed=42)
    visualizer = EducationalFractalVisualizer(env)
    
    print(f"Environment Setup:")
    print(f"  Grid Size: {env.base_size}x{env.base_size}")
    print(f"  Portals: {len(env.base_portal_coords)} at {env.base_portal_coords}")
    print(f"  Max Depth: {env.max_depth}")
    print(f"  Goal: {env.base_goal}")
    print()
    
    # Create agent
    agent = SelfObservingAgent(env, alpha=0.1, gamma=0.95, epsilon_start=0.9, epsilon_decay=0.995)
    
    print("Training agent with educational tracking...")
    
    # Train with detailed tracking
    trajectory_data = []
    state = env.reset()
    total_episodes = 50  # Reduced for demonstration
    steps_per_episode = 100
    
    for episode in range(total_episodes):
        episode_trajectory = []
        state = env.reset()
        
        for step in range(steps_per_episode):
            # Get current Q-values for educational purposes
            depth = state[2]
            x, y = int(state[0]), int(state[1])
            q_values = {}
            if depth < len(agent.q_tables) and x < agent.q_tables[depth].shape[0] and y < agent.q_tables[depth].shape[1]:
                for action in env.actions.keys():
                    q_values[action] = agent.q_tables[depth][x, y, action]
            else:
                q_values = {action: 0.0 for action in env.actions.keys()}
            
            # Choose action
            action = agent.choose_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Get observations
            observations = env.get_observation_perspective()
            
            # Record step for educational analysis
            visualizer.record_step(state, action, next_state, reward, info, q_values, observations)
            episode_trajectory.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'info': info,
                'q_values': q_values,
                'observations': observations
            })
            
            # Update agent
            agent.learn_from_experience(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                print(f"  Episode {episode + 1}: Goal reached in {step + 1} steps!")
                break
        
        trajectory_data.extend(episode_trajectory)
        
        # Show progress
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{total_episodes} episodes")
    
    print("\nTraining complete! Generating educational visualizations...")
    
    # Create comprehensive educational overview
    print("\n1. Creating educational overview...")
    fig1 = visualizer.create_educational_overview(current_step=len(trajectory_data)-1)
    plt.show()
    
    # Analyze results
    insights = agent.get_self_observation_insights()
    print(f"\n📊 FINAL ANALYSIS:")
    print(f"  Total steps: {len(trajectory_data)}")
    print(f"  Depth transitions: {len(visualizer.depth_transitions)}")
    print(f"  Self-observation events: {len(visualizer.self_observation_events)}")
    print(f"  Scale transitions: {insights.get('total_scale_transitions', 0)}")
    print(f"  Exploration depth ratio: {insights.get('exploration_depth_ratio', 0):.1%}")
    
    if len(visualizer.depth_transitions) > 5:
        print(f"\n🎯 SUCCESS: Agent developed fractal navigation capabilities!")
        if len(visualizer.self_observation_events) > 2:
            print(f"🧠 ENHANCED: Agent demonstrates multi-scale self-observation!")
    
    print(f"\n✨ Educational demonstration complete!")
    print(f"The agent has learned to navigate fractal environments and observe")
    print(f"itself from multiple scales, demonstrating enhanced spatial awareness.")
    
    return agent, visualizer, trajectory_data


if __name__ == "__main__":
    # Run the educational demonstration
    agent, visualizer, trajectory = run_educational_demonstration() 