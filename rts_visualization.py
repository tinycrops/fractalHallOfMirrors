#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import time
from typing import List, Dict, Tuple, Optional
import os

from rts_environment import RTSEnvironment, UnitType, ResourceType, StructureType, MAP_SIZE

class RTSVisualization:
    """Visualization dashboard for the RTS environment with fractal attention"""
    
    def __init__(self, env: RTSEnvironment, agent=None, save_path=None):
        self.env = env
        self.agent = agent
        self.save_path = save_path
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
        
        # Main game view
        self.game_ax = plt.subplot(gs[0, :2])
        self.game_ax.set_title("RTS Environment", fontsize=14)
        
        # Attention dashboard
        self.attention_ax = plt.subplot(gs[0, 2])
        self.attention_ax.set_title("Fractal Attention", fontsize=14)
        
        # Attention history
        self.attention_history_ax = plt.subplot(gs[1, :])
        self.attention_history_ax.set_title("Attention Evolution", fontsize=14)
        
        # Hierarchical goals
        self.hierarchy_ax = plt.subplot(gs[2, 0])
        self.hierarchy_ax.set_title("Hierarchical Goals", fontsize=14)
        
        # Performance metrics
        self.metrics_ax = plt.subplot(gs[2, 1:])
        self.metrics_ax.set_title("Performance Metrics", fontsize=14)
        
        # History buffers
        self.attention_history = []
        self.crystal_history = []
        self.unit_count_history = []
        self.enemy_count_history = []
        self.time_markers = []
        self.event_markers = []
        
        # Set up initial plots
        self._setup_plots()
        
        plt.tight_layout()
    
    def _setup_plots(self):
        """Initialize all plots"""
        # Game view
        self.game_ax.set_xlim(0, MAP_SIZE)
        self.game_ax.set_ylim(0, MAP_SIZE)
        
        # Attention dashboard - create pie chart
        self.attention_pie = self.attention_ax.pie(
            [1, 1, 1],  # Equal initial values
            labels=['Micro', 'Meso', 'Super'],
            colors=['#FF9999', '#99FF99', '#9999FF'],
            autopct='%1.1f%%',
            startangle=90
        )
        self.attention_ax.axis('equal')
        
        # Attention history - create line plot
        self.micro_line, = self.attention_history_ax.plot([], [], 'r-', label='Micro')
        self.meso_line, = self.attention_history_ax.plot([], [], 'g-', label='Meso')
        self.super_line, = self.attention_history_ax.plot([], [], 'b-', label='Super')
        
        self.attention_history_ax.set_xlim(0, 100)  # Will auto-expand
        self.attention_history_ax.set_ylim(0, 1)
        self.attention_history_ax.set_xlabel('Time Steps')
        self.attention_history_ax.set_ylabel('Attention Weight')
        self.attention_history_ax.legend(loc='upper right')
        self.attention_history_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add dummy event marker for legend
        event_marker = self.attention_history_ax.axvline(0, color='red', linestyle='--', alpha=0)
        self.event_marker_legend = self.attention_history_ax.legend(
            [event_marker], ['Events'], loc='upper left')
        self.attention_history_ax.add_artist(self.event_marker_legend)
        self.attention_history_ax.add_artist(self.attention_history_ax.legend(loc='upper right'))
        
        # Hierarchical goals - text display
        self.hierarchy_ax.axis('off')
        self.hierarchy_text = self.hierarchy_ax.text(
            0.05, 0.95, "Hierarchical Goals\n\nNo data available", 
            transform=self.hierarchy_ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        # Performance metrics - line plots
        self.metrics_ax.set_xlim(0, 100)  # Will auto-expand
        self.metrics_ax.set_xlabel('Time Steps')
        self.metrics_ax.set_ylabel('Value')
        self.metrics_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create line objects for metrics
        self.crystal_line, = self.metrics_ax.plot([], [], 'g-', label='Crystals')
        self.unit_count_line, = self.metrics_ax.plot([], [], 'b-', label='Units')
        self.enemy_count_line, = self.metrics_ax.plot([], [], 'r-', label='Enemies')
        
        self.metrics_ax.legend(loc='upper left')
    
    def update(self, state, attention_weights=None, goals=None):
        """Update the visualization with current state"""
        self._update_game_view(state)
        
        # Record history for plots
        self.crystal_history.append(state['crystal_count'])
        self.unit_count_history.append(len(state['player_units']))
        self.enemy_count_history.append(len([e for e in state['enemy_units'] if self.env.is_visible(e.position)]))
        
        # Check for events to mark on timeline
        active_events = state.get('active_events', [])
        newly_active = False
        for event in active_events:
            if event.get('is_active', False) and event.get('trigger_time', 0) == state['time'] - 1:
                newly_active = True
                self.event_markers.append((state['time'], event['type']))
        
        # Update attention displays if agent data available
        if attention_weights is not None:
            self._update_attention_dashboard(attention_weights)
            self._update_attention_history(state['time'], attention_weights)
        
        # Update hierarchical goals if available
        if goals is not None:
            self._update_hierarchical_goals(goals)
        
        # Update performance metrics
        self._update_performance_metrics(state['time'])
        
        # Add vertical line for current time
        if newly_active:
            # Add event marker at this time
            self.attention_history_ax.axvline(
                state['time'], 
                color='red', 
                linestyle='--', 
                alpha=0.5
            )
        
        plt.tight_layout()
        
        # Save if path provided
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            plt.savefig(f"{self.save_path}/frame_{state['time']:04d}.png")
    
    def _update_game_view(self, state):
        """Update the main game view"""
        self.game_ax.clear()
        self.game_ax.set_title("RTS Environment", fontsize=14)
        self.game_ax.set_xlim(0, MAP_SIZE)
        self.game_ax.set_ylim(0, MAP_SIZE)
        
        # Draw fog of war
        visible_mask = (state['visibility'] > 0.5)
        fog_img = np.ones((MAP_SIZE, MAP_SIZE, 4))  # RGBA
        fog_img[~visible_mask, 3] = 0.7  # Alpha channel for non-visible cells
        
        self.game_ax.imshow(fog_img, extent=(0, MAP_SIZE, 0, MAP_SIZE), origin='lower')
        
        # Draw resources
        for resource in state['resources']:
            if resource.type == ResourceType.CRYSTAL:
                color = 'blue'
            else:  # VESPENE
                color = 'green'
            self.game_ax.scatter(resource.position[0] + 0.5, resource.position[1] + 0.5, 
                        color=color, s=100 * (resource.amount / 100))
        
        # Draw structures
        for structure in state['structures']:
            if structure.is_alive():
                if structure.type == StructureType.NEXUS:
                    color = 'gold'
                    size = 3
                else:  # TURRET
                    color = 'gray'
                    size = 2
                
                rect = patches.Rectangle((structure.position[0], structure.position[1]), 
                                        size, size, linewidth=1, edgecolor='black', 
                                        facecolor=color, alpha=0.7)
                self.game_ax.add_patch(rect)
                
                # Draw health bar
                health_pct = structure.health / (1000 if structure.type == StructureType.NEXUS else 300)
                self.game_ax.plot(
                    [structure.position[0], structure.position[0] + size * health_pct], 
                    [structure.position[1] - 0.2, structure.position[1] - 0.2], 
                    color='red', linewidth=2
                )
        
        # Draw player units
        for unit in state['player_units']:
            if unit.is_alive():
                if unit.type == UnitType.HARVESTER:
                    color = 'cyan'
                    marker = 'o'
                else:  # WARRIOR
                    color = 'blue'
                    marker = 's'
                
                self.game_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                            color=color, s=80, marker=marker)
                
                # Draw resource carried (for harvesters)
                if unit.type == UnitType.HARVESTER and unit.resources > 0:
                    self.game_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                                color='yellow', s=20)
                
                # Draw health bar
                max_health = 100 if unit.type == UnitType.HARVESTER else 200
                health_pct = unit.health / max_health
                self.game_ax.plot(
                    [unit.position[0], unit.position[0] + health_pct], 
                    [unit.position[1] - 0.2, unit.position[1] - 0.2], 
                    color='red', linewidth=2
                )
        
        # Draw enemy units (only if visible)
        for unit in state['enemy_units']:
            if self.env.is_visible(unit.position):
                if unit.type == UnitType.ELITE_RAIDER:
                    color = 'darkred'
                    marker = '*'
                    size = 100
                else:  # Regular RAIDER
                    color = 'red'
                    marker = 'x'
                    size = 80
                
                self.game_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                            color=color, s=size, marker=marker)
        
        # Draw event notifications
        for notif in state.get('event_notifications', []):
            if 'position' in notif and 'text' in notif:
                x, y = notif['position']
                if self.env.is_visible((x, y)):
                    color = notif.get('color', 'white')
                    # Add a marker for the event location
                    self.game_ax.scatter(x + 0.5, y + 0.5, color=color, 
                                s=150, marker='o', alpha=0.7, edgecolors='black')
                    # Add an arrow pointing to it
                    self.game_ax.annotate(notif['text'], xy=(x + 0.5, y + 0.5), 
                                xytext=(x + 5, y + 5),
                                arrowprops=dict(facecolor=color, shrink=0.05),
                                color=color, fontweight='bold', 
                                bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
        
        # Draw grid
        for i in range(MAP_SIZE + 1):
            self.game_ax.axhline(y=i, color='black', linestyle='-', alpha=0.2)
            self.game_ax.axvline(x=i, color='black', linestyle='-', alpha=0.2)
        
        # Add game stats text
        stats_text = f"Time: {state['time']}\nCrystals: {state['crystal_count']}\nVespene: {state['vespene_count']}\n"
        stats_text += f"Units: {len(state['player_units'])}\nEnemies: {len([e for e in state['enemy_units'] if self.env.is_visible(e.position)])}"
        
        self.game_ax.text(2, MAP_SIZE - 6, stats_text, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7))
    
    def _update_attention_dashboard(self, attention_weights):
        """Update the attention pie chart"""
        # Update pie chart
        self.attention_ax.clear()
        self.attention_ax.set_title("Fractal Attention", fontsize=14)
        
        self.attention_pie = self.attention_ax.pie(
            attention_weights,
            labels=['Micro', 'Meso', 'Super'],
            colors=['#FF9999', '#99FF99', '#9999FF'],
            autopct='%1.1f%%',
            startangle=90
        )
        self.attention_ax.axis('equal')
        
        # Add visual guidance
        dominant_level = np.argmax(attention_weights)
        level_names = ['Micro-level', 'Meso-level', 'Super-level']
        explanation = {
            0: "Focused on individual unit control and immediate reactions",
            1: "Focused on tactical objectives and group coordination",
            2: "Focused on strategic planning and resource allocation"
        }
        
        self.attention_ax.text(0, -1.2, 
                            f"Dominant: {level_names[dominant_level]}\n{explanation[dominant_level]}",
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7))
    
    def _update_attention_history(self, time_step, attention_weights):
        """Update the attention history plot"""
        # Add current attention weights to history
        self.attention_history.append(attention_weights)
        self.time_markers.append(time_step)
        
        # Update history plots with full history
        times = self.time_markers
        micro_history = [weights[0] for weights in self.attention_history]
        meso_history = [weights[1] for weights in self.attention_history]
        super_history = [weights[2] for weights in self.attention_history]
        
        # Update line data
        self.micro_line.set_data(times, micro_history)
        self.meso_line.set_data(times, meso_history)
        self.super_line.set_data(times, super_history)
        
        # Adjust axes limits if needed
        self.attention_history_ax.set_xlim(0, max(times) + 10)
        
        # Add event annotations
        for time, event_type in self.event_markers:
            if event_type not in self._get_existing_event_annotations():
                self.attention_history_ax.annotate(
                    event_type, 
                    xy=(time, 0.5),
                    xytext=(time, 0.8),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=8,
                    rotation=90,
                    ha='center'
                )
    
    def _get_existing_event_annotations(self):
        """Get list of existing event annotations to avoid duplicates"""
        annotations = []
        for child in self.attention_history_ax.get_children():
            if isinstance(child, plt.Annotation):
                annotations.append(child.get_text())
        return annotations
    
    def _update_hierarchical_goals(self, goals):
        """Update the hierarchical goals display"""
        self.hierarchy_ax.clear()
        self.hierarchy_ax.set_title("Hierarchical Goals", fontsize=14)
        self.hierarchy_ax.axis('off')
        
        # Create hierarchical text representation
        hierarchy_text = "HIERARCHICAL GOALS\n\n"
        
        # Strategic focus
        hierarchy_text += f"Strategic Focus: {goals.get('strategic_focus', 'Unknown')}\n"
        
        # Super goals
        super_goal = goals.get('super_goal')
        if super_goal is not None:
            hierarchy_text += "\nSuper-level Goals:\n"
            if isinstance(super_goal, dict):
                for key, value in super_goal.items():
                    hierarchy_text += f"  • {key}: {value:.2f if isinstance(value, float) else value}\n"
            elif isinstance(super_goal, (list, tuple, np.ndarray)):
                hierarchy_text += f"  • Resource ratios: {super_goal[0]:.2f}, {super_goal[1]:.2f}\n"
                hierarchy_text += f"  • Unit ratios: {super_goal[2]:.2f}, {super_goal[3]:.2f}\n"
                hierarchy_text += f"  • Expansion priority: {super_goal[4]:.2f}\n"
                hierarchy_text += f"  • Defense priority: {super_goal[5]:.2f}\n"
        
        # Tactical objective
        hierarchy_text += f"\nTactical Objective: {goals.get('tactical_objective', 'Unknown')}\n"
        
        # Meso goals
        meso_goal = goals.get('meso_goal')
        if meso_goal is not None:
            hierarchy_text += "\nMeso-level Goals:\n"
            if isinstance(meso_goal, dict):
                for key, value in meso_goal.items():
                    hierarchy_text += f"  • {key}: {value:.2f if isinstance(value, float) else value}\n"
            elif isinstance(meso_goal, (list, tuple, np.ndarray)):
                hierarchy_text += f"  • Rally point: ({meso_goal[0]*MAP_SIZE:.1f}, {meso_goal[1]*MAP_SIZE:.1f})\n"
                hierarchy_text += f"  • Priority resource: ({meso_goal[2]*MAP_SIZE:.1f}, {meso_goal[3]*MAP_SIZE:.1f})\n"
                hierarchy_text += f"  • Exploration target: ({meso_goal[4]*MAP_SIZE:.1f}, {meso_goal[5]*MAP_SIZE:.1f})\n"
                hierarchy_text += f"  • Harvester aggression: {meso_goal[6]:.2f}\n"
                hierarchy_text += f"  • Warrior aggression: {meso_goal[7]:.2f}\n"
        
        self.hierarchy_text = self.hierarchy_ax.text(
            0.05, 0.95, hierarchy_text, 
            transform=self.hierarchy_ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
    
    def _update_performance_metrics(self, time_step):
        """Update the performance metrics plot"""
        times = list(range(len(self.crystal_history)))
        
        # Update line data
        self.crystal_line.set_data(times, self.crystal_history)
        self.unit_count_line.set_data(times, self.unit_count_history)
        self.enemy_count_line.set_data(times, self.enemy_count_history)
        
        # Adjust axes limits if needed
        self.metrics_ax.set_xlim(0, max(times) + 10)
        max_crystal = max(self.crystal_history) if self.crystal_history else 100
        max_units = max(max(self.unit_count_history) if self.unit_count_history else 5,
                      max(self.enemy_count_history) if self.enemy_count_history else 5)
        self.metrics_ax.set_ylim(0, max(max_crystal * 1.1, max_units * 1.1))
    
    def save_gif(self, filename="rts_visualization.gif", fps=10):
        """Save the accumulated frames as a GIF animation"""
        if not self.save_path:
            print("No frames were saved. Set save_path in constructor to enable GIF creation.")
            return
        
        try:
            import imageio
            import glob
            
            filenames = sorted(glob.glob(f"{self.save_path}/frame_*.png"))
            
            if not filenames:
                print("No frames found in the save directory.")
                return
            
            with imageio.get_writer(filename, mode='I', fps=fps) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            
            print(f"GIF saved as {filename}")
            
        except ImportError:
            print("Could not create GIF. Please install imageio: pip install imageio")

# Example usage
if __name__ == "__main__":
    # Create environment with a scenario
    env = RTSEnvironment(seed=42, scenario="peaceful_start_and_ambush")
    
    # For demonstration, we'll just use random attention weights
    viz = RTSVisualization(env, save_path="viz_frames")
    
    # Simulate some steps
    for step in range(50):
        state = env.get_state()
        
        # Generate random attention weights for demonstration
        attention_weights = np.random.dirichlet(np.ones(3))
        
        # Generate random goals for demonstration
        goals = {
            'strategic_focus': 'ECONOMY' if step < 25 else 'DEFENSE',
            'super_goal': np.random.rand(8),
            'tactical_objective': 'BUILD_HARVESTER' if step < 25 else 'RALLY_WARRIORS',
            'meso_goal': np.random.rand(12)
        }
        
        # Update visualization
        viz.update(state, attention_weights, goals)
        
        # Step environment
        env.step()
        
        # Small delay for visualization
        time.sleep(0.1)
    
    # Create GIF from saved frames
    viz.save_gif() 