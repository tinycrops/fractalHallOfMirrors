#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any
import torch
from tqdm import tqdm

from rts_environment import RTSEnvironment, UnitType, StructureType, ActionType
from rts_fractal_agent import FractalAgent
from rts_fractal_attention_agent import FractalAttentionAgent

# Simple scripted agent for baseline comparison
class ScriptedAgent:
    def __init__(self):
        self.build_order = ["harvester", "harvester", "harvester", "warrior", "harvester", "warrior"]
        self.build_index = 0
        self.last_build_time = 0
    
    def act(self, state, env):
        # Basic scripted behavior:
        # 1. Build units according to build order
        # 2. Send harvesters to nearest resources
        # 3. Send warriors to attack nearest enemies or defend nexus
        
        # Find nexus
        nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
        if not nexus:
            return  # Game over if no nexus
        
        # Handle unit production
        if self.build_index < len(self.build_order) and state['time'] - self.last_build_time > 10:
            if self.build_order[self.build_index] == "harvester" and state['crystal_count'] >= 50:
                if nexus.produce_unit(UnitType.HARVESTER, env):
                    self.last_build_time = state['time']
                    self.build_index += 1
            elif self.build_order[self.build_index] == "warrior" and state['crystal_count'] >= 100:
                if nexus.produce_unit(UnitType.WARRIOR, env):
                    self.last_build_time = state['time']
                    self.build_index += 1
        
        # Handle harvesters
        harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
        resources = state['resources']
        
        for harvester in harvesters:
            # If carrying resources and near nexus, return them
            if harvester.resources > 0:
                dx = nexus.position[0] - harvester.position[0]
                dy = nexus.position[1] - harvester.position[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    harvester.return_resources(nexus, env)
                    continue
                
                # Move toward nexus
                if abs(dx) > abs(dy):
                    if dx < 0:
                        harvester.move(ActionType.MOVE_LEFT, env)
                    else:
                        harvester.move(ActionType.MOVE_RIGHT, env)
                else:
                    if dy < 0:
                        harvester.move(ActionType.MOVE_UP, env)
                    else:
                        harvester.move(ActionType.MOVE_DOWN, env)
                continue
            
            # Find nearest resource
            if resources:
                nearest_resource = min(resources, 
                                    key=lambda r: ((r.position[0] - harvester.position[0])**2 + 
                                                 (r.position[1] - harvester.position[1])**2))
                
                # If adjacent to resource, harvest it
                dx = nearest_resource.position[0] - harvester.position[0]
                dy = nearest_resource.position[1] - harvester.position[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    harvester.harvest(nearest_resource, env)
                    continue
                
                # Move toward resource
                if abs(dx) > abs(dy):
                    if dx < 0:
                        harvester.move(ActionType.MOVE_LEFT, env)
                    else:
                        harvester.move(ActionType.MOVE_RIGHT, env)
                else:
                    if dy < 0:
                        harvester.move(ActionType.MOVE_UP, env)
                    else:
                        harvester.move(ActionType.MOVE_DOWN, env)
        
        # Handle warriors
        warriors = [u for u in state['player_units'] if u.type == UnitType.WARRIOR]
        enemies = state['enemy_units']
        
        for warrior in warriors:
            # If enemies are visible, attack nearest one
            if enemies:
                nearest_enemy = min(enemies, 
                                   key=lambda e: ((e.position[0] - warrior.position[0])**2 + 
                                                (e.position[1] - warrior.position[1])**2))
                
                # If adjacent to enemy, attack it
                dx = nearest_enemy.position[0] - warrior.position[0]
                dy = nearest_enemy.position[1] - warrior.position[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    warrior.attack(nearest_enemy, env)
                    continue
                
                # Move toward enemy
                if abs(dx) > abs(dy):
                    if dx < 0:
                        warrior.move(ActionType.MOVE_LEFT, env)
                    else:
                        warrior.move(ActionType.MOVE_RIGHT, env)
                else:
                    if dy < 0:
                        warrior.move(ActionType.MOVE_UP, env)
                    else:
                        warrior.move(ActionType.MOVE_DOWN, env)
            else:
                # No enemies visible, patrol around nexus
                dx = nexus.position[0] - warrior.position[0]
                dy = nexus.position[1] - warrior.position[1]
                
                # If far from nexus, move toward it
                if abs(dx) > 5 or abs(dy) > 5:
                    if abs(dx) > abs(dy):
                        if dx < 0:
                            warrior.move(ActionType.MOVE_LEFT, env)
                        else:
                            warrior.move(ActionType.MOVE_RIGHT, env)
                    else:
                        if dy < 0:
                            warrior.move(ActionType.MOVE_UP, env)
                        else:
                            warrior.move(ActionType.MOVE_DOWN, env)
                else:
                    # Random patrol movement
                    move_dir = np.random.randint(0, 4)
                    if move_dir == 0:
                        warrior.move(ActionType.MOVE_UP, env)
                    elif move_dir == 1:
                        warrior.move(ActionType.MOVE_DOWN, env)
                    elif move_dir == 2:
                        warrior.move(ActionType.MOVE_LEFT, env)
                    else:
                        warrior.move(ActionType.MOVE_RIGHT, env)

# Run a single episode with an agent
def run_episode(agent, max_steps=1000, seed=None, render_every=None, verbose=False):
    env = RTSEnvironment(seed=seed)
    
    # Metrics to track
    survival_time = 0
    crystal_collected = 0
    vespene_collected = 0
    enemies_killed = 0
    units_lost = 0
    exploration_pct = 0
    attention_history = []
    
    # For attention agent only
    if hasattr(agent, 'attention_weights'):
        is_attention_agent = True
    else:
        is_attention_agent = False
    
    # Initial counts
    initial_enemy_count = len(env.enemy_units)
    initial_player_count = len(env.player_units)
    
    # Run the episode
    for step in tqdm(range(max_steps), disable=not verbose):
        # Get the current state
        state = env.get_state()
        
        # Record metrics
        crystal_collected = state['crystal_count']
        vespene_collected = state['vespene_count']
        exploration_pct = np.mean(state['visibility'])
        
        # Record attention weights if applicable
        if is_attention_agent:
            attention_history.append(agent.attention_weights.copy())
        
        # Render if needed
        if render_every is not None and step % render_every == 0:
            env.render()
            if is_attention_agent and verbose:
                print(f"Step {step}, Attention weights: {agent.attention_weights}")
        
        # Agent acts
        agent.act(state, env)
        
        # Environment step
        game_over = env.step()
        
        # Update survival time
        survival_time = step + 1
        
        # Calculate enemies killed and units lost
        current_enemy_count = len(env.enemy_units)
        current_player_count = len(env.player_units)
        
        enemies_killed += max(0, initial_enemy_count - current_enemy_count)
        units_lost += max(0, initial_player_count - current_player_count)
        
        # Update counts for next iteration
        initial_enemy_count = current_enemy_count
        initial_player_count = current_player_count
        
        # Check if game is over
        if game_over:
            if verbose:
                print(f"Game over at step {step}!")
            break
    
    # Final render
    if render_every is not None:
        env.render()
    
    # Compute economic score
    economic_score = crystal_collected + (vespene_collected * 3)  # Vespene is worth 3x crystals
    
    # Compute combat score
    combat_score = enemies_killed * 10 - units_lost * 5
    
    # Compile results
    results = {
        'survival_time': survival_time,
        'economic_score': economic_score,
        'combat_score': combat_score,
        'exploration_pct': exploration_pct,
        'crystal_collected': crystal_collected,
        'vespene_collected': vespene_collected,
        'enemies_killed': enemies_killed,
        'units_lost': units_lost
    }
    
    if is_attention_agent:
        results['attention_history'] = attention_history
    
    return results

# Run multiple episodes with different agent types
def compare_agents(n_episodes=5, max_steps=500, seeds=None, render_final=True):
    if seeds is None:
        seeds = [42 + i for i in range(n_episodes)]
    
    # Results for each agent type
    scripted_results = []
    fractal_results = []
    attention_results = []
    
    # Run episodes
    for i, seed in enumerate(seeds):
        print(f"\nEpisode {i+1}/{n_episodes} (seed={seed}):")
        
        # Scripted agent
        print("Running Scripted Agent...")
        scripted_agent = ScriptedAgent()
        scripted_result = run_episode(scripted_agent, max_steps=max_steps, seed=seed, 
                                      render_every=None, verbose=False)
        scripted_results.append(scripted_result)
        print(f"  Survival time: {scripted_result['survival_time']}")
        print(f"  Economic score: {scripted_result['economic_score']}")
        print(f"  Combat score: {scripted_result['combat_score']}")
        
        # Fractal agent
        print("Running Fractal Agent...")
        fractal_agent = FractalAgent()
        fractal_result = run_episode(fractal_agent, max_steps=max_steps, seed=seed, 
                                     render_every=None, verbose=False)
        fractal_results.append(fractal_result)
        print(f"  Survival time: {fractal_result['survival_time']}")
        print(f"  Economic score: {fractal_result['economic_score']}")
        print(f"  Combat score: {fractal_result['combat_score']}")
        
        # Attention agent
        print("Running Fractal Attention Agent...")
        attention_agent = FractalAttentionAgent()
        attention_result = run_episode(attention_agent, max_steps=max_steps, seed=seed, 
                                       render_every=None, verbose=False)
        attention_results.append(attention_result)
        print(f"  Survival time: {attention_result['survival_time']}")
        print(f"  Economic score: {attention_result['economic_score']}")
        print(f"  Combat score: {attention_result['combat_score']}")
    
    # Render final episodes if requested
    if render_final:
        print("\nRendering final episodes...")
        
        print("Scripted Agent:")
        scripted_agent = ScriptedAgent()
        run_episode(scripted_agent, max_steps=max_steps, seed=seeds[-1], 
                    render_every=50, verbose=True)
        
        print("Fractal Agent:")
        fractal_agent = FractalAgent()
        run_episode(fractal_agent, max_steps=max_steps, seed=seeds[-1], 
                    render_every=50, verbose=True)
        
        print("Fractal Attention Agent:")
        attention_agent = FractalAttentionAgent()
        result = run_episode(attention_agent, max_steps=max_steps, seed=seeds[-1], 
                             render_every=50, verbose=True)
        
        # Plot attention evolution for the attention agent
        if 'attention_history' in result:
            plot_attention_evolution(result['attention_history'])
    
    # Compile and print results
    print("\n=== RESULTS ===")
    
    # Calculate average metrics
    avg_scripted = {key: np.mean([r[key] for r in scripted_results]) 
                   for key in scripted_results[0] if key != 'attention_history'}
    avg_fractal = {key: np.mean([r[key] for r in fractal_results]) 
                  for key in fractal_results[0] if key != 'attention_history'}
    avg_attention = {key: np.mean([r[key] for r in attention_results]) 
                    for key in attention_results[0] if key != 'attention_history'}
    
    print("\nAverage Metrics:")
    print(f"Scripted Agent: {avg_scripted}")
    print(f"Fractal Agent: {avg_fractal}")
    print(f"Fractal Attention Agent: {avg_attention}")
    
    # Plot comparative results
    plot_comparative_results(avg_scripted, avg_fractal, avg_attention)
    
    return {
        'scripted': scripted_results,
        'fractal': fractal_results,
        'attention': attention_results
    }

# Plot the evolution of attention weights
def plot_attention_evolution(attention_history):
    """Visualize how attention shifts over the course of an episode."""
    # Convert to numpy array for easier manipulation
    attn_history = np.array(attention_history)
    
    # Smooth the data
    window_size = max(1, len(attn_history) // 20)  # 20 data points for smoothness
    
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Apply smoothing if we have enough data
    if len(attn_history) > window_size * 2:
        smooth_micro = smooth(attn_history[:, 0], window_size)
        smooth_meso = smooth(attn_history[:, 1], window_size)
        smooth_super = smooth(attn_history[:, 2], window_size)
        x = range(window_size-1, len(attn_history))
    else:
        smooth_micro = attn_history[:, 0]
        smooth_meso = attn_history[:, 1]
        smooth_super = attn_history[:, 2]
        x = range(len(attn_history))
    
    # Plot the evolution of attention weights
    plt.figure(figsize=(12, 6))
    plt.plot(x, smooth_micro, 'r-', linewidth=2, label='Micro (Immediate)')
    plt.plot(x, smooth_meso, 'b-', linewidth=2, label='Meso (Tactical)')
    plt.plot(x, smooth_super, 'g-', linewidth=2, label='Super (Strategic)')
    
    plt.title('Evolution of Attention Allocation During Episode', fontsize=16)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot comparative results
def plot_comparative_results(scripted, fractal, attention):
    """Plot comparative results for the three agent types."""
    # Metrics to plot
    metrics = ['survival_time', 'economic_score', 'combat_score', 'exploration_pct']
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Get values for each agent
        values = [scripted[metric], fractal[metric], attention[metric]]
        
        # Create bar chart
        bars = axes[i].bar(['Scripted', 'Fractal', 'Attention'], values, 
                          color=['gray', 'green', 'blue'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # Add title and labels
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional metrics in a separate figure
    other_metrics = ['crystal_collected', 'vespene_collected', 'enemies_killed', 'units_lost']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(other_metrics):
        # Get values for each agent
        values = [scripted[metric], fractal[metric], attention[metric]]
        
        # Create bar chart
        bars = axes[i].bar(['Scripted', 'Fractal', 'Attention'], values, 
                          color=['gray', 'green', 'blue'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # Add title and labels
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Entry point
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Quick test of a single agent
    print("Quick test of Fractal Attention Agent...")
    agent = FractalAttentionAgent()
    result = run_episode(agent, max_steps=100, seed=42, render_every=20, verbose=True)
    print(f"Survival time: {result['survival_time']}")
    print(f"Economic score: {result['economic_score']}")
    print(f"Combat score: {result['combat_score']}")
    
    # Plot attention evolution
    plot_attention_evolution(result['attention_history'])
    
    # Run full comparison
    print("\nRunning full agent comparison...")
    compare_agents(n_episodes=2, max_steps=200, render_final=True) 