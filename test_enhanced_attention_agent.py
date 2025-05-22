#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from rts_environment import RTSEnvironment
from rts_fractal_attention_agent import FractalAttentionAgent
from enhanced_fractal_attention_agent import EnhancedFractalAttentionAgent

def run_simulation(agent_class, steps=300, render_interval=50, seed=42):
    """Run a simulation with the specified agent"""
    # Initialize environment and agent
    env = RTSEnvironment(seed=seed)
    agent = agent_class()
    
    # Metrics to track
    crystal_history = []
    unit_count_history = []
    enemy_count_history = []
    attention_history = []
    
    # Run simulation
    for step in range(steps):
        state = env.get_state()
        
        # Track metrics
        crystal_history.append(state['crystal_count'])
        unit_count_history.append(len(state['player_units']))
        enemy_count_history.append(len(state['enemy_units']))
        
        # Agent action
        attn_weights = agent.act(state, env)
        attention_history.append(attn_weights.copy() if hasattr(attn_weights, 'copy') else attn_weights)
        
        # Environment step
        game_over = env.step()
        
        # Render periodically
        if step % render_interval == 0:
            env.render()
            print(f"Step {step}:")
            print(f"  Crystal: {state['crystal_count']}")
            print(f"  Units: {len(state['player_units'])}")
            print(f"  Enemies: {len(state['enemy_units'])}")
            print(f"  Attention: {attn_weights}")
        
        if game_over:
            print("Game over!")
            break
    
    # Final render
    env.render()
    print("Final state:")
    state = env.get_state()
    print(f"  Crystal: {state['crystal_count']}")
    print(f"  Units: {len(state['player_units'])}")
    print(f"  Enemies: {len(state['enemy_units'])}")
    print(f"  Attention: {agent.attention_weights}")
    
    # Return metrics
    return {
        'crystal_history': crystal_history,
        'unit_count_history': unit_count_history,
        'enemy_count_history': enemy_count_history,
        'attention_history': attention_history,
        'steps_survived': min(steps, step + 1),
        'final_crystal': state['crystal_count'],
        'final_units': len(state['player_units']),
        'final_enemies': len(state['enemy_units']),
    }

def plot_metrics(metrics, title="Agent Performance"):
    """Plot the metrics from a simulation run"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    
    # Plot crystal count
    axes[0, 0].plot(metrics['crystal_history'])
    axes[0, 0].set_title('Crystal Count')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Amount')
    
    # Plot unit counts
    axes[0, 1].plot(metrics['unit_count_history'], label='Player Units')
    axes[0, 1].plot(metrics['enemy_count_history'], label='Enemy Units')
    axes[0, 1].set_title('Unit Counts')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Plot attention weights
    attention_history = np.array(metrics['attention_history'])
    if attention_history.ndim == 2 and attention_history.shape[1] == 3:
        axes[1, 0].stackplot(
            range(len(attention_history)),
            attention_history[:, 0],  # Micro
            attention_history[:, 1],  # Meso
            attention_history[:, 2],  # Super
            labels=['Micro', 'Meso', 'Super'],
            alpha=0.7
        )
        axes[1, 0].set_title('Attention Allocation')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].legend()
    
    # Summary text
    summary_text = (
        f"Steps Survived: {metrics['steps_survived']}\n"
        f"Final Crystal: {metrics['final_crystal']}\n"
        f"Final Units: {metrics['final_units']}\n"
        f"Final Enemies: {metrics['final_enemies']}"
    )
    axes[1, 1].text(0.5, 0.5, summary_text, 
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axes[1, 1].transAxes,
                  fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_agents(steps=300, render_interval=50, seed=42):
    """Compare the standard and enhanced agents"""
    print("Running standard FractalAttentionAgent...")
    standard_metrics = run_simulation(FractalAttentionAgent, steps, render_interval, seed)
    
    print("\nRunning EnhancedFractalAttentionAgent...")
    enhanced_metrics = run_simulation(EnhancedFractalAttentionAgent, steps, render_interval, seed)
    
    # Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Agent Comparison: Standard vs Enhanced')
    
    # Crystal comparison
    axes[0, 0].plot(standard_metrics['crystal_history'], label='Standard')
    axes[0, 0].plot(enhanced_metrics['crystal_history'], label='Enhanced')
    axes[0, 0].set_title('Crystal Count')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Amount')
    axes[0, 0].legend()
    
    # Unit count comparison
    axes[0, 1].plot(standard_metrics['unit_count_history'], label='Standard Units')
    axes[0, 1].plot(enhanced_metrics['unit_count_history'], label='Enhanced Units')
    axes[0, 1].set_title('Player Unit Count')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Enemy count comparison
    axes[1, 0].plot(standard_metrics['enemy_count_history'], label='Standard Enemies')
    axes[1, 0].plot(enhanced_metrics['enemy_count_history'], label='Enhanced Enemies')
    axes[1, 0].set_title('Enemy Unit Count')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Summary statistics
    standard_summary = (
        f"Standard Agent:\n"
        f"Steps Survived: {standard_metrics['steps_survived']}\n"
        f"Final Crystal: {standard_metrics['final_crystal']}\n"
        f"Final Units: {standard_metrics['final_units']}\n"
        f"Final Enemies: {standard_metrics['final_enemies']}\n"
    )
    
    enhanced_summary = (
        f"Enhanced Agent:\n"
        f"Steps Survived: {enhanced_metrics['steps_survived']}\n"
        f"Final Crystal: {enhanced_metrics['final_crystal']}\n"
        f"Final Units: {enhanced_metrics['final_units']}\n"
        f"Final Enemies: {enhanced_metrics['final_enemies']}\n"
    )
    
    axes[1, 1].text(0.5, 0.7, standard_summary, 
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axes[1, 1].transAxes,
                  fontsize=10)
    
    axes[1, 1].text(0.5, 0.3, enhanced_summary, 
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=axes[1, 1].transAxes,
                  fontsize=10)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return standard_metrics, enhanced_metrics

if __name__ == "__main__":
    # Test the enhanced agent on its own
    print("Testing EnhancedFractalAttentionAgent...")
    metrics = run_simulation(EnhancedFractalAttentionAgent, steps=300, render_interval=50)
    plot_metrics(metrics, title="Enhanced Fractal Attention Agent Performance")
    
    # Uncomment to compare agents
    print("\nComparing agents...")
    standard_metrics, enhanced_metrics = compare_agents(steps=300, render_interval=50) 