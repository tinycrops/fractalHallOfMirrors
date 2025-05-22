#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from enum import Enum

from rts_environment import RTSEnvironment
from enhanced_fractal_attention_agent import EnhancedFractalAttentionAgent, StrategicFocus, TacticalObjective
from rts_visualization import RTSVisualization

# Scenario definitions
SCENARIOS = {
    "peaceful_start_and_ambush": {
        "description": "Initial peaceful phase followed by a sudden enemy attack wave",
        "steps": 300,
        "render_interval": 5
    },
    "resource_scarcity_expansion": {
        "description": "Start with limited resources, requiring expansion to new discoveries",
        "steps": 300,
        "render_interval": 5
    },
    "opportunity_and_threat": {
        "description": "Alternating opportunities and threats requiring adaptive strategy",
        "steps": 350,
        "render_interval": 5
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RTS Dynamic Scenario Orchestrator with Fractal Attention')
    
    parser.add_argument('--scenario', type=str, default="peaceful_start_and_ambush",
                        choices=list(SCENARIOS.keys()),
                        help='Scenario to run')
    
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps to run (overrides scenario default)')
    
    parser.add_argument('--render_interval', type=int, default=None,
                        help='Interval between renders (overrides scenario default)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization frames for creating GIFs')
    
    parser.add_argument('--create_gif', action='store_true',
                        help='Create a GIF of the simulation at the end')
    
    parser.add_argument('--pause_on_events', action='store_true',
                        help='Pause execution when new events occur')
    
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode, stepping through with user input')
    
    return parser.parse_args()

def run_scenario(scenario_name, steps=None, render_interval=None, seed=42, 
                save_path=None, create_gif=False, pause_on_events=False, interactive=False):
    """Run a specific scenario with the enhanced fractal attention agent"""
    
    # Use scenario defaults if not specified
    if steps is None:
        steps = SCENARIOS[scenario_name]["steps"]
    
    if render_interval is None:
        render_interval = SCENARIOS[scenario_name]["render_interval"]
    
    print(f"Running scenario: {scenario_name}")
    print(f"Description: {SCENARIOS[scenario_name]['description']}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    print()
    
    # Create environment with the specified scenario
    env = RTSEnvironment(seed=seed, scenario=scenario_name)
    
    # Create agent
    agent = EnhancedFractalAttentionAgent()
    
    # Create visualization
    viz = RTSVisualization(env, agent, save_path=save_path)
    
    # Metrics to track
    crystal_history = []
    unit_count_history = []
    enemy_count_history = []
    attention_history = []
    event_log = []
    
    # Set up interactive mode if requested
    if interactive:
        plt.ion()  # Turn on interactive mode for matplotlib
    
    # Main simulation loop
    for step in range(steps):
        # Get current state
        state = env.get_state()
        
        # Track metrics
        crystal_history.append(state['crystal_count'])
        unit_count_history.append(len(state['player_units']))
        enemy_count_history.append(len(state['enemy_units']))
        
        # Check for active events
        current_events = env.get_current_events()
        newly_active_events = [e for e in current_events if e['status'] == 'active' and e['trigger_time'] == step]
        
        if newly_active_events:
            for event in newly_active_events:
                event_log.append((step, event))
                print(f"Step {step}: Event triggered - {event['type']}")
            
            if pause_on_events:
                print("Press Enter to continue...")
                input()
        
        # Agent action
        attn_weights = agent.act(state, env)
        attention_history.append(attn_weights.copy() if hasattr(attn_weights, 'copy') else attn_weights)
        
        # Prepare goal information for visualization
        if hasattr(agent, 'current_focus') and hasattr(agent, 'current_super_goal') and hasattr(agent, 'current_meso_goal'):
            goals = {
                'strategic_focus': agent.current_focus.name if isinstance(agent.current_focus, Enum) else agent.current_focus,
                'super_goal': agent.current_super_goal,
                'tactical_objective': agent.current_tactical_objective.name if hasattr(agent, 'current_tactical_objective') and isinstance(agent.current_tactical_objective, Enum) else "Unknown",
                'meso_goal': agent.current_meso_goal
            }
        else:
            goals = None
        
        # Update visualization
        if step % render_interval == 0 or newly_active_events:
            viz.update(state, attn_weights, goals)
            
            if interactive:
                print(f"Step {step} - Press Enter to continue (or 'q' to quit)...")
                user_input = input()
                if user_input.lower() == 'q':
                    break
        
        # Environment step
        game_over = env.step()
        
        if game_over:
            print(f"Game over at step {step}! Nexus destroyed.")
            # One final render
            viz.update(state, attn_weights, goals)
            break
    
    # Final render
    state = env.get_state()
    viz.update(state, attn_weights, goals)
    
    print("\nSimulation completed!")
    print(f"Final state - Crystals: {state['crystal_count']}, Units: {len(state['player_units'])}, Enemies: {len(state['enemy_units'])}")
    
    # Create GIF if requested
    if create_gif and save_path:
        gif_filename = f"{scenario_name}_{seed}.gif"
        print(f"Creating GIF animation: {gif_filename}")
        viz.save_gif(filename=gif_filename)
    
    # Calculate attention statistics
    attention_stats = calculate_attention_stats(attention_history, event_log)
    
    # Plot and save metrics
    plot_metrics(steps, crystal_history, unit_count_history, enemy_count_history, 
                attention_history, event_log, scenario_name, seed)
    
    return {
        'crystal_history': crystal_history,
        'unit_count_history': unit_count_history,
        'enemy_count_history': enemy_count_history,
        'attention_history': attention_history,
        'event_log': event_log,
        'attention_stats': attention_stats,
        'steps_survived': min(steps, step + 1),
        'final_crystal': state['crystal_count'],
        'final_units': len(state['player_units']),
        'final_enemies': len(state['enemy_units']),
    }

def calculate_attention_stats(attention_history, event_log):
    """Calculate statistics about the attention allocation, especially around events"""
    stats = {
        'avg_attention': np.mean(attention_history, axis=0),
        'event_responses': []
    }
    
    # Analyze attention shifts around events
    for step, event in event_log:
        # Get attention before and after event
        before_window = max(0, step - 10)
        after_window = min(len(attention_history) - 1, step + 10)
        
        if before_window < step and after_window > step:
            before_attn = np.mean(attention_history[before_window:step], axis=0)
            after_attn = np.mean(attention_history[step:after_window+1], axis=0)
            
            # Calculate the change in attention
            attn_shift = after_attn - before_attn
            
            stats['event_responses'].append({
                'step': step,
                'event_type': event['type'],
                'before_attention': before_attn,
                'after_attention': after_attn,
                'attention_shift': attn_shift,
                'dominant_before': np.argmax(before_attn),
                'dominant_after': np.argmax(after_attn),
                'shift_magnitude': np.linalg.norm(attn_shift)
            })
    
    return stats

def plot_metrics(steps, crystal_history, unit_count_history, enemy_count_history, 
                attention_history, event_log, scenario_name, seed):
    """Plot performance metrics and attention analysis"""
    plt.figure(figsize=(15, 10))
    
    # Plot crystal count
    plt.subplot(3, 2, 1)
    plt.plot(crystal_history)
    plt.title('Crystal Count')
    plt.xlabel('Steps')
    plt.ylabel('Amount')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add event markers
    for step, event in event_log:
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
    
    # Plot unit counts
    plt.subplot(3, 2, 2)
    plt.plot(unit_count_history, label='Player Units')
    plt.plot(enemy_count_history, label='Enemy Units')
    plt.title('Unit Counts')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add event markers
    for step, event in event_log:
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
    
    # Plot attention weights over time
    plt.subplot(3, 2, (3, 4))
    attention_history = np.array(attention_history)
    
    # Create a time series line plot
    plt.plot(attention_history[:, 0], 'r-', label='Micro')
    plt.plot(attention_history[:, 1], 'g-', label='Meso')
    plt.plot(attention_history[:, 2], 'b-', label='Super')
    
    plt.title('Attention Weights Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add event markers with annotations
    for step, event in event_log:
        plt.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        plt.text(step, 0.9, event['type'], rotation=90, fontsize=8)
    
    # Plot attention distribution statistics
    plt.subplot(3, 2, 5)
    avg_attention = np.mean(attention_history, axis=0)
    plt.bar(['Micro', 'Meso', 'Super'], avg_attention, color=['red', 'green', 'blue'])
    plt.title('Average Attention Distribution')
    plt.ylabel('Attention Weight')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot attention shifts around events
    plt.subplot(3, 2, 6)
    event_types = list(set(event['type'] for _, event in event_log))
    event_type_to_idx = {event_type: i for i, event_type in enumerate(event_types)}
    
    # Calculate average shift for each event type
    avg_shifts = np.zeros((len(event_types), 3))  # 3 for micro, meso, super
    event_counts = np.zeros(len(event_types))
    
    attention_stats = calculate_attention_stats(attention_history, event_log)
    for response in attention_stats['event_responses']:
        event_idx = event_type_to_idx[response['event_type']]
        avg_shifts[event_idx] += response['attention_shift']
        event_counts[event_idx] += 1
    
    # Normalize by event count
    for i in range(len(event_types)):
        if event_counts[i] > 0:
            avg_shifts[i] /= event_counts[i]
    
    # Plot as a grouped bar chart
    bar_width = 0.25
    indices = np.arange(len(event_types))
    
    plt.bar(indices - bar_width, avg_shifts[:, 0], bar_width, label='Micro Shift', color='red')
    plt.bar(indices, avg_shifts[:, 1], bar_width, label='Meso Shift', color='green')
    plt.bar(indices + bar_width, avg_shifts[:, 2], bar_width, label='Super Shift', color='blue')
    
    plt.xlabel('Event Type')
    plt.ylabel('Attention Shift')
    plt.title('Attention Shifts by Event Type')
    plt.xticks(indices, event_types, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(f"Scenario: {scenario_name} (Seed: {seed})", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    plt.savefig(f"metrics_{scenario_name}_{seed}.png")
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run the specified scenario
    results = run_scenario(
        scenario_name=args.scenario,
        steps=args.steps,
        render_interval=args.render_interval,
        seed=args.seed,
        save_path=args.save_path,
        create_gif=args.create_gif,
        pause_on_events=args.pause_on_events,
        interactive=args.interactive
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Steps Survived: {results['steps_survived']}")
    print(f"Final Crystal Count: {results['final_crystal']}")
    print(f"Final Unit Count: {results['final_units']}")
    print(f"Final Enemy Count: {results['final_enemies']}")
    
    # Print attention statistics
    print("\nAttention Statistics:")
    avg_attention = results['attention_stats']['avg_attention']
    print(f"Average Attention: Micro={avg_attention[0]:.2f}, Meso={avg_attention[1]:.2f}, Super={avg_attention[2]:.2f}")
    
    if results['attention_stats']['event_responses']:
        print("\nAttention Responses to Events:")
        for response in results['attention_stats']['event_responses']:
            print(f"Step {response['step']} - {response['event_type']}:")
            print(f"  Before: {response['before_attention']}")
            print(f"  After:  {response['after_attention']}")
            print(f"  Shift:  {response['attention_shift']}")
            print(f"  Dominant level changed: {response['dominant_before'] != response['dominant_after']}")
            print(f"  Shift magnitude: {response['shift_magnitude']:.4f}")
            print()
    
    # Pause to keep the final visualization open
    if args.interactive:
        print("Press Enter to exit...")
        input() 