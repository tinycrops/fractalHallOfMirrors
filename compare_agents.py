#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import subprocess
from tqdm import tqdm

def run_agents():
    """Run both flat and fractal agents if results don't already exist"""
    
    # Check if we need to run the flat agent
    if not os.path.exists('flat_agent_log.npy'):
        print("Running flat agent training (this may take a while)...")
        subprocess.run(["python", "flat_agent_baseline.py"], check=True)
    
    # Check if we need to run the fractal agent
    if not os.path.exists('fractal_agent_log.npy'):
        print("Running fractal agent training (this may take a while)...")
        # First modify the fractal agent to save its log
        with open('fractal_agent_sandbox.py', 'r') as f:
            fractal_code = f.read()
        
        # Check if the save code is already there
        if "np.save('fractal_agent_log.npy'" not in fractal_code:
            # Add code to save the log
            save_code = "\n    # Save log data for comparison\n    np.save('fractal_agent_log.npy', np.array(log))\n"
            fractal_code = fractal_code.replace('    return ani\n\ndef plot_learning_curve', 
                                              '    return ani{}\n\ndef plot_learning_curve'.format(save_code))
            
            with open('fractal_agent_sandbox.py', 'w') as f:
                f.write(fractal_code)
        
        # Run the fractal agent
        subprocess.run(["python", "fractal_agent_sandbox.py"], check=True)

def compare_learning_curves():
    """Compare the learning curves of both agents"""
    
    # Load log data
    flat_log = np.load('flat_agent_log.npy')
    fractal_log = np.load('fractal_agent_log.npy')
    
    # Create the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw learning curves
    plt.subplot(2, 1, 1)
    plt.plot(flat_log, 'b-', alpha=0.3, label='Flat Agent (raw)')
    plt.plot(fractal_log, 'r-', alpha=0.3, label='Fractal Agent (raw)')
    
    # Calculate rolling averages
    window_size = 30
    if len(flat_log) >= window_size and len(fractal_log) >= window_size:
        flat_rolling = np.convolve(flat_log, np.ones(window_size)/window_size, mode='valid')
        fractal_rolling = np.convolve(fractal_log, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(range(window_size-1, len(flat_log)), flat_rolling, 'b-', linewidth=2, 
                 label=f'Flat Agent (rolling avg, window={window_size})')
        plt.plot(range(window_size-1, len(fractal_log)), fractal_rolling, 'r-', linewidth=2, 
                 label=f'Fractal Agent (rolling avg, window={window_size})')
    
    plt.title('Learning Curves: Flat vs Fractal Agent')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Log scale learning curves
    plt.subplot(2, 1, 2)
    plt.semilogy(flat_log, 'b-', alpha=0.3, label='Flat Agent (log scale)')
    plt.semilogy(fractal_log, 'r-', alpha=0.3, label='Fractal Agent (log scale)')
    
    # Calculate exponential moving averages
    alpha = 0.1  # Smoothing factor
    flat_ema = [flat_log[0]]
    fractal_ema = [fractal_log[0]]
    
    for i in range(1, len(flat_log)):
        flat_ema.append(alpha * flat_log[i] + (1 - alpha) * flat_ema[-1])
    
    for i in range(1, len(fractal_log)):
        fractal_ema.append(alpha * fractal_log[i] + (1 - alpha) * fractal_ema[-1])
    
    plt.semilogy(flat_ema, 'b-', linewidth=2, label='Flat Agent (EMA)')
    plt.semilogy(fractal_ema, 'r-', linewidth=2, label='Fractal Agent (EMA)')
    
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal (log scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curve_comparison.png', dpi=300)
    plt.show()

def calculate_metrics():
    """Calculate and display performance metrics for both agents"""
    
    # Load log data
    flat_log = np.load('flat_agent_log.npy')
    fractal_log = np.load('fractal_agent_log.npy')
    
    # Calculate metrics
    metrics = {
        'Flat Agent': {
            'Final Performance (steps)': flat_log[-1],
            'Best Performance (steps)': min(flat_log),
            'Average Performance (last 50 episodes)': np.mean(flat_log[-50:]),
            'Standard Deviation (last 50 episodes)': np.std(flat_log[-50:]),
            'Total Steps Taken': np.sum(flat_log),
            'Episodes to Reach Good Performance': None,
        },
        'Fractal Agent': {
            'Final Performance (steps)': fractal_log[-1],
            'Best Performance (steps)': min(fractal_log),
            'Average Performance (last 50 episodes)': np.mean(fractal_log[-50:]),
            'Standard Deviation (last 50 episodes)': np.std(fractal_log[-50:]),
            'Total Steps Taken': np.sum(fractal_log),
            'Episodes to Reach Good Performance': None,
        }
    }
    
    # Calculate episodes to reach good performance (defined as consistently under 100 steps)
    # We'll define "good" as a moving average of under 100 steps for 10 consecutive episodes
    window = 10
    threshold = 100
    
    for agent, log in [('Flat Agent', flat_log), ('Fractal Agent', fractal_log)]:
        for i in range(len(log) - window + 1):
            if np.mean(log[i:i+window]) < threshold:
                metrics[agent]['Episodes to Reach Good Performance'] = i
                break
    
    # Print the comparison table
    print("\n===== PERFORMANCE COMPARISON: FLAT vs FRACTAL AGENT =====\n")
    
    # Get all metric names
    metric_names = list(metrics['Flat Agent'].keys())
    
    # Calculate column widths
    metric_width = max(len(name) for name in metric_names) + 2
    value_width = 25
    
    # Print header
    header = f"{'Metric':<{metric_width}} | {'Flat Agent':<{value_width}} | {'Fractal Agent':<{value_width}} | {'Improvement':<{value_width}}"
    print(header)
    print("-" * len(header))
    
    # Print each metric
    for metric in metric_names:
        flat_value = metrics['Flat Agent'][metric]
        fractal_value = metrics['Fractal Agent'][metric]
        
        # Calculate improvement
        if flat_value is not None and fractal_value is not None:
            if metric in ['Total Steps Taken', 'Episodes to Reach Good Performance']:
                # For these metrics, lower is better
                if flat_value > 0:
                    improvement = f"{(flat_value - fractal_value) / flat_value * 100:.2f}% less"
                else:
                    improvement = "N/A"
            else:
                # For other metrics like final performance (steps to goal), lower is better
                if flat_value > 0:
                    improvement = f"{(flat_value - fractal_value) / flat_value * 100:.2f}% better"
                else:
                    improvement = "N/A"
        else:
            improvement = "N/A"
        
        # Format the values
        if flat_value is None:
            flat_str = "N/A"
        elif isinstance(flat_value, (int, np.integer)):
            flat_str = str(flat_value)
        else:
            flat_str = f"{flat_value:.2f}"
            
        if fractal_value is None:
            fractal_str = "N/A"
        elif isinstance(fractal_value, (int, np.integer)):
            fractal_str = str(fractal_value)
        else:
            fractal_str = f"{fractal_value:.2f}"
        
        print(f"{metric:<{metric_width}} | {flat_str:<{value_width}} | {fractal_str:<{value_width}} | {improvement:<{value_width}}")
    
    print("\n=== CONCLUSION ===\n")
    # Draw conclusion based on the data
    if metrics['Fractal Agent']['Total Steps Taken'] < metrics['Flat Agent']['Total Steps Taken']:
        print("The Fractal Hierarchical agent demonstrates better sample efficiency, requiring fewer total steps to learn.")
    
    if (metrics['Fractal Agent']['Episodes to Reach Good Performance'] is not None and 
        metrics['Flat Agent']['Episodes to Reach Good Performance'] is not None and
        metrics['Fractal Agent']['Episodes to Reach Good Performance'] < metrics['Flat Agent']['Episodes to Reach Good Performance']):
        print("The Fractal Hierarchical agent learns faster, reaching good performance in fewer episodes.")
    
    if metrics['Fractal Agent']['Average Performance (last 50 episodes)'] < metrics['Flat Agent']['Average Performance (last 50 episodes)']:
        print("The Fractal Hierarchical agent achieves better final performance (shorter paths to goal).")
    
    if metrics['Fractal Agent']['Standard Deviation (last 50 episodes)'] < metrics['Flat Agent']['Standard Deviation (last 50 episodes)']:
        print("The Fractal Hierarchical agent's performance is more stable (lower variance).")
        
    # Save the metrics to a text file
    with open('agent_comparison_metrics.txt', 'w') as f:
        f.write("===== PERFORMANCE COMPARISON: FLAT vs FRACTAL AGENT =====\n\n")
        for metric in metric_names:
            f.write(f"{metric}:\n")
            f.write(f"  Flat Agent: {metrics['Flat Agent'][metric]}\n")
            f.write(f"  Fractal Agent: {metrics['Fractal Agent'][metric]}\n\n")
            
def main():
    print("Comparing Flat vs Fractal Hierarchical Q-Learning Agents")
    run_agents()
    calculate_metrics()
    compare_learning_curves()
    print("\nComparison complete! Results saved to 'learning_curve_comparison.png' and 'agent_comparison_metrics.txt'")

if __name__ == "__main__":
    main() 