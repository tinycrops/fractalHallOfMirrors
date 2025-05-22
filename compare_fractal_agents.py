#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time

# Import the agents
import flat_agent_baseline as flat
import corrected_fractal_agent as fractal
import fractal_attention_agent as attn

def compare_agents(n_runs=5):
    """
    Compare the performance of flat, fractal, and fractal-attention agents
    over multiple runs.
    """
    print("Running comparison of agent architectures...")
    
    # Arrays to store results
    flat_logs = []
    fractal_logs = []
    attn_logs = []
    
    flat_times = []
    fractal_times = []
    attn_times = []
    
    # Run multiple trials
    for i in trange(n_runs, desc="Comparison runs"):
        # Reset the random seed for each run to ensure fairness
        np.random.seed(i)
        
        # Train the agents
        print(f"\nRun {i+1}/{n_runs}:")
        
        print("Training Flat Agent...")
        flat_log, flat_time = flat.train_flat_agent()
        flat_logs.append(flat_log)
        flat_times.append(flat_time)
        
        print("Training Fractal Agent...")
        fractal_log, fractal_time = fractal.train_agent()
        fractal_logs.append(fractal_log)
        fractal_times.append(fractal_time)
        
        print("Training Fractal Agent with Attention...")
        attn_log, attn_time, _ = attn.train_agent()
        attn_logs.append(attn_log)
        attn_times.append(attn_time)
    
    # Calculate average performance metrics
    avg_flat_steps = np.mean([np.mean(log[-100:]) for log in flat_logs])
    avg_fractal_steps = np.mean([np.mean(log[-100:]) for log in fractal_logs])
    avg_attn_steps = np.mean([np.mean(log[-100:]) for log in attn_logs])
    
    avg_flat_time = np.mean(flat_times)
    avg_fractal_time = np.mean(fractal_times)
    avg_attn_time = np.mean(attn_times)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Flat Agent: {avg_flat_steps:.2f} avg steps, {avg_flat_time:.2f}s training time")
    print(f"Fractal Agent: {avg_fractal_steps:.2f} avg steps, {avg_fractal_time:.2f}s training time")
    print(f"Fractal-Attention Agent: {avg_attn_steps:.2f} avg steps, {avg_attn_time:.2f}s training time")
    
    if avg_attn_steps < avg_fractal_steps:
        improvement = (avg_fractal_steps - avg_attn_steps) / avg_fractal_steps * 100
        print(f"\nAttention mechanism improved performance by {improvement:.1f}%")
    
    # Plot learning curves
    plot_comparison(flat_logs, fractal_logs, attn_logs)
    
    return {
        'flat': {'logs': flat_logs, 'times': flat_times},
        'fractal': {'logs': fractal_logs, 'times': fractal_times},
        'attention': {'logs': attn_logs, 'times': attn_times}
    }

def smooth(data, window=30):
    """Apply a moving average to smooth the data."""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def plot_comparison(flat_logs, fractal_logs, attn_logs):
    """Plot the learning curves for each agent type."""
    plt.figure(figsize=(15, 10))
    
    # Average across runs
    avg_flat = np.mean(flat_logs, axis=0)
    avg_fractal = np.mean(fractal_logs, axis=0)
    avg_attn = np.mean(attn_logs, axis=0)
    
    # Smooth the curves
    window = 30
    if len(avg_flat) >= window:
        smooth_flat = smooth(avg_flat, window)
        smooth_fractal = smooth(avg_fractal, window)
        smooth_attn = smooth(avg_attn, window)
        
        # Plot smoothed curves
        x = range(window-1, len(avg_flat))
        plt.plot(x, smooth_flat, 'b-', linewidth=2, label='Flat Agent')
        plt.plot(x, smooth_fractal, 'g-', linewidth=2, label='Fractal Agent')
        plt.plot(x, smooth_attn, 'r-', linewidth=2, label='Fractal-Attention Agent')
    else:
        # If not enough data points, plot raw data
        plt.plot(avg_flat, 'b-', alpha=0.7, label='Flat Agent')
        plt.plot(avg_fractal, 'g-', alpha=0.7, label='Fractal Agent')
        plt.plot(avg_attn, 'r-', alpha=0.7, label='Fractal-Attention Agent')
    
    # Plot individual runs with light colors
    for log in flat_logs:
        plt.plot(log, 'b-', alpha=0.1)
    for log in fractal_logs:
        plt.plot(log, 'g-', alpha=0.1)
    for log in attn_logs:
        plt.plot(log, 'r-', alpha=0.1)
    
    plt.title('Learning Curve Comparison: Steps to Goal vs Episode', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Steps to Goal', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add a log-scale version in a subplot
    plt.figure(figsize=(15, 10))
    plt.semilogy(avg_flat, 'b-', alpha=0.4, label='Flat Agent')
    plt.semilogy(avg_fractal, 'g-', alpha=0.4, label='Fractal Agent')
    plt.semilogy(avg_attn, 'r-', alpha=0.4, label='Fractal-Attention Agent')
    
    # Apply exponential moving average for smoother log plot
    alpha = 0.05
    ema_flat = [avg_flat[0]]
    ema_fractal = [avg_fractal[0]]
    ema_attn = [avg_attn[0]]
    
    for i in range(1, len(avg_flat)):
        ema_flat.append(alpha * avg_flat[i] + (1 - alpha) * ema_flat[-1])
        ema_fractal.append(alpha * avg_fractal[i] + (1 - alpha) * ema_fractal[-1])
        ema_attn.append(alpha * avg_attn[i] + (1 - alpha) * ema_attn[-1])
    
    plt.semilogy(ema_flat, 'b-', linewidth=3, label='Flat Agent (EMA)')
    plt.semilogy(ema_fractal, 'g-', linewidth=3, label='Fractal Agent (EMA)')
    plt.semilogy(ema_attn, 'r-', linewidth=3, label='Fractal-Attention Agent (EMA)')
    
    plt.title('Learning Curve Comparison (Log Scale)', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Steps to Goal (log scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # Run with a single comparison for demonstration
    results = compare_agents(n_runs=1)
    
    # Visualize the agents (one example of each)
    print("\nVisualizing Flat Agent behavior...")
    flat.visualize_flat_agent()
    
    print("\nVisualizing Fractal Agent behavior...")
    fractal.visualize_agent()
    
    print("\nVisualizing Fractal-Attention Agent behavior...")
    attn.visualize_agent()
    
    # Plot attention evolution
    print("\nVisualizing Attention Evolution...")
    attn.plot_attention_evolution(results['attention']['logs'][0])

if __name__ == "__main__":
    main() 