#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import time
from collections import deque
import sys

# Check if the required log files exist
flat_log_path = 'flat_agent_log.npy'
fractal_log_path = 'corrected_fractal_agent_log.npy'

if not os.path.exists(flat_log_path) or not os.path.exists(fractal_log_path):
    print("Error: Required log files not found!")
    print(f"Looking for: {flat_log_path} and {fractal_log_path}")
    print("Please run both agents before comparing.")
    sys.exit(1)

# Load log data
flat_log = np.load(flat_log_path)
fractal_log = np.load(fractal_log_path)

def compare_learning_curves():
    """Compare the learning curves of both agents with primitive action counts"""
    print("\n===== COMPARING LEARNING CURVES (PRIMITIVE ACTIONS) =====")
    
    # Determine episode limit for fair comparison
    min_episodes = min(len(flat_log), len(fractal_log))
    print(f"Comparing first {min_episodes} episodes from both agents")
    
    # Calculate statistics
    flat_min = np.min(flat_log[:min_episodes])
    flat_final = flat_log[min_episodes-1]
    flat_mean = np.mean(flat_log[:min_episodes])
    
    fractal_min = np.min(fractal_log[:min_episodes])
    fractal_final = fractal_log[min_episodes-1]
    fractal_mean = np.mean(fractal_log[:min_episodes])
    
    # Performance metrics
    print("\nPerformance Metrics:")
    print(f"{'Metric':<25} {'Flat Agent':<15} {'Fractal Agent':<15} {'Improvement':<15}")
    print("-" * 75)
    
    min_improvement = (flat_min - fractal_min) / flat_min * 100 if flat_min > 0 else 0
    final_improvement = (flat_final - fractal_final) / flat_final * 100 if flat_final > 0 else 0
    mean_improvement = (flat_mean - fractal_mean) / flat_mean * 100 if flat_mean > 0 else 0
    
    print(f"{'Best performance (steps)':<25} {flat_min:<15.2f} {fractal_min:<15.2f} {min_improvement:>14.2f}%")
    print(f"{'Final performance (steps)':<25} {flat_final:<15.2f} {fractal_final:<15.2f} {final_improvement:>14.2f}%")
    print(f"{'Mean performance (steps)':<25} {flat_mean:<15.2f} {fractal_mean:<15.2f} {mean_improvement:>14.2f}%")
    
    # Additional statistics
    optimal_path_length = 38  # From BFS analysis
    
    # Calculate how close each agent gets to optimal path
    flat_optimal_ratio = flat_min / optimal_path_length
    fractal_optimal_ratio = fractal_min / optimal_path_length
    
    print(f"\nBest performance compared to optimal ({optimal_path_length} steps):")
    print(f"Flat Agent: {flat_optimal_ratio:.2f}x optimal")
    print(f"Fractal Agent: {fractal_optimal_ratio:.2f}x optimal")
    
    # Generate comparison plots
    plt.figure(figsize=(15, 12))
    
    # Raw data plot with limited range
    plt.subplot(3, 1, 1)
    plt.plot(flat_log[:min_episodes], 'b-', alpha=0.3, label='Flat Agent')
    plt.plot(fractal_log[:min_episodes], 'r-', alpha=0.3, label='Fractal Agent')
    
    # Add rolling averages
    window = 30
    if min_episodes >= window:
        flat_avg = np.convolve(flat_log[:min_episodes], np.ones(window)/window, mode='valid')
        fractal_avg = np.convolve(fractal_log[:min_episodes], np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, min_episodes), flat_avg, 'b-', linewidth=2, label='Flat Agent (30-ep avg)')
        plt.plot(range(window-1, min_episodes), fractal_avg, 'r-', linewidth=2, label='Fractal Agent (30-ep avg)')
    
    plt.title('Learning Curves: Primitive Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot for better comparison of improvements
    plt.subplot(3, 1, 2)
    plt.semilogy(flat_log[:min_episodes], 'b-', alpha=0.3, label='Flat Agent')
    plt.semilogy(fractal_log[:min_episodes], 'r-', alpha=0.3, label='Fractal Agent')
    
    # Add rolling averages on log scale
    if min_episodes >= window:
        plt.semilogy(range(window-1, min_episodes), flat_avg, 'b-', linewidth=2, label='Flat Agent (30-ep avg)')
        plt.semilogy(range(window-1, min_episodes), fractal_avg, 'r-', linewidth=2, label='Fractal Agent (30-ep avg)')
    
    # Add horizontal line for optimal path length
    plt.axhline(y=optimal_path_length, color='g', linestyle='--', label=f'Optimal Path ({optimal_path_length} steps)')
    
    plt.title('Learning Curves (Log Scale)')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Grouped bar chart comparing key metrics
    plt.subplot(3, 1, 3)
    metrics = ['Best', 'Final', 'Mean']
    flat_values = [flat_min, flat_final, flat_mean]
    fractal_values = [fractal_min, fractal_final, fractal_mean]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, flat_values, width, label='Flat Agent')
    plt.bar(x + width/2, fractal_values, width, label='Fractal Agent')
    
    plt.axhline(y=optimal_path_length, color='g', linestyle='--', label=f'Optimal Path ({optimal_path_length} steps)')
    
    plt.title('Performance Metrics Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Steps to Goal')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fair_comparison.png', dpi=300)
    plt.show()

def analyze_sample_efficiency():
    """Analyze the sample efficiency of both approaches"""
    print("\n===== ANALYZING SAMPLE EFFICIENCY =====")
    
    # Define thresholds for measuring learning speed
    thresholds = [200, 150, 100, 75, 50]
    
    print("\nEpisodes needed to reach performance thresholds:")
    print(f"{'Threshold (steps)':<20} {'Flat Agent':<15} {'Fractal Agent':<15} {'Speedup Factor':<15}")
    print("-" * 70)
    
    for threshold in thresholds:
        # Find first episode where performance is consistently below threshold
        # (using a window of 5 episodes to avoid flukes)
        window_size = 5
        
        flat_ep = None
        for i in range(len(flat_log) - window_size + 1):
            if np.mean(flat_log[i:i+window_size]) < threshold:
                flat_ep = i
                break
                
        fractal_ep = None
        for i in range(len(fractal_log) - window_size + 1):
            if np.mean(fractal_log[i:i+window_size]) < threshold:
                fractal_ep = i
                break
        
        if flat_ep is not None and fractal_ep is not None:
            speedup = flat_ep / fractal_ep if fractal_ep > 0 else float('inf')
            print(f"{f'< {threshold} steps':<20} {flat_ep:<15} {fractal_ep:<15} {speedup:<15.2f}x")
        else:
            if flat_ep is None and fractal_ep is None:
                print(f"{f'< {threshold} steps':<20} {'Never':<15} {'Never':<15} {'-':<15}")
            elif flat_ep is None:
                print(f"{f'< {threshold} steps':<20} {'Never':<15} {fractal_ep:<15} {'-':<15}")
            else:
                print(f"{f'< {threshold} steps':<20} {flat_ep:<15} {'Never':<15} {'-':<15}")
    
    # Plot episodes to threshold
    plt.figure(figsize=(10, 6))
    
    flat_thresholds = []
    fractal_thresholds = []
    valid_thresholds = []
    
    for threshold in thresholds:
        flat_ep = None
        for i in range(len(flat_log) - 5 + 1):
            if np.mean(flat_log[i:i+5]) < threshold:
                flat_ep = i
                break
                
        fractal_ep = None
        for i in range(len(fractal_log) - 5 + 1):
            if np.mean(fractal_log[i:i+5]) < threshold:
                fractal_ep = i
                break
        
        if flat_ep is not None and fractal_ep is not None:
            flat_thresholds.append(flat_ep)
            fractal_thresholds.append(fractal_ep)
            valid_thresholds.append(threshold)
    
    if valid_thresholds:
        x = np.arange(len(valid_thresholds))
        width = 0.35
        
        plt.bar(x - width/2, flat_thresholds, width, label='Flat Agent')
        plt.bar(x + width/2, fractal_thresholds, width, label='Fractal Agent')
        
        plt.title('Sample Efficiency: Episodes to Reach Performance Threshold')
        plt.xlabel('Performance Threshold (steps)')
        plt.ylabel('Episodes Required')
        plt.xticks(x, valid_thresholds)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_efficiency.png', dpi=300)
        plt.show()
    else:
        print("Not enough valid thresholds to plot sample efficiency")

def segment_analysis():
    """Analyze performance in different segments of training"""
    print("\n===== SEGMENT ANALYSIS =====")
    
    # Define segments
    segments = [0, 50, 100, 200, 300, 400, 500, 600]
    segment_labels = [f"{segments[i]}-{segments[i+1]}" for i in range(len(segments)-1)]
    
    # Ensure we don't go beyond available data
    max_ep = min(len(flat_log), len(fractal_log))
    valid_segments = [seg for seg in segments if seg <= max_ep]
    
    if len(valid_segments) < 2:
        print("Not enough episodes for segment analysis")
        return
    
    # Recalculate segments if needed
    if valid_segments[-1] < segments[-1]:
        segments = valid_segments
        segment_labels = [f"{segments[i]}-{segments[i+1]}" for i in range(len(segments)-1)]
    
    print("\nAverage steps per segment:")
    print(f"{'Segment':<15} {'Flat Agent':<15} {'Fractal Agent':<15} {'Improvement':<15}")
    print("-" * 65)
    
    flat_segment_avgs = []
    fractal_segment_avgs = []
    
    for i in range(len(segments)-1):
        start, end = segments[i], min(segments[i+1], max_ep)
        
        flat_avg = np.mean(flat_log[start:end])
        fractal_avg = np.mean(fractal_log[start:end])
        
        improvement = (flat_avg - fractal_avg) / flat_avg * 100 if flat_avg > 0 else 0
        
        print(f"{segment_labels[i]:<15} {flat_avg:<15.2f} {fractal_avg:<15.2f} {improvement:>14.2f}%")
        
        flat_segment_avgs.append(flat_avg)
        fractal_segment_avgs.append(fractal_avg)
    
    # Plot segment analysis
    plt.figure(figsize=(12, 10))
    
    # Bar chart
    plt.subplot(2, 1, 1)
    x = np.arange(len(segment_labels))
    width = 0.35
    
    plt.bar(x - width/2, flat_segment_avgs, width, label='Flat Agent')
    plt.bar(x + width/2, fractal_segment_avgs, width, label='Fractal Agent')
    
    plt.title('Average Steps per Segment')
    plt.xlabel('Episode Segment')
    plt.ylabel('Average Steps to Goal')
    plt.xticks(x, segment_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Line chart for improvement percentage
    plt.subplot(2, 1, 2)
    improvements = []
    
    for i in range(len(flat_segment_avgs)):
        imp = (flat_segment_avgs[i] - fractal_segment_avgs[i]) / flat_segment_avgs[i] * 100 if flat_segment_avgs[i] > 0 else 0
        improvements.append(imp)
    
    plt.plot(x, improvements, 'g-o')
    
    plt.title('Improvement Percentage by Segment (Fractal vs Flat)')
    plt.xlabel('Episode Segment')
    plt.ylabel('Improvement (%)')
    plt.xticks(x, segment_labels)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('segment_analysis.png', dpi=300)
    plt.show()

def distribution_analysis():
    """Analyze the distribution of steps across episodes"""
    print("\n===== DISTRIBUTION ANALYSIS =====")
    
    # Create step frequency histograms
    plt.figure(figsize=(15, 10))
    
    # Define bins based on the range of data
    max_steps = max(np.max(flat_log), np.max(fractal_log))
    bins = np.linspace(0, max_steps, 30)
    
    plt.subplot(2, 1, 1)
    plt.hist(flat_log, bins=bins, alpha=0.5, label='Flat Agent')
    plt.hist(fractal_log, bins=bins, alpha=0.5, label='Fractal Agent')
    plt.title('Distribution of Steps to Goal')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Focus on the lower range for better visibility
    plt.subplot(2, 1, 2)
    
    lower_limit = min(np.min(flat_log), np.min(fractal_log))
    upper_limit = min(200, max_steps)  # Cap at 200 for better visibility
    
    bins_detail = np.linspace(lower_limit, upper_limit, 30)
    
    plt.hist(flat_log, bins=bins_detail, alpha=0.5, label='Flat Agent', range=(lower_limit, upper_limit))
    plt.hist(fractal_log, bins=bins_detail, alpha=0.5, label='Fractal Agent', range=(lower_limit, upper_limit))
    plt.title(f'Distribution of Steps to Goal (Limited to {upper_limit} steps)')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300)
    plt.show()
    
    # Print additional distribution statistics
    print("\nDistribution Statistics:")
    print(f"{'Statistic':<15} {'Flat Agent':<15} {'Fractal Agent':<15}")
    print("-" * 50)
    
    # Episode counts by step ranges
    step_ranges = [(0, 50), (51, 100), (101, 200), (201, 500), (501, float('inf'))]
    
    for start, end in step_ranges:
        range_label = f"{start}-{end}" if end != float('inf') else f"{start}+"
        
        flat_count = np.sum((flat_log >= start) & (flat_log <= end))
        flat_pct = flat_count / len(flat_log) * 100
        
        fractal_count = np.sum((fractal_log >= start) & (fractal_log <= end))
        fractal_pct = fractal_count / len(fractal_log) * 100
        
        print(f"{range_label + ' steps':<15} {f'{flat_count} ({flat_pct:.1f}%)':<15} {f'{fractal_count} ({fractal_pct:.1f}%)':<15}")

def main():
    print("===== FAIR COMPARISON OF FLAT VS CORRECTED FRACTAL AGENT =====")
    print(f"Flat agent log: {len(flat_log)} episodes")
    print(f"Fractal agent log: {len(fractal_log)} episodes")
    
    compare_learning_curves()
    analyze_sample_efficiency()
    segment_analysis()
    distribution_analysis()
    
    print("\nFair comparison analysis complete. Results saved to image files.")

if __name__ == "__main__":
    main() 