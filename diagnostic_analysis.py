#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
import random
import time

# Load log data if it exists
fractal_log = np.load('fractal_agent_log.npy') if os.path.exists('fractal_agent_log.npy') else None
flat_log = np.load('flat_agent_log.npy') if os.path.exists('flat_agent_log.npy') else None

def analyze_step_counting():
    """Analyze how steps are counted in both implementations"""
    print("\n===== ANALYSIS OF STEP COUNTING METHODOLOGY =====")
    
    # Check fractal agent implementation
    with open('fractal_agent_sandbox.py', 'r') as f:
        fractal_code = f.read()
    
    # Check flat agent implementation
    with open('flat_agent_baseline.py', 'r') as f:
        flat_code = f.read()
    
    # Look for how steps are counted in both implementations
    print("\nFractal Agent Step Counting:")
    step_counting_lines = []
    for i, line in enumerate(fractal_code.split('\n')):
        if 'steps' in line and ('+=' in line or '=' in line):
            step_counting_lines.append(f"Line {i+1}: {line.strip()}")
    
    for line in step_counting_lines:
        print(f"  {line}")
    
    print("\nFlat Agent Step Counting:")
    step_counting_lines = []
    for i, line in enumerate(flat_code.split('\n')):
        if 'steps' in line and ('+=' in line or '=' in line):
            step_counting_lines.append(f"Line {i+1}: {line.strip()}")
    
    for line in step_counting_lines:
        print(f"  {line}")
    
    # Analyze how the logs are populated
    print("\nHow logs are populated in Fractal Agent:")
    log_lines = []
    for i, line in enumerate(fractal_code.split('\n')):
        if 'log.append' in line:
            log_lines.append(f"Line {i+1}: {line.strip()}")
    
    for line in log_lines:
        print(f"  {line}")
    
    print("\nHow logs are populated in Flat Agent:")
    log_lines = []
    for i, line in enumerate(flat_code.split('\n')):
        if 'log.append' in line:
            log_lines.append(f"Line {i+1}: {line.strip()}")
    
    for line in log_lines:
        print(f"  {line}")

def analyze_reward_structure():
    """Analyze the reward structure of both implementations"""
    print("\n===== ANALYSIS OF REWARD STRUCTURE =====")
    
    # Check fractal agent implementation
    with open('fractal_agent_sandbox.py', 'r') as f:
        fractal_code = f.read()
    
    # Check flat agent implementation
    with open('flat_agent_baseline.py', 'r') as f:
        flat_code = f.read()
    
    # Extract reward-related code from fractal agent
    print("\nFractal Agent Reward Structure:")
    reward_lines = []
    in_reward_section = False
    for i, line in enumerate(fractal_code.split('\n')):
        if 'r_mac =' in line or 'r_super =' in line or 'rwd =' in line:
            reward_lines.append(f"Line {i+1}: {line.strip()}")
        elif 'Determine' in line and 'reward' in line:
            reward_lines.append(f"Line {i+1}: {line.strip()}")
            in_reward_section = True
        elif in_reward_section and ('if' in line or 'elif' in line or 'else' in line) and ':' in line:
            reward_lines.append(f"Line {i+1}: {line.strip()}")
        elif in_reward_section and 'r_' in line and '=' in line:
            reward_lines.append(f"Line {i+1}: {line.strip()}")
        elif in_reward_section and len(line.strip()) == 0:
            in_reward_section = False
    
    for line in reward_lines:
        print(f"  {line}")
    
    # Extract reward-related code from flat agent
    print("\nFlat Agent Reward Structure:")
    reward_lines = []
    for i, line in enumerate(flat_code.split('\n')):
        if 'rwd =' in line:
            reward_lines.append(f"Line {i+1}: {line.strip()}")
    
    for line in reward_lines:
        print(f"  {line}")

def analyze_log_statistics():
    """Analyze statistics of the logged data"""
    print("\n===== STATISTICAL ANALYSIS OF LOGS =====")
    
    if fractal_log is not None and flat_log is not None:
        print("\nFirst 10 episode steps for Fractal Agent:")
        print(fractal_log[:10])
        
        print("\nFirst 10 episode steps for Flat Agent:")
        print(flat_log[:10])
        
        print("\nLast 10 episode steps for Fractal Agent:")
        print(fractal_log[-10:])
        
        print("\nLast 10 episode steps for Flat Agent:")
        print(flat_log[-10:])
        
        print("\nStatistics for Fractal Agent Log:")
        print(f"  Mean: {np.mean(fractal_log):.2f}")
        print(f"  Median: {np.median(fractal_log):.2f}")
        print(f"  Min: {np.min(fractal_log)}")
        print(f"  Max: {np.max(fractal_log)}")
        print(f"  Standard Deviation: {np.std(fractal_log):.2f}")
        print(f"  Episodes with 1 step: {np.sum(fractal_log == 1)} ({np.sum(fractal_log == 1)/len(fractal_log)*100:.2f}%)")
        
        print("\nStatistics for Flat Agent Log:")
        print(f"  Mean: {np.mean(flat_log):.2f}")
        print(f"  Median: {np.median(flat_log):.2f}")
        print(f"  Min: {np.min(flat_log)}")
        print(f"  Max: {np.max(flat_log)}")
        print(f"  Standard Deviation: {np.std(flat_log):.2f}")
        print(f"  Episodes with ≤ 50 steps: {np.sum(flat_log <= 50)} ({np.sum(flat_log <= 50)/len(flat_log)*100:.2f}%)")
    else:
        print("Log data not available. Run the agents first.")

def analyze_agent_structure():
    """Analyze the structural differences between agents"""
    print("\n===== STRUCTURAL ANALYSIS OF AGENTS =====")
    
    # Calculate the state-action space sizes
    SIZE = 20  # Grid size
    block_micro = 5
    block_macro = 10
    
    micro_states = SIZE * SIZE
    macro_states = (SIZE // block_micro) ** 2
    super_states = (SIZE // block_macro) ** 2
    num_actions = 4
    
    print(f"\nState-Action Space Sizes:")
    print(f"  Flat Agent: {micro_states} states × {num_actions} actions = {micro_states * num_actions} Q-values")
    print(f"  Fractal Agent:")
    print(f"    Micro Level: {micro_states} states × {num_actions} actions = {micro_states * num_actions} Q-values")
    print(f"    Macro Level: {macro_states} states × {num_actions} actions = {macro_states * num_actions} Q-values")
    print(f"    Super Level: {super_states} states × {num_actions} actions = {super_states * num_actions} Q-values")
    print(f"    Total Q-values: {micro_states * num_actions + macro_states * num_actions + super_states * num_actions}")
    
    # Compare action selection at different levels
    print("\nAction Selection Logic:")
    
    # Check fractal agent implementation
    with open('fractal_agent_sandbox.py', 'r') as f:
        fractal_code = f.read()
    
    # Check flat agent implementation
    with open('flat_agent_baseline.py', 'r') as f:
        flat_code = f.read()
    
    # Extract action selection code
    print("\nFractal Agent Action Selection:")
    for i, line in enumerate(fractal_code.split('\n')):
        if 'a_super =' in line or 'a_mac =' in line or 'a_mic =' in line:
            print(f"  Line {i+1}: {line.strip()}")
    
    print("\nFlat Agent Action Selection:")
    for i, line in enumerate(flat_code.split('\n')):
        if 'a =' in line and 'choose' in line:
            print(f"  Line {i+1}: {line.strip()}")

def plot_detailed_comparison():
    """Create more detailed comparison plots"""
    if fractal_log is not None and flat_log is not None:
        # Plot first 100 episodes in detail
        plt.figure(figsize=(15, 10))
        
        # Plot raw data for first 100 episodes
        plt.subplot(2, 1, 1)
        plt.plot(range(min(100, len(flat_log))), flat_log[:100], 'b-', marker='o', markersize=4, alpha=0.7, label='Flat Agent')
        plt.plot(range(min(100, len(fractal_log))), fractal_log[:100], 'r-', marker='o', markersize=4, alpha=0.7, label='Fractal Agent')
        plt.title('First 100 Episodes: Steps to Goal')
        plt.xlabel('Episode')
        plt.ylabel('Steps to Goal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot episode segments for better pattern analysis
        plt.subplot(2, 1, 2)
        segments = [0, 20, 50, 100, 200, 300, 400, 500, 600] if len(flat_log) >= 500 else [0, 20, 50, 100, 150, 200]
        segment_labels = [f"{segments[i]}-{segments[i+1]}" for i in range(len(segments)-1)]
        
        flat_segment_avgs = []
        fractal_segment_avgs = []
        
        for i in range(len(segments)-1):
            start, end = segments[i], segments[i+1]
            if start < len(flat_log) and start < len(fractal_log):
                flat_end = min(end, len(flat_log))
                fractal_end = min(end, len(fractal_log))
                flat_segment_avgs.append(np.mean(flat_log[start:flat_end]))
                fractal_segment_avgs.append(np.mean(fractal_log[start:fractal_end]))
        
        x = np.arange(len(segment_labels))
        width = 0.35
        
        plt.bar(x - width/2, flat_segment_avgs, width, label='Flat Agent')
        plt.bar(x + width/2, fractal_segment_avgs, width, label='Fractal Agent')
        
        plt.xlabel('Episode Segments')
        plt.ylabel('Average Steps to Goal')
        plt.title('Performance by Training Segment')
        plt.xticks(x, segment_labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('detailed_comparison.png', dpi=300)
        plt.show()
        
        # Plot performance improvement over time (percentage reduction in steps)
        if len(flat_log) > 1 and len(fractal_log) > 1:
            plt.figure(figsize=(15, 5))
            
            # Calculate smoothed improvement for flat agent
            window = 30
            flat_smoothed = np.convolve(flat_log, np.ones(window)/window, mode='valid')
            initial_flat = flat_smoothed[0]
            flat_improvement = [(initial_flat - val) / initial_flat * 100 for val in flat_smoothed]
            
            # Calculate smoothed improvement for fractal agent
            fractal_smoothed = np.convolve(fractal_log, np.ones(window)/window, mode='valid')
            initial_fractal = fractal_smoothed[0]
            fractal_improvement = [(initial_fractal - val) / initial_fractal * 100 for val in fractal_smoothed]
            
            plt.plot(range(window-1, len(flat_log)), flat_improvement, 'b-', label='Flat Agent')
            plt.plot(range(window-1, len(fractal_log)), fractal_improvement, 'r-', label='Fractal Agent')
            
            plt.xlabel('Episode')
            plt.ylabel('Improvement (% reduction in steps)')
            plt.title('Performance Improvement Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('improvement_comparison.png', dpi=300)
            plt.show()
    else:
        print("Log data not available for detailed plotting. Run the agents first.")

def run_simulated_paths():
    """Run a simplified simulation to check the minimum possible path length"""
    print("\n===== SIMULATED OPTIMAL PATH ANALYSIS =====")
    
    SIZE = 20
    GOAL = (SIZE-1, SIZE-1)
    START = (0, 0)
    
    # Create obstacles
    def create_obstacles():
        obstacles = set()
        
        # Add some walls
        for i in range(5, 15):
            obstacles.add((i, 5))    # Horizontal wall
            obstacles.add((5, i))    # Vertical wall
        
        # Add a maze-like structure in the bottom right
        for i in range(12, 18):
            obstacles.add((i, 12))   # Horizontal wall
        for i in range(12, 18):
            obstacles.add((12, i))   # Vertical wall
        
        # Make sure the goal is accessible
        if GOAL in obstacles:
            obstacles.remove(GOAL)
        
        # Make sure the start is accessible
        if START in obstacles:
            obstacles.remove(START)
        
        return obstacles

    OBSTACLES = create_obstacles()
    
    # Create grid for visualization
    grid = np.zeros((SIZE, SIZE))
    for obs in OBSTACLES:
        grid[obs] = 1
    grid[START] = 2  # Start position
    grid[GOAL] = 3   # Goal position
    
    print(f"Grid size: {SIZE}x{SIZE}")
    print(f"Start position: {START}")
    print(f"Goal position: {GOAL}")
    print(f"Number of obstacles: {len(OBSTACLES)}")
    
    # BFS to find shortest path
    from collections import deque
    
    def bfs_shortest_path(start, goal, obstacles):
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            (node, path) = queue.popleft()
            
            if node == goal:
                return path
            
            # Try all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = node[0] + dx, node[1] + dy
                next_node = (nx, ny)
                
                if (0 <= nx < SIZE and 0 <= ny < SIZE and 
                    next_node not in visited and 
                    next_node not in obstacles):
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None  # No path found
    
    start_time = time.time()
    shortest_path = bfs_shortest_path(START, GOAL, OBSTACLES)
    end_time = time.time()
    
    if shortest_path:
        print(f"Shortest path found with {len(shortest_path) - 1} steps (took {end_time - start_time:.4f} seconds)")
        
        # Plot the shortest path
        path_grid = grid.copy()
        for x, y in shortest_path:
            if (x, y) != START and (x, y) != GOAL:
                path_grid[x, y] = 4  # Path
        
        plt.figure(figsize=(10, 10))
        plt.imshow(path_grid, cmap='viridis')
        plt.colorbar(ticks=[0, 1, 2, 3, 4], 
                    label='Cell Type', 
                    boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
        plt.clim(-0.5, 4.5)
        labels = ['Empty', 'Obstacle', 'Start', 'Goal', 'Path']
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(labels)
        
        plt.title(f'Shortest Path: {len(shortest_path) - 1} steps')
        plt.savefig('shortest_path.png', dpi=300)
        plt.show()
        
        # Also print the path coordinates
        print("\nShortest path coordinates:")
        for i, (x, y) in enumerate(shortest_path):
            print(f"Step {i}: ({x}, {y})")
    else:
        print("No path found from start to goal!")

def main():
    print("===== DIAGNOSTIC ANALYSIS OF FLAT VS FRACTAL AGENT COMPARISON =====")
    analyze_step_counting()
    analyze_reward_structure()
    analyze_log_statistics()
    analyze_agent_structure()
    plot_detailed_comparison()
    run_simulated_paths()
    print("\nDiagnostic analysis complete. Results saved to 'detailed_comparison.png', 'improvement_comparison.png', and 'shortest_path.png'")

if __name__ == "__main__":
    main() 