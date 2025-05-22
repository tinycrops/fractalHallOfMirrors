#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import deque
import random
from tqdm import trange

# ---------- Environment setup (copied from the corrected_fractal_agent.py) -------------
SIZE = 20
GOAL = (SIZE-1, SIZE-1)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

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
    start = (0, 0)
    if start in obstacles:
        obstacles.remove(start)
    
    return obstacles

OBSTACLES = create_obstacles()

def step(pos, a):
    nxt = tuple(np.clip(np.add(pos, ACTIONS[a]), 0, SIZE-1))
    
    # Check if the next position is an obstacle
    if nxt in OBSTACLES:
        nxt = pos  # Cannot move into an obstacle
        rwd = -1   # Use consistent sparse rewards
    else:
        rwd = 10 if nxt == GOAL else -1  # Sparse reward structure
    
    done = nxt == GOAL
    return nxt, rwd, done

# ---------- Sparse-reward Fractal Agent -----------------------------
block_micro = 5
block_macro = 10
super_states = (SIZE//block_macro)**2
macro_states = (SIZE//block_micro)**2
micro_states = SIZE*SIZE
α, γ, ε_start, ε_end, ε_decay = 0.2, 0.95, 0.3, 0.05, 0.995

# Experience replay buffers
BUFFER_SIZE = 10000
BATCH_SIZE = 64
micro_buffer = deque(maxlen=BUFFER_SIZE)
macro_buffer = deque(maxlen=BUFFER_SIZE)
super_buffer = deque(maxlen=BUFFER_SIZE)

def idx_super(pos):
    return (pos[0]//block_macro) * (SIZE//block_macro) + pos[1]//block_macro

def idx_macro(pos):
    return (pos[0]//block_micro) * (SIZE//block_micro) + pos[1]//block_micro

def idx_micro(pos): 
    return pos[0]*SIZE + pos[1]

def choose(Q, s, epsilon):
    return np.random.choice(4) if np.random.rand() < epsilon else Q[s].argmax()

def update_from_buffer(Q, buffer, batch_size, alpha):
    if len(buffer) < batch_size:
        return
    
    batch = random.sample(buffer, batch_size)
    for s, a, r, s_next, done in batch:
        Q[s, a] += alpha * (r + γ * np.max(Q[s_next]) * (1-done) - Q[s, a])

def train_sparse_fractal_agent():
    """Train a fractal agent with sparse rewards (like the flat agent)"""
    Q_super = np.zeros((super_states, 4))
    Q_macro = np.zeros((macro_states, 4))
    Q_micro = np.zeros((micro_states, 4))
    
    micro_buffer = deque(maxlen=BUFFER_SIZE)
    macro_buffer = deque(maxlen=BUFFER_SIZE)
    super_buffer = deque(maxlen=BUFFER_SIZE)
    
    episodes, horizon = 600, 500
    log = []
    epsilon = ε_start
    training_time = 0
    total_micro_steps = 0
    
    start_time = time.time()
    for ep in trange(episodes, desc="Training Sparse Fractal Agent"):
        pos, done = (0, 0), False
        primitive_steps = 0
        high_level_decisions = 0
        
        # Decay epsilon
        epsilon = max(ε_end, epsilon * ε_decay)
        
        for t in range(horizon):
            if done:
                break
                
            high_level_decisions += 1
            
            # ---- super level pick neighbor super-block (super-goal) -----
            s_super = idx_super(pos)
            a_super = choose(Q_super, s_super, epsilon)
            target_super_block = np.add(divmod(s_super, SIZE//block_macro), ACTIONS[a_super])
            target_super_block = np.clip(target_super_block, 0, SIZE//block_macro-1)
            super_goal = tuple(target_super_block*block_macro + block_macro//2)
            
            # ---- high level navigate within super-block to macro-goal ---
            steps_within_super = 0
            max_steps_super = block_macro*block_macro
            super_done = False
            
            while steps_within_super < max_steps_super and not super_done and not done:
                # ---- high level pick neighbour patch (macro-goal) -------
                s_mac = idx_macro(pos)
                a_mac = choose(Q_macro, s_mac, epsilon)
                target_block = np.add(divmod(s_mac, SIZE//block_micro), ACTIONS[a_mac])
                target_block = np.clip(target_block, 0, SIZE//block_micro-1)
                macro_goal = tuple(target_block*block_micro + block_micro//2)

                # ---- low level navigate to macro-goal or until timeout --
                macro_done = False
                for _ in range(block_micro*block_micro):
                    if done:
                        break
                        
                    s_mic = idx_micro(pos)
                    a_mic = choose(Q_micro, s_mic, epsilon)
                    
                    # Execute primitive action - THIS IS WHERE ACTUAL STEPS HAPPEN
                    nxt, rwd, done = step(pos, a_mic)
                    primitive_steps += 1
                    total_micro_steps += 1
                    
                    s2_mic = idx_micro(nxt)
                    
                    # Store experience in micro buffer with UNCHANGED SPARSE REWARD
                    micro_buffer.append((s_mic, a_mic, rwd, s2_mic, done))
                    
                    update_from_buffer(Q_micro, micro_buffer, min(BATCH_SIZE, len(micro_buffer)), α)
                    
                    pos = nxt
                    steps_within_super += 1
                    
                    if done or pos == macro_goal:
                        macro_done = True
                        break
                
                # SPARSE REWARD FOR MACRO LEVEL - just pass micro reward through
                r_mac = 10 if done else -1
                
                s2_mac = idx_macro(pos)
                
                macro_buffer.append((s_mac, a_mac, r_mac, s2_mac, done or macro_done))
                
                update_from_buffer(Q_macro, macro_buffer, min(BATCH_SIZE, len(macro_buffer)), α)
                
                if done or pos == super_goal:
                    super_done = True
                    break
            
            # SPARSE REWARD FOR SUPER LEVEL - just pass micro reward through  
            r_super = 10 if done else -1
            
            s2_super = idx_super(pos)
            
            super_buffer.append((s_super, a_super, r_super, s2_super, done or super_done))
            
            update_from_buffer(Q_super, super_buffer, min(BATCH_SIZE, len(super_buffer)), α)
        
        # Log the PRIMITIVE STEPS (micro actions) for this episode
        log.append(primitive_steps)
        
        # Additional batch updates at the end of each episode
        for _ in range(5):
            update_from_buffer(Q_micro, micro_buffer, min(BATCH_SIZE, len(micro_buffer)), α)
            update_from_buffer(Q_macro, macro_buffer, min(BATCH_SIZE, len(macro_buffer)), α)
            update_from_buffer(Q_super, super_buffer, min(BATCH_SIZE, len(super_buffer)), α)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Total primitive actions: {total_micro_steps}")
    print(f"Average primitive actions per episode: {total_micro_steps/episodes:.2f}")
    
    return log, training_time, Q_micro, Q_macro, Q_super

# ---------- Flat Agent (for reference) -----------------------------
def train_flat_agent():
    """Train a flat agent (from scratch) to use for comparison"""
    Q = np.zeros((SIZE*SIZE, 4))
    buffer = deque(maxlen=BUFFER_SIZE)
    
    episodes, horizon = 600, 500
    log = []
    epsilon = ε_start
    training_time = 0
    
    start_time = time.time()
    for ep in trange(episodes, desc="Training Flat Agent"):
        pos, done = (0, 0), False
        steps = 0
        
        # Decay epsilon
        epsilon = max(ε_end, epsilon * ε_decay)
        
        for t in range(horizon):
            if done:
                break
                
            # Choose action
            s = idx_micro(pos)
            a = choose(Q, s, epsilon)
            
            # Take action
            nxt, rwd, done = step(pos, a)
            steps += 1
            
            # Update Q-values
            s2 = idx_micro(nxt)
            buffer.append((s, a, rwd, s2, done))
            update_from_buffer(Q, buffer, min(BATCH_SIZE, len(buffer)), α)
            
            pos = nxt
        
        log.append(steps)
        
        # Additional batch updates at the end of each episode
        for _ in range(5):
            update_from_buffer(Q, buffer, min(BATCH_SIZE, len(buffer)), α)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return log, training_time, Q

def compare_results(flat_log, sparse_fractal_log):
    """Compare the learning curves of both agents with a focus on reward fairness"""
    print("\n===== COMPARING LEARNING CURVES WITH SPARSE REWARDS =====")
    
    # Calculate statistics
    flat_min = np.min(flat_log)
    flat_final = flat_log[-1]
    flat_mean = np.mean(flat_log)
    
    fractal_min = np.min(sparse_fractal_log)
    fractal_final = sparse_fractal_log[-1]
    fractal_mean = np.mean(sparse_fractal_log)
    
    # Performance metrics
    print("\nPerformance Metrics with Sparse Rewards:")
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
    
    # Sample efficiency analysis
    print("\nSample Efficiency Analysis:")
    thresholds = [200, 150, 100, 75, 50]
    
    print(f"{'Threshold (steps)':<20} {'Flat Agent':<15} {'Fractal Agent':<15} {'Speedup Factor':<15}")
    print("-" * 70)
    
    for threshold in thresholds:
        window_size = 5
        
        flat_ep = None
        for i in range(len(flat_log) - window_size + 1):
            if np.mean(flat_log[i:i+window_size]) < threshold:
                flat_ep = i
                break
                
        fractal_ep = None
        for i in range(len(sparse_fractal_log) - window_size + 1):
            if np.mean(sparse_fractal_log[i:i+window_size]) < threshold:
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
    
    # Generate comparison plots
    plt.figure(figsize=(15, 12))
    
    # Raw data plot
    plt.subplot(3, 1, 1)
    plt.plot(flat_log, 'b-', alpha=0.3, label='Flat Agent')
    plt.plot(sparse_fractal_log, 'r-', alpha=0.3, label='Sparse Fractal Agent')
    
    # Add rolling averages
    window = 30
    if len(flat_log) >= window and len(sparse_fractal_log) >= window:
        flat_avg = np.convolve(flat_log, np.ones(window)/window, mode='valid')
        fractal_avg = np.convolve(sparse_fractal_log, np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(flat_log)), flat_avg, 'b-', linewidth=2, label='Flat Agent (30-ep avg)')
        plt.plot(range(window-1, len(sparse_fractal_log)), fractal_avg, 'r-', linewidth=2, label='Sparse Fractal Agent (30-ep avg)')
    
    plt.title('Learning Curves: Primitive Steps per Episode (Sparse Rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot for better comparison of improvements
    plt.subplot(3, 1, 2)
    plt.semilogy(flat_log, 'b-', alpha=0.3, label='Flat Agent')
    plt.semilogy(sparse_fractal_log, 'r-', alpha=0.3, label='Sparse Fractal Agent')
    
    # Add rolling averages on log scale
    if len(flat_log) >= window and len(sparse_fractal_log) >= window:
        plt.semilogy(range(window-1, len(flat_log)), flat_avg, 'b-', linewidth=2, label='Flat Agent (30-ep avg)')
        plt.semilogy(range(window-1, len(sparse_fractal_log)), fractal_avg, 'r-', linewidth=2, label='Sparse Fractal Agent (30-ep avg)')
    
    # Add horizontal line for optimal path length
    plt.axhline(y=optimal_path_length, color='g', linestyle='--', label=f'Optimal Path ({optimal_path_length} steps)')
    
    plt.title('Learning Curves (Log Scale) with Sparse Rewards')
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
    plt.bar(x + width/2, fractal_values, width, label='Sparse Fractal Agent')
    
    plt.axhline(y=optimal_path_length, color='g', linestyle='--', label=f'Optimal Path ({optimal_path_length} steps)')
    
    plt.title('Performance Metrics Comparison (Sparse Rewards)')
    plt.xlabel('Metric')
    plt.ylabel('Steps to Goal')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparse_reward_comparison.png', dpi=300)
    plt.show()

def main():
    print("===== SPARSE REWARD COMPARISON OF FLAT VS FRACTAL AGENT =====")
    print("This will train both agents with identical sparse reward structures")
    
    # Train both agents
    flat_log, flat_time, flat_Q = train_flat_agent()
    sparse_fractal_log, fractal_time, Q_micro, Q_macro, Q_super = train_sparse_fractal_agent()
    
    # Save logs for future analysis
    np.save('sparse_flat_agent_log.npy', np.array(flat_log))
    np.save('sparse_fractal_agent_log.npy', np.array(sparse_fractal_log))
    
    print(f"\nFlat agent training time: {flat_time:.2f} seconds")
    print(f"Fractal agent training time: {fractal_time:.2f} seconds")
    
    # Compare results
    compare_results(flat_log, sparse_fractal_log)
    
    print("\nSparse reward comparison complete. Results saved to sparse_reward_comparison.png")

if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility
    main() 