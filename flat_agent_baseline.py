#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import animation
from tqdm import trange
from collections import deque
import random
import time

# Set random seed for reproducibility
np.random.seed(0)

# ---------- Environment (20×20 grid with obstacles, 1 goal square) -------------
SIZE = 20
GOAL = (SIZE-1, SIZE-1)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

# Create obstacles (identical to fractal agent for fair comparison)
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
        rwd = -2   # Higher penalty for bumping into walls
    else:
        rwd = 10 if nxt == GOAL else -1
    
    done = nxt == GOAL
    return nxt, rwd, done

# ---------- Flat (non-hierarchical) Q-Learner with Experience Replay ---------------------------
# State space is the full micro-state space (SIZE*SIZE positions)
micro_states = SIZE*SIZE
α, γ, ε_start, ε_end, ε_decay = 0.2, 0.95, 0.3, 0.05, 0.995

# Experience replay buffer
BUFFER_SIZE = 10000
BATCH_SIZE = 64
experience_buffer = deque(maxlen=BUFFER_SIZE)

# Single flat Q-table for all states
Q_flat = np.zeros((micro_states, 4))  # 4 primitive moves

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

def train_flat_agent():
    # ---------- Training loop -------------------------------------------
    episodes, horizon = 600, 500  # Same as fractal agent for fair comparison
    log = []
    epsilon = ε_start
    training_time = 0

    start_time = time.time()
    for ep in trange(episodes):
        pos, done = (0, 0), False
        steps = 0
        
        # Decay epsilon
        epsilon = max(ε_end, epsilon * ε_decay)
        
        for t in range(horizon):
            steps += 1
            
            # Simple flat Q-learning: choose action directly from current state
            s = idx_micro(pos)
            a = choose(Q_flat, s, epsilon)
            
            # Execute action and observe next state and reward
            nxt, rwd, done = step(pos, a)
            s_next = idx_micro(nxt)
            
            # Store experience in buffer
            experience_buffer.append((s, a, rwd, s_next, done))
            
            # Update Q-table from buffer
            update_from_buffer(Q_flat, experience_buffer, min(BATCH_SIZE, len(experience_buffer)), α)
            
            pos = nxt
            
            if done:
                break
        
        log.append(steps)
        
        # Additional batch updates at the end of each episode
        for _ in range(5):
            update_from_buffer(Q_flat, experience_buffer, min(BATCH_SIZE, len(experience_buffer)), α)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return log, training_time

def visualize_flat_agent():
    # ---------- Visualization of a single run with the trained agent ------
    pos = (0, 0)
    frames = []
    path = [pos]  # Track the agent's path
    
    while True:
        grid = np.zeros((SIZE, SIZE))
        obstacle_grid = np.zeros((SIZE, SIZE))
        
        # Mark obstacles
        for obs in OBSTACLES:
            obstacle_grid[obs] = 1
        
        # Mark goal
        goal_grid = np.zeros((SIZE, SIZE))
        goal_grid[GOAL] = 1
        
        # Mark agent position
        grid[pos] = 1
        
        # Add path history
        path_grid = np.zeros((SIZE, SIZE))
        for p in path:
            path_grid[p] = 0.5  # Mark the path with half intensity
        
        frames.append((grid.copy(), obstacle_grid.copy(), goal_grid.copy(), path_grid.copy()))
        
        if pos == GOAL:
            break
            
        a = choose(Q_flat, idx_micro(pos), 0.05)  # Small epsilon for some exploration
        pos, _, _ = step(pos, a)
        path.append(pos)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    
    im_agent = ax.imshow(frames[0][0], cmap='Reds', alpha=0.7)
    im_obstacles = ax.imshow(frames[0][1], cmap='binary', alpha=0.8)
    im_goal = ax.imshow(frames[0][2], cmap='Oranges', alpha=1.0)
    im_path = ax.imshow(frames[0][3], cmap='viridis', alpha=0.3)
    
    ax.set_title('Flat Q-Learning Agent')
    ax.axis('off')

    def animate(i):
        im_agent.set_data(frames[i][0])
        im_obstacles.set_data(frames[i][1])
        im_goal.set_data(frames[i][2])
        im_path.set_data(frames[i][3])
        return im_agent, im_obstacles, im_goal, im_path

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200)
    plt.tight_layout()
    plt.show()
    return ani

def plot_q_values():
    # Visualize Q-values as heatmaps
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, action in enumerate(ACTIONS):
        q_values = np.zeros((SIZE, SIZE))
        for x in range(SIZE):
            for y in range(SIZE):
                pos = (x, y)
                idx = idx_micro(pos)
                q_values[x, y] = Q_flat[idx, i]
        
        im = axs[i].imshow(q_values, cmap='hot')
        axs[i].set_title(f'Q-values: Action {i} {ACTIONS[i]}')
        fig.colorbar(im, ax=axs[i])
    
    plt.tight_layout()
    plt.suptitle('Flat Q-Learning: Q-Values by Action', y=1.05, fontsize=16)
    plt.show()

def plot_learning_curve(log):
    plt.figure(figsize=(12, 8))
    
    # Plot actual steps per episode
    plt.subplot(2, 1, 1)
    plt.plot(log, '-o', alpha=0.4, markersize=2, label='Steps per Episode')
    
    # Calculate rolling average
    window_size = 30
    if len(log) >= window_size:
        rolling_avg = np.convolve(log, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(log)), rolling_avg, 'r-', linewidth=2, label=f'Rolling Avg (window={window_size})')
    
    plt.title('Flat Q-Learning: Steps to Goal vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot smoothed learning curve with log scale
    plt.subplot(2, 1, 2)
    plt.semilogy(log, '-', alpha=0.3, label='Steps (log scale)')
    
    # Calculate and plot exponential moving average
    alpha = 0.1  # Smoothing factor
    ema = [log[0]]
    for i in range(1, len(log)):
        ema.append(alpha * log[i] + (1 - alpha) * ema[-1])
    
    plt.semilogy(ema, 'g-', linewidth=2, label='Exp Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal (log scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    print("Training the flat (non-hierarchical) Q-learning agent with obstacles and experience replay...")
    log, training_time = train_flat_agent()
    
    print("\nTraining complete!")
    print(f"Final performance: {log[-1]} steps to goal")
    print(f"Best performance: {min(log)} steps to goal")
    print(f"Training time: {training_time:.2f} seconds")
    
    print("\nPlotting learning curve...")
    plot_learning_curve(log)
    
    print("\nVisualizing Q-values...")
    plot_q_values()
    
    print("\nVisualizing agent behavior...")
    visualize_flat_agent()
    
    print("\nExperiment complete!")
    
    # Save log data for comparison with fractal agent
    np.save('flat_agent_log.npy', np.array(log))

if __name__ == "__main__":
    main() 