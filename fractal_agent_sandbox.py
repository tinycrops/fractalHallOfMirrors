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

# ---------- Toy environment (20×20 grid with obstacles, 1 goal square) -------------
SIZE = 20  # Increased from 10 to 20 for deeper hierarchy
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
        rwd = -2   # Higher penalty for bumping into walls
    else:
        rwd = 10 if nxt == GOAL else -1
    
    done = nxt == GOAL
    return nxt, rwd, done

# ---------- Three-level (fractal) Q-Learner with Experience Replay ---------------------------
block_micro = 5               # micro patch size (5×5)
block_macro = 10              # macro patch size (10×10)
super_states = (SIZE//block_macro)**2  # 2×2 super blocks
macro_states = (SIZE//block_micro)**2  # 4×4 macro blocks
micro_states = SIZE*SIZE      # 20×20 fine-grained states
α, γ, ε_start, ε_end, ε_decay = 0.2, 0.95, 0.3, 0.05, 0.995

# Experience replay buffers
BUFFER_SIZE = 10000
BATCH_SIZE = 64
micro_buffer = deque(maxlen=BUFFER_SIZE)
macro_buffer = deque(maxlen=BUFFER_SIZE)
super_buffer = deque(maxlen=BUFFER_SIZE)

Q_super = np.zeros((super_states, 4))  # 4 super actions = neighbor super patch
Q_macro = np.zeros((macro_states, 4))  # 4 macro actions = neighbor patch
Q_micro = np.zeros((micro_states, 4))  # 4 primitive moves

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

def train_agent():
    # ---------- Training loop -------------------------------------------
    episodes, horizon = 600, 500  # Increased for larger environment with obstacles
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
            
            # ---- super level pick neighbor super-block (super-goal) -----
            s_super = idx_super(pos)
            a_super = choose(Q_super, s_super, epsilon)
            # translate super action to desired super-block centre
            target_super_block = np.add(divmod(s_super, SIZE//block_macro), ACTIONS[a_super])
            target_super_block = np.clip(target_super_block, 0, SIZE//block_macro-1)
            super_goal = tuple(target_super_block*block_macro + block_macro//2)
            
            # ---- high level navigate within super-block to macro-goal ---
            steps_within_super = 0
            max_steps_super = block_macro*block_macro
            super_done = False
            
            while steps_within_super < max_steps_super and not super_done:
                # ---- high level pick neighbour patch (macro-goal) -------
                s_mac = idx_macro(pos)
                a_mac = choose(Q_macro, s_mac, epsilon)
                # translate macro action to desired patch centre
                target_block = np.add(divmod(s_mac, SIZE//block_micro), ACTIONS[a_mac])
                target_block = np.clip(target_block, 0, SIZE//block_micro-1)
                macro_goal = tuple(target_block*block_micro + block_micro//2)

                # ---- low level navigate to macro-goal or until timeout --
                macro_done = False
                for _ in range(block_micro*block_micro):
                    s_mic = idx_micro(pos)
                    a_mic = choose(Q_micro, s_mic, epsilon)
                    nxt, rwd, done = step(pos, a_mic)
                    s2_mic = idx_micro(nxt)
                    
                    # Store experience in micro buffer
                    micro_buffer.append((s_mic, a_mic, rwd, s2_mic, done))
                    
                    # Update Q_micro from buffer
                    update_from_buffer(Q_micro, micro_buffer, min(BATCH_SIZE, len(micro_buffer)), α)
                    
                    pos = nxt
                    steps_within_super += 1
                    
                    if done or pos == macro_goal:
                        macro_done = True
                        break
                
                # Determine macro reward
                if done:
                    r_mac = 10  # Goal reached
                elif pos == macro_goal:
                    r_mac = 5   # Subgoal reached
                elif pos == super_goal:
                    r_mac = 3   # Super goal reached
                else:
                    # Check if we're getting closer to the macro_goal
                    prev_dist = abs(pos[0] - macro_goal[0]) + abs(pos[1] - macro_goal[1])
                    r_mac = -1 + 0.1 * (1.0 / (prev_dist + 1))  # Shaped reward based on distance
                
                s2_mac = idx_macro(pos)
                
                # Store experience in macro buffer
                macro_buffer.append((s_mac, a_mac, r_mac, s2_mac, done or macro_done))
                
                # Update Q_macro from buffer
                update_from_buffer(Q_macro, macro_buffer, min(BATCH_SIZE, len(macro_buffer)), α)
                
                if done or pos == super_goal:
                    super_done = True
                    break
            
            # Determine super reward
            if done:
                r_super = 10  # Goal reached
            elif pos == super_goal:
                r_super = 5   # Super goal reached
            else:
                # Check if we're getting closer to the super_goal
                prev_dist = abs(pos[0] - super_goal[0]) + abs(pos[1] - super_goal[1])
                r_super = -1 + 0.1 * (1.0 / (prev_dist + 1))  # Shaped reward based on distance
            
            s2_super = idx_super(pos)
            
            # Store experience in super buffer
            super_buffer.append((s_super, a_super, r_super, s2_super, done or super_done))
            
            # Update Q_super from buffer
            update_from_buffer(Q_super, super_buffer, min(BATCH_SIZE, len(super_buffer)), α)
            
            if done: 
                break
        
        log.append(steps)
        
        # Additional batch updates at the end of each episode for more thorough learning
        for _ in range(5):
            update_from_buffer(Q_micro, micro_buffer, min(BATCH_SIZE, len(micro_buffer)), α)
            update_from_buffer(Q_macro, macro_buffer, min(BATCH_SIZE, len(macro_buffer)), α)
            update_from_buffer(Q_super, super_buffer, min(BATCH_SIZE, len(super_buffer)), α)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return log, training_time

def visualize_agent():
    # ---------- Educational visualisation (1 finale run, animated) ------
    pos = (0, 0)
    frames = []
    path = [pos]  # Track the agent's path
    
    while True:
        super_grid = np.zeros((SIZE, SIZE))
        macro_grid = np.zeros((SIZE, SIZE))
        micro_grid = np.zeros((SIZE, SIZE))
        obstacle_grid = np.zeros((SIZE, SIZE))
        
        # Mark obstacles
        for obs in OBSTACLES:
            obstacle_grid[obs] = 1
        
        # Mark goal
        goal_grid = np.zeros((SIZE, SIZE))
        goal_grid[GOAL] = 1
        
        # Update super grid
        spos = idx_super(pos)
        sr, sc = divmod(spos, SIZE//block_macro)
        super_grid[sr*block_macro:(sr+1)*block_macro, sc*block_macro:(sc+1)*block_macro] = 1
        
        # Update macro grid
        mpos = idx_macro(pos)
        mr, mc = divmod(mpos, SIZE//block_micro)
        macro_grid[mr*block_micro:(mr+1)*block_micro, mc*block_micro:(mc+1)*block_micro] = 1
        
        # Update micro grid (agent position)
        micro_grid[pos] = 1
        
        # Add path history
        path_grid = np.zeros((SIZE, SIZE))
        for p in path:
            path_grid[p] = 0.5  # Mark the path with half intensity
        
        frames.append((super_grid.copy(), macro_grid.copy(), micro_grid.copy(), 
                      obstacle_grid.copy(), goal_grid.copy(), path_grid.copy()))
        
        if pos == GOAL: 
            break
            
        a = choose(Q_micro, idx_micro(pos), 0.05)  # Small epsilon for some exploration
        pos, _, _ = step(pos, a)
        path.append(pos)

    fig = plt.figure(figsize=(15, 15))
    g = gs.GridSpec(3, 2, width_ratios=[3, 1])
    
    ax1 = fig.add_subplot(g[0, 0])
    ax2 = fig.add_subplot(g[1, 0])
    ax3 = fig.add_subplot(g[2, 0])
    
    # Add a combined view showing all levels at once
    ax_combined = fig.add_subplot(g[:, 1])
    
    im1 = ax1.imshow(frames[0][0], cmap='Blues', alpha=0.7)
    im2 = ax2.imshow(frames[0][1], cmap='Greens', alpha=0.7)
    im3 = ax3.imshow(frames[0][2], cmap='Reds', alpha=0.7)
    
    # Add obstacles and goal to each view
    im1_obs = ax1.imshow(frames[0][3], cmap='binary', alpha=0.8)
    im2_obs = ax2.imshow(frames[0][3], cmap='binary', alpha=0.8)
    im3_obs = ax3.imshow(frames[0][3], cmap='binary', alpha=0.8)
    
    im1_goal = ax1.imshow(frames[0][4], cmap='Oranges', alpha=1.0)
    im2_goal = ax2.imshow(frames[0][4], cmap='Oranges', alpha=1.0)
    im3_goal = ax3.imshow(frames[0][4], cmap='Oranges', alpha=1.0)
    
    # Add path history
    im3_path = ax3.imshow(frames[0][5], cmap='viridis', alpha=0.3)
    
    # Combined view with all levels
    combined = np.zeros((SIZE, SIZE, 3))
    combined[:,:,0] = frames[0][0] * 0.5  # Super (blue channel)
    combined[:,:,1] = frames[0][1] * 0.5  # Macro (green channel)
    combined[:,:,2] = frames[0][2]        # Micro (red channel)
    
    # Add obstacles to combined view
    obstacle_mask = frames[0][3] > 0
    combined[obstacle_mask] = [0.3, 0.3, 0.3]  # Gray for obstacles
    
    # Add goal to combined view
    goal_mask = frames[0][4] > 0
    combined[goal_mask] = [1.0, 0.65, 0.0]  # Orange for goal
    
    im_combined = ax_combined.imshow(combined)
    
    ax1.set_title('Super-Coarse (strategic) view')
    ax2.set_title('Coarse (tactical) view')
    ax3.set_title('Fine (immediate) view')
    ax_combined.set_title('Combined View')
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax_combined.axis('off')

    def animate(i):
        im1.set_data(frames[i][0])
        im2.set_data(frames[i][1])
        im3.set_data(frames[i][2])
        
        im1_obs.set_data(frames[i][3])
        im2_obs.set_data(frames[i][3])
        im3_obs.set_data(frames[i][3])
        
        im1_goal.set_data(frames[i][4])
        im2_goal.set_data(frames[i][4])
        im3_goal.set_data(frames[i][4])
        
        im3_path.set_data(frames[i][5])
        
        # Update combined view
        combined = np.zeros((SIZE, SIZE, 3))
        combined[:,:,0] = frames[i][0] * 0.5  # Super (blue channel)
        combined[:,:,1] = frames[i][1] * 0.5  # Macro (green channel)
        combined[:,:,2] = frames[i][2]        # Micro (red channel)
        
        # Re-add obstacles to combined view
        obstacle_mask = frames[i][3] > 0
        combined[obstacle_mask] = [0.3, 0.3, 0.3]  # Gray for obstacles
        
        # Re-add goal to combined view
        goal_mask = frames[i][4] > 0
        combined[goal_mask] = [1.0, 0.65, 0.0]  # Orange for goal
        
        # Add path history
        path_mask = frames[i][5] > 0
        combined[path_mask] = combined[path_mask] + [0.2, 0.2, 0.0]  # Brighten the path
        
        im_combined.set_array(combined)
        
        return im1, im2, im3, im1_obs, im2_obs, im3_obs, im1_goal, im2_goal, im3_goal, im3_path, im_combined

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200)
    plt.tight_layout()
    plt.show()
    
    return ani

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
    
    plt.title('Learning Curve: Steps to Goal vs Episode')
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

def plot_q_values():
    # Visualize Q-values as heatmaps
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    
    # For micro level, we'll sample some states
    for i, action in enumerate(ACTIONS):
        q_values = np.zeros((SIZE, SIZE))
        for x in range(SIZE):
            for y in range(SIZE):
                pos = (x, y)
                idx = idx_micro(pos)
                q_values[x, y] = Q_micro[idx, i]
        
        im = axs[0, i].imshow(q_values, cmap='hot')
        axs[0, i].set_title(f'Micro Q-values: Action {i} {ACTIONS[i]}')
        fig.colorbar(im, ax=axs[0, i])
    
    # For macro level
    macro_size = SIZE // block_micro
    for i, action in enumerate(ACTIONS):
        q_values = np.zeros((macro_size, macro_size))
        for x in range(macro_size):
            for y in range(macro_size):
                idx = x * macro_size + y
                q_values[x, y] = Q_macro[idx, i]
        
        im = axs[1, i].imshow(q_values, cmap='viridis')
        axs[1, i].set_title(f'Macro Q-values: Action {i} {ACTIONS[i]}')
        fig.colorbar(im, ax=axs[1, i])
    
    # For super level
    super_size = SIZE // block_macro
    for i, action in enumerate(ACTIONS):
        q_values = np.zeros((super_size, super_size))
        for x in range(super_size):
            for y in range(super_size):
                idx = x * super_size + y
                q_values[x, y] = Q_super[idx, i]
        
        im = axs[2, i].imshow(q_values, cmap='plasma')
        axs[2, i].set_title(f'Super Q-values: Action {i} {ACTIONS[i]}')
        fig.colorbar(im, ax=axs[2, i])
    
    plt.tight_layout()
    plt.show()

def main():
    print("Training the three-level fractal Q-learning agent with obstacles and experience replay...")
    log, training_time = train_agent()
    
    print("\nTraining complete!")
    print(f"Final performance: {log[-1]} steps to goal")
    print(f"Best performance: {min(log)} steps to goal")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Save log data for comparison
    np.save('fractal_agent_log.npy', np.array(log))
    
    print("\nPlotting learning curve...")
    plot_learning_curve(log)
    
    print("\nVisualizing Q-values...")
    plot_q_values()
    
    print("\nVisualizing agent behavior...")
    visualize_agent()
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main() 