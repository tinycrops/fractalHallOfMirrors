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

# ---------- Fractal Agent with Attention Mechanisms ---------------------------
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

# Attention weights for each level (initialized equally)
# These weights determine how much the agent focuses on each level of abstraction
attention_weights = np.array([1/3, 1/3, 1/3])  # [micro, macro, super]
attention_history = []  # To track how attention changes during training

def idx_super(pos):
    return (pos[0]//block_macro) * (SIZE//block_macro) + pos[1]//block_macro

def idx_macro(pos):
    return (pos[0]//block_micro) * (SIZE//block_micro) + pos[1]//block_micro

def idx_micro(pos): 
    return pos[0]*SIZE + pos[1]

def softmax(x):
    """Compute softmax values for the array x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

def choose(Q, s, epsilon):
    return np.random.choice(4) if np.random.rand() < epsilon else Q[s].argmax()

def update_from_buffer(Q, buffer, batch_size, alpha):
    if len(buffer) < batch_size:
        return
    
    batch = random.sample(buffer, batch_size)
    for s, a, r, s_next, done in batch:
        Q[s, a] += alpha * (r + γ * np.max(Q[s_next]) * (1-done) - Q[s, a])

def compute_attention_weights(pos, super_goal, macro_goal):
    """
    Dynamically compute attention weights based on the current state and goals.
    
    The key insight: When the agent is far from the goal, it should focus more
    on higher-level abstractions. As it gets closer to immediate goals, 
    it should shift attention to finer details.
    """
    global attention_weights
    
    # Calculate distances to goals at different scales
    dist_to_super_goal = abs(pos[0] - super_goal[0]) + abs(pos[1] - super_goal[1])
    dist_to_macro_goal = abs(pos[0] - macro_goal[0]) + abs(pos[1] - macro_goal[1])
    dist_to_final_goal = abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])
    
    # Normalize distances to [0, 1] range
    max_dist = SIZE * 2  # Maximum possible Manhattan distance in the grid
    norm_dist_super = dist_to_super_goal / max_dist
    norm_dist_macro = dist_to_macro_goal / max_dist
    norm_dist_final = dist_to_final_goal / max_dist
    
    # Compute attention logits (pre-softmax)
    # When far from a goal, attend more to that level
    # When close to a goal, reduce attention to that level
    attention_logits = np.array([
        1.0 - norm_dist_final * 0.5,  # Micro attention (fine details matter more when close to final goal)
        1.0 - norm_dist_macro * 0.5,  # Macro attention
        1.0 - norm_dist_super * 0.5   # Super attention
    ])
    
    # Add obstacle detection logic - if near an obstacle, increase micro attention
    # Check adjacent positions for obstacles
    has_nearby_obstacle = False
    for a in ACTIONS.values():
        adj_pos = tuple(np.clip(np.add(pos, a), 0, SIZE-1))
        if adj_pos in OBSTACLES:
            has_nearby_obstacle = True
            break
    
    if has_nearby_obstacle:
        attention_logits[0] += 0.5  # Boost micro attention when obstacles are nearby
    
    # Apply softmax to get final attention weights
    attention_weights = softmax(attention_logits)
    attention_history.append(attention_weights.copy())
    
    return attention_weights

def train_agent():
    # ---------- Training loop -------------------------------------------
    episodes, horizon = 600, 500  # Same parameters as flat agent for fair comparison
    log = []
    epsilon = ε_start
    training_time = 0
    total_micro_steps = 0  # Track total primitive actions across all episodes
    
    start_time = time.time()
    for ep in trange(episodes):
        pos, done = (0, 0), False
        primitive_steps = 0  # Count of primitive actions for this episode
        
        # Decay epsilon
        epsilon = max(ε_end, epsilon * ε_decay)
        
        for t in range(horizon):
            if done:
                break
                
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
            
            while steps_within_super < max_steps_super and not super_done and not done:
                # ---- high level pick neighbour patch (macro-goal) -------
                s_mac = idx_macro(pos)
                a_mac = choose(Q_macro, s_mac, epsilon)
                # translate macro action to desired patch centre
                target_block = np.add(divmod(s_mac, SIZE//block_micro), ACTIONS[a_mac])
                target_block = np.clip(target_block, 0, SIZE//block_micro-1)
                macro_goal = tuple(target_block*block_micro + block_micro//2)

                # ---- low level navigate to macro-goal or until timeout --
                macro_done = False
                
                # Compute dynamic attention weights
                attn_weights = compute_attention_weights(pos, super_goal, macro_goal)
                
                for _ in range(block_micro*block_micro):
                    if done:
                        break
                        
                    s_mic = idx_micro(pos)
                    
                    # Compute actions from all levels
                    a_mic_from_micro = Q_micro[s_mic].argmax()
                    a_mic_from_macro = a_mac  # Macro action directly (simplification)
                    a_mic_from_super = a_super  # Super action directly (simplification)
                    
                    # Combine actions using attention weights
                    action_preferences = np.zeros(4)
                    action_preferences[a_mic_from_micro] += attn_weights[0]
                    action_preferences[a_mic_from_macro] += attn_weights[1]
                    action_preferences[a_mic_from_super] += attn_weights[2]
                    
                    # Choose the action with highest combined preference
                    a_mic = action_preferences.argmax()
                    
                    # Epsilon-greedy override
                    if np.random.rand() < epsilon:
                        a_mic = np.random.choice(4)
                    
                    # Execute primitive action
                    nxt, rwd, done = step(pos, a_mic)
                    primitive_steps += 1
                    total_micro_steps += 1
                    
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
    
    return log, training_time, attention_history

def visualize_agent():
    # ---------- Educational visualisation (1 finale run, animated) ------
    pos = (0, 0)
    frames = []
    path = [pos]  # Track the agent's path
    attention_frames = []  # Track attention weights during visualization
    
    done = False
    while not done:
        # Create visualization grids
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
        
        # Mark agent position
        micro_grid[pos] = 1
        
        # Add path history
        path_grid = np.zeros((SIZE, SIZE))
        for p in path:
            path_grid[p] = 0.5  # Mark the path with half intensity
        
        # Compute super goal
        s_super = idx_super(pos)
        a_super = Q_super[s_super].argmax()
        target_super_block = np.add(divmod(s_super, SIZE//block_macro), ACTIONS[a_super])
        target_super_block = np.clip(target_super_block, 0, SIZE//block_macro-1)
        super_goal = tuple(target_super_block*block_macro + block_macro//2)
        
        # Compute macro goal
        s_mac = idx_macro(pos)
        a_mac = Q_macro[s_mac].argmax()
        target_block = np.add(divmod(s_mac, SIZE//block_micro), ACTIONS[a_mac])
        target_block = np.clip(target_block, 0, SIZE//block_micro-1)
        macro_goal = tuple(target_block*block_micro + block_micro//2)
        
        # Highlight super and macro blocks with different intensities
        # Super block (10x10)
        super_block_x = (pos[0] // block_macro) * block_macro
        super_block_y = (pos[1] // block_macro) * block_macro
        for i in range(block_macro):
            for j in range(block_macro):
                x, y = super_block_x + i, super_block_y + j
                if 0 <= x < SIZE and 0 <= y < SIZE:
                    super_grid[x, y] = 0.3
        
        # Macro block (5x5)
        macro_block_x = (pos[0] // block_micro) * block_micro
        macro_block_y = (pos[1] // block_micro) * block_micro
        for i in range(block_micro):
            for j in range(block_micro):
                x, y = macro_block_x + i, macro_block_y + j
                if 0 <= x < SIZE and 0 <= y < SIZE:
                    macro_grid[x, y] = 0.6
        
        # Mark goals
        super_grid[super_goal] = 0.8
        macro_grid[macro_goal] = 0.8
        
        # Compute attention weights
        attn_weights = compute_attention_weights(pos, super_goal, macro_goal)
        attention_frames.append(attn_weights.copy())
        
        # Store frame for animation
        frames.append((micro_grid.copy(), macro_grid.copy(), super_grid.copy(), 
                      obstacle_grid.copy(), goal_grid.copy(), path_grid.copy()))
        
        # Choose next action based on attention-weighted policy
        s_mic = idx_micro(pos)
        
        # Compute actions from all levels
        a_mic_from_micro = Q_micro[s_mic].argmax()
        a_mic_from_macro = a_mac
        a_mic_from_super = a_super
        
        # Combine actions using attention weights
        action_preferences = np.zeros(4)
        action_preferences[a_mic_from_micro] += attn_weights[0]
        action_preferences[a_mic_from_macro] += attn_weights[1]
        action_preferences[a_mic_from_super] += attn_weights[2]
        
        # Choose the action with highest combined preference
        a_mic = action_preferences.argmax()
        
        # Take step
        pos, _, done = step(pos, a_mic)
        path.append(pos)
    
    # Plotting
    fig = plt.figure(figsize=(16, 12))
    gs_obj = gs.GridSpec(3, 3, figure=fig)
    
    # Main grid display
    ax_grid = fig.add_subplot(gs_obj[0:2, 0:2])
    ax_super = fig.add_subplot(gs_obj[0, 2])
    ax_macro = fig.add_subplot(gs_obj[1, 2])
    ax_micro = fig.add_subplot(gs_obj[2, 0])
    ax_attention = fig.add_subplot(gs_obj[2, 1:])
    
    # Initialize grid images
    im_micro = ax_grid.imshow(frames[0][0], cmap='Reds', alpha=0.7)
    im_macro = ax_grid.imshow(frames[0][1], cmap='Blues', alpha=0.5)
    im_super = ax_grid.imshow(frames[0][2], cmap='Greens', alpha=0.3)
    im_obstacles = ax_grid.imshow(frames[0][3], cmap='binary', alpha=0.8)
    im_goal = ax_grid.imshow(frames[0][4], cmap='Oranges', alpha=1.0)
    im_path = ax_grid.imshow(frames[0][5], cmap='viridis', alpha=0.3)
    
    # Initialize level-specific displays
    im_super_view = ax_super.imshow(frames[0][2], cmap='Greens', alpha=0.7)
    im_macro_view = ax_macro.imshow(frames[0][1], cmap='Blues', alpha=0.7)
    im_micro_view = ax_micro.imshow(frames[0][0], cmap='Reds', alpha=0.7)
    
    # Initialize attention bars
    bar_container = ax_attention.bar(['Micro', 'Macro', 'Super'], attention_frames[0], color=['red', 'blue', 'green'])
    
    # Set titles
    ax_grid.set_title('Fractal Agent with Attention Mechanisms', fontsize=16)
    ax_super.set_title('Super Grid (Strategic)')
    ax_macro.set_title('Macro Grid (Tactical)')
    ax_micro.set_title('Micro Grid (Immediate)')
    ax_attention.set_title('Attention Allocation')
    
    # Remove axis ticks
    ax_grid.axis('off')
    ax_super.axis('off')
    ax_macro.axis('off')
    ax_micro.axis('off')
    
    # Set attention axis limits
    ax_attention.set_ylim(0, 1)
    
    def animate(i):
        im_micro.set_data(frames[i][0])
        im_macro.set_data(frames[i][1])
        im_super.set_data(frames[i][2])
        im_obstacles.set_data(frames[i][3])
        im_goal.set_data(frames[i][4])
        im_path.set_data(frames[i][5])
        
        im_super_view.set_data(frames[i][2])
        im_macro_view.set_data(frames[i][1])
        im_micro_view.set_data(frames[i][0])
        
        # Update attention bars
        for j, bar in enumerate(bar_container):
            bar.set_height(attention_frames[i][j])
        
        # Add frame number as subtitle
        ax_grid.set_xlabel(f'Step: {i+1}/{len(frames)}')
        
        return (im_micro, im_macro, im_super, im_obstacles, im_goal, im_path, 
                im_super_view, im_macro_view, im_micro_view, bar_container)
    
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def plot_attention_evolution(attention_history):
    """Visualize how attention shifts over the course of training."""
    # Convert attention history to numpy array for easier manipulation
    attn_history = np.array(attention_history)
    
    # Calculate average attention weights over fixed intervals
    window_size = len(attn_history) // 20  # 20 data points for smoothness
    if window_size < 1:
        window_size = 1
    
    avg_attention = []
    for i in range(0, len(attn_history), window_size):
        end = min(i + window_size, len(attn_history))
        avg_attention.append(np.mean(attn_history[i:end], axis=0))
    
    avg_attention = np.array(avg_attention)
    
    # Plot the evolution of attention weights
    plt.figure(figsize=(12, 6))
    plt.plot(avg_attention[:, 0], 'r-', linewidth=2, label='Micro (Immediate)')
    plt.plot(avg_attention[:, 1], 'b-', linewidth=2, label='Macro (Tactical)')
    plt.plot(avg_attention[:, 2], 'g-', linewidth=2, label='Super (Strategic)')
    
    plt.title('Evolution of Attention Allocation During Training', fontsize=16)
    plt.xlabel('Training Progress', fontsize=14)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Training Fractal Agent with Attention Mechanisms...")
    log, training_time, attention_history = train_agent()
    
    print("\nVisualizing agent behavior...")
    ani = visualize_agent()
    
    print("\nPlotting attention evolution...")
    plot_attention_evolution(attention_history)
    
    return log, training_time

if __name__ == "__main__":
    main() 