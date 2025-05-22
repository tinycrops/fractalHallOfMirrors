"""
Grid-world Q-learning agents with flat and hierarchical structures.
"""

import numpy as np
import time
from tqdm import trange
from ..common.q_learning_utils import (
    choose_action, update_from_buffer, create_experience_buffer, 
    softmax, decay_epsilon
)


class BaseQLearner:
    """
    Base Q-learning agent with common functionality.
    """
    
    def __init__(self, env, alpha=0.2, gamma=0.95, epsilon_start=0.3, 
                 epsilon_end=0.05, epsilon_decay=0.995, buffer_size=10000, 
                 batch_size=64):
        """
        Initialize the base Q-learner.
        
        Args:
            env: Grid environment
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Experience replay buffer size
            batch_size: Batch size for experience replay
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Initialize Q-tables (to be overridden by subclasses)
        self.Q = None
        
        # Experience replay buffer
        self.experience_buffer = create_experience_buffer(buffer_size)
        
    def get_state_index(self, pos):
        """Convert position to state index. Override in subclasses if needed."""
        return pos[0] * self.env.size + pos[1]
    
    def choose_action(self, state, epsilon):
        """Choose action using epsilon-greedy policy."""
        return choose_action(self.Q, state, epsilon)
    
    def update_q_table(self):
        """Update Q-table from experience buffer."""
        update_from_buffer(self.Q, self.experience_buffer, 
                          min(self.batch_size, len(self.experience_buffer)), 
                          self.alpha, self.gamma)
    
    def train(self, episodes=600, horizon=500):
        """
        Train the agent and return performance log.
        
        Args:
            episodes: Number of training episodes
            horizon: Maximum steps per episode
            
        Returns:
            log: List of steps per episode
            training_time: Time taken for training
        """
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training"):
            pos = self.env.reset()
            done = False
            steps = 0
            
            # Decay epsilon
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                steps += 1
                
                # Get action and execute step
                state = self.get_state_index(pos)
                action = self.choose_action(state, epsilon)
                next_pos, reward, done = self.env.step(pos, action)
                next_state = self.get_state_index(next_pos)
                
                # Store experience
                self.experience_buffer.append((state, action, reward, next_state, done))
                
                # Update Q-table
                self.update_q_table()
                
                pos = next_pos
            
            log.append(steps)
            
            # Additional batch updates at episode end
            for _ in range(5):
                self.update_q_table()
                
        training_time = time.time() - start_time
        return log, training_time
    
    def run_episode(self, epsilon=0.05):
        """Run a single episode and return the path."""
        pos = self.env.reset()
        path = [pos]
        
        while pos != self.env.goal:
            state = self.get_state_index(pos)
            action = self.choose_action(state, epsilon)
            pos, _, _ = self.env.step(pos, action)
            path.append(pos)
            
            # Safety check to prevent infinite loops
            if len(path) > self.env.size * self.env.size:
                break
                
        return path


class FlatAgent(BaseQLearner):
    """
    Flat (non-hierarchical) Q-learning agent.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Single flat Q-table for all states
        self.Q = np.zeros((self.env.state_space_size, self.env.num_actions))


class FractalAgent(BaseQLearner):
    """
    Fractal (hierarchical) Q-learning agent with configurable reward shaping.
    """
    
    def __init__(self, env, block_micro=5, block_macro=10, 
                 reward_shaping='shaped', **kwargs):
        """
        Initialize the fractal agent.
        
        Args:
            env: Grid environment
            block_micro: Micro patch size
            block_macro: Macro patch size  
            reward_shaping: Reward shaping strategy ('shaped' or 'sparse')
            **kwargs: Additional arguments for BaseQLearner
        """
        super().__init__(env, **kwargs)
        
        self.block_micro = block_micro
        self.block_macro = block_macro
        self.reward_shaping = reward_shaping
        
        # Calculate hierarchical state spaces
        self.super_states = (env.size // block_macro) ** 2
        self.macro_states = (env.size // block_micro) ** 2
        self.micro_states = env.state_space_size
        
        # Initialize hierarchical Q-tables
        self.Q_super = np.zeros((self.super_states, env.num_actions))
        self.Q_macro = np.zeros((self.macro_states, env.num_actions))
        self.Q_micro = np.zeros((self.micro_states, env.num_actions))
        
        # Hierarchical experience buffers
        self.super_buffer = create_experience_buffer(self.buffer_size)
        self.macro_buffer = create_experience_buffer(self.buffer_size)
        self.micro_buffer = create_experience_buffer(self.buffer_size)
        
        # Use Q_micro as the main Q-table for compatibility
        self.Q = self.Q_micro
    
    def idx_super(self, pos):
        """Get super-level state index."""
        super_x = min(pos[0] // self.block_macro, (self.env.size // self.block_macro) - 1)
        super_y = min(pos[1] // self.block_macro, (self.env.size // self.block_macro) - 1)
        return super_x * (self.env.size // self.block_macro) + super_y
    
    def idx_macro(self, pos):
        """Get macro-level state index."""
        macro_x = min(pos[0] // self.block_micro, (self.env.size // self.block_micro) - 1)
        macro_y = min(pos[1] // self.block_micro, (self.env.size // self.block_micro) - 1)
        return macro_x * (self.env.size // self.block_micro) + macro_y
    
    def idx_micro(self, pos):
        """Get micro-level state index."""
        return pos[0] * self.env.size + pos[1]
    
    def get_state_index(self, pos):
        """Override to use micro-level indexing."""
        return self.idx_micro(pos)
    
    def _compute_shaped_reward(self, pos, goal, level='macro'):
        """Compute shaped reward based on distance to goal."""
        if self.reward_shaping == 'sparse':
            return 0  # No shaping, only terminal rewards
        
        # Distance-based reward shaping
        dist = self.env.manhattan_distance(pos, goal)
        return 0.1 * (1.0 / (dist + 1))
    
    def update_hierarchical_q_tables(self):
        """Update all hierarchical Q-tables."""
        update_from_buffer(self.Q_micro, self.micro_buffer, 
                          min(self.batch_size, len(self.micro_buffer)), 
                          self.alpha, self.gamma)
        update_from_buffer(self.Q_macro, self.macro_buffer, 
                          min(self.batch_size, len(self.macro_buffer)), 
                          self.alpha, self.gamma)
        update_from_buffer(self.Q_super, self.super_buffer, 
                          min(self.batch_size, len(self.super_buffer)), 
                          self.alpha, self.gamma)
    
    def train(self, episodes=600, horizon=500):
        """Train the hierarchical agent."""
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training Fractal Agent"):
            pos = self.env.reset()
            done = False
            primitive_steps = 0
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                # Super level: pick neighbor super-block
                s_super = self.idx_super(pos)
                a_super = choose_action(self.Q_super, s_super, epsilon)
                
                # Translate super action to desired super-block center
                target_super_block = np.add(divmod(s_super, self.env.size // self.block_macro), 
                                          self.env.actions[a_super])
                target_super_block = np.clip(target_super_block, 0, 
                                           self.env.size // self.block_macro - 1)
                super_goal = tuple(target_super_block * self.block_macro + self.block_macro // 2)
                
                # Navigate within super-block
                steps_within_super = 0
                max_steps_super = self.block_macro * self.block_macro
                super_done = False
                
                while (steps_within_super < max_steps_super and 
                       not super_done and not done):
                    
                    # Macro level: pick neighbor patch
                    s_mac = self.idx_macro(pos)
                    a_mac = choose_action(self.Q_macro, s_mac, epsilon)
                    
                    # Translate macro action to desired patch center
                    target_block = np.add(divmod(s_mac, self.env.size // self.block_micro), 
                                        self.env.actions[a_mac])
                    target_block = np.clip(target_block, 0, 
                                         self.env.size // self.block_micro - 1)
                    macro_goal = tuple(target_block * self.block_micro + self.block_micro // 2)
                    
                    # Low level: navigate to macro-goal
                    macro_done = False
                    for _ in range(self.block_micro * self.block_micro):
                        if done:
                            break
                            
                        s_mic = self.idx_micro(pos)
                        a_mic = choose_action(self.Q_micro, s_mic, epsilon)
                        
                        # Execute primitive action
                        nxt, rwd, done = self.env.step(pos, a_mic)
                        primitive_steps += 1
                        
                        s2_mic = self.idx_micro(nxt)
                        
                        # Store micro experience
                        self.micro_buffer.append((s_mic, a_mic, rwd, s2_mic, done))
                        
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
                        r_mac = -1 + self._compute_shaped_reward(pos, macro_goal, 'macro')
                    
                    s2_mac = self.idx_macro(pos)
                    self.macro_buffer.append((s_mac, a_mac, r_mac, s2_mac, done or macro_done))
                    
                    if done or pos == super_goal:
                        super_done = True
                        break
                
                # Determine super reward
                if done:
                    r_super = 10  # Goal reached
                elif pos == super_goal:
                    r_super = 5   # Super goal reached
                else:
                    r_super = -1 + self._compute_shaped_reward(pos, super_goal, 'super')
                
                s2_super = self.idx_super(pos)
                self.super_buffer.append((s_super, a_super, r_super, s2_super, done or super_done))
                
                # Update all Q-tables
                self.update_hierarchical_q_tables()
            
            log.append(primitive_steps)
            
            # Additional batch updates
            for _ in range(5):
                self.update_hierarchical_q_tables()
                
        training_time = time.time() - start_time
        return log, training_time
    
    def choose_action(self, state, epsilon):
        """Choose action at micro level for compatibility."""
        return choose_action(self.Q_micro, state, epsilon)


class FractalAttentionAgent(FractalAgent):
    """
    Fractal agent with attention mechanism for weighting hierarchical levels.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Attention weights for each level [micro, macro, super]
        self.attention_weights = np.array([1/3, 1/3, 1/3])
        self.attention_history = []
    
    def compute_attention_weights(self, pos, super_goal, macro_goal):
        """
        Dynamically compute attention weights based on current state and goals.
        """
        # Calculate distances to goals at different scales
        dist_to_super_goal = self.env.manhattan_distance(pos, super_goal)
        dist_to_macro_goal = self.env.manhattan_distance(pos, macro_goal)
        dist_to_final_goal = self.env.manhattan_distance(pos, self.env.goal)
        
        # Normalize distances
        max_dist = self.env.size * 2
        norm_dist_super = dist_to_super_goal / max_dist
        norm_dist_macro = dist_to_macro_goal / max_dist
        norm_dist_final = dist_to_final_goal / max_dist
        
        # Compute attention logits
        attention_logits = np.array([
            1.0 - norm_dist_final * 0.5,  # Micro attention
            1.0 - norm_dist_macro * 0.5,  # Macro attention  
            1.0 - norm_dist_super * 0.5   # Super attention
        ])
        
        # Boost micro attention when near obstacles
        has_nearby_obstacle = False
        for action_delta in self.env.actions.values():
            adj_pos = tuple(np.clip(np.add(pos, action_delta), 0, self.env.size - 1))
            if adj_pos in self.env.obstacles:
                has_nearby_obstacle = True
                break
        
        if has_nearby_obstacle:
            attention_logits[0] += 0.5
        
        # Apply softmax
        self.attention_weights = softmax(attention_logits)
        self.attention_history.append(self.attention_weights.copy())
        
        return self.attention_weights
    
    def choose_action_with_attention(self, pos, super_goal, macro_goal, epsilon):
        """Choose action using attention-weighted combination of Q-values."""
        # Compute attention weights
        attn_weights = self.compute_attention_weights(pos, super_goal, macro_goal)
        
        # Get Q-values from each level
        s_mic = self.idx_micro(pos)
        s_mac = self.idx_macro(pos)
        s_sup = self.idx_super(pos)
        
        q_mic = self.Q_micro[s_mic]
        q_mac = self.Q_macro[s_mac] if s_mac < self.Q_macro.shape[0] else np.zeros(4)
        q_sup = self.Q_super[s_sup] if s_sup < self.Q_super.shape[0] else np.zeros(4)
        
        # Weighted combination
        combined_q = (attn_weights[0] * q_mic + 
                     attn_weights[1] * q_mac + 
                     attn_weights[2] * q_sup)
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.num_actions)
        else:
            return combined_q.argmax()
    
    def train(self, episodes=600, horizon=500):
        """Train the attention-based fractal agent."""
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training Fractal Attention Agent"):
            pos = self.env.reset()
            done = False
            primitive_steps = 0
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                # Super level planning (same as fractal agent)
                s_super = self.idx_super(pos)
                a_super = choose_action(self.Q_super, s_super, epsilon)
                
                target_super_block = np.add(divmod(s_super, self.env.size // self.block_macro), 
                                          self.env.actions[a_super])
                target_super_block = np.clip(target_super_block, 0, 
                                           self.env.size // self.block_macro - 1)
                super_goal = tuple(target_super_block * self.block_macro + self.block_macro // 2)
                
                # Macro level planning
                s_mac = self.idx_macro(pos)
                a_mac = choose_action(self.Q_macro, s_mac, epsilon)
                
                target_block = np.add(divmod(s_mac, self.env.size // self.block_micro), 
                                    self.env.actions[a_mac])
                target_block = np.clip(target_block, 0, 
                                     self.env.size // self.block_micro - 1)
                macro_goal = tuple(target_block * self.block_micro + self.block_micro // 2)
                
                # Execute micro actions using attention mechanism
                for _ in range(self.block_micro * self.block_micro):
                    if done:
                        break
                        
                    # Use attention-weighted action selection
                    a_mic = self.choose_action_with_attention(pos, super_goal, macro_goal, epsilon)
                    
                    nxt, rwd, done = self.env.step(pos, a_mic)
                    primitive_steps += 1
                    
                    # Store experience and update as in base fractal agent
                    s_mic = self.idx_micro(pos)
                    s2_mic = self.idx_micro(nxt)
                    self.micro_buffer.append((s_mic, a_mic, rwd, s2_mic, done))
                    
                    pos = nxt
                    
                    if done or pos == macro_goal:
                        break
                
                # Update all Q-tables (macro and super level updates similar to base fractal)
                # Determine macro reward
                if done:
                    r_mac = 10
                elif pos == macro_goal:
                    r_mac = 5
                elif pos == super_goal:
                    r_mac = 3
                else:
                    r_mac = -1 + self._compute_shaped_reward(pos, macro_goal, 'macro')
                
                s2_mac = self.idx_macro(pos)
                self.macro_buffer.append((s_mac, a_mac, r_mac, s2_mac, done))
                
                # Determine super reward  
                if done:
                    r_super = 10
                elif pos == super_goal:
                    r_super = 5
                else:
                    r_super = -1 + self._compute_shaped_reward(pos, super_goal, 'super')
                
                s2_super = self.idx_super(pos)
                self.super_buffer.append((s_super, a_super, r_super, s2_super, done))
                
                self.update_hierarchical_q_tables()
            
            log.append(primitive_steps)
            
            # Additional batch updates
            for _ in range(5):
                self.update_hierarchical_q_tables()
                
        training_time = time.time() - start_time
        return log, training_time 