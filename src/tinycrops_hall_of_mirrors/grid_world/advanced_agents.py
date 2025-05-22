"""
Advanced grid-world agents with novel approaches and innovations.

This module contains experimental agents that push beyond traditional hierarchical RL:
- Adaptive hierarchical structures
- Curiosity-driven exploration
- Multi-head attention mechanisms
- Meta-learning capabilities
"""

import numpy as np
import time
from tqdm import trange
from collections import defaultdict, deque
from .agents import FractalAgent, FractalAttentionAgent
from ..common.q_learning_utils import (
    choose_action, update_from_buffer, create_experience_buffer, 
    softmax, decay_epsilon
)


class AdaptiveFractalAgent(FractalAgent):
    """
    Fractal agent with adaptive hierarchical structure.
    
    Novel Innovation: The agent can dynamically adjust its block sizes
    based on performance and environment complexity.
    """
    
    def __init__(self, env, min_block_size=3, max_block_size=8, 
                 adaptation_rate=0.1, **kwargs):
        """
        Initialize with adaptive parameters.
        
        Args:
            min_block_size: Minimum hierarchical block size
            max_block_size: Maximum hierarchical block size
            adaptation_rate: Rate of adaptation for block sizes
        """
        # Start with default block sizes
        super().__init__(env, block_micro=min_block_size, 
                        block_macro=min_block_size*2, **kwargs)
        
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking for adaptation
        self.performance_history = deque(maxlen=50)
        self.adaptation_episode = 0
        
    def adapt_hierarchy(self):
        """
        Adapt hierarchical structure based on recent performance.
        
        Novel approach: Use performance variance to determine optimal granularity.
        High variance suggests need for finer control (smaller blocks).
        Low variance suggests larger blocks might be more efficient.
        """
        if len(self.performance_history) < 20:
            return  # Need enough data
            
        # Calculate performance metrics
        recent_perf = list(self.performance_history)[-10:]
        variance = np.var(recent_perf)
        mean_perf = np.mean(recent_perf)
        
        # Adaptation logic
        if variance > mean_perf * 0.5:  # High variance - need finer control
            new_micro = max(self.min_block_size, self.block_micro - 1)
        elif variance < mean_perf * 0.1:  # Low variance - can use coarser control
            new_micro = min(self.max_block_size, self.block_micro + 1)
        else:
            new_micro = self.block_micro  # No change
            
        new_macro = min(new_micro * 2, self.env.size // 2)
        
        # Apply adaptation if significant change
        if new_micro != self.block_micro:
            print(f"  Adapting hierarchy: micro {self.block_micro}->{new_micro}, "
                  f"macro {self.block_macro}->{new_macro}")
            
            self.block_micro = new_micro
            self.block_macro = new_macro
            
            # Rebuild Q-tables with new dimensions
            self._rebuild_q_tables()
    
    def _rebuild_q_tables(self):
        """Rebuild Q-tables when hierarchy changes."""
        old_micro_Q = self.Q_micro.copy()
        old_macro_Q = self.Q_macro.copy()
        old_super_Q = self.Q_super.copy()
        
        # Recalculate state spaces
        self.super_states = (self.env.size // self.block_macro) ** 2
        self.macro_states = (self.env.size // self.block_micro) ** 2
        
        # Initialize new Q-tables
        self.Q_super = np.zeros((self.super_states, self.env.num_actions))
        self.Q_macro = np.zeros((self.macro_states, self.env.num_actions))
        self.Q_micro = np.zeros((self.micro_states, self.env.num_actions))
        
        # Transfer knowledge from old Q-tables using nearest neighbor mapping
        self._transfer_q_knowledge(old_micro_Q, old_macro_Q, old_super_Q)
        
        # Update main Q reference
        self.Q = self.Q_micro
    
    def _transfer_q_knowledge(self, old_micro_Q, old_macro_Q, old_super_Q):
        """Transfer knowledge from old Q-tables to new ones."""
        # For micro level - direct transfer (same state space)
        min_states = min(old_micro_Q.shape[0], self.Q_micro.shape[0])
        self.Q_micro[:min_states] = old_micro_Q[:min_states]
        
        # For macro and super - interpolate/aggregate
        # This is a simplified transfer - could be more sophisticated
        if old_macro_Q.shape[0] <= self.Q_macro.shape[0]:
            # Expanding - interpolate
            ratio = self.Q_macro.shape[0] / old_macro_Q.shape[0]
            for i in range(old_macro_Q.shape[0]):
                new_idx = min(int(i * ratio), self.Q_macro.shape[0] - 1)
                self.Q_macro[new_idx] = old_macro_Q[i]
        else:
            # Contracting - aggregate
            ratio = old_macro_Q.shape[0] / self.Q_macro.shape[0]
            for i in range(self.Q_macro.shape[0]):
                old_start = int(i * ratio)
                old_end = min(int((i + 1) * ratio), old_macro_Q.shape[0])
                self.Q_macro[i] = np.mean(old_macro_Q[old_start:old_end], axis=0)
    
    def train(self, episodes=600, horizon=500):
        """Train with adaptive hierarchy."""
        log, training_time = super().train(episodes, horizon)
        
        # Update performance history and adapt
        for steps in log:
            self.performance_history.append(steps)
            
        # Periodic adaptation
        if len(log) % 50 == 0 and len(log) > 0:
            self.adapt_hierarchy()
            
        return log, training_time


class CuriosityDrivenAgent(FractalAttentionAgent):
    """
    Fractal attention agent with curiosity-driven exploration.
    
    Novel Innovation: Uses intrinsic motivation based on prediction error
    to drive exploration in addition to extrinsic rewards.
    """
    
    def __init__(self, env, curiosity_weight=0.1, prediction_lr=0.01, **kwargs):
        super().__init__(env, **kwargs)
        
        self.curiosity_weight = curiosity_weight
        self.prediction_lr = prediction_lr
        
        # Simple forward model for prediction
        self.state_prediction_errors = defaultdict(list)
        self.state_visit_counts = defaultdict(int)
        
        # Intrinsic reward history
        self.intrinsic_rewards = []
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward based on prediction error and novelty.
        
        Novel approach: Combines state prediction error with visit count
        to encourage exploration of both unpredictable and novel states.
        """
        state_key = tuple(state) if hasattr(state, '__iter__') else state
        
        # Novelty bonus (inverse of visit count)
        novelty_bonus = 1.0 / (1 + self.state_visit_counts[state_key])
        
        # Prediction error (simplified - could use actual neural network)
        expected_next = self._predict_next_state(state, action)
        prediction_error = abs(expected_next - next_state) if isinstance(next_state, (int, float)) else 0.1
        
        # Combine novelty and prediction error
        intrinsic_reward = self.curiosity_weight * (novelty_bonus + prediction_error)
        
        # Update state tracking
        self.state_visit_counts[state_key] += 1
        self.state_prediction_errors[state_key].append(prediction_error)
        self.intrinsic_rewards.append(intrinsic_reward)
        
        return intrinsic_reward
    
    def _predict_next_state(self, state, action):
        """Simple state prediction model."""
        # For grid world, this could be more sophisticated
        # Here we use a simple heuristic
        if hasattr(state, '__iter__'):
            return state + action * 0.1  # Simplified
        return state + action * 0.1
    
    def train(self, episodes=600, horizon=500):
        """Train with curiosity-driven exploration."""
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training Curiosity-Driven Agent"):
            pos = self.env.reset()
            done = False
            primitive_steps = 0
            episode_intrinsic_reward = 0
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                # Standard hierarchical planning
                s_super = self.idx_super(pos)
                a_super = choose_action(self.Q_super, s_super, epsilon)
                
                target_super_block = np.add(divmod(s_super, self.env.size // self.block_macro), 
                                          self.env.actions[a_super])
                target_super_block = np.clip(target_super_block, 0, 
                                           self.env.size // self.block_macro - 1)
                super_goal = tuple(target_super_block * self.block_macro + self.block_macro // 2)
                
                s_mac = self.idx_macro(pos)
                a_mac = choose_action(self.Q_macro, s_mac, epsilon)
                
                target_block = np.add(divmod(s_mac, self.env.size // self.block_micro), 
                                    self.env.actions[a_mac])
                target_block = np.clip(target_block, 0, 
                                     self.env.size // self.block_micro - 1)
                macro_goal = tuple(target_block * self.block_micro + self.block_micro // 2)
                
                # Execute micro actions with curiosity-driven exploration
                for _ in range(self.block_micro * self.block_micro):
                    if done:
                        break
                        
                    s_mic = self.idx_micro(pos)
                    
                    # Choose action with attention mechanism
                    a_mic = self.choose_action_with_attention(pos, super_goal, macro_goal, epsilon)
                    
                    nxt, extrinsic_reward, done = self.env.step(pos, a_mic)
                    primitive_steps += 1
                    
                    # Compute intrinsic reward
                    intrinsic_reward = self.compute_intrinsic_reward(s_mic, a_mic, self.idx_micro(nxt))
                    
                    # Combine extrinsic and intrinsic rewards
                    total_reward = extrinsic_reward + intrinsic_reward
                    episode_intrinsic_reward += intrinsic_reward
                    
                    s2_mic = self.idx_micro(nxt)
                    self.micro_buffer.append((s_mic, a_mic, total_reward, s2_mic, done))
                    
                    pos = nxt
                    
                    if done or pos == macro_goal:
                        break
                
                # Update Q-tables (same as parent class)
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
        
        print(f"  Average intrinsic reward per episode: {np.mean(self.intrinsic_rewards):.4f}")
        print(f"  States explored: {len(self.state_visit_counts)}")
        
        return log, training_time


class MultiHeadAttentionAgent(FractalAttentionAgent):
    """
    Fractal agent with multi-head attention mechanism.
    
    Novel Innovation: Uses multiple attention heads to focus on different
    aspects of the hierarchical state simultaneously.
    """
    
    def __init__(self, env, num_heads=3, **kwargs):
        super().__init__(env, **kwargs)
        
        self.num_heads = num_heads
        
        # Multi-head attention weights [heads, levels]
        self.multi_attention_weights = np.ones((num_heads, 3)) / 3
        self.attention_head_history = []
        
    def compute_multi_head_attention(self, pos, super_goal, macro_goal):
        """
        Compute attention weights for multiple heads focusing on different aspects.
        
        Novel approach: Each head specializes in different environmental factors:
        - Head 0: Distance-based attention
        - Head 1: Obstacle-based attention  
        - Head 2: Goal-progress attention
        """
        attention_heads = []
        
        # Head 0: Distance-based attention
        dist_to_super_goal = self.env.manhattan_distance(pos, super_goal)
        dist_to_macro_goal = self.env.manhattan_distance(pos, macro_goal)
        dist_to_final_goal = self.env.manhattan_distance(pos, self.env.goal)
        
        max_dist = self.env.size * 2
        dist_logits = np.array([
            1.0 - dist_to_final_goal / max_dist,
            1.0 - dist_to_macro_goal / max_dist,
            1.0 - dist_to_super_goal / max_dist
        ])
        attention_heads.append(softmax(dist_logits))
        
        # Head 1: Obstacle-based attention
        obstacle_complexity = self._compute_local_obstacle_complexity(pos)
        obstacle_logits = np.array([
            obstacle_complexity,  # High complexity needs micro attention
            obstacle_complexity * 0.5,  # Medium for macro
            0.1  # Low for super
        ])
        attention_heads.append(softmax(obstacle_logits))
        
        # Head 2: Goal-progress attention
        progress_to_goal = 1.0 - (dist_to_final_goal / (self.env.size * 2))
        progress_logits = np.array([
            progress_to_goal,  # High progress needs fine control
            (1 - progress_to_goal) * 0.8,  # Low progress can use macro
            (1 - progress_to_goal)  # Very low progress can use super
        ])
        attention_heads.append(softmax(progress_logits))
        
        # Store for analysis
        self.multi_attention_weights = np.array(attention_heads)
        self.attention_head_history.append(self.multi_attention_weights.copy())
        
        return self.multi_attention_weights
    
    def _compute_local_obstacle_complexity(self, pos):
        """Compute local obstacle density around current position."""
        complexity = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (pos[0] + dx, pos[1] + dy)
                if (check_pos in self.env.obstacles or 
                    check_pos[0] < 0 or check_pos[0] >= self.env.size or
                    check_pos[1] < 0 or check_pos[1] >= self.env.size):
                    complexity += 1
        return complexity / 9.0
    
    def choose_action_with_multi_head_attention(self, pos, super_goal, macro_goal, epsilon):
        """Choose action using multi-head attention mechanism."""
        # Compute multi-head attention
        attention_heads = self.compute_multi_head_attention(pos, super_goal, macro_goal)
        
        # Get Q-values from each level
        s_mic = self.idx_micro(pos)
        s_mac = self.idx_macro(pos)
        s_sup = self.idx_super(pos)
        
        q_mic = self.Q_micro[s_mic]
        q_mac = self.Q_macro[s_mac] if s_mac < self.Q_macro.shape[0] else np.zeros(4)
        q_sup = self.Q_super[s_sup] if s_sup < self.Q_super.shape[0] else np.zeros(4)
        
        q_values = np.array([q_mic, q_mac, q_sup])
        
        # Combine using multi-head attention
        combined_q = np.zeros(4)
        for head in range(self.num_heads):
            head_contribution = np.sum(attention_heads[head][:, None] * q_values, axis=0)
            combined_q += head_contribution / self.num_heads
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.num_actions)
        else:
            return combined_q.argmax()
    
    def train(self, episodes=600, horizon=500):
        """Train with multi-head attention mechanism."""
        log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        for ep in trange(episodes, desc="Training Multi-Head Attention Agent"):
            pos = self.env.reset()
            done = False
            primitive_steps = 0
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            for t in range(horizon):
                if done:
                    break
                    
                # Hierarchical planning (same as parent)
                s_super = self.idx_super(pos)
                a_super = choose_action(self.Q_super, s_super, epsilon)
                
                target_super_block = np.add(divmod(s_super, self.env.size // self.block_macro), 
                                          self.env.actions[a_super])
                target_super_block = np.clip(target_super_block, 0, 
                                           self.env.size // self.block_macro - 1)
                super_goal = tuple(target_super_block * self.block_macro + self.block_macro // 2)
                
                s_mac = self.idx_macro(pos)
                a_mac = choose_action(self.Q_macro, s_mac, epsilon)
                
                target_block = np.add(divmod(s_mac, self.env.size // self.block_micro), 
                                    self.env.actions[a_mac])
                target_block = np.clip(target_block, 0, 
                                     self.env.size // self.block_micro - 1)
                macro_goal = tuple(target_block * self.block_micro + self.block_micro // 2)
                
                # Execute micro actions with multi-head attention
                for _ in range(self.block_micro * self.block_micro):
                    if done:
                        break
                        
                    # Use multi-head attention for action selection
                    a_mic = self.choose_action_with_multi_head_attention(pos, super_goal, macro_goal, epsilon)
                    
                    nxt, rwd, done = self.env.step(pos, a_mic)
                    primitive_steps += 1
                    
                    s_mic = self.idx_micro(pos)
                    s2_mic = self.idx_micro(nxt)
                    self.micro_buffer.append((s_mic, a_mic, rwd, s2_mic, done))
                    
                    pos = nxt
                    
                    if done or pos == macro_goal:
                        break
                
                # Update Q-tables (same as parent class)
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


class MetaLearningAgent(FractalAttentionAgent):
    """
    Meta-learning agent that adapts its learning strategy based on environment characteristics.
    
    Novel Innovation: Uses few-shot learning to quickly adapt to new environments
    and maintains a library of successful strategies.
    """
    
    def __init__(self, env, meta_lr=0.01, strategy_memory_size=100, **kwargs):
        super().__init__(env, **kwargs)
        
        self.meta_lr = meta_lr
        self.strategy_memory_size = strategy_memory_size
        
        # Strategy library
        self.strategy_library = []
        self.current_strategy = None
        
        # Environment characteristics
        self.env_characteristics = {}
        
    def analyze_environment(self):
        """
        Analyze current environment to determine its characteristics.
        
        Novel approach: Creates a signature of the environment based on
        obstacle density, spatial distribution, and connectivity.
        """
        obstacles = self.env.obstacles
        total_cells = self.env.size ** 2
        
        characteristics = {
            'obstacle_density': len(obstacles) / total_cells,
            'connectivity': self._compute_connectivity(),
            'obstacle_clustering': self._compute_clustering(),
            'path_complexity': self._estimate_path_complexity()
        }
        
        self.env_characteristics = characteristics
        return characteristics
    
    def _compute_connectivity(self):
        """Compute how connected the free space is."""
        # Simple connectivity measure - could use proper graph analysis
        free_cells = self.env.size ** 2 - len(self.env.obstacles)
        return free_cells / (self.env.size ** 2)
    
    def _compute_clustering(self):
        """Compute how clustered the obstacles are."""
        if not self.env.obstacles:
            return 0
            
        # Measure average distance between obstacles
        obstacles = list(self.env.obstacles)
        total_dist = 0
        count = 0
        
        for i, obs1 in enumerate(obstacles):
            for obs2 in obstacles[i+1:]:
                total_dist += self.env.manhattan_distance(obs1, obs2)
                count += 1
                
        avg_dist = total_dist / max(count, 1)
        max_possible_dist = self.env.size * 2
        
        return 1.0 - (avg_dist / max_possible_dist)  # Higher = more clustered
    
    def _estimate_path_complexity(self):
        """Estimate complexity of optimal path to goal."""
        # A* or simple heuristic to estimate path complexity
        return len(self.env.obstacles) / (self.env.size ** 2) * 2.0
    
    def select_strategy(self):
        """
        Select best strategy from library based on environment characteristics.
        
        Novel approach: Uses similarity matching to find the most appropriate
        learned strategy for the current environment.
        """
        if not self.strategy_library:
            return self._create_default_strategy()
            
        current_chars = self.env_characteristics
        best_strategy = None
        best_similarity = -1
        
        for strategy in self.strategy_library:
            similarity = self._compute_similarity(current_chars, strategy['characteristics'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_strategy = strategy
                
        if best_similarity > 0.7:  # High similarity threshold
            return best_strategy
        else:
            return self._create_default_strategy()
    
    def _compute_similarity(self, chars1, chars2):
        """Compute similarity between two environment characteristic sets."""
        total_diff = 0
        for key in chars1:
            if key in chars2:
                total_diff += abs(chars1[key] - chars2[key])
                
        return 1.0 - (total_diff / len(chars1))
    
    def _create_default_strategy(self):
        """Create a default strategy for new environments."""
        return {
            'characteristics': self.env_characteristics.copy(),
            'block_micro': 5,
            'block_macro': 10,
            'learning_rate': 0.2,
            'exploration_weight': 1.0,
            'performance': []
        }
    
    def update_strategy(self, performance):
        """Update current strategy based on performance."""
        if self.current_strategy is not None:
            self.current_strategy['performance'].append(performance)
            
            # Add to library if performance is good
            avg_performance = np.mean(self.current_strategy['performance'])
            if avg_performance < 200:  # Good performance threshold
                self._add_to_library(self.current_strategy)
    
    def _add_to_library(self, strategy):
        """Add strategy to library, maintaining size limit."""
        # Avoid duplicates
        for existing in self.strategy_library:
            if self._compute_similarity(strategy['characteristics'], 
                                      existing['characteristics']) > 0.9:
                return
                
        self.strategy_library.append(strategy.copy())
        
        # Maintain size limit
        if len(self.strategy_library) > self.strategy_memory_size:
            # Remove worst performing strategy
            self.strategy_library.sort(key=lambda s: np.mean(s['performance']), reverse=True)
            self.strategy_library = self.strategy_library[:self.strategy_memory_size]
    
    def train(self, episodes=600, horizon=500):
        """Train with meta-learning strategy adaptation."""
        # Analyze environment and select strategy
        self.analyze_environment()
        self.current_strategy = self.select_strategy()
        
        print(f"  Selected strategy for environment with {self.env_characteristics['obstacle_density']:.2f} obstacle density")
        
        # Adapt agent parameters based on strategy
        if self.current_strategy:
            self.block_micro = self.current_strategy.get('block_micro', 5)
            self.block_macro = self.current_strategy.get('block_macro', 10)
            self.alpha = self.current_strategy.get('learning_rate', 0.2)
        
        # Train with selected strategy
        log, training_time = super().train(episodes, horizon)
        
        # Update strategy based on performance
        avg_performance = np.mean(log[-20:])  # Last 20 episodes
        self.update_strategy(avg_performance)
        
        print(f"  Strategy performance: {avg_performance:.1f} steps (library size: {len(self.strategy_library)})")
        
        return log, training_time


class CuriosityDrivenAdaptiveAgent(AdaptiveFractalAgent):
    """
    Advanced agent combining curiosity-driven exploration with adaptive hierarchy.
    
    Novel Innovation: Combines intrinsic motivation through curiosity with
    dynamic hierarchical structure adaptation for maximum exploration efficiency.
    """
    
    def __init__(self, env, curiosity_weight=0.1, prediction_lr=0.01, 
                 novelty_threshold=0.8, **kwargs):
        super().__init__(env, **kwargs)
        
        self.curiosity_weight = curiosity_weight
        self.prediction_lr = prediction_lr
        self.novelty_threshold = novelty_threshold
        
        # Curiosity components
        self.visit_counts = np.zeros(self.env.state_space_size)
        self.prediction_errors = []
        self.novelty_map = np.zeros((env.size, env.size))
        
        # Adaptive components from parent class are inherited
        # Initialize hierarchy tracking
        self.hierarchy_history = []
        
    def get_novelty_bonus(self, state_idx):
        """Compute novelty bonus for curiosity-driven exploration."""
        visit_count = self.visit_counts[state_idx]
        novelty = 1.0 / np.sqrt(visit_count + 1)
        return self.curiosity_weight * novelty
        
    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward combining curiosity and prediction error.
        """
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        # Novelty bonus
        novelty_bonus = self.get_novelty_bonus(state_idx)
        
        # Prediction error (simplified)
        predicted_next = self._predict_next_state(state, action)
        prediction_error = abs(predicted_next - next_state_idx) / self.env.state_space_size
        
        # Update prediction model (simple running average)
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 1000:
            self.prediction_errors.pop(0)
            
        # Update novelty map
        x, y = state
        self.novelty_map[x, y] = novelty_bonus
        
        return novelty_bonus + prediction_error * 0.5
        
    def _predict_next_state(self, state, action):
        """Simple state prediction for intrinsic motivation."""
        # Simplified prediction - could use neural network
        action_delta = list(self.env.actions.values())[action]
        predicted_pos = (state[0] + action_delta[0], state[1] + action_delta[1])
        
        # Bound to environment
        predicted_pos = (
            max(0, min(self.env.size - 1, predicted_pos[0])),
            max(0, min(self.env.size - 1, predicted_pos[1]))
        )
        
        return self.get_state_index(predicted_pos)
        
    def update_micro_q_table(self):
        """Enhanced micro Q-table update with curiosity bonuses."""
        super().update_micro_q_table()
        
        # Additional curiosity-based updates
        if hasattr(self, 'micro_buffer') and self.micro_buffer:
            for s, a, r, s2, done in self.micro_buffer[-10:]:  # Last 10 transitions
                # Add intrinsic reward
                pos = self.get_position_from_state_index(s)
                next_pos = self.get_position_from_state_index(s2)
                intrinsic_r = self.compute_intrinsic_reward(pos, a, next_pos)
                
                # Update with combined reward
                total_reward = r + intrinsic_r
                old_q = self.Q_micro[s, a]
                max_next_q = np.max(self.Q_micro[s2, :])
                self.Q_micro[s, a] = old_q + self.alpha * (total_reward + self.gamma * max_next_q - old_q)
                
    def get_position_from_state_index(self, state_idx):
        """Convert state index back to position."""
        return (state_idx // self.env.size, state_idx % self.env.size)
        
    def adapt_hierarchy_with_curiosity(self):
        """
        Adapt hierarchy based on both performance and exploration efficiency.
        """
        # Get base adaptation from parent
        self.adapt_hierarchy()
        
        # Additional curiosity-based adaptation
        if len(self.performance_history) > 20:
            recent_performance = list(self.performance_history)[-10:]
            older_performance = list(self.performance_history)[-20:-10]
            
            # Check exploration efficiency
            recent_novelty = np.mean([np.max(self.novelty_map) for _ in recent_performance])
            
            # If we're not exploring well, try smaller blocks for more granular control
            if recent_novelty < self.novelty_threshold:
                if self.block_micro > self.min_block_size:
                    self.block_micro = max(self.min_block_size, self.block_micro - 1)
                    self.hierarchy_history.append({
                        'episode': len(self.performance_history),
                        'block_micro': self.block_micro,
                        'block_macro': self.block_macro,
                        'trigger': 'low_exploration',
                        'performance': recent_performance[-1]
                    })
                    print(f"    Adapted hierarchy for exploration: micro={self.block_micro}")
                    self._rebuild_q_tables()
                    
    def train(self, episodes=600, horizon=500):
        """Train with both curiosity and adaptive hierarchy."""
        start_time = time.time()
        log = []
        epsilon = self.epsilon_start  # Initialize epsilon
        
        print(f"Training CuriosityDrivenAdaptiveAgent with curiosity_weight={self.curiosity_weight}")
        print(f"Initial hierarchy: micro={self.block_micro}, macro={self.block_macro}")
        
        for episode in range(episodes):
            pos = self.env.start
            step = 0
            
            while pos != self.env.goal and step < horizon:
                # Update visit counts for curiosity
                state_idx = self.get_state_index(pos)
                self.visit_counts[state_idx] += 1
                
                # Enhanced epsilon-greedy with curiosity bonuses
                if np.random.random() < epsilon:
                    action = np.random.choice(len(self.env.actions))
                else:
                    # Combine Q-values with curiosity bonuses
                    base_q_values = self.Q_micro[state_idx]
                    curiosity_bonuses = np.array([
                        self.get_novelty_bonus(self.get_state_index(self.env.step(pos, a)[0]))
                        for a in range(len(self.env.actions))
                    ])
                    combined_values = base_q_values + curiosity_bonuses * 0.5
                    action = np.argmax(combined_values)
                
                next_pos, _, _ = self.env.step(pos, action)
                
                # Compute rewards
                reward = 100 if next_pos == self.env.goal else -1
                intrinsic_reward = self.compute_intrinsic_reward(pos, action, next_pos)
                total_reward = reward + intrinsic_reward
                
                # Store transitions (similar to parent class)
                done = (next_pos == self.env.goal)
                
                # Update Q-tables with curiosity
                state_idx = self.get_state_index(pos)
                next_state_idx = self.get_state_index(next_pos)
                
                old_q = self.Q_micro[state_idx, action]
                max_next_q = np.max(self.Q_micro[next_state_idx])
                self.Q_micro[state_idx, action] = old_q + self.alpha * (
                    total_reward + self.gamma * max_next_q - old_q
                )
                
                pos = next_pos
                step += 1
                
                if done:
                    break
                    
            log.append(step)
            self.performance_history.append(step)
            
            # Adapt hierarchy with curiosity considerations
            if episode > 50 and episode % 25 == 0:
                self.adapt_hierarchy_with_curiosity()
                
            # Decay epsilon
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            
            if episode % 100 == 0:
                avg_performance = np.mean(log[-20:])
                avg_novelty = np.mean(self.novelty_map[self.novelty_map > 0])
                print(f"  Episode {episode}: avg_steps={avg_performance:.1f}, "
                      f"exploration_rate={avg_novelty:.3f}, "
                      f"hierarchy=({self.block_micro},{self.block_macro})")
                      
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Final hierarchy: micro={self.block_micro}, macro={self.block_macro}")
        print(f"Total hierarchy adaptations: {len(self.hierarchy_history)}")
        
        return log 