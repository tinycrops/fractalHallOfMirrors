"""
Fractal Depth Environment for testing self-observation and multi-scale awareness.

This environment implements a true fractal structure where entering portals
leads to nested, scaled versions of the same environment. The hypothesis is
that agents capable of observing themselves from different scales will
develop enhanced awareness and transferable knowledge.
"""

import numpy as np
import random
from collections import deque


class FractalDepthEnvironment:
    """
    An environment with fractal, nested depths.
    Entering a "portal" cell at one depth leads to a new instance of the
    entire environment, conceptually scaled within that portal.
    
    Key innovation: Agent can observe itself from different scales,
    potentially leading to enhanced awareness and knowledge transfer.
    """
    
    def __init__(self, base_size=15, num_portals=1, max_depth=2, seed=0):
        self.base_size = base_size
        self.num_portals_per_level = num_portals
        self.max_depth = max_depth
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.actions = {
            0: (-1, 0), 1: (1, 0),  # UP, DOWN
            2: (0, -1), 3: (0, 1)   # LEFT, RIGHT
        }

        # Create base layout that will be replicated at all fractal depths
        self.base_obstacles = self._create_base_obstacles()
        self.base_goal = (base_size - 2, base_size - 2)  # Ensure goal is not on edge
        if self.base_goal in self.base_obstacles:
            self.base_obstacles.remove(self.base_goal)
        if (0, 0) in self.base_obstacles:  # Ensure start is clear
            self.base_obstacles.remove((0, 0))

        self.base_portal_coords = self._select_portal_locations(self.base_obstacles, self.base_goal)
        if not self.base_portal_coords and self.num_portals_per_level > 0:
            # Failsafe portal placement
            self.base_portal_coords = [(base_size // 2, base_size // 2)]
            if self.base_portal_coords[0] in self.base_obstacles:
                self.base_obstacles.discard(self.base_portal_coords[0])
            if self.base_portal_coords[0] == self.base_goal:
                self.base_portal_coords = [(base_size // 2 + 1, base_size // 2)]

        # Agent state: position and depth
        self.current_pos = (0, 0)
        self.current_depth = 0
        # Stack tracking portal entry path for returning to parent levels
        self.entry_portal_path = []

        print(f"FractalDepthEnv: Base Size={base_size}, Portals/Level={self.num_portals_per_level}, Max Depth={max_depth}")
        print(f"  Base Obstacles: {len(self.base_obstacles)}")
        print(f"  Base Goal (Depth 0): {self.base_goal}")
        print(f"  Base Portal Coords: {self.base_portal_coords}")

    def _create_base_obstacles(self):
        """Create obstacles that will appear at all fractal depths."""
        obstacles = set()
        
        # Create an interesting maze pattern that will be self-similar
        # Vertical walls with gaps
        for i in range(self.base_size):
            if i % 4 == 1:
                for j in range(self.base_size // 4, 3 * self.base_size // 4):
                    obstacles.add((j, i))
            if i % 4 == 3:
                for j in range(self.base_size // 4, 3 * self.base_size // 4):
                    obstacles.add((i, j))
        
        # Clear central area for portal placement
        center_x, center_y = self.base_size // 2, self.base_size // 2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                obstacles.discard((center_x + dx, center_y + dy))
        
        return obstacles

    def _select_portal_locations(self, obstacles, goal_pos):
        """Select locations for portals that lead to deeper fractal levels."""
        portals = []
        candidate_locations = []
        
        for r in range(1, self.base_size - 1):  # Avoid edges
            for c in range(1, self.base_size - 1):
                if (r, c) not in obstacles and (r, c) != goal_pos and (r, c) != (0, 0):
                    candidate_locations.append((r, c))
        
        if not candidate_locations:
            return [(self.base_size // 2, self.base_size // 2)]

        if len(candidate_locations) < self.num_portals_per_level:
            return random.sample(candidate_locations, len(candidate_locations))
        
        return random.sample(candidate_locations, self.num_portals_per_level)

    def get_current_layout_elements(self):
        """Get obstacles, goal, and portals for current depth.
        Key insight: Same layout at all depths - this is the fractal self-similarity."""
        return self.base_obstacles, self.base_goal, self.base_portal_coords

    def reset(self):
        """Reset environment to starting state."""
        self.current_pos = (0, 0)
        self.current_depth = 0
        self.entry_portal_path = []
        return self.get_state()

    def get_state(self):
        """Get current state as (x, y, depth) tuple."""
        return (self.current_pos[0], self.current_pos[1], self.current_depth)

    def step(self, action_idx):
        """
        Execute action in the fractal environment.
        
        Returns:
            next_state: (x, y, depth)
            reward: float
            done: bool
            info: dict with action type and depth information
        """
        obstacles, goal_pos, portal_coords = self.get_current_layout_elements()
        reward = -0.01  # Small step cost
        done = False
        info = {
            'action_type': 'move',
            'prev_depth': self.current_depth,
            'new_depth': self.current_depth
        }

        ax, ay = self.actions[action_idx]
        prev_pos = self.current_pos
        
        next_x = self.current_pos[0] + ax
        next_y = self.current_pos[1] + ay

        # Check for fractal edge transition (zoom out)
        if not (0 <= next_x < self.base_size and 0 <= next_y < self.base_size):
            if self.current_depth > 0:
                # Exit to parent level - emerge at portal that was entered
                parent_portal_x, parent_portal_y, _ = self.entry_portal_path.pop()
                self.current_pos = (parent_portal_x, parent_portal_y)
                self.current_depth -= 1
                reward += 1.0  # Reward for successful depth navigation
                info['action_type'] = 'zoom_out'
                info['new_depth'] = self.current_depth
                print(f"  Zoomed out to depth {self.current_depth}, pos {self.current_pos}")
            else:
                # Hit boundary at depth 0
                self.current_pos = prev_pos
                reward -= 0.5
        elif (next_x, next_y) in obstacles:
            # Hit obstacle
            self.current_pos = prev_pos
            reward -= 0.5
        else:
            # Valid move
            self.current_pos = (next_x, next_y)
            
            # Check for portal entry (zoom in)
            if self.current_pos in portal_coords and self.current_depth < self.max_depth:
                portal_idx = portal_coords.index(self.current_pos)
                
                # Enter deeper fractal level
                self.entry_portal_path.append((self.current_pos[0], self.current_pos[1], portal_idx))
                self.current_depth += 1
                self.current_pos = (0, 0)  # Start at origin of new fractal level
                reward += 2.0  # Reward for exploring deeper
                info['action_type'] = 'zoom_in'
                info['new_depth'] = self.current_depth
                print(f"  Zoomed in to depth {self.current_depth}")

        # Check for goal achievement
        if self.current_pos == goal_pos:
            if self.current_depth == 0:
                # Global goal at base level
                reward += 100.0
                done = True
                print(f"  Reached global goal!")
            else:
                # Fractal sub-goal at deeper level
                reward += 10.0
                print(f"  Reached fractal sub-goal at depth {self.current_depth}")

        info['current_pos'] = self.current_pos
        info['current_depth'] = self.current_depth
        
        return self.get_state(), reward, done, info

    @property
    def num_actions(self):
        """Number of possible actions."""
        return len(self.actions)

    def get_observation_perspective(self):
        """
        Get observation from current fractal perspective.
        This could be expanded to include multi-scale observations.
        """
        obs = {
            'position': self.current_pos,
            'depth': self.current_depth,
            'scale_factor': 1.0 / (2.0 ** self.current_depth),  # Scale decreases with depth
            'parent_positions': [pos[:2] for pos in self.entry_portal_path],  # Path through fractal
            'can_zoom_deeper': self.current_depth < self.max_depth and self.current_pos in self.base_portal_coords
        }
        return obs


class SelfObservingAgent:
    """
    Agent that can observe itself from multiple fractal perspectives.
    
    This agent leverages the fractal environment to potentially develop
    enhanced awareness through multi-scale self-observation.
    """
    
    def __init__(self, env: FractalDepthEnvironment, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-tables for each depth level
        self.q_tables = [
            np.zeros((env.base_size, env.base_size, env.num_actions))
            for _ in range(env.max_depth + 1)
        ]
        
        # Self-observation memory for potential cross-scale learning
        self.observation_memory = deque(maxlen=1000)
        self.cross_scale_experiences = deque(maxlen=500)
        
        print(f"SelfObservingAgent: {len(self.q_tables)} Q-tables for depths 0-{env.max_depth}")

    def choose_action(self, state_tuple):
        """Choose action based on current state and depth."""
        x, y, depth = state_tuple
        if random.random() < self.epsilon:
            return random.choice(list(self.env.actions.keys()))
        else:
            return np.argmax(self.q_tables[depth][int(x), int(y), :])

    def learn_from_experience(self, state_tuple, action, reward, next_state_tuple, done):
        """Learn from experience, potentially transferring knowledge across scales."""
        x, y, depth = state_tuple
        nx, ny, ndepth = next_state_tuple
        
        # Ensure integer indexing
        x, y, nx, ny = int(x), int(y), int(nx), int(ny)

        # Standard Q-learning update
        current_q = self.q_tables[depth][x, y, action]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_tables[ndepth][nx, ny, :])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_tables[depth][x, y, action] = new_q
        
        # Store cross-scale experience for potential knowledge transfer
        if depth != ndepth:  # Depth change occurred
            self.cross_scale_experiences.append({
                'from_state': (x, y, depth),
                'to_state': (nx, ny, ndepth),
                'action': action,
                'reward': reward,
                'transition_type': 'zoom_in' if ndepth > depth else 'zoom_out'
            })
        
        # Enhanced: Cross-scale Q-value sharing for knowledge transfer
        self._apply_cross_scale_learning(x, y, depth, action, new_q)

    def _apply_cross_scale_learning(self, x, y, depth, action, q_value):
        """Apply knowledge transfer across fractal scales."""
        transfer_rate = 0.1  # Rate of cross-scale learning
        
        # Share Q-values with other depths for same spatial position
        for other_depth in range(len(self.q_tables)):
            if other_depth != depth and x < self.env.base_size and y < self.env.base_size:
                # Current Q-value at other depth
                other_q = self.q_tables[other_depth][x, y, action]
                
                # Blend Q-values across scales (bidirectional transfer)
                transfer_amount = transfer_rate * (q_value - other_q)
                self.q_tables[other_depth][x, y, action] += transfer_amount
                
        # Enhanced awareness: Track successful cross-scale patterns
        if len(self.cross_scale_experiences) > 10:
            self._reinforce_successful_patterns()

    def _reinforce_successful_patterns(self):
        """Reinforce Q-values for successful cross-scale patterns."""
        # Find high-reward cross-scale transitions
        successful_transitions = [
            exp for exp in self.cross_scale_experiences 
            if exp['reward'] > 1.0  # Positive transitions (portal usage, goals)
        ]
        
        if len(successful_transitions) > 5:
            reinforcement_rate = 0.05
            for exp in successful_transitions[-5:]:  # Recent successful patterns
                from_x, from_y, from_depth = exp['from_state']
                action = exp['action']
                reward = exp['reward']
                
                # Reinforce similar states across all depths
                for depth in range(len(self.q_tables)):
                    if (from_x < self.env.base_size and from_y < self.env.base_size):
                        current_q = self.q_tables[depth][from_x, from_y, action]
                        bonus = reinforcement_rate * reward
                        self.q_tables[depth][from_x, from_y, action] += bonus

    def get_self_observation_insights(self):
        """Analyze self-observation patterns across scales."""
        if not self.cross_scale_experiences:
            return {}
        
        zoom_ins = sum(1 for exp in self.cross_scale_experiences if exp['transition_type'] == 'zoom_in')
        zoom_outs = sum(1 for exp in self.cross_scale_experiences if exp['transition_type'] == 'zoom_out')
        
        return {
            'total_scale_transitions': len(self.cross_scale_experiences),
            'zoom_ins': zoom_ins,
            'zoom_outs': zoom_outs,
            'exploration_depth_ratio': zoom_ins / max(1, zoom_ins + zoom_outs),
            'max_depth_reached': max([exp['to_state'][2] for exp in self.cross_scale_experiences] + [0])
        }

    def train(self, episodes=1000, horizon_per_episode=300, verbose=True):
        """Train agent with fractal self-observation capabilities."""
        if verbose:
            print(f"Training SelfObservingAgent for {episodes} episodes...")
        
        all_rewards = []
        all_steps = []
        max_depths_reached = []
        successful_episodes = 0

        for ep in range(episodes):
            current_state = self.env.reset()
            episode_reward = 0
            episode_max_depth = 0

            for step in range(horizon_per_episode):
                # Get current observation perspective
                obs = self.env.get_observation_perspective()
                self.observation_memory.append(obs)
                
                action = self.choose_action(current_state)
                next_state, reward, done, info = self.env.step(action)
                
                self.learn_from_experience(current_state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_max_depth = max(episode_max_depth, current_state[2])
                
                current_state = next_state
                if done:
                    successful_episodes += 1
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            all_rewards.append(episode_reward)
            all_steps.append(step + 1)
            max_depths_reached.append(episode_max_depth)

            if verbose and ep % 100 == 0:
                insights = self.get_self_observation_insights()
                print(f"Ep {ep}: Avg Reward (last 100): {np.mean(all_rewards[-100:]):.2f}, "
                      f"Avg Steps: {np.mean(all_steps[-100:]):.1f}, "
                      f"Max Depth: {episode_max_depth}, "
                      f"Scale Transitions: {insights.get('total_scale_transitions', 0)}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        final_insights = self.get_self_observation_insights()
        if verbose:
            print(f"Training complete. Final insights:")
            print(f"  Success rate: {successful_episodes/episodes:.2%}")
            print(f"  Total scale transitions: {final_insights.get('total_scale_transitions', 0)}")
            print(f"  Max depth reached: {final_insights.get('max_depth_reached', 0)}")
            print(f"  Exploration depth ratio: {final_insights.get('exploration_depth_ratio', 0):.2%}")
        
        return {
            'rewards': all_rewards,
            'steps': all_steps,
            'max_depths': max_depths_reached,
            'insights': final_insights
        }

    def test_policy(self, num_episodes=10, horizon=300, verbose=True):
        """Test learned policy without exploration."""
        if verbose:
            print(f"\nTesting learned policy for {num_episodes} episodes...")
        
        successes = 0
        avg_steps = []
        avg_rewards = []
        depth_explorations = []

        original_epsilon = self.epsilon
        self.epsilon = 0.0  # Greedy policy

        for ep in range(num_episodes):
            current_state = self.env.reset()
            ep_reward = 0
            max_depth_reached = 0
            
            for step in range(horizon):
                action = self.choose_action(current_state)
                next_state, reward, done, info = self.env.step(action)
                ep_reward += reward
                max_depth_reached = max(max_depth_reached, current_state[2])
                current_state = next_state
                
                if done:
                    if reward > 50:  # Assuming high reward for goal
                        successes += 1
                    avg_steps.append(step + 1)
                    break
            
            if not done:
                avg_steps.append(horizon)
            
            avg_rewards.append(ep_reward)
            depth_explorations.append(max_depth_reached)

        self.epsilon = original_epsilon
        
        success_rate = successes / num_episodes
        mean_steps = np.mean(avg_steps) if avg_steps else horizon
        mean_depth = np.mean(depth_explorations)
        
        if verbose:
            print(f"Test Results:")
            print(f"  Success Rate: {success_rate*100:.1f}%")
            print(f"  Avg Steps: {mean_steps:.1f}")
            print(f"  Avg Reward: {np.mean(avg_rewards):.2f}")
            print(f"  Avg Max Depth Explored: {mean_depth:.1f}")
        
        return success_rate, mean_steps, mean_depth 