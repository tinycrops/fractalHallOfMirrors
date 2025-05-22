"""
Grid-world environment for reinforcement learning experiments.
"""

import numpy as np


class GridEnvironment:
    """
    A 2D grid environment with obstacles and a goal.
    
    Features:
    - Configurable grid size
    - Static obstacles
    - Single goal location
    - Manhattan distance-based rewards
    """
    
    def __init__(self, size=20, obstacles=None, start=None, goal=None, seed=0):
        """
        Initialize the grid environment.
        
        Args:
            size: Grid size (size x size)
            obstacles: List/set of obstacle positions (x, y). If None, creates default obstacles.
            start: Starting position (x, y). Default is (0, 0).
            goal: Goal position (x, y). Default is (size-1, size-1).
            seed: Random seed for reproducibility
        """
        self.size = size
        self.start = start if start is not None else (0, 0)
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        
        # Set random seed
        np.random.seed(seed)
        
        # Create obstacles
        if obstacles is not None:
            self.obstacles = set(obstacles)
        else:
            self.obstacles = self._create_obstacles()
        
    def _create_obstacles(self):
        """Create a set of obstacle positions."""
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
        if self.goal in obstacles:
            obstacles.remove(self.goal)
        
        # Make sure the start is accessible
        if self.start in obstacles:
            obstacles.remove(self.start)
        
        return obstacles
    
    def step(self, pos, action):
        """
        Execute an action in the environment.
        
        Args:
            pos: Current position (x, y)
            action: Action index (0-3)
            
        Returns:
            next_pos: Next position after action
            reward: Reward for the transition
            done: Whether the episode is finished
        """
        # Calculate next position
        next_pos = tuple(np.clip(np.add(pos, self.actions[action]), 0, self.size - 1))
        
        # Check if the next position is an obstacle
        if next_pos in self.obstacles:
            next_pos = pos  # Cannot move into an obstacle
            reward = -2     # Higher penalty for bumping into walls
        else:
            reward = 10 if next_pos == self.goal else -1
        
        done = next_pos == self.goal
        return next_pos, reward, done
    
    def reset(self):
        """Reset the environment to the starting position."""
        return self.start
    
    def is_valid_position(self, pos):
        """Check if a position is valid (within bounds and not an obstacle)."""
        x, y = pos
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                pos not in self.obstacles)
    
    def get_valid_actions(self, pos):
        """Get list of valid actions from a given position."""
        valid_actions = []
        for action, delta in self.actions.items():
            next_pos = tuple(np.add(pos, delta))
            if self.is_valid_position(next_pos):
                valid_actions.append(action)
        return valid_actions
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @property
    def num_actions(self):
        """Number of possible actions."""
        return len(self.actions)
    
    @property
    def state_space_size(self):
        """Total number of possible states."""
        return self.size * self.size 