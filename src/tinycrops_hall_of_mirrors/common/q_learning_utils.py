"""
Common Q-learning utilities for reinforcement learning agents.
"""

import numpy as np
import random
from collections import deque


def choose_action(Q, state, epsilon):
    """
    Choose an action using epsilon-greedy policy.
    
    Args:
        Q: Q-table (numpy array)
        state: Current state index (int)
        epsilon: Exploration rate (float)
        
    Returns:
        action: Selected action index (int)
    """
    if np.random.rand() < epsilon:
        return np.random.choice(Q.shape[1])  # Random action
    else:
        return Q[state].argmax()  # Greedy action


def update_from_buffer(Q, buffer, batch_size, alpha, gamma):
    """
    Update Q-table using experience replay from buffer.
    
    Args:
        Q: Q-table to update
        buffer: Experience replay buffer (deque)
        batch_size: Number of experiences to sample
        alpha: Learning rate
        gamma: Discount factor
    """
    if len(buffer) < batch_size:
        return
    
    batch = random.sample(buffer, batch_size)
    for s, a, r, s_next, done in batch:
        # Check bounds to prevent index errors
        if s < Q.shape[0] and s_next < Q.shape[0] and a < Q.shape[1]:
            target = r + gamma * np.max(Q[s_next]) * (1 - done)
            Q[s, a] += alpha * (target - Q[s, a])


def create_experience_buffer(buffer_size):
    """Create an experience replay buffer."""
    return deque(maxlen=buffer_size)


def softmax(x):
    """Compute softmax values for array x with numerical stability."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


def decay_epsilon(epsilon, epsilon_min, decay_rate):
    """Decay epsilon for exploration schedule."""
    return max(epsilon_min, epsilon * decay_rate) 