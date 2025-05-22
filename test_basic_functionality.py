#!/usr/bin/env python3
"""
Basic functionality test for consolidated grid-world agents.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.environment import GridEnvironment
from tinycrops_hall_of_mirrors.grid_world.agents import (
    FlatAgent, FractalAgent, FractalAttentionAgent
)


def test_environment():
    """Test the grid environment."""
    print("Testing GridEnvironment...")
    env = GridEnvironment(size=20, seed=0)
    print(f"  Environment size: {env.size}x{env.size}")
    print(f"  Goal position: {env.goal}")
    print(f"  Number of obstacles: {len(env.obstacles)}")
    print(f"  Actions: {env.actions}")
    
    # Test a few steps
    pos = env.reset()
    print(f"  Starting position: {pos}")
    
    for action in range(4):
        next_pos, reward, done = env.step(pos, action)
        print(f"    Action {action}: {pos} -> {next_pos}, reward={reward}, done={done}")
    
    print("  ✓ Environment test passed\n")


def test_flat_agent():
    """Test the flat agent."""
    print("Testing FlatAgent...")
    env = GridEnvironment(size=10, seed=0)  # Smaller for faster testing
    agent = FlatAgent(env)
    
    print(f"  Q-table shape: {agent.Q.shape}")
    
    # Quick training test
    log, training_time = agent.train(episodes=5, horizon=50)
    print(f"  Trained for {len(log)} episodes in {training_time:.2f} seconds")
    print(f"  Steps per episode: {log}")
    
    # Test episode run
    path = agent.run_episode(epsilon=0.1)
    print(f"  Test episode path length: {len(path)}")
    print("  ✓ FlatAgent test passed\n")


def test_fractal_agent():
    """Test the fractal agent."""
    print("Testing FractalAgent...")
    env = GridEnvironment(size=10, seed=0)
    agent = FractalAgent(env, block_micro=3, block_macro=6, reward_shaping='shaped')
    
    print(f"  Q_micro shape: {agent.Q_micro.shape}")
    print(f"  Q_macro shape: {agent.Q_macro.shape}")
    print(f"  Q_super shape: {agent.Q_super.shape}")
    
    # Quick training test
    log, training_time = agent.train(episodes=3, horizon=30)
    print(f"  Trained for {len(log)} episodes in {training_time:.2f} seconds")
    print(f"  Steps per episode: {log}")
    
    print("  ✓ FractalAgent test passed\n")


def test_fractal_attention_agent():
    """Test the fractal attention agent."""
    print("Testing FractalAttentionAgent...")
    env = GridEnvironment(size=10, seed=0)
    agent = FractalAttentionAgent(env, block_micro=3, block_macro=6, reward_shaping='shaped')
    
    print(f"  Initial attention weights: {agent.attention_weights}")
    
    # Quick training test
    log, training_time = agent.train(episodes=3, horizon=30)
    print(f"  Trained for {len(log)} episodes in {training_time:.2f} seconds")
    print(f"  Steps per episode: {log}")
    print(f"  Attention history length: {len(agent.attention_history)}")
    
    if agent.attention_history:
        final_weights = agent.attention_history[-1]
        print(f"  Final attention weights: {final_weights}")
    
    print("  ✓ FractalAttentionAgent test passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        test_environment()
        test_flat_agent()
        test_fractal_agent()
        test_fractal_attention_agent()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 