#!/usr/bin/env python3
"""
Test script for enhanced grid-world visualizations.
Demonstrates the new frame-by-frame visualization capabilities including:
- Dynamic Q-value/policy overlays
- Hierarchical goal visualization
- Attention mechanism visualization
- Curiosity-driven exploration visualization
- Adaptive hierarchy visualization
- Meta-learning insights
- Optimal path comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tinycrops_hall_of_mirrors.grid_world.environment import GridEnvironment
from tinycrops_hall_of_mirrors.grid_world.agents import (
    FlatAgent, FractalAgent, FractalAttentionAgent
)
from tinycrops_hall_of_mirrors.grid_world.advanced_agents import (
    AdaptiveFractalAgent, CuriosityDrivenAgent, MultiHeadAttentionAgent, 
    MetaLearningAgent, CuriosityDrivenAdaptiveAgent
)
from tinycrops_hall_of_mirrors.grid_world.visualization import (
    animate_agent_step_by_step, visualize_adaptive_hierarchy,
    visualize_meta_learning_strategies, visualize_multihead_attention_analysis
)


def test_enhanced_visualizations():
    """Test all enhanced visualization capabilities."""
    print("Testing Enhanced Grid-World Visualizations")
    print("=" * 60)
    
    # Create environment
    env = GridEnvironment(
        size=10,
        obstacles=[(3, 3), (3, 4), (4, 3), (6, 6), (6, 7), (7, 6)],
        start=(0, 0),
        goal=(9, 9)
    )
    print(f"Environment: {env.size}x{env.size} grid with {len(env.obstacles)} obstacles")
    
    # Test 1: Basic agent with policy overlay
    print("\n1. Testing FlatAgent with Dynamic Policy Overlay...")
    flat_agent = FlatAgent(env)
    flat_agent.train(episodes=200)
    
    print("   Creating animation with policy arrows and optimal path...")
    ani1 = animate_agent_step_by_step(
        flat_agent, env,
        title="FlatAgent: Policy Arrows & Optimal Path",
        show_policy=True,
        show_values=False,
        show_hierarchical=False,
        show_optimal_path=True,
        interval=800
    )
    plt.close()
    
    # Test 2: Fractal agent with hierarchical visualization
    print("\n2. Testing FractalAgent with Hierarchical Goals...")
    fractal_agent = FractalAgent(env, block_micro=2, block_macro=4, reward_shaping='shaped')
    fractal_agent.train(episodes=200)
    
    print("   Creating animation with hierarchical goals and value heatmap...")
    ani2 = animate_agent_step_by_step(
        fractal_agent, env,
        title="FractalAgent: Hierarchical Goals & Value Heatmap",
        show_policy=False,
        show_values=True,
        show_hierarchical=True,
        show_optimal_path=True,
        interval=1000
    )
    plt.close()
    
    # Test 3: Attention agent with live attention weights
    print("\n3. Testing FractalAttentionAgent with Live Attention Visualization...")
    attention_agent = FractalAttentionAgent(env, block_micro=2, block_macro=4)
    attention_agent.train(episodes=200)
    
    print("   Creating animation with attention weights and hierarchical goals...")
    ani3 = animate_agent_step_by_step(
        attention_agent, env,
        title="FractalAttentionAgent: Live Attention Weights",
        show_policy=True,
        show_values=False,
        show_hierarchical=True,
        show_attention=True,
        interval=1200
    )
    plt.close()
    
    # Test 4: Curiosity-driven agent with exploration visualization
    print("\n4. Testing CuriosityDrivenAgent with Exploration Visualization...")
    curiosity_agent = CuriosityDrivenAgent(env)
    curiosity_agent.train(episodes=200)
    
    print("   Creating animation with curiosity bonuses...")
    ani4 = animate_agent_step_by_step(
        curiosity_agent, env,
        title="CuriosityDrivenAgent: Exploration & Novelty",
        show_policy=False,
        show_values=True,
        show_curiosity=True,
        interval=800
    )
    plt.close()
    
    # Test 5: Multi-head attention agent
    print("\n5. Testing MultiHeadAttentionAgent Analysis...")
    multihead_agent = MultiHeadAttentionAgent(env)
    multihead_agent.train(episodes=200)
    
    # Collect episode data for detailed analysis
    print("   Running episode to collect attention data...")
    pos = env.start
    episode_data = {
        'positions': [pos],
        'attention_heads': [],
        'head_weights': []
    }
    
    step = 0
    while pos != env.goal and step < 50:
        state_idx = multihead_agent.get_state_index(pos)
        
        # Get attention head activation
        if hasattr(multihead_agent, 'active_head'):
            episode_data['attention_heads'].append(multihead_agent.active_head)
        
        # Get head weights
        if hasattr(multihead_agent, 'head_weights'):
            episode_data['head_weights'].append(multihead_agent.head_weights.copy())
        
        # Take action
        action = multihead_agent.choose_action(state_idx, epsilon=0.05)
        pos, _, _ = env.step(pos, action)
        episode_data['positions'].append(pos)
        step += 1
    
    print("   Creating detailed multi-head attention analysis...")
    visualize_multihead_attention_analysis(multihead_agent, env, episode_data)
    plt.close()
    
    # Test 6: Adaptive hierarchy visualization
    print("\n6. Testing AdaptiveFractalAgent Hierarchy Adaptation...")
    adaptive_agent = AdaptiveFractalAgent(env)
    adaptive_agent.train(episodes=300)
    
    print("   Creating adaptive hierarchy analysis...")
    visualize_adaptive_hierarchy(adaptive_agent, env)
    plt.close()
    
    # Test 7: Meta-learning strategy visualization
    print("\n7. Testing MetaLearningAgent Strategy Selection...")
    meta_agent = MetaLearningAgent(env)
    meta_agent.train(episodes=300)
    
    print("   Creating meta-learning strategy analysis...")
    visualize_meta_learning_strategies(meta_agent, env)
    plt.close()
    
    # Test 8: Comprehensive comparison with all features
    print("\n8. Testing Comprehensive Visualization (All Features)...")
    comprehensive_agent = CuriosityDrivenAdaptiveAgent(env)
    comprehensive_agent.train(episodes=250)
    
    print("   Creating comprehensive animation with all features...")
    ani8 = animate_agent_step_by_step(
        comprehensive_agent, env,
        title="CuriosityDrivenAdaptiveAgent: All Features",
        show_policy=True,
        show_values=True,
        show_hierarchical=True,
        show_attention=False,  # This agent doesn't have attention
        show_curiosity=True,
        show_optimal_path=True,
        interval=1500
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("Enhanced Visualization Testing Complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Dynamic Q-value/Policy Overlays")
    print("✓ Hierarchical Goal Visualization")
    print("✓ Live Attention Mechanism Display")
    print("✓ Curiosity-Driven Exploration Heatmaps")
    print("✓ Adaptive Hierarchy Analysis")
    print("✓ Meta-Learning Strategy Selection")
    print("✓ Optimal Path Comparison")
    print("✓ Comprehensive Multi-Panel Displays")


def test_interactive_features():
    """Test interactive features and data logging capabilities."""
    print("\n" + "=" * 60)
    print("Testing Interactive Features & Data Logging")
    print("=" * 60)
    
    # Create a complex environment for testing
    env = GridEnvironment(
        size=12,
        obstacles=[(2, 2), (2, 3), (3, 2), (5, 5), (5, 6), (6, 5), 
                  (8, 8), (8, 9), (9, 8), (10, 3), (10, 4)],
        start=(0, 0),
        goal=(11, 11)
    )
    
    # Test different animation speeds and features
    print("\n1. Testing Animation Speed Controls...")
    
    # Fast animation
    agent = FractalAgent(env, block_micro=3, block_macro=6)
    agent.train(episodes=150)
    
    print("   Fast animation (300ms intervals)...")
    ani_fast = animate_agent_step_by_step(
        agent, env,
        title="Fast Animation Demo",
        show_policy=True,
        show_hierarchical=True,
        interval=300
    )
    plt.close()
    
    # Slow animation with detailed analysis
    print("   Slow animation (2000ms intervals) with all features...")
    ani_slow = animate_agent_step_by_step(
        agent, env,
        title="Detailed Analysis (Slow)",
        show_policy=True,
        show_values=True,
        show_hierarchical=True,
        show_optimal_path=True,
        interval=2000
    )
    plt.close()
    
    print("\n2. Testing Data Logging Capabilities...")
    
    # Demonstrate logging capabilities
    curiosity_agent = CuriosityDrivenAgent(env)
    result = curiosity_agent.train(episodes=200)
    training_log = result[0] if isinstance(result, tuple) else result
    
    print(f"   Training completed: {len(training_log)} episodes logged")
    print(f"   Final performance: {training_log[-1]} steps")
    print(f"   Best performance: {min(training_log)} steps")
    
    # Create visualization from logged data
    print("   Creating visualization from logged training data...")
    ani_logged = animate_agent_step_by_step(
        curiosity_agent, env,
        title="Post-Training Analysis",
        show_curiosity=True,
        show_policy=True,
        interval=1000
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("Interactive Features Testing Complete!")


if __name__ == "__main__":
    print("Enhanced Grid-World Visualization Test Suite")
    print("=" * 60)
    print("This script demonstrates the new frame-by-frame visualization")
    print("capabilities including dynamic overlays and hierarchical analysis.")
    print()
    
    try:
        # Run main visualization tests
        test_enhanced_visualizations()
        
        # Run interactive features tests
        test_interactive_features()
        
        print("\n🎉 All visualization tests completed successfully!")
        print("\nThe enhanced visualization system is ready for:")
        print("• Research analysis and debugging")
        print("• Agent behavior understanding") 
        print("• Performance demonstration")
        print("• Educational presentations")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 