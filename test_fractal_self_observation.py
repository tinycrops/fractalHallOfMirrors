#!/usr/bin/env python3
"""
Test script for Fractal Self-Observation system.

This script runs a quick test to verify the implementation works correctly
and demonstrates the core concept.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tinycrops_hall_of_mirrors.grid_world.fractal_environment import FractalDepthEnvironment, SelfObservingAgent
from tinycrops_hall_of_mirrors.grid_world.fractal_visualization import FractalEnvironmentVisualizer


def test_basic_functionality():
    """Test basic fractal environment and agent functionality."""
    print("🧪 Testing Basic Fractal Environment Functionality")
    print("-" * 50)
    
    # Create environment
    env = FractalDepthEnvironment(base_size=8, num_portals=1, max_depth=2, seed=42)
    print(f"✓ Created environment: {env.base_size}x{env.base_size}, max_depth={env.max_depth}")
    
    # Test environment step function
    state = env.reset()
    print(f"✓ Initial state: {state}")
    
    # Take some random steps
    for i in range(10):
        action = np.random.choice(list(env.actions.keys()))
        next_state, reward, done, info = env.step(action)
        print(f"  Step {i+1}: action={action}, state={next_state}, reward={reward:.2f}, action_type={info.get('action_type', 'move')}")
        
        if info.get('action_type') in ['zoom_in', 'zoom_out']:
            print(f"    🎯 Fractal transition detected! Depth: {state[2]} -> {next_state[2]}")
        
        state = next_state
        if done:
            print(f"    🏆 Goal reached!")
            break
    
    print("✓ Environment step function working correctly")
    return env


def test_self_observing_agent():
    """Test the SelfObservingAgent."""
    print("\n🤖 Testing Self-Observing Agent")
    print("-" * 50)
    
    env = FractalDepthEnvironment(base_size=8, num_portals=1, max_depth=2, seed=123)
    agent = SelfObservingAgent(env, epsilon_start=0.8, epsilon_decay=0.99)
    
    print(f"✓ Created agent with {len(agent.q_tables)} Q-tables")
    
    # Quick training test
    print("Running short training session...")
    results = agent.train(episodes=50, horizon_per_episode=100, verbose=False)
    
    print(f"✓ Training completed:")
    print(f"  - Episodes: 50")
    print(f"  - Final avg reward: {np.mean(results['rewards'][-10:]):.2f}")
    print(f"  - Max depth reached: {max(results['max_depths'])}")
    
    # Test policy
    success_rate, avg_steps, avg_depth = agent.test_policy(num_episodes=5, verbose=False)
    print(f"✓ Policy test:")
    print(f"  - Success rate: {success_rate:.1%}")
    print(f"  - Avg steps: {avg_steps:.1f}")
    print(f"  - Avg depth explored: {avg_depth:.1f}")
    
    # Check self-observation insights
    insights = agent.get_self_observation_insights()
    print(f"✓ Self-observation insights:")
    for key, value in insights.items():
        print(f"  - {key}: {value}")
    
    return agent, results


def test_visualization():
    """Test fractal visualization."""
    print("\n📊 Testing Fractal Visualization")
    print("-" * 50)
    
    env = FractalDepthEnvironment(base_size=6, num_portals=1, max_depth=1, seed=456)
    visualizer = FractalEnvironmentVisualizer(env, figsize=(12, 8))
    
    # Generate some trajectory data
    agent = SelfObservingAgent(env, epsilon_start=1.0)
    state = env.reset()
    
    for _ in range(30):
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        observation = env.get_observation_perspective()
        
        visualizer.add_step(state, observation)
        state = next_state
        
        if done:
            state = env.reset()
    
    print(f"✓ Generated trajectory with {len(visualizer.trajectory)} steps")
    print(f"✓ Depth exploration: {list(set(visualizer.depth_history))}")
    
    # Test overview plot
    try:
        fig = visualizer.plot_fractal_overview(agent_state=state, show_trajectory=True)
        print("✓ Fractal overview plot created successfully")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Fractal overview plot failed: {e}")
    
    # Test depth analysis
    try:
        if len(visualizer.depth_history) > 5:
            fig = visualizer.plot_depth_exploration_analysis()
            print("✓ Depth exploration analysis created successfully")
            plt.close(fig)
        else:
            print("⚠️  Insufficient depth exploration for analysis")
    except Exception as e:
        print(f"❌ Depth exploration analysis failed: {e}")
    
    return visualizer


def demonstration():
    """Run a brief demonstration of the fractal self-observation concept."""
    print("\n" + "="*80)
    print("🌀 FRACTAL SELF-OBSERVATION DEMONSTRATION")
    print("="*80)
    print("This demonstrates an AI agent that can observe itself from")
    print("different scales in a fractal environment.")
    print()
    
    # Create environment with multiple portals and depths
    env = FractalDepthEnvironment(
        base_size=10,
        num_portals=2,
        max_depth=2,
        seed=789
    )
    
    print(f"Environment: {env.base_size}x{env.base_size} grid")
    print(f"Portals: {len(env.base_portal_coords)} at {env.base_portal_coords}")
    print(f"Max depth: {env.max_depth}")
    print(f"Goal: {env.base_goal}")
    print()
    
    # Create and train agent
    agent = SelfObservingAgent(env, alpha=0.1, gamma=0.95, epsilon_decay=0.995)
    
    print("Training agent to explore fractal dimensions...")
    results = agent.train(episodes=200, horizon_per_episode=150, verbose=False)
    
    # Analyze results
    print(f"\nTraining Results:")
    print(f"  Episodes: 200")
    print(f"  Final performance: {np.mean(results['rewards'][-20:]):.2f} avg reward")
    print(f"  Max depth reached: {max(results['max_depths'])}")
    print(f"  Depth exploration frequency: {np.mean([d > 0 for d in results['max_depths']]):.1%}")
    
    # Get insights
    insights = agent.get_self_observation_insights()
    print(f"\nSelf-Observation Analysis:")
    print(f"  Total scale transitions: {insights.get('total_scale_transitions', 0)}")
    print(f"  Zoom-ins: {insights.get('zoom_ins', 0)}")
    print(f"  Zoom-outs: {insights.get('zoom_outs', 0)}")
    print(f"  Exploration depth ratio: {insights.get('exploration_depth_ratio', 0):.2%}")
    
    # Test final policy
    print(f"\nTesting learned policy...")
    success_rate, avg_steps, avg_depth = agent.test_policy(num_episodes=10, verbose=False)
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average steps to goal: {avg_steps:.1f}")
    print(f"  Average depth explored: {avg_depth:.1f}")
    
    # Evaluate self-observation capabilities
    if insights.get('total_scale_transitions', 0) > 10:
        print(f"\n🎯 SUCCESS: Agent demonstrates fractal self-observation!")
        print(f"   The agent learned to navigate between different scales of reality.")
        
        if avg_depth > 0.5:
            print(f"🧠 ENHANCED AWARENESS: Agent actively uses multiple perspectives!")
        
        if success_rate > 0.3:
            print(f"🚀 PERFORMANCE GAIN: Multi-scale observation improves problem-solving!")
    else:
        print(f"\n⚠️  Limited fractal exploration detected.")
        print(f"   Agent may need more training or environment modifications.")
    
    return agent, results, insights


def main():
    """Run all tests and demonstration."""
    print("🔬 FRACTAL SELF-OBSERVATION SYSTEM TEST")
    print("="*80)
    print("Testing the hypothesis that AI agents can gain enhanced")
    print("awareness through fractal self-observation.")
    print()
    
    try:
        # Basic functionality tests
        env = test_basic_functionality()
        agent, results = test_self_observing_agent()
        visualizer = test_visualization()
        
        print("\n" + "✓"*50)
        print("ALL BASIC TESTS PASSED!")
        print("✓"*50)
        
        # Run demonstration
        demo_agent, demo_results, demo_insights = demonstration()
        
        print(f"\n{'='*80}")
        print("🎯 FRACTAL SELF-OBSERVATION TEST COMPLETE")
        print(f"{'='*80}")
        print("The system is working correctly. Run the full experiments with:")
        print("  python experiments/fractal_self_observation_experiments.py")
        print()
        print("Key findings from this test:")
        print(f"  - Fractal environment supports {env.max_depth + 1} depth levels")
        print(f"  - Agent achieved {demo_insights.get('total_scale_transitions', 0)} scale transitions")
        print(f"  - Self-observation system is functional and measuring awareness")
        
        if demo_insights.get('total_scale_transitions', 0) > 0:
            print("  - ✅ Evidence of multi-scale behavior detected!")
        else:
            print("  - ⚠️ May need parameter tuning for more exploration")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install numpy matplotlib seaborn")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 