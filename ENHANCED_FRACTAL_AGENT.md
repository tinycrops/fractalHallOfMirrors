# Enhanced Fractal Attention Agent for RTS Environments

This document describes the enhancements made to the RTS Fractal Attention Agent to create a more sophisticated, learning-enabled hierarchical agent with dynamic attention allocation.

## Key Enhancements

### 1. Learnable Attention Mechanism

The original agent used a rule-based method for computing attention weights. The enhanced agent:

- Implements a neural network (`AttentionNetwork`) that learns to allocate attention across different levels of abstraction
- Extracts context-aware features from the environment state to serve as input to the attention network
- Includes a reward mechanism that reinforces good attention allocation based on outcomes
- Maintains a balance between exploration (using the rule-based method occasionally) and exploitation (using the learned network)

### 2. Explicit Goal Passing Between Levels

Instead of implicit communication between levels, the enhanced agent:

- Implements goal networks (`GoalNetwork`) for both super-to-meso and meso-to-micro communication
- Super level generates strategic parameters (resource ratios, unit composition targets, etc.)
- Meso level generates tactical parameters (rally points, priority targets, etc.)
- Goals are learned through reinforcement, with experience buffers tracking which goals led to positive outcomes

### 3. Deep Spatial Awareness

Enhanced state representation through:

- CNN-based spatial processing (`SpatialFeatureExtractor`) that understands the 2D game world
- Multi-channel input representing different entity types (units, structures, resources)
- Integration of spatial features with traditional numeric features for richer state representation
- Enhanced Q-networks that combine both feature types for better decision-making

### 4. More Sophisticated Tactical Behavior

The meso level (tactical) has been enhanced with:

- More advanced unit coordination based on learnable goal parameters
- Adaptive aggression levels for combat units based on the situation
- Dynamic resource prioritization based on goals from the strategic level
- Improved scout behavior with better exploration target selection

## Components

### AttentionNetwork

Neural network that learns to generate attention weights across the three levels (micro, meso, super) based on the current game state.

```python
class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        # Network that outputs a probability distribution over the three levels
```

### GoalNetwork

Networks that generate explicit goals for lower levels to follow.

```python
class GoalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, level='super'):
        # Different architectures for super and meso levels
```

### SpatialFeatureExtractor

CNN-based network that processes the 2D game grid to extract spatial features.

```python
class SpatialFeatureExtractor(nn.Module):
    def __init__(self, grid_size=MAP_SIZE, output_dim=32):
        # Processes 4-channel input (player units, enemy units, structures, resources)
```

### EnhancedQNetwork

Q-network that combines traditional features with spatial features for better decision-making.

```python
class EnhancedQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, use_spatial=True):
        # Combines traditional and spatial features
```

## Performance Metrics and Evaluation

The enhanced agent tracks:

- Attention weights over time
- Goal parameter effectiveness
- Resource collection efficiency
- Combat effectiveness
- Exploration coverage

## Usage

```python
from enhanced_fractal_attention_agent import EnhancedFractalAttentionAgent
from rts_environment import RTSEnvironment

# Initialize environment and agent
env = RTSEnvironment()
agent = EnhancedFractalAttentionAgent()

# Run simulation
for step in range(300):
    state = env.get_state()
    attn_weights = agent.act(state, env)
    game_over = env.step()
    
    if game_over:
        break
```

## Future Improvements

1. **Experience Replay with Spatial States**: Store spatial states with experiences for more accurate learning
2. **Actor-Critic Architecture**: Transition to actor-critic methods for more stable policy learning
3. **Meta-Learning for Attention**: Learn to adapt attention allocation through meta-learning techniques
4. **Attention Visualization**: Tools to visualize how attention shifts during different game phases 