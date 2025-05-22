# Dynamic Fractal Attention Agent for Real-Time Strategy

This project demonstrates a novel concept called "Fractal Awareness with Dynamic Attention" where an agent perceives and interacts with its environment at multiple scales simultaneously, using attention mechanisms to dynamically allocate computational resources based on the current situation.

## Concept Overview

The core concept is that intelligent agents can benefit from representing the world at different levels of abstraction:

1. **Micro-level (Immediate)**: High-resolution details of individual unit control, movement, and targeting
2. **Meso-level (Tactical)**: Coordination of unit groups, resource allocation, and local objectives
3. **Macro-level (Strategic)**: Overall strategy, economy vs. defense focus, and global resource management

The key innovation in this implementation is the **dynamic attention mechanism** that allows the agent to:
- Shift focus between different levels of abstraction based on the current situation
- Allocate more "attention" to micro-level control when precise unit actions are needed (e.g., during combat)
- Focus on higher-level patterns when strategic planning is more important (e.g., when resources are scarce)

## RTS Environment

The project includes a simplified Real-Time Strategy (RTS) environment with:

- **Map**: A 2D grid (64×64) with fog of war
- **Resources**: Crystal patches and vespene geysers
- **Units**: Harvesters for resource collection, Warriors for combat
- **Structures**: Nexus (main base), Turrets (defensive structures)
- **Enemies**: Raiders that attack the base in increasingly difficult waves

## Agent Implementations

The project includes three agent types:

1. **Scripted Agent**: A hand-coded agent with predefined behaviors for resource collection and combat
2. **Fractal Agent**: A hierarchical agent with policies at three scales (micro, meso, macro) but fixed attention allocation
3. **Fractal Attention Agent**: Our novel implementation that dynamically shifts attention between the three levels

## Hierarchical Decision Making

The Fractal Attention Agent makes decisions at multiple levels:

### Strategic Level (Super)
- Chooses between high-level strategies:
  - ECONOMY: Focus on resource gathering and harvester production
  - DEFENSE: Focus on warrior production and defensive positioning
  - EXPANSION: Focus on exploring for new resources
  - VESPENE: Focus on securing and harvesting the valuable vespene geyser

### Tactical Level (Meso)
- Translates strategic focus into specific objectives:
  - ASSIGN_HARVESTERS: Assign harvesters to specific resource patches
  - BUILD_HARVESTER/WARRIOR: Produce new units
  - BUILD_TURRET: Construct defensive structures
  - RALLY_WARRIORS: Coordinate warriors for attack or defense
  - SCOUT: Send units to explore the map

### Unit Level (Micro)
- Controls individual unit actions:
  - Movement (up, down, left, right)
  - Harvesting resources
  - Attacking enemies
  - Returning resources to base

## Dynamic Attention Mechanism

The attention mechanism works by:

1. Analyzing the current game state (enemies, resources, unit states)
2. Computing attention logits for each level based on:
   - Combat proximity (boosts micro attention)
   - Resource needs (affects strategic attention)
   - Unit coordination requirements (affects tactical attention)
   - Harvesting precision needs (boosts micro attention)
3. Applying softmax to produce normalized attention weights
4. Using these weights in a voting system where each level suggests actions

The key insight is that **different situations require different balances of attention**:
- During combat, micro-level control becomes more important
- When resources are scarce, strategic decisions become critical
- When coordinating multiple units, tactical considerations take priority

## Visualizations

The project includes visualizations to demonstrate how the attention mechanism works:

- **RTS Game View**: Shows units, resources, structures, and fog of war
- **Attention Weights Chart**: Displays the dynamic allocation of attention across scales
- **Evolution of Attention**: Tracks how attention shifts during gameplay
- **Comparative Results**: Compares performance metrics between agent types

## Running the Code

To run the RTS environment with the Fractal Attention Agent:

```bash
python rts_test_attention.py
```

This will:
1. Initialize the RTS environment
2. Create and run the Fractal Attention Agent
3. Display the game progress and attention weights
4. Plot performance metrics and attention evolution

For a full comparison between agent types:

```python
# In rts_test_attention.py, uncomment:
compare_agents(n_episodes=3, max_steps=300, render_final=True)
```

## Project Structure

- `rts_environment.py`: The RTS game environment
- `rts_fractal_agent.py`: Base implementation of the fractal agent
- `rts_fractal_attention_agent.py`: Implementation of the dynamic attention mechanism
- `rts_test_attention.py`: Testing and comparison script

## Dependencies

- Python 3.6+
- NumPy
- Matplotlib
- PyTorch
- tqdm

## Future Directions

This implementation demonstrates the concept in a simplified RTS environment, but the principles could be extended to:

- Full-scale commercial RTS games
- Robotic control with multiple levels of abstraction
- Multi-agent coordination in complex environments
- Any domain where decisions must be made at multiple scales simultaneously

The dynamic attention mechanism provides a flexible framework for efficiently allocating computational resources across different levels of abstraction, making it a promising approach for developing more capable and adaptive AI systems. 