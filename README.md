# Fractal Hall of Mirrors - Agent Visualization

This project demonstrates a self-improving agent inside a "fractal" FPS environment, where each combat encounter is a Task-Instance-Projection (TIP) of the Fractal Operations Manifold. The visualization helps to understand the agent's learning process across different combat patterns.

## What You're Seeing

The visualization contains four main panels:

### 1. Canonical Space Map (Top Left)
- **Blue points**: Navigation waypoints in the canonical space
- **Green circle**: The agent's current position
- **Green arrow**: The agent's facing direction (yaw)
- **Red/Orange circles**: Opponents with different weapons (shotgun/rocket)
- **Dashed lines**: Planned paths the agent is considering
- **Dotted green lines**: Line of sight to visible opponents

### 2. Pattern Learning Progress (Top Right)
- Tracks how the agent's learned parameters evolve over time
- **Solid lines**: Aim offset learning (adjusting for recoil)
- **Dashed lines**: Flank bias learning (tactical positioning)
- Different colors represent different weapon patterns

### 3. Combat Performance (Bottom Left)
- Bar chart showing accuracy by weapon type
- Horizontal line at 60% shows the initial expected accuracy
- As the agent learns, its accuracy should improve beyond this baseline

### 4. Learning Metrics (Bottom Right)
- Textual summary of all learning parameters
- Shows exact values for aim_offset and flank_bias
- Tracks combat statistics (shots fired, hit rate, etc.)
- Displays active and queued tasks

## Key Concepts

- **Fractal Pattern**: A repeatable encounter type (e.g., "combat/shotgun") that has learnable parameters
- **TIP (Task-Instance-Projection)**: A specific instance of an encounter with an opponent
- **Universal Learning Propagator (ULP)**: The mechanism that shares learning between all instances of the same pattern

## Running the Visualization

```
python fractal_meta_bot.py
```

The simulation will run for 30 ticks, showing the agent's learning progress in real-time. You'll see the visualization update as the agent encounters opponents and improves its strategies.

## Requirements

- Python 3.6+
- matplotlib
- numpy

## How It Works

1. The agent detects opponents in the environment
2. For each opponent, it creates a TIP of the appropriate pattern type
3. The TIP plans a path considering learned flank_bias
4. The TIP aims with the learned aim_offset
5. Results of shots feed back into the Universal Learning Propagator
6. All future TIPs benefit from this accumulated learning

This demonstrates how learning propagates across the "fractal" space, allowing the agent to improve across all similar encounters. 