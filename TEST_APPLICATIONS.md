# Test Applications and Evaluation Scripts

This document provides an overview of the various test applications, experiment scripts, and demonstration tools available within the Fractal Hall of Mirrors framework. These tools are essential for verifying functionality, evaluating performance, reproducing research, and understanding the core concepts and contributions of the project.

The scripts are broadly categorized into:
*   **Unit Tests**: For verifying individual components.
*   **Experiment Scripts**: For running systematic evaluations and comparisons.
*   **Demonstration and Evaluation Scripts**: For showcasing specific features and research contributions.

For detailed instructions on command-line options and specific configurations, please refer to the main `README.md` file.

## Unit Tests

Unit tests are used to verify the correctness of individual components, modules, or functions within the framework, ensuring that each part of the code behaves as expected.

Here are the main unit test files:

*   `test_basic_functionality.py`: Tests core framework functionality.
*   `test_advanced_agents.py`: Test suite for advanced and novel agent implementations.
*   `test_rts_environment.py`: Tests for the enhanced Real-Time Strategy (RTS) environment.
*   `test_enhanced_visualizations.py`: Tests for the advanced visualization features.
*   `test_fractal_self_observation.py`: Tests specifically for fractal self-observation capabilities.
*   `test_novel_rts_agents.py`: Tests for novel RTS agents (e.g., `CuriosityDrivenRTSAgent`, `MultiHeadRTSAgent`).

To run the tests, use the following commands:

```bash
python test_basic_functionality.py
python test_advanced_agents.py
python test_rts_environment.py
python test_enhanced_visualizations.py
python test_fractal_self_observation.py
python test_novel_rts_agents.py
```

## Experiment Scripts

Experiment scripts are designed to run systematic evaluations, compare the performance of different agents or configurations, and reproduce research findings. They often generate logs and data for further analysis.

Here are the main experiment script files:

*   `experiments/run_grid_experiments.py`: For running standard grid-world experiments, typically with visualizations.
*   `experiments/run_grid_experiments_no_viz.py`: For grid-world experiments without visualizations, suitable for servers or faster execution.
*   `experiments/advanced_agent_experiments.py`: For testing and showcasing novel advanced agents.
*   `experiments/fractal_self_observation_experiments.py`: For experiments specifically focused on fractal self-observation.
*   `experiments/parameter_sweep_experiments.py`: For running experiments that sweep through different parameter values.

To run the scripts, use commands like the following:

```bash
# Standard grid-world comparison
python experiments/run_grid_experiments_no_viz.py all_agents_shaped --episodes 50

# Novel approaches showcase
python experiments/advanced_agent_experiments.py novel_approaches_showcase --episodes 25

# Curiosity vs standard exploration
python experiments/advanced_agent_experiments.py curiosity_exploration --episodes 30
```

Command-line options are available for customization (e.g., `--episodes`, `--horizon`, `--env-size`). Refer to the main `README.md` for more details.

## Demonstration and Evaluation Scripts

These scripts are used to demonstrate specific research contributions, visualize unique capabilities of the agents or environments, and perform targeted evaluations of framework features. They are often more narrative or focused than general experiment scripts.

Here are the main demonstration and evaluation script files:

*   `demonstrate_research_contributions.py`: Showcases key research breakthroughs, particularly fractal self-observation and claims of consciousness emergence. This script runs a series of demonstrations illustrating the core concepts and results.
*   `educational_fractal_visualization.py`: Provides detailed, educational visualizations of fractal self-observation and multi-perspective learning. It is designed to help users understand the mechanics and implications of these concepts through visual aids.
*   `fractal_capabilities_evaluation.py`: Rigorously evaluates whether fractal self-observation enhances AI agent capabilities compared to baselines, addressing learning efficiency, generalization, and robustness. This script runs specific tests and benchmarks to provide quantitative evidence.
*   `fractal_evidence_demonstration.py`: Offers an "honest assessment" of what fractal self-observation accomplishes, its benefits, and limitations, aiming to provide a balanced view of the technology.

To run these scripts, typically execute them directly:

```bash
python demonstrate_research_contributions.py
python educational_fractal_visualization.py
python fractal_capabilities_evaluation.py
python fractal_evidence_demonstration.py
```

Please refer to the main `README.md` or individual script documentation for more specific instructions or options.
