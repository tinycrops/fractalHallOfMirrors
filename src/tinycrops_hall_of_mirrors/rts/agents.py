"""
RTS Agents with Novel Hierarchical and Attention-Based Approaches.

This module brings the proven grid-world innovations to the complex RTS domain:
- BaseRTSAgent: Foundation for all RTS agents
- AdaptiveRTSAgent: Dynamic strategic focus adaptation
- CuriosityDrivenRTSAgent: Exploration of tech trees and map areas
- MultiHeadRTSAgent: Specialized attention for economy, military, defense
- MetaLearningRTSAgent: Strategy adaptation based on game patterns
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time

from .environment import (
    RTSEnvironment, UnitType, StructureType, ResourceType, ActionType, 
    Unit, Structure, Resource
)
from ..common.q_learning_utils import (
    choose_action, update_from_buffer, create_experience_buffer,
    softmax, decay_epsilon
)


class BaseRTSAgent:
    """
    Base class for RTS agents with hierarchical action decomposition.
    
    Novel Innovation: Adapts the proven hierarchical Q-learning approach 
    to the complex multi-dimensional RTS action space.
    """
    
    def __init__(self, env: RTSEnvironment, 
                 learning_rate=0.1, epsilon_start=0.9, epsilon_end=0.1, 
                 epsilon_decay=0.995, gamma=0.9):
        self.env = env
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # RTS-specific hierarchical decomposition
        self.strategic_horizon = 300   # Long-term strategic planning
        self.tactical_horizon = 100    # Medium-term tactical decisions
        self.operational_horizon = 30  # Short-term unit operations
        
        # Multi-dimensional action space
        self.strategic_actions = [
            'expand_economy', 'build_military', 'tech_research', 
            'defensive_posture', 'aggressive_expansion'
        ]
        self.tactical_actions = [
            'scout_map', 'gather_resources', 'train_units', 
            'build_structures', 'attack_enemy'
        ]
        self.operational_actions = [
            'move_harvester', 'attack_with_warrior', 'build_turret',
            'explore_area', 'defend_position'
        ]
        
        # Q-tables for each hierarchical level
        self.Q_strategic = defaultdict(lambda: np.zeros(len(self.strategic_actions)))
        self.Q_tactical = defaultdict(lambda: np.zeros(len(self.tactical_actions)))
        self.Q_operational = defaultdict(lambda: np.zeros(len(self.operational_actions)))
        
        # Experience buffers
        self.strategic_buffer = create_experience_buffer(10000)
        self.tactical_buffer = create_experience_buffer(10000)
        self.operational_buffer = create_experience_buffer(10000)
        
        # Strategy tracking
        self.current_strategy = None
        self.strategy_performance = []
        
    def get_strategic_state(self, game_state: Dict) -> str:
        """
        Extract strategic-level state representation.
        
        Focuses on high-level game metrics like resource ratios, 
        unit counts, and map control.
        """
        crystal_ratio = min(game_state['crystal_count'] / 1000, 1.0)
        unit_count = len(game_state['player_units'])
        enemy_count = len(game_state['enemy_units'])
        
        # Map control estimate
        map_control = 0.5  # TODO: Implement based on unit positions
        
        # Game phase detection
        if game_state['time'] < 200:
            phase = 'early'
        elif game_state['time'] < 600:
            phase = 'mid'
        else:
            phase = 'late'
        
        # Strategic state encoding
        state_key = f"{phase}_{int(crystal_ratio*10)}_{unit_count//5}_{enemy_count//5}_{int(map_control*10)}"
        
        return state_key
    
    def get_tactical_state(self, game_state: Dict) -> str:
        """
        Extract tactical-level state representation.
        
        Focuses on immediate threats, opportunities, and resource needs.
        """
        immediate_threats = len([u for u in game_state['enemy_units'] 
                               if self._distance_to_nexus(u) < 20])
        
        resource_urgency = 1 if game_state['crystal_count'] < 100 else 0
        production_capacity = len([s for s in game_state['structures'] 
                                 if s.type == StructureType.NEXUS])
        
        # Recent events influence tactical decisions
        active_events = len(game_state.get('events', []))
        
        state_key = f"threat_{immediate_threats}_resource_{resource_urgency}_prod_{production_capacity}_events_{active_events}"
        
        return state_key
    
    def get_operational_state(self, game_state: Dict) -> str:
        """
        Extract operational-level state representation.
        
        Focuses on individual unit positions and immediate actions.
        """
        # Simplified operational state based on unit composition
        harvesters = len([u for u in game_state['player_units'] 
                         if u.type == UnitType.HARVESTER])
        warriors = len([u for u in game_state['player_units'] 
                       if u.type == UnitType.WARRIOR])
        
        # Immediate resource collection opportunities
        nearby_resources = len([r for r in game_state['resources'] 
                              if r.amount > 0])  # Simplified
        
        state_key = f"h_{harvesters}_w_{warriors}_res_{nearby_resources}"
        
        return state_key
    
    def _distance_to_nexus(self, unit) -> float:
        """Calculate distance from unit to player's nexus."""
        nexus = next((s for s in self.env.structures 
                     if s.type == StructureType.NEXUS), None)
        if nexus is None:
            return float('inf')
        
        return ((unit.position[0] - nexus.position[0])**2 + 
                (unit.position[1] - nexus.position[1])**2)**0.5
    
    def choose_strategic_action(self, strategic_state: str, epsilon: float) -> int:
        """Choose strategic action using epsilon-greedy policy."""
        Q_values = self.Q_strategic[strategic_state]
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.strategic_actions))
        else:
            return np.argmax(Q_values)
    
    def choose_tactical_action(self, tactical_state: str, epsilon: float) -> int:
        """Choose tactical action using epsilon-greedy policy."""
        Q_values = self.Q_tactical[tactical_state]
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.tactical_actions))
        else:
            return np.argmax(Q_values)
    
    def choose_operational_action(self, operational_state: str, epsilon: float) -> int:
        """Choose operational action using epsilon-greedy policy."""
        Q_values = self.Q_operational[operational_state]
        if np.random.rand() < epsilon:
            return np.random.choice(len(self.operational_actions))
        else:
            return np.argmax(Q_values)
    
    def execute_strategic_action(self, action_idx: int, game_state: Dict):
        """
        Execute strategic action by setting tactical goals.
        
        Strategic actions influence the weighting of tactical decisions.
        """
        action = self.strategic_actions[action_idx]
        self.current_strategy = action
        
        # Strategic actions modify tactical preferences
        tactical_weights = np.ones(len(self.tactical_actions))
        
        if action == 'expand_economy':
            tactical_weights[1] *= 2.0  # Prioritize gather_resources
            tactical_weights[3] *= 1.5  # Prioritize build_structures
        elif action == 'build_military':
            tactical_weights[2] *= 2.0  # Prioritize train_units
        elif action == 'tech_research':
            tactical_weights[3] *= 1.5  # Prioritize build_structures (for tech)
        elif action == 'defensive_posture':
            tactical_weights[3] *= 1.8  # Prioritize build_structures (turrets)
            tactical_weights[4] *= 0.5  # Reduce attack_enemy
        elif action == 'aggressive_expansion':
            tactical_weights[4] *= 2.0  # Prioritize attack_enemy
            tactical_weights[0] *= 1.5  # Prioritize scout_map
        
        return tactical_weights
    
    def execute_tactical_action(self, action_idx: int, game_state: Dict, strategic_weights=None):
        """Execute tactical action by influencing operational decisions."""
        action = self.tactical_actions[action_idx]
        
        operational_weights = np.ones(len(self.operational_actions))
        
        if action == 'scout_map':
            operational_weights[3] *= 2.0  # Prioritize explore_area
        elif action == 'gather_resources':
            operational_weights[0] *= 2.0  # Prioritize move_harvester
        elif action == 'train_units':
            # Influence unit production (handled in operational execution)
            pass
        elif action == 'build_structures':
            operational_weights[2] *= 2.0  # Prioritize build_turret
        elif action == 'attack_enemy':
            operational_weights[1] *= 2.0  # Prioritize attack_with_warrior
        
        return operational_weights
    
    def execute_operational_action(self, action_idx: int, game_state: Dict):
        """
        Execute operational action by directly controlling units.
        
        This is where abstract actions are converted to concrete RTS commands.
        """
        action = self.operational_actions[action_idx]
        
        if action == 'move_harvester':
            return self._command_harvester_movement(game_state)
        elif action == 'attack_with_warrior':
            return self._command_warrior_attack(game_state)
        elif action == 'build_turret':
            return self._command_build_turret(game_state)
        elif action == 'explore_area':
            return self._command_exploration(game_state)
        elif action == 'defend_position':
            return self._command_defense(game_state)
        
        return False  # No action taken
    
    def _command_harvester_movement(self, game_state: Dict) -> bool:
        """Command harvesters to move toward resources or return to nexus."""
        harvesters = [u for u in game_state['player_units'] 
                     if u.type == UnitType.HARVESTER]
        
        if not harvesters:
            return False
        
        harvester = random.choice(harvesters)
        
        # If carrying resources, return to nexus
        if harvester.resources > 0:
            nexus = next((s for s in game_state['structures'] 
                         if s.type == StructureType.NEXUS), None)
            if nexus:
                # Move toward nexus (simplified)
                dx = np.sign(nexus.position[0] - harvester.position[0])
                dy = np.sign(nexus.position[1] - harvester.position[1])
                
                if abs(dx) > abs(dy):
                    direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
                else:
                    direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
                
                return harvester.move(direction, self.env)
        else:
            # Find nearest resource
            resources = [r for r in game_state['resources'] if r.amount > 0]
            if resources:
                nearest_resource = min(resources, 
                                     key=lambda r: ((harvester.position[0] - r.position[0])**2 + 
                                                   (harvester.position[1] - r.position[1])**2))
                
                # Move toward resource or harvest if adjacent
                distance = ((harvester.position[0] - nearest_resource.position[0])**2 + 
                           (harvester.position[1] - nearest_resource.position[1])**2)**0.5
                
                if distance <= 1.5:
                    return harvester.harvest(nearest_resource, self.env)
                else:
                    dx = np.sign(nearest_resource.position[0] - harvester.position[0])
                    dy = np.sign(nearest_resource.position[1] - harvester.position[1])
                    
                    if abs(dx) > abs(dy):
                        direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
                    else:
                        direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
                    
                    return harvester.move(direction, self.env)
        
        return False
    
    def _command_warrior_attack(self, game_state: Dict) -> bool:
        """Command warriors to attack enemies."""
        warriors = [u for u in game_state['player_units'] 
                   if u.type == UnitType.WARRIOR]
        enemies = game_state['enemy_units']
        
        if not warriors or not enemies:
            return False
        
        warrior = random.choice(warriors)
        nearest_enemy = min(enemies, 
                          key=lambda e: ((warrior.position[0] - e.position[0])**2 + 
                                        (warrior.position[1] - e.position[1])**2))
        
        # Attack if in range, otherwise move toward enemy
        distance = ((warrior.position[0] - nearest_enemy.position[0])**2 + 
                   (warrior.position[1] - nearest_enemy.position[1])**2)**0.5
        
        if distance <= 2:
            return warrior.attack(nearest_enemy, self.env)
        else:
            dx = np.sign(nearest_enemy.position[0] - warrior.position[0])
            dy = np.sign(nearest_enemy.position[1] - warrior.position[1])
            
            if abs(dx) > abs(dy):
                direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
            else:
                direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
            
            return warrior.move(direction, self.env)
    
    def _command_build_turret(self, game_state: Dict) -> bool:
        """Command turret construction near nexus."""
        nexus = next((s for s in game_state['structures'] 
                     if s.type == StructureType.NEXUS), None)
        
        if not nexus or game_state['crystal_count'] < 150:  # Turret cost
            return False
        
        # Find position near nexus for turret
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                pos = (nexus.position[0] + dx, nexus.position[1] + dy)
                if not self.env.is_position_occupied(pos):
                    # Create turret (simplified - in full implementation, 
                    # this would go through proper construction process)
                    turret = Structure(self.env.next_structure_id, 
                                     StructureType.TURRET, pos)
                    self.env.structures.append(turret)
                    self.env.next_structure_id += 1
                    game_state['crystal_count'] -= 150
                    return True
        
        return False
    
    def _command_exploration(self, game_state: Dict) -> bool:
        """Command units to explore unknown areas."""
        units = [u for u in game_state['player_units'] 
                if u.type in [UnitType.WARRIOR, UnitType.HARVESTER]]
        
        if not units:
            return False
        
        unit = random.choice(units)
        
        # Simple exploration - move toward map edges
        map_center = (self.env.visibility.shape[0] // 2, 
                     self.env.visibility.shape[1] // 2)
        
        # Move away from center
        dx = 1 if unit.position[0] < map_center[0] else -1
        dy = 1 if unit.position[1] < map_center[1] else -1
        
        if random.random() > 0.5:
            direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
        else:
            direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
        
        return unit.move(direction, self.env)
    
    def _command_defense(self, game_state: Dict) -> bool:
        """Command units to defend key positions."""
        nexus = next((s for s in game_state['structures'] 
                     if s.type == StructureType.NEXUS), None)
        
        if not nexus:
            return False
        
        # Move warriors toward nexus for defense
        warriors = [u for u in game_state['player_units'] 
                   if u.type == UnitType.WARRIOR]
        
        if not warriors:
            return False
        
        warrior = random.choice(warriors)
        
        # Move toward nexus if far away
        distance = ((warrior.position[0] - nexus.position[0])**2 + 
                   (warrior.position[1] - nexus.position[1])**2)**0.5
        
        if distance > 5:
            dx = np.sign(nexus.position[0] - warrior.position[0])
            dy = np.sign(nexus.position[1] - warrior.position[1])
            
            if abs(dx) > abs(dy):
                direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
            else:
                direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
            
            return warrior.move(direction, self.env)
        
        return False
    
    def update_q_tables(self):
        """Update all Q-tables from their respective experience buffers."""
        # Update strategic Q-table
        if len(self.strategic_buffer) > 0:
            batch_size = min(10, len(self.strategic_buffer))
            batch = random.sample(self.strategic_buffer, batch_size)
            
            for state, action, reward, next_state, done in batch:
                current_q = self.Q_strategic[state][action]
                next_q = np.max(self.Q_strategic[next_state]) if not done else 0
                target_q = reward + self.gamma * next_q
                self.Q_strategic[state][action] = current_q + self.alpha * (target_q - current_q)
        
        # Update tactical Q-table
        if len(self.tactical_buffer) > 0:
            batch_size = min(10, len(self.tactical_buffer))
            batch = random.sample(self.tactical_buffer, batch_size)
            
            for state, action, reward, next_state, done in batch:
                current_q = self.Q_tactical[state][action]
                next_q = np.max(self.Q_tactical[next_state]) if not done else 0
                target_q = reward + self.gamma * next_q
                self.Q_tactical[state][action] = current_q + self.alpha * (target_q - current_q)
        
        # Update operational Q-table
        if len(self.operational_buffer) > 0:
            batch_size = min(10, len(self.operational_buffer))
            batch = random.sample(self.operational_buffer, batch_size)
            
            for state, action, reward, next_state, done in batch:
                current_q = self.Q_operational[state][action]
                next_q = np.max(self.Q_operational[next_state]) if not done else 0
                target_q = reward + self.gamma * next_q
                self.Q_operational[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def act(self, game_state: Dict, epsilon: float = 0.1) -> bool:
        """
        Main action selection method that coordinates all hierarchical levels.
        
        Returns True if any action was successfully executed.
        """
        # Get state representations
        strategic_state = self.get_strategic_state(game_state)
        tactical_state = self.get_tactical_state(game_state)
        operational_state = self.get_operational_state(game_state)
        
        # Choose actions at each level
        strategic_action = self.choose_strategic_action(strategic_state, epsilon)
        tactical_action = self.choose_tactical_action(tactical_state, epsilon)
        operational_action = self.choose_operational_action(operational_state, epsilon)
        
        # Execute hierarchical actions
        tactical_weights = self.execute_strategic_action(strategic_action, game_state)
        operational_weights = self.execute_tactical_action(tactical_action, game_state, tactical_weights)
        action_success = self.execute_operational_action(operational_action, game_state)
        
        # Store states for learning updates
        self.last_strategic_state = strategic_state
        self.last_tactical_state = tactical_state
        self.last_operational_state = operational_state
        self.last_strategic_action = strategic_action
        self.last_tactical_action = tactical_action
        self.last_operational_action = operational_action
        
        return action_success
    
    def learn(self, reward: float, next_game_state: Dict, done: bool):
        """
        Update Q-tables based on received reward and next state.
        
        Hierarchical reward decomposition assigns different rewards 
        to different planning levels.
        """
        if hasattr(self, 'last_strategic_state'):
            # Get next states
            next_strategic_state = self.get_strategic_state(next_game_state)
            next_tactical_state = self.get_tactical_state(next_game_state)
            next_operational_state = self.get_operational_state(next_game_state)
            
            # Decompose reward across levels
            strategic_reward = reward * 0.1  # Long-term component
            tactical_reward = reward * 0.3   # Medium-term component
            operational_reward = reward * 0.6  # Immediate component
            
            # Add experiences to buffers
            self.strategic_buffer.append((
                self.last_strategic_state, self.last_strategic_action,
                strategic_reward, next_strategic_state, done
            ))
            
            self.tactical_buffer.append((
                self.last_tactical_state, self.last_tactical_action,
                tactical_reward, next_tactical_state, done
            ))
            
            self.operational_buffer.append((
                self.last_operational_state, self.last_operational_action,
                operational_reward, next_operational_state, done
            ))
            
            # Update Q-tables
            self.update_q_tables()
    
    def train(self, episodes: int = 500, max_steps_per_episode: int = 1000):
        """
        Train the RTS agent through multiple episodes.
        
        Returns training log with episode metrics.
        """
        training_log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Reset environment
            self.env = RTSEnvironment(seed=episode, enable_novel_features=True)
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Get current game state
                game_state = self.env.get_state()
                
                # Take action
                action_taken = self.act(game_state, epsilon)
                
                # Environment step
                self.env.step()
                episode_steps += 1
                
                # Get reward and next state
                reward = self.env.get_reward()
                next_game_state = self.env.get_state()
                done = self.env.is_game_over()
                
                # Learn from experience
                self.learn(reward, next_game_state, done)
                
                episode_reward += reward
                
                if done:
                    break
            
            # Decay epsilon
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            # Log episode metrics
            training_log.append({
                'episode': episode,
                'steps': episode_steps,
                'reward': episode_reward,
                'epsilon': epsilon,
                'strategy': self.current_strategy
            })
            
            # Periodic reporting
            if episode % 50 == 0:
                avg_reward = np.mean([log['reward'] for log in training_log[-50:]])
                avg_steps = np.mean([log['steps'] for log in training_log[-50:]])
                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.1f}, ε={epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final performance: {training_log[-1]['reward']:.2f} reward in {training_log[-1]['steps']} steps")
        
        return training_log, training_time 