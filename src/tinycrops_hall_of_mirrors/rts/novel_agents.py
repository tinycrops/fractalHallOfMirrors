"""
Novel RTS Agents: Porting Grid-World Innovations to RTS Domain.

This module implements the cutting-edge agent innovations proven in grid-world,
adapted for the complexity of the RTS environment:

- CuriosityDrivenRTSAgent: Explores tech trees, map areas, and unit compositions
- MultiHeadRTSAgent: Specialized attention for economy, military, defense, scouting
- AdaptiveRTSAgent: Dynamically adjusts strategic focus and tactical horizons
- MetaLearningRTSAgent: Cross-game strategy transfer and pattern recognition
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import time
import json

from .agents import BaseRTSAgent
from .environment import (
    RTSEnvironment, UnitType, StructureType, ResourceType, ActionType,
    MAP_SIZE
)
from ..common.q_learning_utils import decay_epsilon


class CuriosityDrivenRTSAgent(BaseRTSAgent):
    """
    Novel Innovation: Curiosity-driven exploration for RTS environments.
    
    Applies intrinsic motivation to explore:
    - Unknown map areas and strategic positions
    - New unit compositions and tactical formations
    - Tech tree advancement and timing strategies
    - Resource management patterns
    """
    
    def __init__(self, env: RTSEnvironment, curiosity_weight=0.1, 
                 map_exploration_bonus=2.0, tech_discovery_bonus=5.0,
                 **kwargs):
        super().__init__(env, **kwargs)
        
        self.curiosity_weight = curiosity_weight
        self.map_exploration_bonus = map_exploration_bonus
        self.tech_discovery_bonus = tech_discovery_bonus
        
        # Exploration tracking
        self.map_exploration_history = np.zeros((MAP_SIZE, MAP_SIZE))
        self.unit_composition_history = defaultdict(int)
        self.tech_sequence_history = []
        self.strategic_timing_history = []
        
        # Novelty detection
        self.map_position_counts = defaultdict(int)
        self.tactical_pattern_counts = defaultdict(int)
        self.strategic_sequence_counts = defaultdict(int)
        
        # Intrinsic reward tracking
        self.intrinsic_rewards = deque(maxlen=10000)
        self.exploration_targets = []
        
    def calculate_map_exploration_bonus(self, game_state: Dict) -> float:
        """Calculate intrinsic reward for exploring new map areas."""
        exploration_bonus = 0.0
        
        # Reward for units in previously unexplored areas
        for unit in game_state['player_units']:
            x, y = unit.position
            visit_count = self.map_position_counts[(x, y)]
            
            if visit_count == 0:
                # First time visiting this area - high bonus
                exploration_bonus += self.map_exploration_bonus
            elif visit_count < 3:
                # Still relatively novel - smaller bonus
                exploration_bonus += self.map_exploration_bonus * 0.3
            
            # Update visit count
            self.map_position_counts[(x, y)] += 1
            self.map_exploration_history[x, y] += 1
        
        return exploration_bonus
    
    def calculate_tactical_novelty_bonus(self, game_state: Dict) -> float:
        """Calculate intrinsic reward for novel tactical patterns."""
        novelty_bonus = 0.0
        
        # Current unit composition
        composition = {}
        for unit in game_state['player_units']:
            composition[unit.type] = composition.get(unit.type, 0) + 1
        
        composition_key = tuple(sorted(composition.items()))
        composition_count = self.unit_composition_history[composition_key]
        
        if composition_count < 5:  # Novel composition
            novelty_bonus += (5 - composition_count) * 0.5
        
        self.unit_composition_history[composition_key] += 1
        
        # Resource allocation patterns
        crystal_ratio = game_state['crystal_count'] / max(1, game_state['time'])
        resource_pattern = f"crystals_{int(crystal_ratio*10)}"
        
        if self.tactical_pattern_counts[resource_pattern] < 3:
            novelty_bonus += 0.3
        
        self.tactical_pattern_counts[resource_pattern] += 1
        
        return novelty_bonus
    
    def calculate_strategic_timing_bonus(self, game_state: Dict) -> float:
        """Calculate intrinsic reward for novel strategic timing."""
        timing_bonus = 0.0
        
        # Track strategic decision timing
        if self.current_strategy:
            timing_key = f"{self.current_strategy}_{game_state['time']//100}"
            
            if self.strategic_sequence_counts[timing_key] < 3:
                timing_bonus += 1.0 - (self.strategic_sequence_counts[timing_key] * 0.3)
            
            self.strategic_sequence_counts[timing_key] += 1
        
        return timing_bonus
    
    def get_intrinsic_reward(self, game_state: Dict) -> float:
        """Calculate total intrinsic reward for curiosity-driven learning."""
        map_bonus = self.calculate_map_exploration_bonus(game_state)
        tactical_bonus = self.calculate_tactical_novelty_bonus(game_state)
        strategic_bonus = self.calculate_strategic_timing_bonus(game_state)
        
        total_intrinsic = map_bonus + tactical_bonus + strategic_bonus
        self.intrinsic_rewards.append(total_intrinsic)
        
        return total_intrinsic * self.curiosity_weight
    
    def learn(self, reward: float, next_game_state: Dict, done: bool):
        """Enhanced learning with intrinsic motivation."""
        # Calculate intrinsic reward
        intrinsic_reward = self.get_intrinsic_reward(next_game_state)
        
        # Combine extrinsic and intrinsic rewards
        total_reward = reward + intrinsic_reward
        
        # Call parent learning with enhanced reward
        super().learn(total_reward, next_game_state, done)


class MultiHeadRTSAgent(BaseRTSAgent):
    """
    Novel Innovation: Multi-head attention for RTS strategic focus.
    
    Implements specialized attention heads:
    - Economy Head: Resource management and expansion
    - Military Head: Unit production and combat tactics  
    - Defense Head: Base protection and threat response
    - Scouting Head: Map exploration and intelligence
    """
    
    def __init__(self, env: RTSEnvironment, num_heads=4, attention_lr=0.05, **kwargs):
        super().__init__(env, **kwargs)
        
        self.num_heads = num_heads
        self.attention_lr = attention_lr
        
        # Attention head names and specializations
        self.head_names = ['economy', 'military', 'defense', 'scouting']
        
        # Attention weights for each head (strategic, tactical, operational)
        self.attention_weights = {
            'economy': np.array([0.4, 0.6, 0.5]),     # Focus on tactical resource management
            'military': np.array([0.3, 0.4, 0.7]),   # Focus on operational unit control
            'defense': np.array([0.2, 0.5, 0.6]),    # Balanced with operational emphasis
            'scouting': np.array([0.1, 0.3, 0.8])    # Heavy operational focus
        }
        
        # Current active head and switching history
        self.active_head = 'economy'  # Start with economy
        self.head_switching_history = []
        self.head_performance = defaultdict(list)
        
        # Head-specific Q-tables for enhanced specialization
        self.head_tactical_preferences = {
            'economy': np.array([0.2, 2.0, 0.5, 1.5, 0.3]),    # gather_resources, build_structures
            'military': np.array([0.5, 0.5, 2.0, 0.8, 1.8]),   # train_units, attack_enemy
            'defense': np.array([0.3, 0.8, 0.5, 2.0, 0.8]),    # build_structures (turrets)
            'scouting': np.array([2.0, 0.3, 0.3, 0.5, 0.5])    # scout_map
        }
        
        # Attention evolution tracking
        self.attention_evolution = []
        self.head_activation_counts = defaultdict(int)
        
    def select_attention_head(self, game_state: Dict) -> str:
        """Dynamically select the most appropriate attention head."""
        # Calculate head relevance scores
        scores = {}
        
        # Economy head relevance
        economy_need = 1.0 - min(game_state['crystal_count'] / 500, 1.0)
        resource_pressure = len([r for r in game_state['resources'] if r.amount < 50]) / max(len(game_state['resources']), 1)
        scores['economy'] = economy_need + resource_pressure
        
        # Military head relevance  
        enemy_strength = len(game_state['enemy_units'])
        player_military = len([u for u in game_state['player_units'] if u.type == UnitType.WARRIOR])
        military_ratio = player_military / max(enemy_strength + 1, 1)
        scores['military'] = max(0, 2.0 - military_ratio)
        
        # Defense head relevance
        immediate_threats = len([u for u in game_state['enemy_units'] 
                               if self._distance_to_nexus(u) < 15])
        turret_count = len([s for s in game_state['structures'] if s.type == StructureType.TURRET])
        defense_need = immediate_threats * 0.5 + max(0, 2 - turret_count) * 0.3
        scores['defense'] = defense_need
        
        # Scouting head relevance
        # Simple heuristic based on game time for exploration need
        time_factor = game_state['time'] / 1000.0
        exploration_need = max(0.2, 1.0 - time_factor)  # Higher need early game
        scores['scouting'] = exploration_need
        
        # Select head with highest relevance (with some randomness)
        if random.random() < 0.8:  # 80% optimal, 20% exploration
            selected_head = max(scores, key=scores.get)
        else:
            selected_head = random.choice(self.head_names)
        
        # Track head switching
        if selected_head != self.active_head:
            self.head_switching_history.append({
                'time': game_state['time'],
                'from': self.active_head,
                'to': selected_head,
                'scores': scores.copy()
            })
        
        self.active_head = selected_head
        self.head_activation_counts[selected_head] += 1
        
        return selected_head
    
    def execute_tactical_action(self, action_idx: int, game_state: Dict, strategic_weights=None):
        """Execute tactical action with head-specific preferences."""
        # Apply active head's tactical preferences
        head_preferences = self.head_tactical_preferences[self.active_head]
        
        # Modify the action based on head preferences
        if random.random() < 0.7:  # 70% chance to follow head preference
            preferred_action = np.argmax(head_preferences)
            action_idx = preferred_action
        
        # Record attention evolution
        attention_snapshot = {
            'time': game_state['time'],
            'active_head': self.active_head,
            'tactical_action': self.tactical_actions[action_idx],
            'head_weights': self.attention_weights[self.active_head].copy()
        }
        self.attention_evolution.append(attention_snapshot)
        
        return super().execute_tactical_action(action_idx, game_state, strategic_weights)
    
    def act(self, game_state: Dict, epsilon: float = 0.1) -> bool:
        """Enhanced action selection with multi-head attention."""
        # Select appropriate attention head
        self.select_attention_head(game_state)
        
        # Apply attention weights to action selection
        attention_weights = self.attention_weights[self.active_head]
        
        # Modify epsilon based on head confidence
        head_confidence = max(0.3, 1.0 - len(self.head_switching_history) * 0.01)
        adjusted_epsilon = epsilon * (2.0 - head_confidence)
        
        return super().act(game_state, adjusted_epsilon)
    
    def get_attention_metrics(self) -> Dict:
        """Get detailed attention analysis metrics."""
        return {
            'active_head': self.active_head,
            'head_activation_counts': dict(self.head_activation_counts),
            'total_switches': len(self.head_switching_history),
            'attention_diversity': len(set(h['to'] for h in self.head_switching_history[-10:])),
            'recent_switches': self.head_switching_history[-5:] if self.head_switching_history else []
        }


class AdaptiveRTSAgent(BaseRTSAgent):
    """
    Novel Innovation: Adaptive strategy and hierarchy adjustment for RTS.
    
    Dynamically adapts:
    - Strategic planning horizons based on game phase
    - Tactical response patterns based on enemy behavior
    - Resource allocation strategies based on map characteristics
    - Unit production priorities based on success patterns
    """
    
    def __init__(self, env: RTSEnvironment, adaptation_rate=0.1, 
                 min_horizon=50, max_horizon=500, **kwargs):
        super().__init__(env, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        
        # Adaptive parameters
        self.current_strategic_horizon = self.strategic_horizon
        self.current_tactical_horizon = self.tactical_horizon
        
        # Performance tracking for adaptation
        self.strategy_performance_history = defaultdict(list)
        self.horizon_effectiveness = defaultdict(list)
        self.adaptation_triggers = []
        
        # Game phase detection
        self.game_phase_metrics = {
            'early': {'resource_focus': 0.7, 'military_focus': 0.2, 'defense_focus': 0.1},
            'mid': {'resource_focus': 0.4, 'military_focus': 0.4, 'defense_focus': 0.2},
            'late': {'resource_focus': 0.2, 'military_focus': 0.5, 'defense_focus': 0.3}
        }
        
        # Adaptation memory
        self.successful_adaptations = []
        self.failed_adaptations = []
        
    def detect_game_phase(self, game_state: Dict) -> str:
        """Detect current game phase for phase-specific adaptations."""
        time = game_state['time']
        enemy_count = len(game_state['enemy_units'])
        player_units = len(game_state['player_units'])
        
        # Multi-factor phase detection
        time_factor = time / 1000.0  # Normalize time
        military_factor = (player_units + enemy_count) / 20.0  # Unit density
        resource_factor = game_state['crystal_count'] / 1000.0  # Resource accumulation
        
        phase_score = time_factor + military_factor + resource_factor
        
        if phase_score < 0.3:
            return 'early'
        elif phase_score < 0.7:
            return 'mid'
        else:
            return 'late'
    
    def analyze_performance_variance(self) -> float:
        """Analyze recent performance to trigger adaptations."""
        if len(self.strategy_performance) < 5:
            return 0.0
        
        recent_performance = self.strategy_performance[-5:]
        variance = np.var(recent_performance)
        
        return variance
    
    def adapt_strategic_horizon(self, game_state: Dict, performance_variance: float):
        """Adapt strategic planning horizon based on game conditions."""
        game_phase = self.detect_game_phase(game_state)
        
        # Base adaptation on game phase
        if game_phase == 'early':
            target_horizon = self.max_horizon  # Long-term planning
        elif game_phase == 'mid':
            target_horizon = (self.min_horizon + self.max_horizon) // 2  # Balanced
        else:
            target_horizon = self.min_horizon  # Tactical focus
        
        # Adjust based on performance variance
        if performance_variance > 100:  # High variance - more adaptive
            target_horizon = int(target_horizon * 0.8)
        elif performance_variance < 20:  # Low variance - more strategic
            target_horizon = int(target_horizon * 1.2)
        
        # Smooth adaptation
        self.current_strategic_horizon = int(
            self.current_strategic_horizon * (1 - self.adaptation_rate) +
            target_horizon * self.adaptation_rate
        )
        
        # Clamp to bounds
        self.current_strategic_horizon = max(self.min_horizon, 
                                           min(self.max_horizon, self.current_strategic_horizon))
    
    def adapt_tactical_preferences(self, game_state: Dict):
        """Adapt tactical action preferences based on current conditions."""
        game_phase = self.detect_game_phase(game_state)
        phase_preferences = self.game_phase_metrics[game_phase]
        
        # Calculate threat level
        immediate_threats = len([u for u in game_state['enemy_units'] 
                               if self._distance_to_nexus(u) < 20])
        threat_level = min(immediate_threats / 5.0, 1.0)
        
        # Adapt strategic action weights
        adapted_weights = np.ones(len(self.strategic_actions))
        
        if threat_level > 0.5:  # High threat - defensive/military focus
            adapted_weights[2] *= (1 + threat_level)  # build_military
            adapted_weights[3] *= (1 + threat_level)  # defensive_posture
            adapted_weights[0] *= (1 - threat_level * 0.5)  # reduce expand_economy
        else:  # Low threat - economic focus
            adapted_weights[0] *= (1 + phase_preferences['resource_focus'])  # expand_economy
            adapted_weights[2] *= phase_preferences['military_focus']  # build_military
        
        return adapted_weights
    
    def record_adaptation(self, adaptation_type: str, before_value, after_value, 
                         expected_improvement: float):
        """Record adaptation for learning purposes."""
        adaptation_record = {
            'type': adaptation_type,
            'time': time.time(),
            'before': before_value,
            'after': after_value,
            'expected_improvement': expected_improvement
        }
        self.adaptation_triggers.append(adaptation_record)
    
    def execute_strategic_action(self, action_idx: int, game_state: Dict):
        """Execute strategic action with adaptive modifications."""
        # Analyze current conditions for adaptation
        performance_variance = self.analyze_performance_variance()
        
        # Trigger adaptations if needed
        if performance_variance > 80:  # Significant variance
            old_horizon = self.current_strategic_horizon
            self.adapt_strategic_horizon(game_state, performance_variance)
            
            if abs(old_horizon - self.current_strategic_horizon) > 10:
                self.record_adaptation('strategic_horizon', old_horizon, 
                                     self.current_strategic_horizon, 0.1)
        
        # Get adapted tactical weights
        adapted_weights = self.adapt_tactical_preferences(game_state)
        
        # Execute with adaptations
        tactical_weights = super().execute_strategic_action(action_idx, game_state)
        
        # Apply adaptive modifications
        for i, weight in enumerate(adapted_weights):
            if i < len(tactical_weights):
                tactical_weights[i] *= weight
        
        return tactical_weights
    
    def get_adaptation_metrics(self) -> Dict:
        """Get detailed adaptation analysis metrics."""
        return {
            'current_strategic_horizon': self.current_strategic_horizon,
            'current_tactical_horizon': self.current_tactical_horizon,
            'total_adaptations': len(self.adaptation_triggers),
            'successful_adaptations': len(self.successful_adaptations),
            'adaptation_rate': self.adaptation_rate,
            'recent_adaptations': self.adaptation_triggers[-5:] if self.adaptation_triggers else []
        }


class MetaLearningRTSAgent(BaseRTSAgent):
    """
    Novel Innovation: Meta-learning for cross-game strategy transfer.
    
    Builds and maintains:
    - Strategy library from successful game patterns
    - Environment characteristic recognition
    - Cross-game knowledge transfer
    - Adaptive strategy selection based on similarity
    """
    
    def __init__(self, env: RTSEnvironment, strategy_memory_size=50, 
                 similarity_threshold=0.7, **kwargs):
        super().__init__(env, **kwargs)
        
        self.strategy_memory_size = strategy_memory_size
        self.similarity_threshold = similarity_threshold
        
        # Meta-learning components
        self.strategy_library = []
        self.environment_characteristics = {}
        self.strategy_effectiveness = defaultdict(list)
        self.cross_game_patterns = defaultdict(list)
        
        # Current game analysis
        self.current_game_signature = {}
        self.strategy_sequence = []
        self.decision_contexts = []
        
        # Learning and transfer metrics
        self.successful_transfers = 0
        self.failed_transfers = 0
        self.novel_discoveries = 0
        
    def analyze_environment_characteristics(self, game_state: Dict) -> Dict:
        """Analyze and characterize the current environment."""
        characteristics = {}
        
        # Map characteristics
        total_resources = sum(r.amount for r in game_state['resources'])
        resource_density = total_resources / (MAP_SIZE * MAP_SIZE)
        characteristics['resource_density'] = resource_density
        
        # Enemy characteristics
        enemy_aggression = len(game_state['enemy_units']) / max(game_state['time'], 1)
        characteristics['enemy_aggression'] = enemy_aggression
        
        # Spatial characteristics
        resource_positions = [r.position for r in game_state['resources']]
        if resource_positions:
            resource_spread = np.std([pos[0] + pos[1] for pos in resource_positions])
            characteristics['resource_spread'] = resource_spread
        else:
            characteristics['resource_spread'] = 0.0
        
        # Temporal characteristics
        characteristics['game_pace'] = game_state['time'] / max(len(game_state['player_units']) + len(game_state['enemy_units']), 1)
        
        return characteristics
    
    def calculate_environment_similarity(self, char1: Dict, char2: Dict) -> float:
        """Calculate similarity between two environment characteristic sets."""
        if not char1 or not char2:
            return 0.0
        
        # Normalize and compare key characteristics
        similarities = []
        
        for key in ['resource_density', 'enemy_aggression', 'resource_spread', 'game_pace']:
            if key in char1 and key in char2:
                # Normalized difference (closer to 1 = more similar)
                max_val = max(char1[key], char2[key], 1e-6)
                similarity = 1.0 - abs(char1[key] - char2[key]) / max_val
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def find_similar_strategy(self, current_characteristics: Dict) -> Optional[Dict]:
        """Find the most similar strategy from the library."""
        best_similarity = 0.0
        best_strategy = None
        
        for strategy in self.strategy_library:
            similarity = self.calculate_environment_similarity(
                current_characteristics, strategy['environment_characteristics']
            )
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_strategy = strategy
        
        return best_strategy
    
    def extract_strategy_pattern(self, strategy_sequence: List, decision_contexts: List) -> Dict:
        """Extract reusable patterns from a strategy sequence."""
        if not strategy_sequence:
            return {}
        
        pattern = {
            'strategy_sequence': strategy_sequence.copy(),
            'dominant_strategy': max(set(strategy_sequence), key=strategy_sequence.count),
            'strategy_transitions': [],
            'timing_patterns': [],
            'context_triggers': []
        }
        
        # Analyze strategy transitions
        for i in range(len(strategy_sequence) - 1):
            transition = (strategy_sequence[i], strategy_sequence[i + 1])
            pattern['strategy_transitions'].append(transition)
        
        # Analyze timing patterns
        strategy_timings = defaultdict(list)
        for i, strategy in enumerate(strategy_sequence):
            if i < len(decision_contexts):
                timing = decision_contexts[i].get('time', i)
                strategy_timings[strategy].append(timing)
        
        pattern['timing_patterns'] = dict(strategy_timings)
        
        return pattern
    
    def store_successful_strategy(self, final_performance: float):
        """Store current game strategy if it was successful."""
        if final_performance > 100:  # Threshold for "successful"
            strategy_record = {
                'environment_characteristics': self.current_game_signature.copy(),
                'strategy_pattern': self.extract_strategy_pattern(self.strategy_sequence, self.decision_contexts),
                'performance': final_performance,
                'timestamp': time.time(),
                'game_length': len(self.strategy_sequence)
            }
            
            # Add to library (with size limit)
            self.strategy_library.append(strategy_record)
            if len(self.strategy_library) > self.strategy_memory_size:
                # Remove oldest strategy
                self.strategy_library.pop(0)
            
            self.novel_discoveries += 1
    
    def apply_transferred_strategy(self, similar_strategy: Dict, game_state: Dict) -> str:
        """Apply insights from a similar strategy."""
        strategy_pattern = similar_strategy['strategy_pattern']
        
        # Get recommended strategy based on game phase
        game_time = game_state['time']
        
        # Find appropriate strategy for current timing
        timing_patterns = strategy_pattern.get('timing_patterns', {})
        current_recommendations = []
        
        for strategy, timings in timing_patterns.items():
            if timings and min(timings) <= game_time <= max(timings):
                current_recommendations.append(strategy)
        
        if current_recommendations:
            selected_strategy = random.choice(current_recommendations)
            self.successful_transfers += 1
            return selected_strategy
        else:
            # Fall back to dominant strategy
            dominant = strategy_pattern.get('dominant_strategy')
            if dominant:
                self.successful_transfers += 1
                return dominant
            else:
                self.failed_transfers += 1
                return None
    
    def execute_strategic_action(self, action_idx: int, game_state: Dict):
        """Execute strategic action with meta-learning insights."""
        # Update current game characteristics
        self.current_game_signature = self.analyze_environment_characteristics(game_state)
        
        # Record current decision context
        context = {
            'time': game_state['time'],
            'crystal_count': game_state['crystal_count'],
            'unit_count': len(game_state['player_units']),
            'enemy_count': len(game_state['enemy_units'])
        }
        self.decision_contexts.append(context)
        
        # Look for similar strategies to transfer
        similar_strategy = self.find_similar_strategy(self.current_game_signature)
        
        if similar_strategy:
            # Try to apply transferred strategy
            recommended_strategy = self.apply_transferred_strategy(similar_strategy, game_state)
            
            if recommended_strategy and recommended_strategy in self.strategic_actions:
                # Override action with transferred knowledge
                transferred_action_idx = self.strategic_actions.index(recommended_strategy)
                action_idx = transferred_action_idx
        
        # Record strategy selection
        selected_strategy = self.strategic_actions[action_idx]
        self.strategy_sequence.append(selected_strategy)
        
        return super().execute_strategic_action(action_idx, game_state)
    
    def train(self, episodes: int = 500, max_steps_per_episode: int = 1000):
        """Enhanced training with meta-learning across episodes."""
        training_log = []
        epsilon = self.epsilon_start
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Reset episode-specific tracking
            self.strategy_sequence = []
            self.decision_contexts = []
            
            # Environment with different characteristics each episode
            self.env = RTSEnvironment(seed=episode, enable_novel_features=True)
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                game_state = self.env.get_state()
                
                action_taken = self.act(game_state, epsilon)
                
                self.env.step()
                episode_steps += 1
                
                reward = self.env.get_reward()
                next_game_state = self.env.get_state()
                done = self.env.is_game_over()
                
                self.learn(reward, next_game_state, done)
                episode_reward += reward
                
                if done:
                    break
            
            # Store successful strategies for future transfer
            self.store_successful_strategy(episode_reward)
            
            epsilon = decay_epsilon(epsilon, self.epsilon_end, self.epsilon_decay)
            
            training_log.append({
                'episode': episode,
                'steps': episode_steps,
                'reward': episode_reward,
                'epsilon': epsilon,
                'strategy': self.current_strategy,
                'transfers': self.successful_transfers,
                'library_size': len(self.strategy_library)
            })
            
            if episode % 50 == 0:
                avg_reward = np.mean([log['reward'] for log in training_log[-50:]])
                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                      f"Library Size={len(self.strategy_library)}, "
                      f"Transfers={self.successful_transfers}")
        
        training_time = time.time() - start_time
        
        print(f"Meta-learning training completed in {training_time:.2f} seconds")
        print(f"Final strategy library size: {len(self.strategy_library)}")
        print(f"Successful transfers: {self.successful_transfers}")
        print(f"Novel discoveries: {self.novel_discoveries}")
        
        return training_log, training_time
    
    def get_meta_learning_metrics(self) -> Dict:
        """Get detailed meta-learning analysis metrics."""
        return {
            'strategy_library_size': len(self.strategy_library),
            'successful_transfers': self.successful_transfers,
            'failed_transfers': self.failed_transfers,
            'novel_discoveries': self.novel_discoveries,
            'current_game_characteristics': self.current_game_signature,
            'recent_strategy_sequence': self.strategy_sequence[-10:] if self.strategy_sequence else [],
            'transfer_success_rate': self.successful_transfers / max(self.successful_transfers + self.failed_transfers, 1)
        } 