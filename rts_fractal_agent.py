#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from enum import Enum, auto
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from rts_environment import RTSEnvironment, UnitType, StructureType, ResourceType, ActionType, MAP_SIZE

# Constants
EXPERIENCE_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# Strategic decisions (super-level)
class StrategicFocus(Enum):
    ECONOMY = auto()       # Focus on resource gathering and harvester production
    DEFENSE = auto()       # Focus on warrior production and defensive positioning
    EXPANSION = auto()     # Focus on exploring and securing new resources
    VESPENE = auto()       # Focus on securing and harvesting the vespene geyser

# Tactical decisions (meso-level)
class TacticalObjective(Enum):
    ASSIGN_HARVESTERS = auto()  # Assign harvesters to specific resource patch
    BUILD_HARVESTER = auto()    # Build a harvester unit
    BUILD_WARRIOR = auto()      # Build a warrior unit
    BUILD_TURRET = auto()       # Build a defensive turret
    RALLY_WARRIORS = auto()     # Rally warriors at a specific location
    SCOUT = auto()              # Send a unit to explore

# Helper function to convert game state to a simplified representation
def extract_features(state, level='micro'):
    """
    Extract features from the game state based on the level of abstraction.
    Returns a numpy array of features.
    """
    # Basic information available at all levels
    time = state['time']
    crystal_count = state['crystal_count']
    vespene_count = state['vespene_count']
    visibility = state['visibility']
    
    player_units = state['player_units']
    enemy_units = state['enemy_units']
    structures = state['structures']
    resources = state['resources']
    
    # Count units by type
    harvester_count = sum(1 for u in player_units if u.type == UnitType.HARVESTER)
    warrior_count = sum(1 for u in player_units if u.type == UnitType.WARRIOR)
    enemy_count = len(enemy_units)
    
    # Find Nexus position
    nexus_pos = next((s.position for s in structures if s.type == StructureType.NEXUS), (0, 0))
    
    if level == 'super':
        # Strategic level features - global information
        features = np.zeros(20)  # Adjust size as needed
        
        # Economic indicators
        features[0] = crystal_count / 1000  # Normalized resource count
        features[1] = vespene_count / 1000
        features[2] = harvester_count / 20  # Normalized unit counts
        features[3] = warrior_count / 20
        
        # Threat indicators
        features[4] = enemy_count / 20
        
        # Resource indicators
        crystal_patches = sum(1 for r in resources if r.type == ResourceType.CRYSTAL)
        vespene_geysers = sum(1 for r in resources if r.type == ResourceType.VESPENE)
        features[5] = crystal_patches / 10
        features[6] = vespene_geysers
        
        # Map exploration (percentage of map explored)
        features[7] = np.mean(visibility)
        
        # Time factor
        features[8] = min(time / 1000, 1.0)  # Normalized time
        
        # Nearest enemy to Nexus
        min_enemy_dist = float('inf')
        for enemy in enemy_units:
            dist = ((enemy.position[0] - nexus_pos[0])**2 + 
                    (enemy.position[1] - nexus_pos[1])**2)**0.5
            min_enemy_dist = min(min_enemy_dist, dist)
        
        features[9] = 1.0 - min(min_enemy_dist / MAP_SIZE, 1.0)  # Closer enemies = higher value
        
        return features
    
    elif level == 'meso':
        # Tactical level features - regional information
        features = np.zeros(30)  # Adjust size as needed
        
        # Resource regions
        crystal_regions = {}
        for r in resources:
            if r.type == ResourceType.CRYSTAL:
                region_key = (r.position[0] // 16, r.position[1] // 16)  # 4 regions per dimension
                if region_key not in crystal_regions:
                    crystal_regions[region_key] = 0
                crystal_regions[region_key] += r.amount
        
        # Enemy regions
        enemy_regions = {}
        for e in enemy_units:
            region_key = (e.position[0] // 16, e.position[1] // 16)
            if region_key not in enemy_regions:
                enemy_regions[region_key] = 0
            enemy_regions[region_key] += 1
        
        # Top resource regions
        for i, (region, amount) in enumerate(sorted(crystal_regions.items(), key=lambda x: x[1], reverse=True)[:3]):
            features[i*3] = region[0] / 4  # Normalize region coordinates
            features[i*3+1] = region[1] / 4
            features[i*3+2] = amount / 1000  # Normalize amount
        
        # Top enemy regions
        for i, (region, count) in enumerate(sorted(enemy_regions.items(), key=lambda x: x[1], reverse=True)[:3]):
            features[10+i*3] = region[0] / 4
            features[10+i*3+1] = region[1] / 4
            features[10+i*3+2] = count / 10
        
        # Current strategic focus (one-hot encoded)
        # This would come from the super-level policy
        # For now, we'll just use a placeholder
        
        # Basic resources and unit counts
        features[20] = crystal_count / 1000
        features[21] = vespene_count / 1000
        features[22] = harvester_count / 20
        features[23] = warrior_count / 20
        features[24] = enemy_count / 20
        
        return features
    
    else:  # micro level
        # Unit-specific features - local information
        unit_features = []
        
        for unit in player_units:
            features = np.zeros(20)  # Adjust size as needed
            
            # Unit type (one-hot encoded)
            if unit.type == UnitType.HARVESTER:
                features[0] = 1
            elif unit.type == UnitType.WARRIOR:
                features[1] = 1
            
            # Unit position (normalized)
            features[2] = unit.position[0] / MAP_SIZE
            features[3] = unit.position[1] / MAP_SIZE
            
            # Unit health (normalized)
            if unit.type == UnitType.HARVESTER:
                features[4] = unit.health / 100
            else:
                features[4] = unit.health / 200
            
            # Resources carried (for harvesters)
            if unit.type == UnitType.HARVESTER:
                features[5] = unit.resources / 10
            
            # Distance to Nexus (normalized)
            dist_to_nexus = ((unit.position[0] - nexus_pos[0])**2 + 
                             (unit.position[1] - nexus_pos[1])**2)**0.5
            features[6] = dist_to_nexus / MAP_SIZE
            
            # Nearest resource
            min_resource_dist = float('inf')
            nearest_resource_type = None
            for r in resources:
                dist = ((unit.position[0] - r.position[0])**2 + 
                        (unit.position[1] - r.position[1])**2)**0.5
                if dist < min_resource_dist:
                    min_resource_dist = dist
                    nearest_resource_type = r.type
            
            features[7] = min_resource_dist / MAP_SIZE
            if nearest_resource_type == ResourceType.CRYSTAL:
                features[8] = 1
            elif nearest_resource_type == ResourceType.VESPENE:
                features[9] = 1
            
            # Nearest enemy
            min_enemy_dist = float('inf')
            for enemy in enemy_units:
                dist = ((unit.position[0] - enemy.position[0])**2 + 
                        (unit.position[1] - enemy.position[1])**2)**0.5
                min_enemy_dist = min(min_enemy_dist, dist)
            
            features[10] = min_enemy_dist / MAP_SIZE
            
            # Current tactical objective (from meso-level)
            # This would come from the meso-level policy
            # For now, we'll just use a placeholder
            
            unit_features.append(features)
        
        # If no units, return a dummy feature vector
        if not unit_features:
            return np.zeros((1, 20))
        
        return np.array(unit_features)

# Simple neural network for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Base class for fractal agent
class FractalAgent:
    def __init__(self):
        # Q-networks for each level
        self.Q_super = QNetwork(20, len(StrategicFocus))  # 20 features, 4 strategic options
        self.Q_meso = QNetwork(30, len(TacticalObjective))  # 30 features, 6 tactical options
        self.Q_micro_harvester = QNetwork(20, 7)  # 20 features, 7 actions (move x4, harvest, return, idle)
        self.Q_micro_warrior = QNetwork(20, 5)    # 20 features, 5 actions (move x4, attack)
        
        # Optimizers
        self.optimizer_super = optim.Adam(self.Q_super.parameters(), lr=LEARNING_RATE)
        self.optimizer_meso = optim.Adam(self.Q_meso.parameters(), lr=LEARNING_RATE)
        self.optimizer_micro_harvester = optim.Adam(self.Q_micro_harvester.parameters(), lr=LEARNING_RATE)
        self.optimizer_micro_warrior = optim.Adam(self.Q_micro_warrior.parameters(), lr=LEARNING_RATE)
        
        # Experience replay buffers
        self.buffer_super = deque(maxlen=EXPERIENCE_BUFFER_SIZE)
        self.buffer_meso = deque(maxlen=EXPERIENCE_BUFFER_SIZE)
        self.buffer_micro_harvester = deque(maxlen=EXPERIENCE_BUFFER_SIZE)
        self.buffer_micro_warrior = deque(maxlen=EXPERIENCE_BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        
        # Current strategic focus and tactical objectives
        self.current_focus = StrategicFocus.ECONOMY  # Default strategy
        self.current_objectives = {}  # Map unit IDs to objectives
        
        # Attention weights (initialize equally)
        self.attention_weights = np.array([1/3, 1/3, 1/3])  # [micro, meso, super]
        self.attention_history = []
    
    def compute_attention_weights(self, state):
        """
        Dynamically compute attention weights based on the current state.
        
        Key insights:
        1. Shift to micro-focus when in combat or precise harvesting
        2. Shift to meso-focus when coordinating unit groups
        3. Shift to super-focus when making strategic decisions or low on resources
        """
        # Extract relevant state information
        crystal_count = state['crystal_count']
        enemy_units = state['enemy_units']
        player_units = state['player_units']
        
        # Find Nexus position
        nexus_pos = next((s.position for s in state['structures'] 
                          if s.type == StructureType.NEXUS), (0, 0))
        
        # Calculate distances and threats
        enemy_near_nexus = False
        combat_happening = False
        harvesting_precision_needed = False
        
        # Check for enemies near Nexus (threat)
        for enemy in enemy_units:
            dist_to_nexus = ((enemy.position[0] - nexus_pos[0])**2 + 
                             (enemy.position[1] - nexus_pos[1])**2)**0.5
            if dist_to_nexus < 10:
                enemy_near_nexus = True
            
            # Check for combat (enemy near player unit)
            for unit in player_units:
                dist_to_enemy = ((unit.position[0] - enemy.position[0])**2 + 
                                 (unit.position[1] - enemy.position[1])**2)**0.5
                if dist_to_enemy < 3:
                    combat_happening = True
        
        # Check for harvesters that are full or close to resources
        for unit in player_units:
            if unit.type == UnitType.HARVESTER:
                if unit.resources >= 8:  # Almost full
                    harvesting_precision_needed = True
                
                # Check distance to resources
                for resource in state['resources']:
                    dist_to_resource = ((unit.position[0] - resource.position[0])**2 + 
                                       (unit.position[1] - resource.position[1])**2)**0.5
                    if dist_to_resource < 2:
                        harvesting_precision_needed = True
        
        # Resource scarcity
        resource_scarcity = crystal_count < 100
        
        # Compute attention weights based on these factors
        attention_logits = np.zeros(3)
        
        # Micro attention (unit control)
        attention_logits[0] = 1.0  # Base value
        if combat_happening:
            attention_logits[0] += 2.0  # High boost for combat
        if harvesting_precision_needed:
            attention_logits[0] += 1.0  # Medium boost for precise harvesting
        
        # Meso attention (tactical)
        attention_logits[1] = 1.0  # Base value
        if enemy_near_nexus:
            attention_logits[1] += 1.5  # High boost for defending nexus
        if len(player_units) > 5:
            attention_logits[1] += 0.5  # Small boost for managing larger groups
        
        # Super attention (strategic)
        attention_logits[2] = 1.0  # Base value
        if resource_scarcity:
            attention_logits[2] += 1.5  # High boost when resources are low
        if state['time'] % 50 == 0:  # Periodically reassess strategy
            attention_logits[2] += 1.0
        
        # Apply softmax to get final attention weights
        def softmax(x):
            e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return e_x / e_x.sum()
        
        self.attention_weights = softmax(attention_logits)
        self.attention_history.append(self.attention_weights.copy())
        
        return self.attention_weights
    
    def choose_action(self, state, unit=None, level='micro'):
        """Choose an action based on the current state and level."""
        if np.random.rand() < self.epsilon:
            # Exploration: choose random action
            if level == 'super':
                return np.random.randint(0, len(StrategicFocus))
            elif level == 'meso':
                return np.random.randint(0, len(TacticalObjective))
            else:  # micro
                if unit.type == UnitType.HARVESTER:
                    return np.random.randint(0, 7)  # 7 actions for harvester
                else:  # WARRIOR
                    return np.random.randint(0, 5)  # 5 actions for warrior
        else:
            # Exploitation: choose best action according to Q-network
            features = extract_features(state, level)
            
            if level == 'super':
                with torch.no_grad():
                    q_values = self.Q_super(torch.FloatTensor(features))
                    return q_values.argmax().item()
            elif level == 'meso':
                with torch.no_grad():
                    q_values = self.Q_meso(torch.FloatTensor(features))
                    return q_values.argmax().item()
            else:  # micro
                unit_idx = next((i for i, u in enumerate(state['player_units']) if u.id == unit.id), 0)
                with torch.no_grad():
                    if unit.type == UnitType.HARVESTER:
                        q_values = self.Q_micro_harvester(torch.FloatTensor(features[unit_idx]))
                        return q_values.argmax().item()
                    else:  # WARRIOR
                        q_values = self.Q_micro_warrior(torch.FloatTensor(features[unit_idx]))
                        return q_values.argmax().item()
    
    def update_q_network(self, buffer, optimizer, q_network, batch_size=BATCH_SIZE):
        """Update Q-network from experience replay buffer."""
        if len(buffer) < batch_size:
            return
        
        # Sample random batch from buffer
        batch = random.sample(buffer, batch_size)
        
        # Extract batch components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Compute current Q values
        current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        next_q_values = q_network(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def update_q_networks(self):
        """Update all Q-networks from their respective experience replay buffers."""
        self.update_q_network(self.buffer_super, self.optimizer_super, self.Q_super)
        self.update_q_network(self.buffer_meso, self.optimizer_meso, self.Q_meso)
        self.update_q_network(self.buffer_micro_harvester, self.optimizer_micro_harvester, self.Q_micro_harvester)
        self.update_q_network(self.buffer_micro_warrior, self.optimizer_micro_warrior, self.Q_micro_warrior)
    
    def decay_epsilon(self):
        """Decay exploration parameter."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def act(self, state, env):
        """Take actions in the environment based on the current state."""
        # Compute dynamic attention weights
        attn_weights = self.compute_attention_weights(state)
        
        # Super-level decision (strategic focus)
        super_action = self.choose_action(state, level='super')
        self.current_focus = list(StrategicFocus)[super_action]
        
        # Meso-level decisions (tactical objectives)
        meso_action = self.choose_action(state, level='meso')
        tactical_objective = list(TacticalObjective)[meso_action]
        
        # Apply the tactical objective based on strategic focus
        self._apply_tactical_objective(tactical_objective, state, env)
        
        # Micro-level decisions (unit actions)
        for unit in state['player_units']:
            micro_action = self.choose_action(state, unit, level='micro')
            self._apply_unit_action(unit, micro_action, state, env, attn_weights)
        
        # Update Q-networks
        self.update_q_networks()
        
        # Decay exploration parameter
        self.decay_epsilon()
    
    def _apply_tactical_objective(self, objective, state, env):
        """Apply a tactical objective based on the current strategic focus."""
        # Implementation depends on the specific objective and focus
        # This is a placeholder for the actual implementation
        pass
    
    def _apply_unit_action(self, unit, action, state, env, attn_weights):
        """Apply a micro-level action to a unit, influenced by attention weights."""
        # Implementation depends on the unit type and action
        # This is a placeholder for the actual implementation
        pass
    
    def save_model(self, path):
        """Save the Q-networks to disk."""
        torch.save({
            'Q_super': self.Q_super.state_dict(),
            'Q_meso': self.Q_meso.state_dict(),
            'Q_micro_harvester': self.Q_micro_harvester.state_dict(),
            'Q_micro_warrior': self.Q_micro_warrior.state_dict()
        }, path)
    
    def load_model(self, path):
        """Load the Q-networks from disk."""
        checkpoint = torch.load(path)
        self.Q_super.load_state_dict(checkpoint['Q_super'])
        self.Q_meso.load_state_dict(checkpoint['Q_meso'])
        self.Q_micro_harvester.load_state_dict(checkpoint['Q_micro_harvester'])
        self.Q_micro_warrior.load_state_dict(checkpoint['Q_micro_warrior']) 