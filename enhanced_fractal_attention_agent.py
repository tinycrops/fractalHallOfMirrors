#!/usr/bin/env python3
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple

from rts_environment import RTSEnvironment, UnitType, StructureType, ResourceType, ActionType, Structure, MAP_SIZE
from rts_fractal_attention_agent import FractalAttentionAgent
from rts_fractal_agent import FractalAgent, StrategicFocus, TacticalObjective, extract_features

class AttentionNetwork(nn.Module):
    """Neural network that learns to generate attention weights based on state features"""
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class GoalNetwork(nn.Module):
    """Neural network that generates goal parameters for lower levels"""
    def __init__(self, input_dim, output_dim, level='super'):
        super().__init__()
        
        # Different architectures based on the hierarchical level
        if level == 'super':
            # Super level generates goals for the meso level
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        elif level == 'meso':
            # Meso level generates goals for the micro level
            self.network = nn.Sequential(
                nn.Linear(input_dim, 48),
                nn.ReLU(),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, output_dim)
            )
    
    def forward(self, x):
        return self.network(x)

class SpatialFeatureExtractor(nn.Module):
    """Neural network that processes spatial information from the game grid"""
    def __init__(self, grid_size=MAP_SIZE, output_dim=32):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Third convolutional layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate the output size after convolutions
        conv_output_size = grid_size // 8  # After 3 layers with stride 2
        self.fc_input_dim = 32 * conv_output_size * conv_output_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # x should be [batch_size, channels, height, width]
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc_layers(x)
        return x
    
    def process_spatial_state(self, state):
        """Process the spatial state to create a tensor suitable for CNN input"""
        # Create channels for different entity types
        grid_size = MAP_SIZE
        
        # Channel 0: Player units
        player_units_grid = np.zeros((grid_size, grid_size))
        for unit in state['player_units']:
            x, y = unit.position
            if 0 <= x < grid_size and 0 <= y < grid_size:
                player_units_grid[x, y] = 1
                
        # Channel 1: Enemy units
        enemy_units_grid = np.zeros((grid_size, grid_size))
        for unit in state['enemy_units']:
            x, y = unit.position
            if 0 <= x < grid_size and 0 <= y < grid_size:
                enemy_units_grid[x, y] = 1
        
        # Channel 2: Structures
        structures_grid = np.zeros((grid_size, grid_size))
        for structure in state['structures']:
            x, y = structure.position
            if 0 <= x < grid_size and 0 <= y < grid_size:
                structures_grid[x, y] = 1 if structure.type == StructureType.NEXUS else 0.5
        
        # Channel 3: Resources
        resources_grid = np.zeros((grid_size, grid_size))
        for resource in state['resources']:
            x, y = resource.position
            if 0 <= x < grid_size and 0 <= y < grid_size:
                resources_grid[x, y] = 1 if resource.type == ResourceType.CRYSTAL else 0.7
        
        # Combine channels
        spatial_state = np.stack([
            player_units_grid,
            enemy_units_grid,
            structures_grid,
            resources_grid
        ], axis=0)
        
        # Convert to tensor
        return torch.FloatTensor(spatial_state).unsqueeze(0)  # Add batch dimension

class EnhancedQNetwork(nn.Module):
    """Enhanced Q-network that combines traditional features with spatial features"""
    def __init__(self, input_dim, output_dim, use_spatial=True):
        super().__init__()
        
        self.use_spatial = use_spatial
        
        if use_spatial:
            self.spatial_extractor = SpatialFeatureExtractor(output_dim=32)
            combined_input_dim = input_dim + 32  # Original features + spatial features
        else:
            combined_input_dim = input_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x, spatial_state=None):
        if self.use_spatial and spatial_state is not None:
            # Extract spatial features
            spatial_features = self.spatial_extractor(spatial_state)
            
            # Make sure dimensions match for concatenation
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            
            if spatial_features.dim() == 1:
                spatial_features = spatial_features.unsqueeze(0)  # Add batch dimension if needed
            
            # Ensure spatial_features has same batch size as x
            if x.size(0) > 1 and spatial_features.size(0) == 1:
                # Repeat spatial features for each item in the batch
                spatial_features = spatial_features.repeat(x.size(0), 1)
            
            # Combine with traditional features
            combined_features = torch.cat([x, spatial_features], dim=1)
            
            # Process through fully connected layers
            return self.fc_layers(combined_features)
        else:
            # Process only traditional features
            return self.fc_layers(x)

class EnhancedFractalAttentionAgent(FractalAttentionAgent):
    """
    Enhanced agent that uses learnable attention mechanisms to focus on different levels
    of abstraction based on the current game state.
    """
    def __init__(self):
        super().__init__()
        
        # Initialize the attention network
        self.attention_feature_dim = 12  # We'll extract these features from the state
        self.attention_network = AttentionNetwork(self.attention_feature_dim)
        self.attention_optimizer = optim.Adam(self.attention_network.parameters(), lr=0.001)
        
        # Buffer for attention learning experiences
        self.attention_buffer = []
        self.attention_batch_size = 32
        
        # Track attention performance for learning
        self.attention_performance_history = []
        
        # Initialize goal networks for explicit goal passing
        self.super_to_meso_goal_dim = 8  # Resource targets, unit ratios, etc.
        self.meso_to_micro_goal_dim = 12  # Specific positions, targets, etc.
        
        # Get dimensions for traditional features
        super_features_dim = len(extract_features({"player_units": [], "enemy_units": [], "structures": [], 
                                "resources": [], "crystal_count": 0, "vespene_count": 0, 
                                "visibility": np.zeros((MAP_SIZE, MAP_SIZE)), "time": 0}, 'super'))
        
        meso_features_dim = len(extract_features({"player_units": [], "enemy_units": [], "structures": [], 
                               "resources": [], "crystal_count": 0, "vespene_count": 0, 
                               "visibility": np.zeros((MAP_SIZE, MAP_SIZE)), "time": 0}, 'meso'))
        
        micro_features_dim = len(extract_features({"player_units": [], "enemy_units": [], "structures": [], 
                                "resources": [], "crystal_count": 0, "vespene_count": 0, 
                                "visibility": np.zeros((MAP_SIZE, MAP_SIZE)), "time": 0}, 'micro')[0])
        
        # Initialize goal networks
        self.super_goal_network = GoalNetwork(super_features_dim, self.super_to_meso_goal_dim, level='super')
        self.meso_goal_network = GoalNetwork(meso_features_dim + self.super_to_meso_goal_dim, 
                                            self.meso_to_micro_goal_dim, level='meso')
        
        self.super_goal_optimizer = optim.Adam(self.super_goal_network.parameters(), lr=0.001)
        self.meso_goal_optimizer = optim.Adam(self.meso_goal_network.parameters(), lr=0.001)
        
        # Buffers for goal learning
        self.super_goal_buffer = []
        self.meso_goal_buffer = []
        self.goal_batch_size = 32
        
        # Current goals
        self.current_super_goal = None
        self.current_meso_goal = None
        self.current_tactical_objective = None  # Track current tactical objective for visualization
        
        # Replace standard Q-networks with enhanced networks that use spatial features
        # Initialize new Q-networks for the different levels
        self.Q_super = EnhancedQNetwork(super_features_dim, len(StrategicFocus), use_spatial=True)
        self.Q_meso = EnhancedQNetwork(meso_features_dim, len(TacticalObjective), use_spatial=True)
        self.Q_micro_harvester = EnhancedQNetwork(micro_features_dim, 7, use_spatial=True)  # 7 actions for harvester
        self.Q_micro_warrior = EnhancedQNetwork(micro_features_dim, 5, use_spatial=True)    # 5 actions for warrior
        
        # Initialize optimizers
        self.optimizer_super = optim.Adam(self.Q_super.parameters(), lr=0.001)
        self.optimizer_meso = optim.Adam(self.Q_meso.parameters(), lr=0.001)
        self.optimizer_micro_harvester = optim.Adam(self.Q_micro_harvester.parameters(), lr=0.001)
        self.optimizer_micro_warrior = optim.Adam(self.Q_micro_warrior.parameters(), lr=0.001)
        
        # Initialize spatial state processor
        self.spatial_processor = SpatialFeatureExtractor()
        
        # Current state's spatial representation
        self.current_spatial_state = None
        
        # Event awareness
        self.last_event_response = None
    
    def extract_attention_features(self, state):
        """
        Extract features from the state that are relevant for determining attention allocation.
        
        These features include:
        1. Resource scarcity (crystal count normalized)
        2. Enemy threat level (number and proximity of enemies)
        3. Combat intensity (number of units engaged in combat)
        4. Harvester efficiency (number of harvesters carrying resources)
        5. Strategic phase (early/mid/late game indicator)
        6. Unit distribution (ratio of warriors to harvesters)
        7. Event awareness (active events)
        """
        features = np.zeros(self.attention_feature_dim)
        
        # 1. Resource features
        crystal_count = state['crystal_count']
        features[0] = min(crystal_count / 1000.0, 1.0)  # Normalized crystal count
        features[1] = 1.0 if crystal_count < 100 else 0.0  # Resource scarcity flag
        
        # 2. Enemy threat features
        enemy_units = state['enemy_units']
        features[2] = min(len(enemy_units) / 10.0, 1.0)  # Normalized enemy count
        
        # Find Nexus position
        nexus_pos = next((s.position for s in state['structures'] 
                         if s.type == StructureType.NEXUS), (0, 0))
        
        # Calculate enemy threat level based on proximity to Nexus
        enemy_threat = 0.0
        for enemy in enemy_units:
            dist_to_nexus = ((enemy.position[0] - nexus_pos[0])**2 + 
                            (enemy.position[1] - nexus_pos[1])**2)**0.5
            if dist_to_nexus < 20:
                enemy_threat += (20 - dist_to_nexus) / 20.0
                
                # Elite raiders are more threatening
                if enemy.type == UnitType.ELITE_RAIDER:
                    enemy_threat += 0.5
        
        features[3] = min(enemy_threat / 5.0, 1.0)  # Normalized enemy threat
        
        # 3. Combat features
        player_units = state['player_units']
        units_in_combat = 0
        for unit in player_units:
            for enemy in enemy_units:
                dist = ((unit.position[0] - enemy.position[0])**2 + 
                       (unit.position[1] - enemy.position[1])**2)**0.5
                if dist < 3:
                    units_in_combat += 1
                    break
        
        features[4] = min(units_in_combat / 5.0, 1.0)  # Normalized combat intensity
        
        # 4. Harvester efficiency
        harvesters = [u for u in player_units if u.type == UnitType.HARVESTER]
        harvesters_with_resources = sum(1 for h in harvesters if h.resources > 0)
        
        features[5] = len(harvesters) / 10.0 if harvesters else 0.0  # Normalized harvester count
        features[6] = harvesters_with_resources / max(len(harvesters), 1)  # Harvester efficiency
        
        # 5. Strategic phase
        time_step = state.get('time', 0)
        features[7] = min(time_step / 1000.0, 1.0)  # Game progression (early/mid/late)
        
        # 6. Unit distribution
        warriors = [u for u in player_units if u.type == UnitType.WARRIOR]
        features[8] = len(warriors) / 10.0 if warriors else 0.0  # Normalized warrior count
        features[9] = len(warriors) / max(len(player_units), 1)  # Warrior ratio
        
        # 7. Map exploration
        features[10] = np.mean(state['visibility'])  # Exploration progress
        
        # 8. Vespene focus
        vespene_count = state.get('vespene_count', 0)
        features[11] = min(vespene_count / 200.0, 1.0)  # Normalized vespene count
        
        # Enhanced event awareness
        active_events = state.get('active_events', [])
        if active_events:
            # Check for attack waves
            attack_wave_active = any(event.get('type') == 'EnemyAttackWave' for event in active_events)
            if attack_wave_active:
                # Boost combat-related features
                features[3] += 0.3  # Increase enemy threat
                features[4] += 0.3  # Increase combat intensity
            
            # Check for resource events
            resource_depletion = any(event.get('type') == 'ResourceDepleted' for event in active_events)
            resource_discovery = any(event.get('type') == 'NewResourceDiscovery' for event in active_events)
            
            if resource_depletion:
                features[1] += 0.3  # Increase resource scarcity
            
            if resource_discovery:
                features[10] += 0.2  # Boost exploration
        
        return features

    def compute_attention_weights(self, state):
        """
        Compute attention weights using the learned attention network.
        
        This replaces the rule-based approach with a neural network that can learn
        to allocate attention based on experience.
        """
        # Extract features for attention
        attn_features = self.extract_attention_features(state)
        
        # During early training, occasionally use the rule-based method to provide 
        # guidance to the neural network
        if np.random.rand() < 0.1:  # 10% of the time, use rule-based method
            # Call the parent class implementation for rule-based weights
            rule_based_weights = super().compute_attention_weights(state)
            
            # Store this experience with the rule-based weights as target
            self.attention_buffer.append((attn_features, rule_based_weights.copy()))
            
            return rule_based_weights
        
        # Convert features to tensor
        features_tensor = torch.FloatTensor(attn_features)
        
        # Get attention weights from network
        with torch.no_grad():
            attention_weights = self.attention_network(features_tensor).numpy()
        
        # Store weights in history for analysis
        self.attention_weights = attention_weights
        self.attention_history.append(self.attention_weights.copy())
        
        return attention_weights
    
    def update_attention_network(self):
        """
        Update the attention network using the experiences in the buffer.
        This trains the network to predict good attention allocations.
        """
        if len(self.attention_buffer) < self.attention_batch_size:
            return
        
        # Sample batch from buffer
        batch_indices = np.random.choice(len(self.attention_buffer), 
                                        size=self.attention_batch_size, 
                                        replace=False)
        batch = [self.attention_buffer[i] for i in batch_indices]
        
        # Extract batch components
        features, targets = zip(*batch)
        
        # Convert to tensors
        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(targets))
        
        # Forward pass
        predictions = self.attention_network(features)
        
        # Compute loss (mean squared error)
        loss = nn.MSELoss()(predictions, targets)
        
        # Backward pass and optimize
        self.attention_optimizer.zero_grad()
        loss.backward()
        self.attention_optimizer.step()
        
        return loss.item() 
    
    def _process_event_information(self, state):
        """Process event information from the state to influence decision making"""
        active_events = state.get('active_events', [])
        event_info = {
            'attack_wave_active': False,
            'resource_depletion_active': False,
            'resource_discovery_active': False,
            'opportunity_active': False,
            'attack_location': None,
            'resource_location': None,
            'opportunity_location': None
        }
        
        for event in active_events:
            event_type = event.get('type', '')
            
            if event_type == 'EnemyAttackWave':
                event_info['attack_wave_active'] = True
                if 'location' in event:
                    event_info['attack_location'] = event['location']
            
            elif event_type == 'ResourceDepleted':
                event_info['resource_depletion_active'] = True
            
            elif event_type == 'NewResourceDiscovery':
                event_info['resource_discovery_active'] = True
                if 'location' in event:
                    event_info['resource_location'] = event['location']
            
            elif event_type == 'SuddenOpportunity':
                event_info['opportunity_active'] = True
                if 'location' in event:
                    event_info['opportunity_location'] = event['location']
                if 'opportunity_type' in event:
                    event_info['opportunity_type'] = event['opportunity_type']
        
        return event_info

    def generate_super_goal(self, state):
        """
        Generate explicit goals from the super level for the meso level.
        The goals include desired resource ratios, unit compositions, etc.
        """
        # Extract features for super level
        super_features = extract_features(state, 'super')
        
        # Process event information
        event_info = self._process_event_information(state)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(super_features)
        
        # Generate goal parameters
        with torch.no_grad():
            goal_params = self.super_goal_network(features_tensor).numpy()
        
        # Adjust goal parameters based on active events
        if event_info['attack_wave_active']:
            # Increase defense priority
            goal_params[5] = min(goal_params[5] * 1.5, 1.0)
            # Adjust unit ratio to favor warriors
            goal_params[3] = min(goal_params[3] * 1.3, 1.0)  # Increase warrior ratio
        
        if event_info['resource_depletion_active']:
            # Increase expansion priority
            goal_params[4] = min(goal_params[4] * 1.3, 1.0)
        
        if event_info['resource_discovery_active']:
            # Increase expansion priority
            goal_params[4] = min(goal_params[4] * 1.2, 1.0)
            # Adjust resource ratio based on discovery type
            if event_info.get('resource_type') == 'VESPENE':
                goal_params[1] = min(goal_params[1] * 1.3, 1.0)  # Favor vespene
        
        # Interpret goal parameters
        # 0-1: Desired resource ratio (crystal vs vespene)
        # 2-3: Desired unit ratio (harvesters vs warriors)
        # 4: Expansion priority (0-1)
        # 5: Defense priority (0-1)
        # 6-7: Reserved for future use
        
        # Store for learning
        self.current_super_goal = goal_params
        
        return goal_params

    def generate_meso_goal(self, state, super_goal):
        """
        Generate explicit goals from the meso level for the micro level.
        Takes into account the super-level goals and active events.
        """
        # Extract features for meso level
        meso_features = extract_features(state, 'meso')
        
        # Process event information
        event_info = self._process_event_information(state)
        
        # Combine with super goal
        combined_features = np.concatenate([meso_features, super_goal])
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(combined_features)
        
        # Generate goal parameters
        with torch.no_grad():
            goal_params = self.meso_goal_network(features_tensor).numpy()
        
        # Adjust goal parameters based on active events
        if event_info['attack_wave_active'] and event_info['attack_location']:
            # Set rally point near the attack
            attack_x, attack_y = event_info['attack_location']
            # Normalized coordinates (0-1)
            goal_params[0] = attack_x / MAP_SIZE  # Rally X
            goal_params[1] = attack_y / MAP_SIZE  # Rally Y
            # Increase warrior aggression
            goal_params[7] = min(goal_params[7] * 1.5, 1.0)
        
        if event_info['resource_discovery_active'] and event_info['resource_location']:
            # Set priority resource location to the new discovery
            resource_x, resource_y = event_info['resource_location']
            # Normalized coordinates (0-1)
            goal_params[2] = resource_x / MAP_SIZE  # Resource X
            goal_params[3] = resource_y / MAP_SIZE  # Resource Y
        
        if event_info['opportunity_active'] and event_info['opportunity_location']:
            # Set exploration target to the opportunity
            opp_x, opp_y = event_info['opportunity_location']
            # Normalized coordinates (0-1)
            goal_params[4] = opp_x / MAP_SIZE  # Exploration X
            goal_params[5] = opp_y / MAP_SIZE  # Exploration Y
            
            # If it's an enemy weakness opportunity, increase warrior aggression
            if event_info.get('opportunity_type') == 'enemy_weakness':
                goal_params[7] = min(goal_params[7] * 1.3, 1.0)
        
        # Interpret goal parameters
        # 0-1: Rally point X, Y for warriors
        # 2-3: Priority resource location X, Y
        # 4-5: Exploration target X, Y
        # 6: Harvester aggression (0-1)
        # 7: Warrior aggression (0-1)
        # 8-9: Defense point X, Y
        # 10-11: Reserved for future use
        
        # Store for learning
        self.current_meso_goal = goal_params
        
        return goal_params

    def update_goal_networks(self):
        """
        Update the goal networks based on outcomes.
        """
        # Update super goal network
        if len(self.super_goal_buffer) >= self.goal_batch_size:
            # Sample batch
            batch_indices = np.random.choice(len(self.super_goal_buffer), 
                                           size=self.goal_batch_size, 
                                           replace=False)
            batch = [self.super_goal_buffer[i] for i in batch_indices]
            
            # Extract batch components
            features, goals, rewards = zip(*batch)
            
            # Convert to tensors
            features = torch.FloatTensor(np.array(features))
            goals = torch.FloatTensor(np.array(goals))
            rewards = torch.FloatTensor(np.array(rewards))
            
            # Compute weighted goals based on rewards
            weighted_goals = goals * rewards.unsqueeze(1)
            
            # Forward pass
            predictions = self.super_goal_network(features)
            
            # Compute loss
            loss = nn.MSELoss()(predictions, weighted_goals)
            
            # Backward pass and optimize
            self.super_goal_optimizer.zero_grad()
            loss.backward()
            self.super_goal_optimizer.step()
        
        # Update meso goal network
        if len(self.meso_goal_buffer) >= self.goal_batch_size:
            # Sample batch
            batch_indices = np.random.choice(len(self.meso_goal_buffer), 
                                           size=self.goal_batch_size, 
                                           replace=False)
            batch = [self.meso_goal_buffer[i] for i in batch_indices]
            
            # Extract batch components
            features, goals, rewards = zip(*batch)
            
            # Convert to tensors
            features = torch.FloatTensor(np.array(features))
            goals = torch.FloatTensor(np.array(goals))
            rewards = torch.FloatTensor(np.array(rewards))
            
            # Compute weighted goals based on rewards
            weighted_goals = goals * rewards.unsqueeze(1)
            
            # Forward pass
            predictions = self.meso_goal_network(features)
            
            # Compute loss
            loss = nn.MSELoss()(predictions, weighted_goals)
            
            # Backward pass and optimize
            self.meso_goal_optimizer.zero_grad()
            loss.backward()
            self.meso_goal_optimizer.step()
    
    def record_attention_outcome(self, attn_features, attn_weights, reward):
        """
        Record the outcome of an attention allocation decision for learning.
        This associates attention weights with a reward signal.
        """
        # Store the features, weights and outcome for learning
        self.attention_performance_history.append((attn_features, attn_weights, reward))
    
    def _calculate_attention_reward(self, state, prev_state=None):
        """
        Calculate a reward signal for the attention mechanism based on
        immediate outcomes.
        """
        reward = 0.0
        
        # If no previous state, return neutral reward
        if prev_state is None:
            return reward
        
        # Reward for successful resource collection
        prev_crystal = prev_state.get('crystal_count', 0)
        curr_crystal = state.get('crystal_count', 0)
        
        if curr_crystal > prev_crystal:
            # If the micro attention was high during resource collection, that's good
            if self.attention_weights[0] > 0.4:  # Micro attention was significant
                reward += 0.1 * (curr_crystal - prev_crystal) / 10.0
        
        # Reward for successful combat
        prev_enemy_count = len(prev_state.get('enemy_units', []))
        curr_enemy_count = len(state.get('enemy_units', []))
        
        if curr_enemy_count < prev_enemy_count:
            # If the micro attention was high during combat, that's good
            if self.attention_weights[0] > 0.4:  # Micro attention was significant
                reward += 0.2 * (prev_enemy_count - curr_enemy_count)
        
        # Reward for good strategic decisions
        # If there are no enemies nearby but we have a good economy focus
        enemy_units = state.get('enemy_units', [])
        nexus_pos = next((s.position for s in state.get('structures', []) 
                         if s.type == StructureType.NEXUS), (0, 0))
        
        enemies_near_nexus = False
        for enemy in enemy_units:
            dist_to_nexus = ((enemy.position[0] - nexus_pos[0])**2 + 
                            (enemy.position[1] - nexus_pos[1])**2)**0.5
            if dist_to_nexus < 15:
                enemies_near_nexus = True
                break
        
        if not enemies_near_nexus and self.current_focus == StrategicFocus.ECONOMY:
            # If the super attention was high during strategic planning, that's good
            if self.attention_weights[2] > 0.4:  # Super attention was significant
                reward += 0.1
        
        # Penalty for unit losses
        prev_unit_count = len(prev_state.get('player_units', []))
        curr_unit_count = len(state.get('player_units', []))
        
        if curr_unit_count < prev_unit_count:
            # If we lost units while micro attention was low, that's bad
            if self.attention_weights[0] < 0.2:  # Micro attention was too low
                reward -= 0.2 * (prev_unit_count - curr_unit_count)
        
        return reward
    
    def choose_action(self, state, unit=None, level='micro'):
        """
        Choose an action based on the current state, unit, and level,
        incorporating spatial features.
        """
        # Process spatial state once per game step
        if self.current_spatial_state is None:
            self.current_spatial_state = self.spatial_processor.process_spatial_state(state)
        
        # Exploration: choose random action
        if np.random.rand() < self.epsilon:
            if level == 'super':
                return np.random.randint(0, len(StrategicFocus))
            elif level == 'meso':
                return np.random.randint(0, len(TacticalObjective))
            else:  # micro
                if unit.type == UnitType.HARVESTER:
                    return np.random.randint(0, 7)  # 7 actions for harvester
                else:  # WARRIOR
                    return np.random.randint(0, 5)  # 5 actions for warrior
        
        # Exploitation: choose best action according to Q-network with spatial features
        features = extract_features(state, level)
        
        if level == 'super':
            with torch.no_grad():
                q_values = self.Q_super(torch.FloatTensor(features), self.current_spatial_state)
                return q_values.argmax().item()
        elif level == 'meso':
            with torch.no_grad():
                q_values = self.Q_meso(torch.FloatTensor(features), self.current_spatial_state)
                return q_values.argmax().item()
        else:  # micro
            unit_idx = next((i for i, u in enumerate(state['player_units']) if u.id == unit.id), 0)
            with torch.no_grad():
                if unit.type == UnitType.HARVESTER:
                    q_values = self.Q_micro_harvester(torch.FloatTensor(features[unit_idx]), self.current_spatial_state)
                    return q_values.argmax().item()
                else:  # WARRIOR
                    q_values = self.Q_micro_warrior(torch.FloatTensor(features[unit_idx]), self.current_spatial_state)
                    return q_values.argmax().item()
    
    def update_q_network(self, buffer, optimizer, q_network, batch_size=32):
        """Update Q-network from experience replay buffer using spatial features."""
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
        
        # Use the current spatial state for all batch items (simplified approximation)
        # In a real implementation, you would store spatial states with each experience
        spatial_state = self.current_spatial_state
        
        # Compute current Q values
        current_q_values = q_network(states, spatial_state).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        next_q_values = q_network(next_states, spatial_state).max(1)[0]
        expected_q_values = rewards + 0.99 * next_q_values * (1 - dones)  # 0.99 is gamma
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def update_q_networks(self):
        """Update all Q-networks from their respective experience replay buffers."""
        self.update_q_network(self.buffer_super, self.optimizer_super, self.Q_super)
        self.update_q_network(self.buffer_meso, self.optimizer_meso, self.Q_meso)
        self.update_q_network(self.buffer_micro_harvester, self.optimizer_micro_harvester, self.Q_micro_harvester)
        self.update_q_network(self.buffer_micro_warrior, self.optimizer_micro_warrior, self.Q_micro_warrior)
    
    def act(self, state, env):
        """
        Override the act method to include attention learning, goal passing, spatial features,
        and improved outcome tracking.
        """
        # Reset the spatial state for this step
        self.current_spatial_state = self.spatial_processor.process_spatial_state(state)
        
        # Store previous state for calculating attention reward
        prev_state = None
        if hasattr(self, 'prev_state'):
            prev_state = self.prev_state
        
        # Compute dynamic attention weights using learned network
        attn_weights = self.compute_attention_weights(state)
        attn_features = self.extract_attention_features(state)
        
        # Process event information
        event_info = self._process_event_information(state)
        
        # Adjust attention weights based on active events (emergency override)
        # This provides an immediate response to critical events
        if event_info['attack_wave_active']:
            # Shift attention toward micro/meso for combat
            # Weight adjustments (mild to avoid overriding learning)
            attn_weights[0] = attn_weights[0] * 1.2  # Boost micro attention
            attn_weights[1] = attn_weights[1] * 1.1  # Slightly boost meso attention
            attn_weights[2] = attn_weights[2] * 0.8  # Reduce super attention
            
            # Renormalize
            attn_weights = attn_weights / np.sum(attn_weights)
        
        # Calculate reward for previous attention allocation
        if prev_state is not None:
            attn_reward = self._calculate_attention_reward(state, prev_state)
            if len(self.attention_history) > 1:
                prev_attn_features = self.extract_attention_features(prev_state)
                prev_attn_weights = self.attention_history[-2]
                self.record_attention_outcome(prev_attn_features, prev_attn_weights, attn_reward)
        
        # Store current state for next iteration
        self.prev_state = state.copy()
        
        # Super-level decision (strategic focus)
        super_action = self.choose_action(state, level='super')
        prev_focus = self.current_focus
        self.current_focus = list(StrategicFocus)[super_action]
        
        # Override strategic focus based on critical events if needed
        if event_info['attack_wave_active'] and self.current_focus != StrategicFocus.DEFENSE:
            # Record original focus for learning
            original_focus = self.current_focus
            
            # Override with DEFENSE
            self.current_focus = StrategicFocus.DEFENSE
            print(f"Strategic focus overridden by attack event: {original_focus.name} -> {self.current_focus.name}")
        
        # Generate explicit super-to-meso goals
        super_goal = self.generate_super_goal(state)
        
        # Record experience for super-level learning
        super_features = extract_features(state, 'super')
        
        # Reward for super-level based on game state
        super_reward = 0.0
        
        # Reward calculations (same as in parent class)
        if self.current_focus == StrategicFocus.ECONOMY:
            harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
            if len(harvesters) > 3:
                super_reward += 0.5
            super_reward += min(state['crystal_count'] / 1000, 1.0)
        elif self.current_focus == StrategicFocus.DEFENSE:
            enemies_near_nexus = False
            nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
            if nexus:
                for enemy in state['enemy_units']:
                    dist_to_nexus = ((enemy.position[0] - nexus.position[0])**2 + 
                                    (enemy.position[1] - nexus.position[1])**2)**0.5
                    if dist_to_nexus < 10:
                        enemies_near_nexus = True
                        break
            
            warriors = [u for u in state['player_units'] if u.type == UnitType.WARRIOR]
            if enemies_near_nexus and len(warriors) > 0:
                super_reward += 1.0
            elif not enemies_near_nexus and self.current_focus != prev_focus:
                super_reward -= 0.5
        elif self.current_focus == StrategicFocus.EXPANSION:
            exploration_pct = np.mean(state['visibility'])
            super_reward += exploration_pct
            if state['crystal_count'] < 50:
                super_reward -= 0.5
        elif self.current_focus == StrategicFocus.VESPENE:
            if state['vespene_count'] > 0:
                super_reward += state['vespene_count'] / 100
            vespene_geysers = [r for r in state['resources'] if r.type == ResourceType.VESPENE]
            if not vespene_geysers:
                super_reward -= 0.5
        
        # Store super goal experience for learning
        self.super_goal_buffer.append((
            super_features,
            super_goal,
            super_reward
        ))
        
        # Add to replay buffer for Q-learning
        self.buffer_super.append((
            super_features,
            super_action,
            super_reward,
            super_features,  # Next state (placeholder)
            False  # Not terminal
        ))
        
        # Meso-level decisions (tactical objectives)
        meso_action = self.choose_action(state, level='meso')
        tactical_objective = list(TacticalObjective)[meso_action]
        self.current_tactical_objective = tactical_objective  # Set current tactical objective for visualization
        
        # Generate explicit meso-to-micro goals, influenced by super goals
        meso_goal = self.generate_meso_goal(state, super_goal)
        
        # Apply the tactical objective based on strategic focus and goals
        objective_success = self._apply_tactical_objective_with_goals(tactical_objective, state, env, super_goal, meso_goal)
        
        # Calculate meso-level reward
        meso_reward = 1.0 if objective_success else -0.1
        
        # Store meso goal experience for learning
        meso_features = extract_features(state, 'meso')
        combined_features = np.concatenate([meso_features, super_goal])
        
        self.meso_goal_buffer.append((
            combined_features,
            meso_goal,
            meso_reward
        ))
        
        # Micro-level decisions (unit actions)
        for unit in state['player_units']:
            micro_action = self.choose_action(state, unit, level='micro')
            self._apply_unit_action_with_goals(unit, micro_action, state, env, attn_weights, meso_goal)
        
        # Update networks
        self.update_q_networks()
        self.update_attention_network()
        self.update_goal_networks()
        
        # Decay exploration parameter
        self.decay_epsilon()
        
        return attn_weights  # Return attention weights for analysis
    
    def _apply_tactical_objective_with_goals(self, objective, state, env, super_goal, meso_goal):
        """
        Enhanced version of _apply_tactical_objective that takes into account the generated goals.
        """
        # Get available units and resources
        harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
        warriors = [u for u in state['player_units'] if u.type == UnitType.WARRIOR]
        crystal_patches = [r for r in state['resources'] if r.type == ResourceType.CRYSTAL]
        vespene_geysers = [r for r in state['resources'] if r.type == ResourceType.VESPENE]
        nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
        
        # Track if objective was successfully applied
        objective_applied = False
        
        # Extract goal parameters
        desired_harvester_ratio = super_goal[2]  # From super goal
        desired_warrior_ratio = super_goal[3]    # From super goal
        expansion_priority = super_goal[4]       # From super goal
        defense_priority = super_goal[5]         # From super goal
        
        rally_point = (int(meso_goal[0] * MAP_SIZE), int(meso_goal[1] * MAP_SIZE))  # From meso goal
        priority_resource = (int(meso_goal[2] * MAP_SIZE), int(meso_goal[3] * MAP_SIZE))  # From meso goal
        exploration_target = (int(meso_goal[4] * MAP_SIZE), int(meso_goal[5] * MAP_SIZE))  # From meso goal
        defense_point = (int(meso_goal[8] * MAP_SIZE), int(meso_goal[9] * MAP_SIZE))  # From meso goal
        
        # Apply objectives based on current strategic focus and goals
        if self.current_focus == StrategicFocus.ECONOMY:
            # Adjust based on desired harvester ratio from super goal
            current_harvester_ratio = len(harvesters) / max(len(state['player_units']), 1)
            
            if objective == TacticalObjective.ASSIGN_HARVESTERS and harvesters and crystal_patches:
                # Prioritize resource locations based on meso goal
                idle_harvesters = [h for h in harvesters if h.action == ActionType.IDLE]
                if idle_harvesters:
                    harvester = random.choice(idle_harvesters)
                    
                    # Find nearest crystal patch to the priority resource location
                    nearest_patch = min(crystal_patches, 
                                      key=lambda r: ((r.position[0] - priority_resource[0])**2 + 
                                                   (r.position[1] - priority_resource[1])**2))
                    
                    # Record the objective for this unit
                    self.current_objectives[harvester.id] = (TacticalObjective.ASSIGN_HARVESTERS, nearest_patch)
                    objective_applied = True
            
            elif objective == TacticalObjective.BUILD_HARVESTER and nexus and state['crystal_count'] >= 50:
                # Only build if we're below the desired ratio
                if current_harvester_ratio < desired_harvester_ratio:
                    if nexus.produce_unit(UnitType.HARVESTER, env):
                        objective_applied = True
        
        elif self.current_focus == StrategicFocus.DEFENSE:
            # Adjust based on defense priority from super goal
            defense_urgency = defense_priority > 0.7  # High defense priority
            
            if objective == TacticalObjective.BUILD_WARRIOR and nexus and state['crystal_count'] >= 100:
                # Build warrior if defense is urgent or we're below desired warrior ratio
                current_warrior_ratio = len(warriors) / max(len(state['player_units']), 1)
                if defense_urgency or current_warrior_ratio < desired_warrior_ratio:
                    if nexus.produce_unit(UnitType.WARRIOR, env):
                        objective_applied = True
            
            elif objective == TacticalObjective.BUILD_TURRET and state['crystal_count'] >= 150:
                # Build a turret near the defense point from meso goal
                existing_turrets = [s for s in state['structures'] if s.type == StructureType.TURRET]
                if not existing_turrets and nexus and defense_urgency:
                    # Find a spot near the defense point
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            pos = (defense_point[0] + dx, defense_point[1] + dy)
                            if 0 <= pos[0] < MAP_SIZE and 0 <= pos[1] < MAP_SIZE:
                                if not env.is_occupied_by_structure(pos):
                                    # Build turret at this position
                                    env.structures.append(Structure(env.next_structure_id, 
                                                                  StructureType.TURRET, pos))
                                    env.next_structure_id += 1
                                    env.crystal_count -= 150
                                    objective_applied = True
                                    break
                        if objective_applied:
                            break
            
            elif objective == TacticalObjective.RALLY_WARRIORS and warriors:
                # Rally warriors to the rally point from meso goal
                for warrior in warriors:
                    # Record the objective for this unit
                    self.current_objectives[warrior.id] = (TacticalObjective.RALLY_WARRIORS, rally_point)
                objective_applied = True
        
        elif self.current_focus == StrategicFocus.EXPANSION:
            # Adjust based on expansion priority from super goal
            if objective == TacticalObjective.SCOUT and (harvesters or warriors):
                # Send a unit to explore the target from meso goal
                scout_unit = None
                if warriors and len(warriors) > 1:  # Keep at least one warrior for defense
                    scout_unit = min(warriors, key=lambda u: u.id)
                elif harvesters and len(harvesters) > 2:  # Keep at least two harvesters for economy
                    scout_unit = min(harvesters, key=lambda u: u.id)
                
                if scout_unit and expansion_priority > 0.3:  # Only scout if expansion priority is sufficient
                    # Record the objective for this unit
                    self.current_objectives[scout_unit.id] = (TacticalObjective.SCOUT, exploration_target)
                    objective_applied = True
        
        elif self.current_focus == StrategicFocus.VESPENE:
            if objective == TacticalObjective.ASSIGN_HARVESTERS and harvesters and vespene_geysers:
                # Assign harvesters to vespene geyser
                for geyser in vespene_geysers:
                    # Assign up to 3 harvesters per geyser, influenced by desired resource ratio
                    max_harvesters = int(3 * super_goal[1])  # Vespene importance
                    assigned_count = 0
                    for harvester in harvesters:
                        if assigned_count >= max_harvesters:
                            break
                        
                        # Check if this harvester is already assigned to this geyser
                        if (harvester.id in self.current_objectives and 
                            self.current_objectives[harvester.id][0] == TacticalObjective.ASSIGN_HARVESTERS and
                            self.current_objectives[harvester.id][1].id == geyser.id):
                            assigned_count += 1
                            continue
                        
                        # Assign this harvester to the geyser
                        self.current_objectives[harvester.id] = (TacticalObjective.ASSIGN_HARVESTERS, geyser)
                        assigned_count += 1
                        objective_applied = True
        
        # Store the experience for meso-level learning
        if objective_applied:
            reward = 1.0  # Reward for successfully applying a tactical objective
        else:
            reward = -0.1  # Small penalty for failing to apply a tactical objective
        
        # Extract features for current and next state
        meso_features = extract_features(state, 'meso')
        
        # Add to replay buffer
        self.buffer_meso.append((
            meso_features,
            list(TacticalObjective).index(objective),
            reward,
            meso_features,  # Next state (placeholder)
            False  # Not terminal
        ))
        
        return objective_applied
    
    def _apply_unit_action_with_goals(self, unit, action, state, env, attn_weights, meso_goal):
        """
        Enhanced version of _apply_unit_action that takes into account the generated goals.
        """
        # Extract relevant goal parameters
        harvester_aggression = meso_goal[6]  # From meso goal
        warrior_aggression = meso_goal[7]    # From meso goal
        
        # Store the micro-level suggested action
        if unit.id not in self.unit_action_memory:
            self.unit_action_memory[unit.id] = {}
        
        self.unit_action_memory[unit.id]['micro'] = action
        
        # Get tactical objective for this unit (meso-level suggestion)
        meso_suggestion = None
        if unit.id in self.current_objectives:
            objective, target = self.current_objectives[unit.id]
            
            # Translate objective to a suggested action
            if objective == TacticalObjective.ASSIGN_HARVESTERS:
                # If harvester is full, return to nexus
                if unit.resources >= 10:
                    nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
                    if nexus:
                        # Move toward nexus
                        dx = nexus.position[0] - unit.position[0]
                        dy = nexus.position[1] - unit.position[1]
                        
                        if abs(dx) > abs(dy):
                            meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                        else:
                            meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
                    else:
                        meso_suggestion = action  # Fall back to micro suggestion
                else:
                    # Move toward resource target
                    dx = target.position[0] - unit.position[0]
                    dy = target.position[1] - unit.position[1]
                    
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        meso_suggestion = 5  # HARVEST
                    elif abs(dx) > abs(dy):
                        meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                    else:
                        meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
            
            elif objective == TacticalObjective.RALLY_WARRIORS:
                # Target could be a position or an enemy
                target_pos = target.position if hasattr(target, 'position') else target
                
                # Move toward target
                dx = target_pos[0] - unit.position[0]
                dy = target_pos[1] - unit.position[1]
                
                # If near enemy and aggression is high, attack
                nearby_enemy = None
                for enemy in state['enemy_units']:
                    dist = ((unit.position[0] - enemy.position[0])**2 + 
                           (unit.position[1] - enemy.position[1])**2)**0.5
                    if dist <= 2:  # Extended attack range based on aggression
                        nearby_enemy = enemy
                        break
                
                if nearby_enemy and warrior_aggression > 0.5:
                    meso_suggestion = 4  # ATTACK
                elif abs(dx) <= 1 and abs(dy) <= 1:
                    # At destination
                    if nearby_enemy:
                        meso_suggestion = 4  # ATTACK
                    else:
                        meso_suggestion = 7  # IDLE
                elif abs(dx) > abs(dy):
                    meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                else:
                    meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
            
            elif objective == TacticalObjective.SCOUT:
                # Move toward exploration target
                target_pos = target if isinstance(target, tuple) else target.position
                
                dx = target_pos[0] - unit.position[0]
                dy = target_pos[1] - unit.position[1]
                
                if abs(dx) <= 1 and abs(dy) <= 1:
                    # At destination, find a new unexplored area
                    unexplored_regions = []
                    for i in range(0, MAP_SIZE, 16):
                        for j in range(0, MAP_SIZE, 16):
                            region = (i, j)
                            # Check if this region is unexplored
                            if np.mean(state['visibility'][i:i+16, j:j+16]) < 0.5:
                                unexplored_regions.append(region)
                    
                    if unexplored_regions:
                        new_target = random.choice(unexplored_regions)
                        self.current_objectives[unit.id] = (TacticalObjective.SCOUT, new_target)
                        
                        # Continue moving in current direction (keep exploring)
                        if abs(dx) > abs(dy):
                            meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                        else:
                            meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
                    else:
                        # All explored, return to nexus
                        nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
                        if nexus:
                            dx = nexus.position[0] - unit.position[0]
                            dy = nexus.position[1] - unit.position[1]
                            
                            if abs(dx) > abs(dy):
                                meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                            else:
                                meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
                        else:
                            meso_suggestion = 7  # IDLE
                elif abs(dx) > abs(dy):
                    meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                else:
                    meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
        
        if meso_suggestion is not None:
            self.unit_action_memory[unit.id]['meso'] = meso_suggestion
        
        # Super-level suggestion based on strategic focus, influenced by goals
        super_suggestion = None
        if self.current_focus == StrategicFocus.ECONOMY:
            if unit.type == UnitType.HARVESTER:
                # Prioritize harvesting or returning resources
                if unit.resources >= 10:
                    super_suggestion = 6  # RETURN_RESOURCES
                else:
                    super_suggestion = 5  # HARVEST
            elif unit.type == UnitType.WARRIOR:
                # Warriors should protect harvesters during ECONOMY focus
                harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
                if harvesters:
                    # Find nearest harvester
                    nearest_harvester = min(harvesters, 
                                          key=lambda h: ((h.position[0] - unit.position[0])**2 + 
                                                       (h.position[1] - unit.position[1])**2))
                    
                    # Move toward that harvester
                    dx = nearest_harvester.position[0] - unit.position[0]
                    dy = nearest_harvester.position[1] - unit.position[1]
                    
                    if abs(dx) > abs(dy):
                        super_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                    else:
                        super_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
        
        elif self.current_focus == StrategicFocus.DEFENSE:
            if unit.type == UnitType.WARRIOR:
                # Look for nearby enemies with aggression based on warrior_aggression
                enemies = state['enemy_units']
                if enemies:
                    # Find nearest enemy
                    nearest_enemy = min(enemies, 
                                      key=lambda e: ((e.position[0] - unit.position[0])**2 + 
                                                   (e.position[1] - unit.position[1])**2))
                    
                    # Move toward that enemy
                    dx = nearest_enemy.position[0] - unit.position[0]
                    dy = nearest_enemy.position[1] - unit.position[1]
                    
                    # Attack range increases with aggression
                    attack_range = 1 + int(warrior_aggression * 2)
                    
                    if abs(dx) <= attack_range and abs(dy) <= attack_range:
                        super_suggestion = 4  # ATTACK
                    elif abs(dx) > abs(dy):
                        super_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                    else:
                        super_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
                else:
                    # No enemies visible, guard the Nexus
                    nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
                    if nexus:
                        # Move toward nexus
                        dx = nexus.position[0] - unit.position[0]
                        dy = nexus.position[1] - unit.position[1]
                        
                        if abs(dx) > 2 or abs(dy) > 2:  # Only move if not already near nexus
                            if abs(dx) > abs(dy):
                                super_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                            else:
                                super_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
        
        if super_suggestion is not None:
            self.unit_action_memory[unit.id]['super'] = super_suggestion
        
        # Now combine all suggestions using attention weights
        final_action = action  # Default to micro suggestion
        
        # Weighted voting system
        action_votes = np.zeros(8)  # 8 possible actions (including IDLE)
        
        # Add micro vote
        action_votes[self.unit_action_memory[unit.id].get('micro', 0)] += attn_weights[0]
        
        # Add meso vote
        if 'meso' in self.unit_action_memory[unit.id]:
            action_votes[self.unit_action_memory[unit.id]['meso']] += attn_weights[1]
        
        # Add super vote
        if 'super' in self.unit_action_memory[unit.id]:
            action_votes[self.unit_action_memory[unit.id]['super']] += attn_weights[2]
        
        # Choose action with highest vote
        final_action = np.argmax(action_votes)
        
        # Apply the chosen action (as in the original method)
        if final_action == 0:  # MOVE_UP
            unit.move(ActionType.MOVE_UP, env)
        elif final_action == 1:  # MOVE_DOWN
            unit.move(ActionType.MOVE_DOWN, env)
        elif final_action == 2:  # MOVE_LEFT
            unit.move(ActionType.MOVE_LEFT, env)
        elif final_action == 3:  # MOVE_RIGHT
            unit.move(ActionType.MOVE_RIGHT, env)
        elif final_action == 4:  # ATTACK
            # Find nearest enemy
            nearest_enemy = None
            min_dist = float('inf')
            for enemy in state['enemy_units']:
                dist = ((unit.position[0] - enemy.position[0])**2 + 
                       (unit.position[1] - enemy.position[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy = enemy
            
            if nearest_enemy:
                unit.attack(nearest_enemy, env)
        elif final_action == 5:  # HARVEST
            # Find nearest resource
            nearest_resource = None
            min_dist = float('inf')
            for resource in state['resources']:
                dist = ((unit.position[0] - resource.position[0])**2 + 
                       (unit.position[1] - resource.position[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_resource = resource
            
            if nearest_resource:
                unit.harvest(nearest_resource, env)
        elif final_action == 6:  # RETURN_RESOURCES
            # Find nexus
            nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
            if nexus:
                unit.return_resources(nexus, env)
        elif final_action == 7:  # IDLE
            unit.action = ActionType.IDLE
        
        # Calculate micro reward and record experience (as in the original method)
        reward = 0.0
        
        if unit.type == UnitType.HARVESTER:
            if final_action == 5 and unit.action == ActionType.HARVEST:
                reward = 0.5
            elif final_action == 6 and unit.action == ActionType.RETURN_RESOURCES:
                reward = 1.0 if unit.resources > 0 else -0.1
            else:
                reward = -0.01
        elif unit.type == UnitType.WARRIOR:
            if final_action == 4 and unit.action == ActionType.ATTACK:
                reward = 1.0
            else:
                reward = -0.01
        
        # Extract features and add to buffer
        unit_features = extract_features(state, 'micro')
        unit_idx = next((i for i, u in enumerate(state['player_units']) if u.id == unit.id), 0)
        
        if unit.type == UnitType.HARVESTER:
            self.buffer_micro_harvester.append((
                unit_features[unit_idx],
                final_action,
                reward,
                unit_features[unit_idx],  # Next state (placeholder)
                False  # Not terminal
            ))
        else:  # WARRIOR
            self.buffer_micro_warrior.append((
                unit_features[unit_idx],
                final_action,
                reward,
                unit_features[unit_idx],  # Next state (placeholder)
                False  # Not terminal
            )) 