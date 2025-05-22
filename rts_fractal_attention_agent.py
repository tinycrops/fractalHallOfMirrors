#!/usr/bin/env python3
import numpy as np
import random
import torch
from typing import List, Dict, Tuple

from rts_environment import RTSEnvironment, UnitType, StructureType, ResourceType, ActionType, Structure, MAP_SIZE
from rts_fractal_agent import FractalAgent, StrategicFocus, TacticalObjective, extract_features

class FractalAttentionAgent(FractalAgent):
    """
    Enhanced agent that uses dynamic attention mechanisms to focus on different levels
    of abstraction based on the current game state.
    """
    def __init__(self):
        super().__init__()
        # Specific memory for attention-based action selection
        self.unit_action_memory = {}  # Remember actions suggested by each level
    
    def _apply_tactical_objective(self, objective, state, env):
        """
        Apply tactical objectives based on the current strategic focus.
        This function translates meso-level decisions into assignments for units.
        """
        # Get available units and resources
        harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
        warriors = [u for u in state['player_units'] if u.type == UnitType.WARRIOR]
        crystal_patches = [r for r in state['resources'] if r.type == ResourceType.CRYSTAL]
        vespene_geysers = [r for r in state['resources'] if r.type == ResourceType.VESPENE]
        nexus = next((s for s in state['structures'] if s.type == StructureType.NEXUS), None)
        
        # Track if objective was successfully applied
        objective_applied = False
        
        # Apply objectives based on current strategic focus
        if self.current_focus == StrategicFocus.ECONOMY:
            if objective == TacticalObjective.ASSIGN_HARVESTERS and harvesters and crystal_patches:
                # Assign idle harvesters to the nearest crystal patch
                idle_harvesters = [h for h in harvesters if h.action == ActionType.IDLE]
                if idle_harvesters:
                    harvester = random.choice(idle_harvesters)
                    
                    # Find nearest crystal patch
                    nearest_patch = min(crystal_patches, 
                                        key=lambda r: ((r.position[0] - harvester.position[0])**2 + 
                                                     (r.position[1] - harvester.position[1])**2))
                    
                    # Record the objective for this unit
                    self.current_objectives[harvester.id] = (TacticalObjective.ASSIGN_HARVESTERS, nearest_patch)
                    objective_applied = True
            
            elif objective == TacticalObjective.BUILD_HARVESTER and nexus and state['crystal_count'] >= 50:
                # Queue a harvester for production
                if nexus.produce_unit(UnitType.HARVESTER, env):
                    objective_applied = True
        
        elif self.current_focus == StrategicFocus.DEFENSE:
            if objective == TacticalObjective.BUILD_WARRIOR and nexus and state['crystal_count'] >= 100:
                # Queue a warrior for production
                if nexus.produce_unit(UnitType.WARRIOR, env):
                    objective_applied = True
            
            elif objective == TacticalObjective.BUILD_TURRET and state['crystal_count'] >= 150:
                # Build a turret near the nexus if none exists yet
                existing_turrets = [s for s in state['structures'] if s.type == StructureType.TURRET]
                if not existing_turrets and nexus:
                    # Find a spot near the nexus
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue  # Skip the nexus position itself
                            
                            pos = (nexus.position[0] + dx, nexus.position[1] + dy)
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
                # Find enemies near the nexus to engage
                if nexus:
                    # Find nearby enemies
                    nearby_enemies = []
                    for enemy in state['enemy_units']:
                        dist_to_nexus = ((enemy.position[0] - nexus.position[0])**2 + 
                                         (enemy.position[1] - nexus.position[1])**2)**0.5
                        if dist_to_nexus < 15:
                            nearby_enemies.append(enemy)
                    
                    if nearby_enemies:
                        # Assign warriors to attack nearby enemies
                        for warrior in warriors:
                            enemy = min(nearby_enemies, 
                                      key=lambda e: ((e.position[0] - warrior.position[0])**2 + 
                                                   (e.position[1] - warrior.position[1])**2))
                            
                            # Record the objective for this unit
                            self.current_objectives[warrior.id] = (TacticalObjective.RALLY_WARRIORS, enemy)
                        objective_applied = True
        
        elif self.current_focus == StrategicFocus.EXPANSION:
            if objective == TacticalObjective.SCOUT and (harvesters or warriors):
                # Send a unit to explore unexplored areas
                # Choose a unit for scouting (prefer warrior, but use harvester if necessary)
                scout_unit = None
                if warriors:
                    scout_unit = min(warriors, key=lambda u: u.id)  # Just pick one
                elif harvesters:
                    scout_unit = min(harvesters, key=lambda u: u.id)  # Just pick one
                
                if scout_unit:
                    # Find unexplored region
                    unexplored_regions = []
                    for i in range(0, MAP_SIZE, 16):
                        for j in range(0, MAP_SIZE, 16):
                            region = (i, j)
                            # Check if this region is unexplored
                            if np.mean(state['visibility'][i:i+16, j:j+16]) < 0.5:
                                unexplored_regions.append(region)
                    
                    if unexplored_regions:
                        target_region = random.choice(unexplored_regions)
                        # Record the objective for this unit
                        self.current_objectives[scout_unit.id] = (TacticalObjective.SCOUT, target_region)
                        objective_applied = True
        
        elif self.current_focus == StrategicFocus.VESPENE:
            if objective == TacticalObjective.ASSIGN_HARVESTERS and harvesters and vespene_geysers:
                # Assign harvesters to vespene geyser
                for geyser in vespene_geysers:
                    # Assign up to 3 harvesters per geyser
                    assigned_count = 0
                    for harvester in harvesters:
                        if assigned_count >= 3:
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
        # We don't have the next state yet, so we'll use the current one as a placeholder
        # In a real implementation, you would store this and update it when the next state is available
        
        # Add to replay buffer
        self.buffer_meso.append((
            meso_features,
            list(TacticalObjective).index(objective),
            reward,
            meso_features,  # Next state (placeholder)
            False  # Not terminal
        ))
        
        return objective_applied
    
    def _apply_unit_action(self, unit, action, state, env, attn_weights):
        """
        Apply micro-level actions to units, influenced by attention weights from all levels.
        The attention mechanism determines how much to listen to each level's suggested action.
        """
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
                # Move toward enemy target
                dx = target.position[0] - unit.position[0]
                dy = target.position[1] - unit.position[1]
                
                if abs(dx) <= 1 and abs(dy) <= 1:
                    meso_suggestion = 4  # ATTACK
                elif abs(dx) > abs(dy):
                    meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                else:
                    meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
            
            elif objective == TacticalObjective.SCOUT:
                # Move toward unexplored region
                dx = target[0] - unit.position[0]
                dy = target[1] - unit.position[1]
                
                if abs(dx) > abs(dy):
                    meso_suggestion = 0 if dx < 0 else 1  # MOVE_UP or MOVE_DOWN
                else:
                    meso_suggestion = 2 if dy < 0 else 3  # MOVE_LEFT or MOVE_RIGHT
        
        if meso_suggestion is not None:
            self.unit_action_memory[unit.id]['meso'] = meso_suggestion
        
        # Super-level suggestion based on strategic focus
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
                # Look for nearby enemies
                enemies = state['enemy_units']
                if enemies:
                    # Find nearest enemy
                    nearest_enemy = min(enemies, 
                                      key=lambda e: ((e.position[0] - unit.position[0])**2 + 
                                                   (e.position[1] - unit.position[1])**2))
                    
                    # Move toward that enemy
                    dx = nearest_enemy.position[0] - unit.position[0]
                    dy = nearest_enemy.position[1] - unit.position[1]
                    
                    if abs(dx) <= 1 and abs(dy) <= 1:
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
        action_votes = np.zeros(7)  # 7 possible actions
        
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
        
        # Apply the chosen action
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
        
        # Record experience for micro-level learning
        # Calculate reward based on unit's action
        reward = 0.0
        
        if unit.type == UnitType.HARVESTER:
            if final_action == 5 and unit.action == ActionType.HARVEST:  # Successfully harvesting
                reward = 0.5
            elif final_action == 6 and unit.action == ActionType.RETURN_RESOURCES:  # Successfully returning resources
                reward = 1.0 if unit.resources > 0 else -0.1
            else:
                reward = -0.01  # Small penalty for other actions
        elif unit.type == UnitType.WARRIOR:
            if final_action == 4 and unit.action == ActionType.ATTACK:  # Successfully attacking
                reward = 1.0
            else:
                reward = -0.01  # Small penalty for other actions
        
        # Extract features for current unit
        unit_features = extract_features(state, 'micro')
        unit_idx = next((i for i, u in enumerate(state['player_units']) if u.id == unit.id), 0)
        
        # Add to appropriate replay buffer
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
    
    def act(self, state, env):
        """Take actions in the environment based on the current state with dynamic attention."""
        # Compute dynamic attention weights
        attn_weights = self.compute_attention_weights(state)
        
        # Super-level decision (strategic focus)
        super_action = self.choose_action(state, level='super')
        prev_focus = self.current_focus
        self.current_focus = list(StrategicFocus)[super_action]
        
        # Record experience for super-level learning
        super_features = extract_features(state, 'super')
        
        # Reward for super-level based on game state
        super_reward = 0.0
        
        # Reward for economy focus
        if self.current_focus == StrategicFocus.ECONOMY:
            # Reward based on resource collection
            harvesters = [u for u in state['player_units'] if u.type == UnitType.HARVESTER]
            if len(harvesters) > 3:
                super_reward += 0.5  # Good to have many harvesters during economy focus
            super_reward += min(state['crystal_count'] / 1000, 1.0)  # Reward for resources
        
        # Reward for defense focus
        elif self.current_focus == StrategicFocus.DEFENSE:
            # Reward based on enemies near nexus and warrior count
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
                super_reward += 1.0  # Good to focus on defense when enemies are near
            elif not enemies_near_nexus and self.current_focus != prev_focus:
                super_reward -= 0.5  # Penalty for switching to defense when not needed
        
        # Reward for expansion focus
        elif self.current_focus == StrategicFocus.EXPANSION:
            # Reward based on map exploration
            exploration_pct = np.mean(state['visibility'])
            super_reward += exploration_pct
            
            # Penalty for focusing on expansion when resources are very low
            if state['crystal_count'] < 50:
                super_reward -= 0.5
        
        # Reward for vespene focus
        elif self.current_focus == StrategicFocus.VESPENE:
            # Reward based on vespene collection
            if state['vespene_count'] > 0:
                super_reward += state['vespene_count'] / 100
            
            # Check if vespene geyser exists and is visible
            vespene_geysers = [r for r in state['resources'] if r.type == ResourceType.VESPENE]
            if not vespene_geysers:
                super_reward -= 0.5  # Penalty for focusing on vespene when none is available
        
        # Add to replay buffer
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

# Example usage (to be implemented in a separate script)
if __name__ == "__main__":
    # Initialize the environment and agent
    env = RTSEnvironment(seed=42)
    agent = FractalAttentionAgent()
    
    # Simple test loop
    for step in range(100):
        state = env.get_state()
        agent.act(state, env)
        game_over = env.step()
        
        if step % 10 == 0:
            env.render()
            print(f"Step {step}, Attention weights: {agent.attention_weights}")
        
        if game_over:
            print("Game over!")
            break
    
    env.render()
    print("Final attention weights:", agent.attention_weights) 