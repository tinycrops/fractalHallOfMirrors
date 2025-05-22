#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
from enum import Enum, auto
from typing import List, Dict, Tuple, Set, Optional
import time

# Constants
MAP_SIZE = 64
NEXUS_HEALTH = 1000
HARVESTER_HEALTH = 100
WARRIOR_HEALTH = 200
TURRET_HEALTH = 300
RAIDER_HEALTH = 150

HARVESTER_COST = 50
WARRIOR_COST = 100
TURRET_COST = 150

HARVESTER_CAPACITY = 10
WARRIOR_ATTACK = 20
TURRET_ATTACK = 30
RAIDER_ATTACK = 15

CRYSTAL_MAX = 100
VESPENE_MAX = 200

VISION_RADIUS = 8
FOG_DECAY_RATE = 0.01  # How quickly fog returns after units leave

# Enumeration types
class UnitType(Enum):
    HARVESTER = auto()
    WARRIOR = auto()
    RAIDER = auto()  # Enemy unit
    ELITE_RAIDER = auto()  # Stronger enemy unit for attack waves

class StructureType(Enum):
    NEXUS = auto()
    TURRET = auto()

class ResourceType(Enum):
    CRYSTAL = auto()
    VESPENE = auto()

class ActionType(Enum):
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    ATTACK = auto()
    HARVEST = auto()
    RETURN_RESOURCES = auto()
    IDLE = auto()

# Event System
class GameEvent:
    """Base class for all game events"""
    def __init__(self, trigger_time: int, duration: int = 1):
        self.trigger_time = trigger_time
        self.duration = duration
        self.is_active = False
        self.is_completed = False
        
    def check_trigger(self, current_time: int) -> bool:
        """Check if the event should trigger at the current time"""
        if not self.is_active and not self.is_completed and current_time >= self.trigger_time:
            self.is_active = True
            return True
        return False
    
    def update(self, env) -> bool:
        """Update the event state, returns True if the event is still active"""
        if self.is_active:
            self.duration -= 1
            if self.duration <= 0:
                self.is_active = False
                self.is_completed = True
                return False
        return self.is_active
    
    def get_event_info(self) -> Dict:
        """Get information about the event for visualization and agent awareness"""
        return {
            "type": self.__class__.__name__,
            "is_active": self.is_active,
            "is_completed": self.is_completed,
            "trigger_time": self.trigger_time,
            "duration": self.duration
        }

class EnemyAttackWave(GameEvent):
    """A wave of enemy units attacking from a specific location"""
    def __init__(self, trigger_time: int, strength: int, location: Tuple[int, int], duration: int = 1):
        super().__init__(trigger_time, duration)
        self.strength = strength  # Number of units in the wave
        self.location = location  # Starting location for the wave
        self.spawned_units = []  # Track spawned units
    
    def update(self, env) -> bool:
        """Spawn enemy units when activated"""
        if self.is_active and not self.spawned_units:
            # Spawn the enemy units
            for _ in range(self.strength):
                # Add some randomness to spawn position
                x = max(0, min(MAP_SIZE-1, self.location[0] + random.randint(-5, 5)))
                y = max(0, min(MAP_SIZE-1, self.location[1] + random.randint(-5, 5)))
                
                # Stronger units for the attack wave
                new_unit = Unit(env.next_unit_id, 
                               UnitType.ELITE_RAIDER if random.random() < 0.3 else UnitType.RAIDER, 
                               (x, y))
                
                # Elite raiders have more health and deal more damage
                if new_unit.type == UnitType.ELITE_RAIDER:
                    new_unit.health = RAIDER_HEALTH * 1.5
                
                env.enemy_units.append(new_unit)
                env.next_unit_id += 1
                self.spawned_units.append(new_unit.id)
            
            # Add visual notification
            env.event_notifications.append({
                "text": f"ALERT: Enemy attack wave ({self.strength} units)!",
                "position": self.location,
                "color": "red",
                "time": env.time,
                "duration": 50  # Show for 50 time steps
            })
        
        return super().update(env)
    
    def get_event_info(self) -> Dict:
        info = super().get_event_info()
        info.update({
            "strength": self.strength,
            "location": self.location,
            "spawned_units": len(self.spawned_units)
        })
        return info

class ResourceDepleted(GameEvent):
    """Event when a key resource patch runs out"""
    def __init__(self, trigger_time: int, resource_id: int, duration: int = 1):
        super().__init__(trigger_time, duration)
        self.resource_id = resource_id
    
    def update(self, env) -> bool:
        """Mark the resource as depleted"""
        if self.is_active and not self.is_completed:
            # Find the resource
            resource = next((r for r in env.resources if r.id == self.resource_id), None)
            if resource:
                # Deplete the resource
                resource.amount = 0
                
                # Add visual notification
                env.event_notifications.append({
                    "text": f"Resource depleted at {resource.position}!",
                    "position": resource.position,
                    "color": "orange",
                    "time": env.time,
                    "duration": 30
                })
        
        return super().update(env)
    
    def get_event_info(self) -> Dict:
        info = super().get_event_info()
        info.update({
            "resource_id": self.resource_id
        })
        return info

class NewResourceDiscovery(GameEvent):
    """Event when a new resource area is discovered"""
    def __init__(self, trigger_time: int, location: Tuple[int, int], resource_type: ResourceType, amount: int, duration: int = 1):
        super().__init__(trigger_time, duration)
        self.location = location
        self.resource_type = resource_type
        self.amount = amount
        self.resource_ids = []
    
    def update(self, env) -> bool:
        """Create new resources at the specified location"""
        if self.is_active and not self.resource_ids:
            # Create a cluster of resources
            num_patches = random.randint(3, 6)
            for i in range(num_patches):
                x = max(0, min(MAP_SIZE-1, self.location[0] + random.randint(-3, 3)))
                y = max(0, min(MAP_SIZE-1, self.location[1] + random.randint(-3, 3)))
                
                new_resource = Resource(env.next_resource_id, self.resource_type, (x, y), self.amount)
                env.resources.append(new_resource)
                self.resource_ids.append(env.next_resource_id)
                env.next_resource_id += 1
            
            # Add visual notification
            env.event_notifications.append({
                "text": f"New {self.resource_type.name} discovered!",
                "position": self.location,
                "color": "blue" if self.resource_type == ResourceType.CRYSTAL else "green",
                "time": env.time,
                "duration": 40
            })
        
        return super().update(env)
    
    def get_event_info(self) -> Dict:
        info = super().get_event_info()
        info.update({
            "location": self.location,
            "resource_type": self.resource_type.name,
            "amount": self.amount,
            "resource_ids": self.resource_ids
        })
        return info

class SuddenOpportunity(GameEvent):
    """Event representing a temporary vulnerability or opportunity"""
    def __init__(self, trigger_time: int, location: Tuple[int, int], opportunity_type: str, duration: int = 50):
        super().__init__(trigger_time, duration)
        self.location = location
        self.opportunity_type = opportunity_type  # e.g., "enemy_weakness", "unguarded_resource"
        self.bonus_applied = False
    
    def update(self, env) -> bool:
        """Create a temporary opportunity"""
        if self.is_active and not self.bonus_applied:
            # Different effects based on opportunity type
            if self.opportunity_type == "enemy_weakness":
                # Temporarily weaken enemies in the area
                for enemy in env.enemy_units:
                    dist = ((enemy.position[0] - self.location[0])**2 + 
                           (enemy.position[1] - self.location[1])**2)**0.5
                    if dist < 10:
                        enemy.health *= 0.7  # Reduce health by 30%
            
            # Add visual notification
            env.event_notifications.append({
                "text": f"Opportunity: {self.opportunity_type}!",
                "position": self.location,
                "color": "green",
                "time": env.time,
                "duration": self.duration
            })
            
            self.bonus_applied = True
        
        # Reset when the opportunity expires
        if self.is_completed and self.bonus_applied:
            self.bonus_applied = False
        
        return super().update(env)
    
    def get_event_info(self) -> Dict:
        info = super().get_event_info()
        info.update({
            "location": self.location,
            "opportunity_type": self.opportunity_type,
            "bonus_applied": self.bonus_applied
        })
        return info

# Classes
class Unit:
    def __init__(self, unit_id: int, unit_type: UnitType, position: Tuple[int, int]):
        self.id = unit_id
        self.type = unit_type
        self.position = position
        self.action = ActionType.IDLE
        self.target = None
        self.resources = 0
        
        # Set health based on unit type
        if unit_type == UnitType.HARVESTER:
            self.health = HARVESTER_HEALTH
        elif unit_type == UnitType.WARRIOR:
            self.health = WARRIOR_HEALTH
        elif unit_type == UnitType.RAIDER:
            self.health = RAIDER_HEALTH
        elif unit_type == UnitType.ELITE_RAIDER:
            self.health = RAIDER_HEALTH * 1.5
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def move(self, direction: ActionType, env):
        dx, dy = 0, 0
        if direction == ActionType.MOVE_UP:
            dy = -1
        elif direction == ActionType.MOVE_DOWN:
            dy = 1
        elif direction == ActionType.MOVE_LEFT:
            dx = -1
        elif direction == ActionType.MOVE_RIGHT:
            dx = 1
        
        new_x = max(0, min(MAP_SIZE-1, self.position[0] + dx))
        new_y = max(0, min(MAP_SIZE-1, self.position[1] + dy))
        
        # Check if position is occupied by a structure
        if not env.is_occupied_by_structure((new_x, new_y)):
            self.position = (new_x, new_y)
            # Update fog of war
            env.update_visibility(self.position, VISION_RADIUS)
    
    def attack(self, target, env):
        self.action = ActionType.ATTACK
        self.target = target
        
        # Calculate distance to target
        if isinstance(target, Unit):
            target_pos = target.position
        else:  # Structure
            target_pos = target.position
        
        dist = ((self.position[0] - target_pos[0])**2 + 
                (self.position[1] - target_pos[1])**2)**0.5
        
        # Check if in range (1 for melee)
        if dist <= 1:
            # Apply damage
            if self.type == UnitType.WARRIOR:
                damage = WARRIOR_ATTACK
            elif self.type == UnitType.RAIDER:
                damage = RAIDER_ATTACK
            elif self.type == UnitType.ELITE_RAIDER:
                damage = RAIDER_ATTACK * 1.5  # Elite raiders deal more damage
            else:
                damage = 5  # Minimal damage for other units
            
            if isinstance(target, Unit):
                target.health -= damage
            else:  # Structure
                target.health -= damage
            
            return True
        return False
    
    def harvest(self, resource, env):
        self.action = ActionType.HARVEST
        self.target = resource
        
        # Check if in range and has capacity
        dist = ((self.position[0] - resource.position[0])**2 + 
                (self.position[1] - resource.position[1])**2)**0.5
        
        if dist <= 1 and self.resources < HARVESTER_CAPACITY:
            # Harvest resource
            amount_to_harvest = min(1, resource.amount)
            amount_to_carry = min(amount_to_harvest, HARVESTER_CAPACITY - self.resources)
            
            resource.amount -= amount_to_carry
            self.resources += amount_to_carry
            
            return True
        return False
    
    def return_resources(self, nexus, env):
        self.action = ActionType.RETURN_RESOURCES
        self.target = nexus
        
        # Check if in range of nexus
        dist = ((self.position[0] - nexus.position[0])**2 + 
                (self.position[1] - nexus.position[1])**2)**0.5
        
        if dist <= 1 and self.resources > 0:
            # Return resources to nexus
            if self.resources > 0:
                env.crystal_count += self.resources
                self.resources = 0
                return True
        return False

class Structure:
    def __init__(self, structure_id: int, structure_type: StructureType, position: Tuple[int, int]):
        self.id = structure_id
        self.type = structure_type
        self.position = position
        self.production_queue = []
        self.production_time = 0
        
        # Set health based on structure type
        if structure_type == StructureType.NEXUS:
            self.health = NEXUS_HEALTH
        elif structure_type == StructureType.TURRET:
            self.health = TURRET_HEALTH
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def attack(self, target, env):
        if self.type != StructureType.TURRET:
            return False
        
        # Calculate distance to target
        if isinstance(target, Unit):
            target_pos = target.position
        else:  # Structure
            target_pos = target.position
        
        dist = ((self.position[0] - target_pos[0])**2 + 
                (self.position[1] - target_pos[1])**2)**0.5
        
        # Check if in range (3 for turrets)
        if dist <= 3:
            # Apply damage
            damage = TURRET_ATTACK
            
            if isinstance(target, Unit):
                target.health -= damage
            else:  # Structure
                target.health -= damage
            
            return True
        return False
    
    def produce_unit(self, unit_type: UnitType, env):
        if self.type != StructureType.NEXUS:
            return False
        
        # Check if we have enough resources
        cost = 0
        if unit_type == UnitType.HARVESTER:
            cost = HARVESTER_COST
        elif unit_type == UnitType.WARRIOR:
            cost = WARRIOR_COST
        
        if env.crystal_count >= cost:
            # Add to production queue
            self.production_queue.append(unit_type)
            env.crystal_count -= cost
            return True
        return False

class Resource:
    def __init__(self, resource_id: int, resource_type: ResourceType, position: Tuple[int, int], amount: int):
        self.id = resource_id
        self.type = resource_type
        self.position = position
        self.amount = amount
    
    def is_depleted(self) -> bool:
        return self.amount <= 0

class RTSEnvironment:
    def __init__(self, seed=None, scenario=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Game state
        self.time = 0
        self.crystal_count = 100  # Starting resources
        self.vespene_count = 0
        self.nexus_position = (MAP_SIZE//4, MAP_SIZE//4)  # Starting position
        self.enemy_base_position = (3*MAP_SIZE//4, 3*MAP_SIZE//4)  # Enemy base
        
        # ID counters
        self.next_unit_id = 0
        self.next_structure_id = 0
        self.next_resource_id = 0
        
        # Game objects
        self.player_units: List[Unit] = []
        self.enemy_units: List[Unit] = []
        self.structures: List[Structure] = []
        self.resources: List[Resource] = []
        
        # Fog of war (0 = unexplored, 1 = visible)
        self.visibility = np.zeros((MAP_SIZE, MAP_SIZE))
        self.last_seen = np.zeros((MAP_SIZE, MAP_SIZE))  # Time when each cell was last seen
        
        # Event notifications
        self.event_notifications = []
        
        # Events system
        self.events = []
        self.active_events = []
        self.completed_events = []
        
        # Initialize the map
        self._init_map()
        
        # Set up scenario if provided
        if scenario:
            self._setup_scenario(scenario)
    
    def _init_map(self):
        # Create the Nexus
        self.structures.append(Structure(self.next_structure_id, StructureType.NEXUS, self.nexus_position))
        self.next_structure_id += 1
        
        # Create initial units
        for _ in range(3):
            x = self.nexus_position[0] + random.randint(-2, 2)
            y = self.nexus_position[1] + random.randint(-2, 2)
            self.player_units.append(Unit(self.next_unit_id, UnitType.HARVESTER, (x, y)))
            self.next_unit_id += 1
        
        # Create crystal patches
        num_patches = 5
        for _ in range(num_patches):
            # Ensure patches aren't too close to bases
            while True:
                x = random.randint(5, MAP_SIZE-6)
                y = random.randint(5, MAP_SIZE-6)
                
                dist_to_nexus = ((x - self.nexus_position[0])**2 + (y - self.nexus_position[1])**2)**0.5
                dist_to_enemy = ((x - self.enemy_base_position[0])**2 + (y - self.enemy_base_position[1])**2)**0.5
                
                if dist_to_nexus > 10 and dist_to_enemy > 10:
                    break
            
            # Create a crystal patch (cluster of crystals)
            for i in range(random.randint(3, 6)):
                patch_x = min(MAP_SIZE-1, max(0, x + random.randint(-2, 2)))
                patch_y = min(MAP_SIZE-1, max(0, y + random.randint(-2, 2)))
                amount = random.randint(CRYSTAL_MAX//2, CRYSTAL_MAX)
                self.resources.append(Resource(self.next_resource_id, ResourceType.CRYSTAL, 
                                              (patch_x, patch_y), amount))
                self.next_resource_id += 1
        
        # Create one vespene geyser (contested resource)
        # Place it roughly in the middle between player and enemy base
        vespene_x = (self.nexus_position[0] + self.enemy_base_position[0]) // 2
        vespene_y = (self.nexus_position[1] + self.enemy_base_position[1]) // 2
        # Add some randomness
        vespene_x += random.randint(-5, 5)
        vespene_y += random.randint(-5, 5)
        self.resources.append(Resource(self.next_resource_id, ResourceType.VESPENE, 
                                      (vespene_x, vespene_y), VESPENE_MAX))
        self.next_resource_id += 1
        
        # Update initial visibility (around Nexus and units)
        self.update_visibility(self.nexus_position, VISION_RADIUS * 2)
        for unit in self.player_units:
            self.update_visibility(unit.position, VISION_RADIUS)
    
    def _setup_scenario(self, scenario_name):
        """Set up a predefined scenario with events"""
        if scenario_name == "peaceful_start_and_ambush":
            # Scenario A: Peaceful Start & Sudden Ambush
            
            # Start with more resources for a peaceful early game
            self.crystal_count = 200
            
            # Add an extra harvester
            x = self.nexus_position[0] + random.randint(-2, 2)
            y = self.nexus_position[1] + random.randint(-2, 2)
            self.player_units.append(Unit(self.next_unit_id, UnitType.HARVESTER, (x, y)))
            self.next_unit_id += 1
            
            # Add a major enemy attack wave at time 100
            attack_location = (random.randint(5, 15), random.randint(5, 15))  # Near player base
            self.add_event(EnemyAttackWave(100, 8, attack_location))
            
            # Add a smaller follow-up attack at time 200
            attack_location2 = (random.randint(10, 20), random.randint(10, 20))
            self.add_event(EnemyAttackWave(200, 5, attack_location2))
            
        elif scenario_name == "resource_scarcity_expansion":
            # Scenario B: Resource Scarcity & Expansion Opportunity
            
            # Start with fewer resources
            self.crystal_count = 50
            
            # Make initial resources scarcer
            for resource in self.resources:
                if resource.type == ResourceType.CRYSTAL:
                    resource.amount = resource.amount // 2
            
            # Schedule resource depletion events
            if self.resources:
                for i, resource in enumerate(self.resources):
                    if i % 2 == 0 and resource.type == ResourceType.CRYSTAL:  # Deplete half of crystal patches
                        self.add_event(ResourceDepleted(50 + i*10, resource.id))
            
            # Add new resource discovery events
            new_resource_x = random.randint(MAP_SIZE//2, 3*MAP_SIZE//4)
            new_resource_y = random.randint(MAP_SIZE//2, 3*MAP_SIZE//4)
            self.add_event(NewResourceDiscovery(150, (new_resource_x, new_resource_y), 
                                              ResourceType.CRYSTAL, CRYSTAL_MAX * 2))
            
            # Add a second resource discovery, this time vespene
            new_vespene_x = random.randint(MAP_SIZE//4, MAP_SIZE//2)
            new_vespene_y = random.randint(MAP_SIZE//4, MAP_SIZE//2)
            self.add_event(NewResourceDiscovery(220, (new_vespene_x, new_vespene_y), 
                                              ResourceType.VESPENE, VESPENE_MAX))
            
        elif scenario_name == "opportunity_and_threat":
            # Scenario C: Alternating Opportunities and Threats
            
            # Start with normal resources
            self.crystal_count = 150
            
            # Add opportunity events
            opportunity_pos = (MAP_SIZE//2, MAP_SIZE//2)
            self.add_event(SuddenOpportunity(80, opportunity_pos, "enemy_weakness", 40))
            
            # Add threat events alternating with opportunities
            attack_pos1 = (random.randint(10, 20), random.randint(10, 20))
            self.add_event(EnemyAttackWave(120, 6, attack_pos1))
            
            opportunity_pos2 = (random.randint(30, 40), random.randint(30, 40))
            self.add_event(SuddenOpportunity(180, opportunity_pos2, "unguarded_resource", 30))
            
            attack_pos2 = (random.randint(15, 25), random.randint(15, 25))
            self.add_event(EnemyAttackWave(240, 10, attack_pos2))
        
        else:
            # Default: add a mix of random events
            print(f"Unknown scenario '{scenario_name}'. Using random events.")
            self._add_random_events()
    
    def _add_random_events(self):
        """Add a set of random events to the environment"""
        # Add 1-3 attack waves
        num_attacks = random.randint(1, 3)
        for i in range(num_attacks):
            trigger_time = random.randint(80, 250)
            strength = random.randint(4, 10)
            x = random.randint(5, MAP_SIZE-5)
            y = random.randint(5, MAP_SIZE-5)
            self.add_event(EnemyAttackWave(trigger_time, strength, (x, y)))
        
        # Add 1-2 resource depletion events
        if self.resources:
            num_depletions = min(len(self.resources), random.randint(1, 2))
            for i in range(num_depletions):
                resource = random.choice(self.resources)
                trigger_time = random.randint(50, 150)
                self.add_event(ResourceDepleted(trigger_time, resource.id))
        
        # Add 1-2 resource discovery events
        num_discoveries = random.randint(1, 2)
        for i in range(num_discoveries):
            trigger_time = random.randint(100, 200)
            x = random.randint(10, MAP_SIZE-10)
            y = random.randint(10, MAP_SIZE-10)
            resource_type = random.choice([ResourceType.CRYSTAL, ResourceType.VESPENE])
            amount = CRYSTAL_MAX if resource_type == ResourceType.CRYSTAL else VESPENE_MAX
            self.add_event(NewResourceDiscovery(trigger_time, (x, y), resource_type, amount))
        
        # Add 0-1 opportunity events
        if random.random() < 0.7:  # 70% chance
            trigger_time = random.randint(150, 250)
            x = random.randint(10, MAP_SIZE-10)
            y = random.randint(10, MAP_SIZE-10)
            opportunity_type = random.choice(["enemy_weakness", "unguarded_resource"])
            self.add_event(SuddenOpportunity(trigger_time, (x, y), opportunity_type))
    
    def add_event(self, event):
        """Add a new event to the environment"""
        self.events.append(event)
    
    def update_events(self):
        """Update all events, trigger new ones and update active ones"""
        # Check for new events to trigger
        for event in self.events[:]:
            if event.check_trigger(self.time):
                self.active_events.append(event)
                self.events.remove(event)
        
        # Update active events
        for event in self.active_events[:]:
            if not event.update(self):
                self.completed_events.append(event)
                self.active_events.remove(event)
        
        # Update event notifications
        self._update_event_notifications()
    
    def _update_event_notifications(self):
        """Update the visual notifications for events"""
        # Remove expired notifications
        self.event_notifications = [
            notif for notif in self.event_notifications 
            if self.time - notif['time'] < notif['duration']
        ]
    
    def get_current_events(self):
        """Get information about all current events"""
        events_info = []
        
        # Add pending events
        for event in self.events:
            info = event.get_event_info()
            info['status'] = 'pending'
            events_info.append(info)
        
        # Add active events
        for event in self.active_events:
            info = event.get_event_info()
            info['status'] = 'active'
            events_info.append(info)
        
        # Add recently completed events
        recent_completed = [e for e in self.completed_events 
                          if self.time - e.trigger_time < 30]  # Only show recently completed
        for event in recent_completed:
            info = event.get_event_info()
            info['status'] = 'completed'
            events_info.append(info)
        
        return events_info

    def update_visibility(self, position: Tuple[int, int], radius: int):
        """Update the fog of war based on unit position."""
        x, y = position
        for i in range(max(0, x-radius), min(MAP_SIZE, x+radius+1)):
            for j in range(max(0, y-radius), min(MAP_SIZE, y+radius+1)):
                dist = ((i - x)**2 + (j - y)**2)**0.5
                if dist <= radius:
                    self.visibility[i, j] = 1
                    self.last_seen[i, j] = self.time
    
    def is_visible(self, position: Tuple[int, int]) -> bool:
        """Check if a position is currently visible."""
        return self.visibility[position[0], position[1]] > 0.5
    
    def is_occupied_by_structure(self, position: Tuple[int, int]) -> bool:
        """Check if a position is occupied by a structure."""
        for structure in self.structures:
            if structure.position == position and structure.is_alive():
                return True
        return False
    
    def spawn_enemy(self):
        """Spawn enemy raiders at random positions along the map edges."""
        # Spawn rate increases over time
        if self.time < 100:
            spawn_prob = 0.05
        elif self.time < 300:
            spawn_prob = 0.1
        else:
            spawn_prob = 0.15
        
        if random.random() < spawn_prob:
            # Decide which edge to spawn from
            edge = random.randint(0, 3)
            if edge == 0:  # Top
                x = random.randint(0, MAP_SIZE-1)
                y = 0
            elif edge == 1:  # Right
                x = MAP_SIZE-1
                y = random.randint(0, MAP_SIZE-1)
            elif edge == 2:  # Bottom
                x = random.randint(0, MAP_SIZE-1)
                y = MAP_SIZE-1
            else:  # Left
                x = 0
                y = random.randint(0, MAP_SIZE-1)
            
            # Create the enemy unit
            self.enemy_units.append(Unit(self.next_unit_id, UnitType.RAIDER, (x, y)))
            self.next_unit_id += 1
    
    def process_production(self):
        """Process production queue for structures."""
        for structure in self.structures:
            if structure.type == StructureType.NEXUS and structure.is_alive():
                if structure.production_queue:
                    structure.production_time += 1
                    
                    # Production complete
                    if structure.production_time >= 5:
                        unit_type = structure.production_queue.pop(0)
                        structure.production_time = 0
                        
                        # Spawn the unit near the Nexus
                        pos = structure.position
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                new_pos = (max(0, min(MAP_SIZE-1, pos[0] + dx)), 
                                          max(0, min(MAP_SIZE-1, pos[1] + dy)))
                                if not self.is_occupied_by_structure(new_pos):
                                    self.player_units.append(Unit(self.next_unit_id, unit_type, new_pos))
                                    self.next_unit_id += 1
                                    return
    
    def update_fog_of_war(self):
        """Update fog of war, slowly returning to unexplored state."""
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if self.visibility[i, j] > 0.5 and self.time - self.last_seen[i, j] > 20:
                    self.visibility[i, j] = max(0, self.visibility[i, j] - FOG_DECAY_RATE)
    
    def update_enemy_ai(self):
        """Simple scripted AI for enemy units."""
        for enemy in self.enemy_units:
            if not enemy.is_alive():
                continue
            
            # Find closest target
            closest_target = None
            closest_dist = float('inf')
            
            # First prioritize structures
            for structure in self.structures:
                if structure.is_alive():
                    dist = ((enemy.position[0] - structure.position[0])**2 + 
                            (enemy.position[1] - structure.position[1])**2)**0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_target = structure
            
            # Then player units
            for unit in self.player_units:
                if unit.is_alive() and self.is_visible(unit.position):
                    dist = ((enemy.position[0] - unit.position[0])**2 + 
                            (enemy.position[1] - unit.position[1])**2)**0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_target = unit
            
            # If no visible target, head toward Nexus
            if closest_target is None:
                for structure in self.structures:
                    if structure.type == StructureType.NEXUS and structure.is_alive():
                        closest_target = structure
                        break
            
            # Attack or move toward target
            if closest_target:
                if closest_dist <= 1:
                    enemy.attack(closest_target, self)
                else:
                    # Move toward target
                    tx, ty = closest_target.position
                    x, y = enemy.position
                    
                    if tx < x:
                        enemy.move(ActionType.MOVE_LEFT, self)
                    elif tx > x:
                        enemy.move(ActionType.MOVE_RIGHT, self)
                    elif ty < y:
                        enemy.move(ActionType.MOVE_UP, self)
                    elif ty > y:
                        enemy.move(ActionType.MOVE_DOWN, self)
    
    def update_turrets(self):
        """Update turret attacks."""
        for structure in self.structures:
            if structure.type == StructureType.TURRET and structure.is_alive():
                # Find closest enemy in range
                closest_enemy = None
                closest_dist = float('inf')
                
                for enemy in self.enemy_units:
                    if enemy.is_alive() and self.is_visible(enemy.position):
                        dist = ((structure.position[0] - enemy.position[0])**2 + 
                                (structure.position[1] - enemy.position[1])**2)**0.5
                        if dist <= 3 and dist < closest_dist:
                            closest_dist = dist
                            closest_enemy = enemy
                
                if closest_enemy:
                    structure.attack(closest_enemy, self)
    
    def remove_dead(self):
        """Remove dead units and structures from the game."""
        self.player_units = [u for u in self.player_units if u.is_alive()]
        self.enemy_units = [u for u in self.enemy_units if u.is_alive()]
        self.structures = [s for s in self.structures if s.is_alive()]
        self.resources = [r for r in self.resources if not r.is_depleted()]
    
    def step(self):
        """Step the environment forward by one time step."""
        self.time += 1
        
        # Process production queues
        self.process_production()
        
        # Spawn enemies
        self.spawn_enemy()
        
        # Update events
        self.update_events()
        
        # Update enemy AI
        self.update_enemy_ai()
        
        # Update turrets
        self.update_turrets()
        
        # Update fog of war
        self.update_fog_of_war()
        
        # Remove dead units, structures, and depleted resources
        self.remove_dead()
        
        # Check if game is over (Nexus destroyed)
        game_over = True
        for structure in self.structures:
            if structure.type == StructureType.NEXUS and structure.is_alive():
                game_over = False
                break
        
        return game_over
    
    def get_state(self):
        """Return the current state of the environment."""
        return {
            'time': self.time,
            'crystal_count': self.crystal_count,
            'vespene_count': self.vespene_count,
            'player_units': self.player_units,
            'enemy_units': [e for e in self.enemy_units if self.is_visible(e.position)],
            'structures': self.structures,
            'resources': [r for r in self.resources if self.is_visible(r.position)],
            'visibility': self.visibility.copy(),
            'game_over': not any(s.type == StructureType.NEXUS and s.is_alive() for s in self.structures),
            'event_notifications': self.event_notifications,
            'active_events': [e.get_event_info() for e in self.active_events]
        }
    
    def render(self):
        """Render the current state of the environment."""
        plt.figure(figsize=(12, 10))
        
        # Create main game grid axes
        main_ax = plt.subplot2grid((6, 8), (0, 0), colspan=6, rowspan=5)
        main_ax.set_xlim(0, MAP_SIZE)
        main_ax.set_ylim(0, MAP_SIZE)
        
        # Draw fog of war
        visible_mask = (self.visibility > 0.5)
        fog_img = np.ones((MAP_SIZE, MAP_SIZE, 4))  # RGBA
        fog_img[~visible_mask, 3] = 0.7  # Alpha channel for non-visible cells
        
        main_ax.imshow(fog_img, extent=(0, MAP_SIZE, 0, MAP_SIZE), origin='lower')
        
        # Draw resources
        for resource in self.resources:
            if self.is_visible(resource.position):
                if resource.type == ResourceType.CRYSTAL:
                    color = 'blue'
                else:  # VESPENE
                    color = 'green'
                main_ax.scatter(resource.position[0] + 0.5, resource.position[1] + 0.5, 
                          color=color, s=100 * (resource.amount / CRYSTAL_MAX))
        
        # Draw structures
        for structure in self.structures:
            if structure.is_alive():
                if structure.type == StructureType.NEXUS:
                    color = 'gold'
                    size = 3
                else:  # TURRET
                    color = 'gray'
                    size = 2
                
                rect = patches.Rectangle((structure.position[0], structure.position[1]), 
                                        size, size, linewidth=1, edgecolor='black', 
                                        facecolor=color, alpha=0.7)
                main_ax.add_patch(rect)
                
                # Draw health bar
                health_pct = structure.health / (NEXUS_HEALTH if structure.type == StructureType.NEXUS else TURRET_HEALTH)
                main_ax.plot([structure.position[0], structure.position[0] + size * health_pct], 
                        [structure.position[1] - 0.2, structure.position[1] - 0.2], 
                        color='red', linewidth=2)
        
        # Draw player units
        for unit in self.player_units:
            if unit.is_alive():
                if unit.type == UnitType.HARVESTER:
                    color = 'cyan'
                    marker = 'o'
                else:  # WARRIOR
                    color = 'blue'
                    marker = 's'
                
                main_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                          color=color, s=80, marker=marker)
                
                # Draw resource carried (for harvesters)
                if unit.type == UnitType.HARVESTER and unit.resources > 0:
                    main_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                              color='yellow', s=20)
                
                # Draw health bar
                max_health = HARVESTER_HEALTH if unit.type == UnitType.HARVESTER else WARRIOR_HEALTH
                health_pct = unit.health / max_health
                main_ax.plot([unit.position[0], unit.position[0] + health_pct], 
                        [unit.position[1] - 0.2, unit.position[1] - 0.2], 
                        color='red', linewidth=2)
        
        # Draw enemy units (only if visible)
        for unit in self.enemy_units:
            if unit.is_alive() and self.is_visible(unit.position):
                if unit.type == UnitType.ELITE_RAIDER:
                    color = 'darkred'
                    marker = '*'
                    size = 100
                else:  # Regular RAIDER
                    color = 'red'
                    marker = 'x'
                    size = 80
                
                main_ax.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                          color=color, s=size, marker=marker)
                
                # Draw health bar
                if unit.type == UnitType.ELITE_RAIDER:
                    max_health = RAIDER_HEALTH * 1.5
                else:
                    max_health = RAIDER_HEALTH
                health_pct = unit.health / max_health
                main_ax.plot([unit.position[0], unit.position[0] + health_pct], 
                        [unit.position[1] - 0.2, unit.position[1] - 0.2], 
                        color='red', linewidth=2)
        
        # Draw event notifications on the map
        for notif in self.event_notifications:
            if 'position' in notif and 'text' in notif:
                x, y = notif['position']
                if self.is_visible((x, y)):
                    color = notif.get('color', 'white')
                    # Add a marker for the event location
                    main_ax.scatter(x + 0.5, y + 0.5, color=color, 
                                s=150, marker='o', alpha=0.7, edgecolors='black')
                    # Add an arrow pointing to it
                    main_ax.annotate(notif['text'], xy=(x + 0.5, y + 0.5), 
                                xytext=(x + 5, y + 5),
                                arrowprops=dict(facecolor=color, shrink=0.05),
                                color=color, fontweight='bold', 
                                bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
        
        # Draw grid
        for i in range(MAP_SIZE + 1):
            main_ax.axhline(y=i, color='black', linestyle='-', alpha=0.2)
            main_ax.axvline(x=i, color='black', linestyle='-', alpha=0.2)
        
        # Create side panel for game stats
        stats_ax = plt.subplot2grid((6, 8), (0, 6), colspan=2, rowspan=2)
        stats_ax.axis('off')
        
        # Display game stats
        stats_text = f"Time: {self.time}\nCrystals: {self.crystal_count}\nVespene: {self.vespene_count}\n"
        stats_text += f"Units: {len(self.player_units)}\nEnemies: {len([e for e in self.enemy_units if self.is_visible(e.position)])}"
        stats_ax.text(0.1, 0.9, stats_text, fontsize=12, va='top', 
                     bbox=dict(facecolor='white', alpha=0.7))
        
        # Create side panel for active events
        events_ax = plt.subplot2grid((6, 8), (2, 6), colspan=2, rowspan=3)
        events_ax.axis('off')
        
        # Display active events
        events_text = "ACTIVE EVENTS:\n"
        active_events = self.get_current_events()
        active_count = 0
        
        for event in active_events:
            if event['status'] == 'active':
                active_count += 1
                events_text += f"\n• {event['type']}"
                if event['type'] == 'EnemyAttackWave':
                    events_text += f" ({event['strength']} units)"
                elif event['type'] == 'NewResourceDiscovery':
                    events_text += f" ({event['resource_type']})"
                elif event['type'] == 'SuddenOpportunity':
                    events_text += f" ({event['opportunity_type']})"
        
        if active_count == 0:
            events_text += "\nNone"
        
        events_text += "\n\nPENDING EVENTS:"
        pending_count = 0
        
        for event in active_events:
            if event['status'] == 'pending':
                pending_count += 1
                events_text += f"\n• {event['type']} (T+{event['trigger_time'] - self.time})"
        
        if pending_count == 0:
            events_text += "\nNone"
        
        events_ax.text(0.1, 0.95, events_text, fontsize=10, va='top', 
                      bbox=dict(facecolor='white', alpha=0.7))
        
        # Create bottom panel for attention visualization
        attention_ax = plt.subplot2grid((6, 8), (5, 0), colspan=8, rowspan=1)
        attention_ax.set_title("Attention Allocation (if agent data available)", fontsize=10)
        attention_ax.set_xlim(0, 1)
        attention_ax.set_ylim(0, 1)
        attention_ax.axis('off')
        
        # This will be populated by the agent visualization code
        
        plt.suptitle("RTS Environment", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.pause(0.1)

# Example usage
if __name__ == "__main__":
    env = RTSEnvironment(seed=42)
    
    # Simple game loop
    for _ in range(100):
        env.render()
        game_over = env.step()
        
        if game_over:
            print("Game Over! Nexus destroyed.")
            break
    
    plt.show() 