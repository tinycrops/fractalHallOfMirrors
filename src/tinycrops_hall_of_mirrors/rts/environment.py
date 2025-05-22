"""
Consolidated RTS Environment with Dynamic Events and Novel Features.

This environment combines the existing RTS functionality with several novel enhancements:
- Dynamic event system with emergent scenarios
- Adaptive AI opponents
- Multi-objective optimization scenarios
- Procedural content generation
"""

import numpy as np
import random
from enum import Enum, auto
from typing import List, Dict, Tuple, Set, Optional
import time
from collections import defaultdict, deque

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
FOG_DECAY_RATE = 0.01


class UnitType(Enum):
    HARVESTER = auto()
    WARRIOR = auto()
    RAIDER = auto()
    ELITE_RAIDER = auto()


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


class EventType(Enum):
    """Novel enhancement: Expanded event system."""
    ENEMY_ATTACK_WAVE = auto()
    RESOURCE_DEPLETION = auto()
    RESOURCE_DISCOVERY = auto()
    SUDDEN_OPPORTUNITY = auto()
    WEATHER_EVENT = auto()      # New: Affects unit movement/visibility
    TECH_BREAKTHROUGH = auto()   # New: Temporarily enhances units
    SABOTAGE = auto()           # New: Enemy sabotages structures
    ALLY_REINFORCEMENT = auto()  # New: Friendly units arrive
    RESOURCE_RUSH = auto()      # New: Time-limited high-value resources


class GameEvent:
    """Enhanced event system with more sophisticated mechanics."""
    
    def __init__(self, event_type: EventType, trigger_time: int, duration: int = 1):
        self.event_type = event_type
        self.trigger_time = trigger_time
        self.duration = duration
        self.is_active = False
        self.is_completed = False
        self.intensity = 1.0  # New: Event intensity scaling
        
    def check_trigger(self, current_time: int) -> bool:
        if not self.is_active and not self.is_completed and current_time >= self.trigger_time:
            self.is_active = True
            return True
        return False
    
    def update(self, env) -> bool:
        if self.is_active:
            self.apply_effect(env)
            self.duration -= 1
            if self.duration <= 0:
                self.is_active = False
                self.is_completed = True
                self.cleanup_effect(env)
                return False
        return self.is_active
    
    def apply_effect(self, env):
        """Apply the event's effect to the environment."""
        pass  # To be overridden by specific event types
    
    def cleanup_effect(self, env):
        """Clean up any temporary effects when the event ends."""
        pass
    
    def get_event_info(self) -> Dict:
        return {
            "type": self.event_type.name,
            "is_active": self.is_active,
            "is_completed": self.is_completed,
            "trigger_time": self.trigger_time,
            "duration": self.duration,
            "intensity": self.intensity
        }


class EnemyAttackWave(GameEvent):
    """Enhanced enemy attack with adaptive difficulty."""
    
    def __init__(self, trigger_time: int, strength: int, location: Tuple[int, int], 
                 duration: int = 1, wave_type: str = "standard"):
        super().__init__(EventType.ENEMY_ATTACK_WAVE, trigger_time, duration)
        self.strength = strength
        self.location = location
        self.wave_type = wave_type  # "standard", "elite", "mixed"
        self.spawned_units = []
    
    def apply_effect(self, env):
        if not self.spawned_units:
            # Adaptive strength based on player performance
            adaptive_strength = max(1, self.strength + env.get_player_strength_modifier())
            
            for _ in range(adaptive_strength):
                x = max(0, min(MAP_SIZE-1, self.location[0] + random.randint(-5, 5)))
                y = max(0, min(MAP_SIZE-1, self.location[1] + random.randint(-5, 5)))
                
                # Determine unit type based on wave type
                if self.wave_type == "elite":
                    unit_type = UnitType.ELITE_RAIDER
                elif self.wave_type == "mixed":
                    unit_type = UnitType.ELITE_RAIDER if random.random() < 0.4 else UnitType.RAIDER
                else:
                    unit_type = UnitType.RAIDER
                
                new_unit = Unit(env.next_unit_id, unit_type, (x, y))
                
                if new_unit.type == UnitType.ELITE_RAIDER:
                    new_unit.health = int(RAIDER_HEALTH * 1.5)
                    new_unit.attack_damage = int(RAIDER_ATTACK * 1.3)
                
                env.enemy_units.append(new_unit)
                env.next_unit_id += 1
                self.spawned_units.append(new_unit.id)
            
            env.add_notification(f"ALERT: {self.wave_type.title()} attack wave ({adaptive_strength} units)!",
                               self.location, "red", 50)


class WeatherEvent(GameEvent):
    """Novel innovation: Weather affects gameplay."""
    
    def __init__(self, trigger_time: int, weather_type: str, duration: int = 100):
        super().__init__(EventType.WEATHER_EVENT, trigger_time, duration)
        self.weather_type = weather_type  # "fog", "storm", "clear"
        self.original_vision_radius = None
        
    def apply_effect(self, env):
        if self.original_vision_radius is None:
            self.original_vision_radius = env.vision_radius
            
        if self.weather_type == "fog":
            env.vision_radius = max(3, env.vision_radius // 2)
            env.fog_decay_rate *= 2  # Fog returns faster
        elif self.weather_type == "storm":
            env.vision_radius = max(4, env.vision_radius // 1.5)
            # Units move slower in storms (implemented in unit movement)
            for unit in env.player_units + env.enemy_units:
                unit.weather_movement_penalty = 0.7
        elif self.weather_type == "clear":
            env.vision_radius = min(MAP_SIZE//4, env.vision_radius * 1.5)
            env.fog_decay_rate *= 0.5  # Fog returns slower
    
    def cleanup_effect(self, env):
        if self.original_vision_radius is not None:
            env.vision_radius = self.original_vision_radius
            env.fog_decay_rate = FOG_DECAY_RATE
            
        # Remove weather penalties
        for unit in env.player_units + env.enemy_units:
            if hasattr(unit, 'weather_movement_penalty'):
                delattr(unit, 'weather_movement_penalty')


class TechBreakthrough(GameEvent):
    """Novel innovation: Temporary technological advantages."""
    
    def __init__(self, trigger_time: int, tech_type: str, duration: int = 200):
        super().__init__(EventType.TECH_BREAKTHROUGH, trigger_time, duration)
        self.tech_type = tech_type  # "enhanced_harvesting", "improved_weapons", "advanced_armor"
        
    def apply_effect(self, env):
        if self.tech_type == "enhanced_harvesting":
            for unit in env.player_units:
                if unit.type == UnitType.HARVESTER:
                    unit.harvesting_bonus = 1.5
        elif self.tech_type == "improved_weapons":
            for unit in env.player_units:
                if unit.type == UnitType.WARRIOR:
                    unit.attack_bonus = 1.3
        elif self.tech_type == "advanced_armor":
            for unit in env.player_units:
                unit.armor_bonus = 0.8  # Take 20% less damage
        
        env.add_notification(f"BREAKTHROUGH: {self.tech_type.replace('_', ' ').title()}!",
                           (MAP_SIZE//2, MAP_SIZE//2), "blue", 30)
    
    def cleanup_effect(self, env):
        # Remove tech bonuses
        for unit in env.player_units:
            for attr in ['harvesting_bonus', 'attack_bonus', 'armor_bonus']:
                if hasattr(unit, attr):
                    delattr(unit, attr)


class Unit:
    """Enhanced unit class with novel features."""
    
    def __init__(self, unit_id: int, unit_type: UnitType, position: Tuple[int, int]):
        self.id = unit_id
        self.type = unit_type
        self.position = position
        self.health = self._get_max_health()
        self.max_health = self.health
        self.resources = 0
        self.action = ActionType.IDLE
        self.target = None
        
        # Novel enhancements
        self.experience = 0  # Units gain experience
        self.morale = 1.0   # Affects performance
        self.fatigue = 0.0  # Accumulates over time
        self.attack_damage = self._get_base_attack()
        
    def _get_max_health(self):
        health_map = {
            UnitType.HARVESTER: HARVESTER_HEALTH,
            UnitType.WARRIOR: WARRIOR_HEALTH,
            UnitType.RAIDER: RAIDER_HEALTH,
            UnitType.ELITE_RAIDER: int(RAIDER_HEALTH * 1.5)
        }
        return health_map.get(self.type, 100)
    
    def _get_base_attack(self):
        attack_map = {
            UnitType.WARRIOR: WARRIOR_ATTACK,
            UnitType.RAIDER: RAIDER_ATTACK,
            UnitType.ELITE_RAIDER: int(RAIDER_ATTACK * 1.3)
        }
        return attack_map.get(self.type, 0)
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def move(self, direction: ActionType, env):
        """Enhanced movement with weather effects and fatigue."""
        if not self.is_alive():
            return False
            
        movement_map = {
            ActionType.MOVE_UP: (0, -1),
            ActionType.MOVE_DOWN: (0, 1),
            ActionType.MOVE_LEFT: (-1, 0),
            ActionType.MOVE_RIGHT: (1, 0)
        }
        
        if direction not in movement_map:
            return False
            
        dx, dy = movement_map[direction]
        
        # Apply weather movement penalty
        if hasattr(self, 'weather_movement_penalty'):
            if random.random() > self.weather_movement_penalty:
                return False  # Movement blocked by weather
        
        # Apply fatigue penalty
        if self.fatigue > 0.8:
            if random.random() < self.fatigue * 0.3:
                return False  # Too tired to move
        
        new_x = max(0, min(MAP_SIZE - 1, self.position[0] + dx))
        new_y = max(0, min(MAP_SIZE - 1, self.position[1] + dy))
        
        # Check if position is occupied
        for other_unit in env.player_units + env.enemy_units:
            if other_unit.id != self.id and other_unit.position == (new_x, new_y):
                return False
                
        for structure in env.structures:
            if structure.position == (new_x, new_y):
                return False
        
        self.position = (new_x, new_y)
        self.action = direction
        
        # Increase fatigue slightly
        self.fatigue = min(1.0, self.fatigue + 0.001)
        
        # Update visibility
        env.update_visibility(self.position, env.vision_radius)
        
        return True
    
    def attack(self, target, env):
        """Enhanced attack with experience and morale effects."""
        if not self.is_alive() or not target.is_alive():
            return False
            
        distance = ((self.position[0] - target.position[0])**2 + 
                   (self.position[1] - target.position[1])**2)**0.5
        
        if distance > 2:  # Attack range
            return False
        
        # Calculate damage with bonuses
        base_damage = self.attack_damage
        
        # Apply tech bonuses
        if hasattr(self, 'attack_bonus'):
            base_damage *= self.attack_bonus
            
        # Apply morale and experience bonuses
        damage = base_damage * self.morale * (1 + self.experience * 0.1)
        
        # Apply target's armor bonus
        if hasattr(target, 'armor_bonus'):
            damage *= target.armor_bonus
            
        target.health -= int(damage)
        self.action = ActionType.ATTACK
        self.target = target
        
        # Gain experience
        self.experience = min(10, self.experience + 0.1)
        
        return True
    
    def harvest(self, resource, env):
        """Enhanced harvesting with tech bonuses."""
        if not self.is_alive() or self.type != UnitType.HARVESTER:
            return False
            
        distance = ((self.position[0] - resource.position[0])**2 + 
                   (self.position[1] - resource.position[1])**2)**0.5
        
        if distance > 1.5:
            return False
            
        if self.resources >= HARVESTER_CAPACITY:
            return False
            
        if resource.amount <= 0:
            return False
        
        # Calculate harvest amount with bonuses
        harvest_amount = 1
        if hasattr(self, 'harvesting_bonus'):
            harvest_amount = int(harvest_amount * self.harvesting_bonus)
        
        actual_harvest = min(harvest_amount, 
                           resource.amount, 
                           HARVESTER_CAPACITY - self.resources)
        
        self.resources += actual_harvest
        resource.amount -= actual_harvest
        self.action = ActionType.HARVEST
        self.target = resource
        
        # Gain experience
        self.experience = min(10, self.experience + 0.05)
        
        return True
    
    def return_resources(self, nexus, env):
        """Return resources to the nexus."""
        if not self.is_alive() or self.type != UnitType.HARVESTER or self.resources <= 0:
            return False
            
        distance = ((self.position[0] - nexus.position[0])**2 + 
                   (self.position[1] - nexus.position[1])**2)**0.5
        
        if distance > 2:
            return False
        
        env.crystal_count += self.resources
        self.resources = 0
        self.action = ActionType.RETURN_RESOURCES
        self.target = nexus
        
        return True
    
    def update_state(self):
        """Novel enhancement: Update unit's internal state."""
        # Reduce fatigue over time when idle
        if self.action == ActionType.IDLE:
            self.fatigue = max(0, self.fatigue - 0.002)
        
        # Morale is affected by health and nearby allies
        health_ratio = self.health / self.max_health
        self.morale = 0.5 + 0.5 * health_ratio


class Structure:
    """Enhanced structure class."""
    
    def __init__(self, structure_id: int, structure_type: StructureType, position: Tuple[int, int]):
        self.id = structure_id
        self.type = structure_type
        self.position = position
        self.health = self._get_max_health()
        self.max_health = self.health
        self.production_queue = deque()
        self.production_timer = 0
        
        # Novel enhancements
        self.efficiency = 1.0  # Production efficiency
        self.damage_taken = 0   # Track cumulative damage
        
    def _get_max_health(self):
        health_map = {
            StructureType.NEXUS: NEXUS_HEALTH,
            StructureType.TURRET: TURRET_HEALTH
        }
        return health_map.get(self.type, 500)
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def produce_unit(self, unit_type: UnitType, env):
        """Enhanced production with efficiency."""
        if not self.is_alive() or self.type != StructureType.NEXUS:
            return False
            
        cost_map = {
            UnitType.HARVESTER: HARVESTER_COST,
            UnitType.WARRIOR: WARRIOR_COST
        }
        
        cost = cost_map.get(unit_type, 0)
        if env.crystal_count >= cost:
            env.crystal_count -= cost
            
            # Production time affected by efficiency
            production_time = int(30 / self.efficiency)  # Base 30 time units
            
            self.production_queue.append((unit_type, production_time))
            return True
        
        return False
    
    def update_production(self, env):
        """Update production with efficiency bonuses."""
        if self.production_queue and self.production_timer <= 0:
            unit_type, time_remaining = self.production_queue[0]
            time_remaining -= int(self.efficiency)  # Efficiency affects production speed
            
            if time_remaining <= 0:
                # Unit is ready
                self.production_queue.popleft()
                
                # Find spawn location
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        spawn_pos = (self.position[0] + dx, self.position[1] + dy)
                        if (0 <= spawn_pos[0] < MAP_SIZE and 
                            0 <= spawn_pos[1] < MAP_SIZE and
                            not env.is_position_occupied(spawn_pos)):
                            
                            new_unit = Unit(env.next_unit_id, unit_type, spawn_pos)
                            env.player_units.append(new_unit)
                            env.next_unit_id += 1
                            break
                    else:
                        continue
                    break
            else:
                self.production_queue[0] = (unit_type, time_remaining)


class Resource:
    """Enhanced resource class."""
    
    def __init__(self, resource_id: int, resource_type: ResourceType, 
                 position: Tuple[int, int], amount: int):
        self.id = resource_id
        self.type = resource_type
        self.position = position
        self.amount = amount
        self.max_amount = amount
        
        # Novel enhancement: Resource regeneration
        self.regeneration_rate = 0.01 if resource_type == ResourceType.VESPENE else 0
        
    def is_depleted(self) -> bool:
        return self.amount <= 0
        
    def update(self):
        """Update resource state including regeneration."""
        if self.regeneration_rate > 0 and self.amount < self.max_amount:
            self.amount = min(self.max_amount, 
                            self.amount + self.regeneration_rate)


class RTSEnvironment:
    """Enhanced RTS Environment with novel features and dynamic events."""
    
    def __init__(self, seed=None, scenario=None, enable_novel_features=True):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.time = 0
        self.crystal_count = 200  # Starting resources
        self.vespene_count = 0
        
        # Environment state
        self.player_units = []
        self.enemy_units = []
        self.structures = []
        self.resources = []
        
        # Enhanced features
        self.events = []
        self.event_notifications = []
        self.visibility = np.zeros((MAP_SIZE, MAP_SIZE))
        self.fog_of_war = np.ones((MAP_SIZE, MAP_SIZE))
        
        # Novel enhancements
        self.enable_novel_features = enable_novel_features
        self.adaptive_difficulty = 1.0
        self.player_performance_history = deque(maxlen=100)
        self.weather_system_active = enable_novel_features
        self.tech_progression = {"enhanced_harvesting": False, "improved_weapons": False, "advanced_armor": False}
        
        # Environment parameters
        self.vision_radius = VISION_RADIUS
        self.fog_decay_rate = FOG_DECAY_RATE
        
        # IDs for new entities
        self.next_unit_id = 1
        self.next_structure_id = 1
        self.next_resource_id = 1
        
        # Initialize map
        self._init_map()
        
        # Setup scenario if provided
        if scenario:
            self._setup_scenario(scenario)
        
        # Add random events if novel features enabled
        if self.enable_novel_features:
            self._add_enhanced_random_events()
    
    def _init_map(self):
        """Initialize the map with structures and resources."""
        # Create the Nexus at a central location
        nexus_pos = (MAP_SIZE//4, MAP_SIZE//4)
        nexus = Structure(self.next_structure_id, StructureType.NEXUS, nexus_pos)
        self.structures.append(nexus)
        self.next_structure_id += 1
        
        # Update visibility around nexus
        self.update_visibility(nexus_pos, self.vision_radius)
        
        # Create starting units
        for i in range(3):  # Start with 3 harvesters
            unit_pos = (nexus_pos[0] + i - 1, nexus_pos[1] + 2)
            harvester = Unit(self.next_unit_id, UnitType.HARVESTER, unit_pos)
            self.player_units.append(harvester)
            self.next_unit_id += 1
            self.update_visibility(unit_pos, self.vision_radius)
        
        # Create initial resources
        self._generate_resources()
    
    def _generate_resources(self):
        """Generate resources across the map with novel distribution patterns."""
        # Crystal patches - clustered distribution
        num_crystal_clusters = random.randint(4, 7)
        for _ in range(num_crystal_clusters):
            cluster_center = (random.randint(10, MAP_SIZE-10), random.randint(10, MAP_SIZE-10))
            cluster_size = random.randint(3, 6)
            
            for _ in range(cluster_size):
                x = max(0, min(MAP_SIZE-1, cluster_center[0] + random.randint(-5, 5)))
                y = max(0, min(MAP_SIZE-1, cluster_center[1] + random.randint(-5, 5)))
                
                if not self.is_position_occupied((x, y)):
                    amount = random.randint(80, 120)
                    crystal = Resource(self.next_resource_id, ResourceType.CRYSTAL, (x, y), amount)
                    self.resources.append(crystal)
                    self.next_resource_id += 1
        
        # Vespene geysers - rare and valuable
        num_vespene = random.randint(2, 4)
        for _ in range(num_vespene):
            while True:
                x = random.randint(5, MAP_SIZE-5)
                y = random.randint(5, MAP_SIZE-5)
                
                if not self.is_position_occupied((x, y)):
                    amount = random.randint(150, 250)
                    vespene = Resource(self.next_resource_id, ResourceType.VESPENE, (x, y), amount)
                    self.resources.append(vespene)
                    self.next_resource_id += 1
                    break
    
    def _add_enhanced_random_events(self):
        """Add enhanced event system with novel event types."""
        # Traditional events
        for i in range(3):
            attack_time = random.randint(300 + i*200, 500 + i*300)
            strength = random.randint(3, 8)
            location = (random.randint(45, MAP_SIZE-5), random.randint(45, MAP_SIZE-5))
            wave_type = random.choice(["standard", "elite", "mixed"])
            
            self.events.append(EnemyAttackWave(attack_time, strength, location, 1, wave_type))
        
        # Novel weather events
        if self.weather_system_active:
            for i in range(2):
                weather_time = random.randint(200 + i*300, 400 + i*400)
                weather_type = random.choice(["fog", "storm", "clear"])
                duration = random.randint(80, 150)
                
                self.events.append(WeatherEvent(weather_time, weather_type, duration))
        
        # Tech breakthrough events
        for i in range(2):
            tech_time = random.randint(400 + i*250, 600 + i*350)
            tech_type = random.choice(["enhanced_harvesting", "improved_weapons", "advanced_armor"])
            duration = random.randint(150, 250)
            
            self.events.append(TechBreakthrough(tech_time, tech_type, duration))
    
    def get_player_strength_modifier(self) -> int:
        """Novel enhancement: Adaptive difficulty based on player performance."""
        if len(self.player_performance_history) < 10:
            return 0
            
        recent_performance = list(self.player_performance_history)[-10:]
        avg_performance = np.mean(recent_performance)
        
        # Scale enemy strength based on player success
        if avg_performance > 0.7:  # Player doing well
            return random.randint(1, 3)
        elif avg_performance < 0.3:  # Player struggling
            return random.randint(-2, 0)
        else:
            return random.randint(-1, 1)
    
    def add_notification(self, text: str, position: Tuple[int, int], color: str, duration: int):
        """Add a notification for display."""
        self.event_notifications.append({
            "text": text,
            "position": position,
            "color": color,
            "time": self.time,
            "duration": duration
        })
    
    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        """Check if a position is occupied by any entity."""
        for unit in self.player_units + self.enemy_units:
            if unit.position == position:
                return True
        for structure in self.structures:
            if structure.position == position:
                return True
        return False
    
    def update_visibility(self, position: Tuple[int, int], radius: int):
        """Update visibility map around a position."""
        x, y = position
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE:
                        self.visibility[new_x, new_y] = 1.0
                        self.fog_of_war[new_x, new_y] = 0.0
    
    def step(self):
        """Enhanced step function with novel features."""
        self.time += 1
        
        # Update events
        self.update_events()
        
        # Update units
        for unit in self.player_units + self.enemy_units:
            unit.update_state()
        
        # Update structures
        for structure in self.structures:
            structure.update_production(self)
        
        # Update resources (with regeneration)
        for resource in self.resources:
            resource.update()
        
        # Update visibility and fog of war
        self.update_fog_of_war()
        
        # Update enemy AI
        self.update_enemy_ai()
        
        # Update turrets
        self.update_turrets()
        
        # Remove dead entities
        self.remove_dead()
        
        # Process production
        self.process_production()
        
        # Update notifications
        self._update_event_notifications()
        
        # Calculate player performance
        self._update_player_performance()
    
    def update_events(self):
        """Update all active events."""
        for event in self.events[:]:
            if event.check_trigger(self.time):
                # Event just triggered
                pass
            
            if not event.update(self):
                # Event finished, remove it
                pass  # Keep completed events for analysis
    
    def _update_player_performance(self):
        """Novel enhancement: Track player performance for adaptive difficulty."""
        # Simple performance metric based on various factors
        performance_score = 0.0
        
        # Resource efficiency
        if self.crystal_count > 500:
            performance_score += 0.3
        elif self.crystal_count > 200:
            performance_score += 0.1
        
        # Unit survival
        alive_units = len([u for u in self.player_units if u.is_alive()])
        performance_score += min(0.3, alive_units * 0.05)
        
        # Structure health
        nexus = next((s for s in self.structures if s.type == StructureType.NEXUS), None)
        if nexus and nexus.is_alive():
            performance_score += (nexus.health / nexus.max_health) * 0.4
        
        self.player_performance_history.append(performance_score)
    
    def update_fog_of_war(self):
        """Update fog of war decay."""
        self.fog_of_war = np.minimum(1.0, self.fog_of_war + self.fog_decay_rate)
        self.visibility = np.maximum(0.0, self.visibility - self.fog_decay_rate)
    
    def update_enemy_ai(self):
        """Enhanced enemy AI with adaptive behavior."""
        nexus = next((s for s in self.structures if s.type == StructureType.NEXUS), None)
        if not nexus:
            return
        
        for enemy in self.enemy_units:
            if not enemy.is_alive():
                continue
            
            # Enhanced AI behavior
            nearby_players = [u for u in self.player_units 
                            if ((u.position[0] - enemy.position[0])**2 + 
                                (u.position[1] - enemy.position[1])**2) <= 25]
            
            if nearby_players:
                # Attack nearby player units
                target = min(nearby_players, 
                           key=lambda u: ((u.position[0] - enemy.position[0])**2 + 
                                         (u.position[1] - enemy.position[1])**2))
                
                if not enemy.attack(target, self):
                    # Move towards target
                    dx = np.sign(target.position[0] - enemy.position[0])
                    dy = np.sign(target.position[1] - enemy.position[1])
                    
                    if abs(dx) > abs(dy):
                        direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
                    else:
                        direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
                    
                    enemy.move(direction, self)
            else:
                # Move towards nexus
                dx = np.sign(nexus.position[0] - enemy.position[0])
                dy = np.sign(nexus.position[1] - enemy.position[1])
                
                if abs(dx) > abs(dy):
                    direction = ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
                else:
                    direction = ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
                
                enemy.move(direction, self)
    
    def update_turrets(self):
        """Update turret behavior."""
        for structure in self.structures:
            if structure.type == StructureType.TURRET and structure.is_alive():
                # Find enemies in range
                enemies_in_range = []
                for enemy in self.enemy_units:
                    distance = ((enemy.position[0] - structure.position[0])**2 + 
                               (enemy.position[1] - structure.position[1])**2)**0.5
                    if distance <= 8:  # Turret range
                        enemies_in_range.append(enemy)
                
                if enemies_in_range:
                    # Attack closest enemy
                    target = min(enemies_in_range, 
                               key=lambda e: ((e.position[0] - structure.position[0])**2 + 
                                             (e.position[1] - structure.position[1])**2))
                    structure.attack(target, self)
    
    def remove_dead(self):
        """Remove dead units and structures."""
        self.player_units = [u for u in self.player_units if u.is_alive()]
        self.enemy_units = [u for u in self.enemy_units if u.is_alive()]
        self.structures = [s for s in self.structures if s.is_alive()]
        self.resources = [r for r in self.resources if not r.is_depleted()]
    
    def process_production(self):
        """Process unit production for all structures."""
        for structure in self.structures:
            if structure.type == StructureType.NEXUS:
                structure.update_production(self)
    
    def _update_event_notifications(self):
        """Update and clean up event notifications."""
        self.event_notifications = [
            notification for notification in self.event_notifications
            if self.time - notification["time"] < notification["duration"]
        ]
    
    def get_state(self):
        """Get current environment state."""
        return {
            'time': self.time,
            'crystal_count': self.crystal_count,
            'vespene_count': self.vespene_count,
            'player_units': self.player_units,
            'enemy_units': self.enemy_units,
            'structures': self.structures,
            'resources': self.resources,
            'visibility': self.visibility,
            'fog_of_war': self.fog_of_war,
            'events': [event.get_event_info() for event in self.events if event.is_active],
            'notifications': self.event_notifications,
            'adaptive_difficulty': self.adaptive_difficulty,
            'player_performance': np.mean(self.player_performance_history) if self.player_performance_history else 0.5
        }
    
    def is_game_over(self):
        """Check if the game is over."""
        nexus = next((s for s in self.structures if s.type == StructureType.NEXUS), None)
        return nexus is None or not nexus.is_alive()
    
    def get_reward(self):
        """Novel enhancement: Multi-objective reward system."""
        reward = 0.0
        
        # Survival reward
        nexus = next((s for s in self.structures if s.type == StructureType.NEXUS), None)
        if nexus and nexus.is_alive():
            reward += 0.1  # Base survival reward
            
            # Health bonus
            health_ratio = nexus.health / nexus.max_health
            reward += health_ratio * 0.1
        
        # Resource accumulation reward
        reward += min(self.crystal_count / 10000, 0.5)
        reward += min(self.vespene_count / 5000, 0.3)
        
        # Unit efficiency reward
        unit_count = len(self.player_units)
        reward += min(unit_count / 20, 0.3)
        
        # Enemy elimination reward
        if hasattr(self, '_last_enemy_count'):
            enemies_eliminated = self._last_enemy_count - len(self.enemy_units)
            reward += enemies_eliminated * 0.5
        self._last_enemy_count = len(self.enemy_units)
        
        return reward 