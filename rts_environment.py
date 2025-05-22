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
    def __init__(self, seed=None):
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
        
        # Initialize the map
        self._init_map()
    
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
            'game_over': not any(s.type == StructureType.NEXUS and s.is_alive() for s in self.structures)
        }
    
    def render(self):
        """Render the current state of the environment."""
        plt.figure(figsize=(10, 10))
        plt.xlim(0, MAP_SIZE)
        plt.ylim(0, MAP_SIZE)
        
        # Draw fog of war
        visible_mask = (self.visibility > 0.5)
        fog_img = np.ones((MAP_SIZE, MAP_SIZE, 4))  # RGBA
        fog_img[~visible_mask, 3] = 0.7  # Alpha channel for non-visible cells
        
        plt.imshow(fog_img, extent=(0, MAP_SIZE, 0, MAP_SIZE), origin='lower')
        
        # Draw resources
        for resource in self.resources:
            if self.is_visible(resource.position):
                if resource.type == ResourceType.CRYSTAL:
                    color = 'blue'
                else:  # VESPENE
                    color = 'green'
                plt.scatter(resource.position[0] + 0.5, resource.position[1] + 0.5, 
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
                plt.gca().add_patch(rect)
                
                # Draw health bar
                health_pct = structure.health / (NEXUS_HEALTH if structure.type == StructureType.NEXUS else TURRET_HEALTH)
                plt.plot([structure.position[0], structure.position[0] + size * health_pct], 
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
                
                plt.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                          color=color, s=80, marker=marker)
                
                # Draw resource carried (for harvesters)
                if unit.type == UnitType.HARVESTER and unit.resources > 0:
                    plt.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                              color='yellow', s=20)
                
                # Draw health bar
                max_health = HARVESTER_HEALTH if unit.type == UnitType.HARVESTER else WARRIOR_HEALTH
                health_pct = unit.health / max_health
                plt.plot([unit.position[0], unit.position[0] + health_pct], 
                        [unit.position[1] - 0.2, unit.position[1] - 0.2], 
                        color='red', linewidth=2)
        
        # Draw enemy units (only if visible)
        for unit in self.enemy_units:
            if unit.is_alive() and self.is_visible(unit.position):
                plt.scatter(unit.position[0] + 0.5, unit.position[1] + 0.5, 
                          color='red', s=80, marker='x')
                
                # Draw health bar
                health_pct = unit.health / RAIDER_HEALTH
                plt.plot([unit.position[0], unit.position[0] + health_pct], 
                        [unit.position[1] - 0.2, unit.position[1] - 0.2], 
                        color='red', linewidth=2)
        
        # Draw grid
        for i in range(MAP_SIZE + 1):
            plt.axhline(y=i, color='black', linestyle='-', alpha=0.2)
            plt.axvline(x=i, color='black', linestyle='-', alpha=0.2)
        
        # Display game stats
        stats_text = f"Time: {self.time}\nCrystals: {self.crystal_count}\nVespene: {self.vespene_count}\n"
        stats_text += f"Units: {len(self.player_units)}\nEnemies: {len([e for e in self.enemy_units if self.is_visible(e.position)])}"
        plt.text(2, MAP_SIZE - 6, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("RTS Environment")
        plt.tight_layout()
        plt.gca().set_aspect('equal')
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