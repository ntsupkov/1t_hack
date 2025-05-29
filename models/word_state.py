import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import time

from .drone import Drone
from .fire import Fire

@dataclass
class WorldBounds:
    min_x: float = -50.0
    max_x: float = 50.0
    min_y: float = -50.0
    max_y: float = 50.0
    min_z: float = 0.0
    max_z: float = 20.0
    
    def is_within_bounds(self, position: np.ndarray) -> bool:
        x, y, z = position
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y and
                self.min_z <= z <= self.max_z)

class WorldState:
    def __init__(self):
        self.drones: Dict[int, Drone] = {}
        self.fires: Dict[str, Fire] = {}
        self.base_position = np.array([0.0, 0.0, 0.0])
        self.bounds = WorldBounds()
        
        self.fire_counter = 0
        self.last_update_time = 0.0
        self.simulation_start_time = time.time()
        
        self.obstacle_map: Dict[tuple, bool] = {}
        self.explored_areas: Set[tuple] = set()
        
    def initialize_drones(self, count: int = 5):
        for i in range(count):
            self.drones[i] = Drone(i)
            
    def update_drone_data(self, drones_data: List[Dict]):
        for drone_data in drones_data:
            drone_id = drone_data.get('id')
            if drone_id in self.drones:
                self.drones[drone_id].update_from_simulator_data(drone_data)
                
    def update_fires_data(self, fires_positions: List[List[float]]):
        current_fire_positions = set()
        
        for i, position in enumerate(fires_positions):
            fire_id = f"fire_{i}"
            fire_pos = np.array(position[:3]) if len(position) >= 3 else np.array([position[0], position[1], 0])
            current_fire_positions.add(fire_id)
            
            if fire_id not in self.fires:
                self.fires[fire_id] = Fire(position=fire_pos)
            else:
                self.fires[fire_id].position = fire_pos
                
        fires_to_remove = []
        for fire_id in self.fires:
            if fire_id not in current_fire_positions:
                fires_to_remove.append(fire_id)
                
        for fire_id in fires_to_remove:
            del self.fires[fire_id]
            
    def get_active_fires(self) -> List[Fire]:
        return [fire for fire in self.fires.values() if fire.state.value != "extinguished"]
        
    def get_available_drones(self) -> List[Drone]:
        return [drone for drone in self.drones.values() 
                if not drone.is_crashed and drone.is_recharged]
                
    def get_drones_needing_recharge(self) -> List[Drone]:
        return [drone for drone in self.drones.values() 
                if not drone.is_recharged or drone.fuel_level < 0.3]
                
    def find_nearest_fire(self, position: np.ndarray) -> Optional[Fire]:
        active_fires = self.get_active_fires()
        if not active_fires:
            return None
            
        nearest_fire = None
        min_distance = float('inf')
        
        for fire in active_fires:
            if fire.assigned_drone_id is None:
                distance = np.linalg.norm(fire.position - position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_fire = fire
                    
        return nearest_fire
        
    def find_nearest_drone(self, position: np.ndarray) -> Optional[Drone]:
        available_drones = self.get_available_drones()
        if not available_drones:
            return None
            
        nearest_drone = None
        min_distance = float('inf')
        
        for drone in available_drones:
            distance = np.linalg.norm(drone.position - position)
            if distance < min_distance:
                min_distance = distance
                nearest_drone = drone
                
        return nearest_drone
        
    def assign_fire_to_drone(self, fire_id: str, drone_id: int) -> bool:
        if fire_id in self.fires and drone_id in self.drones:
            self.fires[fire_id].assign_drone(drone_id)
            return True
        return False
        
    def unassign_fire(self, fire_id: str):
        if fire_id in self.fires:
            self.fires[fire_id].unassign_drone()
            
    def extinguish_fire(self, fire_id: str):
        if fire_id in self.fires:
            self.fires[fire_id].extinguish()
            
    def update_obstacle_map(self, drone_id: int):
        if drone_id not in self.drones:
            return
            
        drone = self.drones[drone_id]
        pos = drone.position
        lidar = drone.lidar_data
        
        directions = {
            'f': np.array([1, 0, 0]),
            'fr': np.array([1, 1, 0]),
            'r': np.array([0, 1, 0]),
            'br': np.array([-1, 1, 0]),
            'b': np.array([-1, 0, 0]),
            'bl': np.array([-1, -1, 0]),
            'l': np.array([0, -1, 0]),
            'fl': np.array([1, -1, 0]),
            'up': np.array([0, 0, 1]),
            'd': np.array([0, 0, -1])
        }
        
        for direction, vector in directions.items():
            distance = getattr(lidar, direction)
            if distance < 10.0:
                obstacle_pos = pos + vector * distance
                grid_pos = tuple(np.round(obstacle_pos).astype(int))
                self.obstacle_map[grid_pos] = True
                
    def is_position_safe(self, position: np.ndarray, safety_radius: float = 2.0) -> bool:
        grid_pos = tuple(np.round(position).astype(int))
        
        for dx in range(-int(safety_radius), int(safety_radius) + 1):
            for dy in range(-int(safety_radius), int(safety_radius) + 1):
                for dz in range(-int(safety_radius), int(safety_radius) + 1):
                    check_pos = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                    if self.obstacle_map.get(check_pos, False):
                        return False
        return True
        
    def get_drones_in_range(self, position: np.ndarray, max_range: float) -> List[Drone]:
        nearby_drones = []
        for drone in self.drones.values():
            distance = np.linalg.norm(drone.position - position)
            if distance <= max_range:
                nearby_drones.append(drone)
        return nearby_drones
        
    def check_drone_collisions(self, drone_id: int, min_distance: float = 3.0) -> List[int]:
        if drone_id not in self.drones:
            return []
            
        current_drone = self.drones[drone_id]
        colliding_drones = []
        
        for other_id, other_drone in self.drones.items():
            if other_id != drone_id and not other_drone.is_crashed:
                distance = np.linalg.norm(current_drone.position - other_drone.position)
                if distance < min_distance:
                    colliding_drones.append(other_id)
                    
        return colliding_drones
        
    def get_world_statistics(self) -> Dict:
        active_fires = len(self.get_active_fires())
        available_drones = len(self.get_available_drones())
        crashed_drones = len([d for d in self.drones.values() if d.is_crashed])
        
        return {
            'total_fires': len(self.fires),
            'active_fires': active_fires,
            'extinguished_fires': len(self.fires) - active_fires,
            'total_drones': len(self.drones),
            'available_drones': available_drones,
            'crashed_drones': crashed_drones,
            'simulation_time': time.time() - self.simulation_start_time,
            'explored_area_size': len(self.explored_areas)
        }