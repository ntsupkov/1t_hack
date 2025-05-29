import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class DroneState(Enum):
    IDLE = "idle"
    MOVING_TO_TARGET = "moving_to_target"
    EXTINGUISHING = "extinguishing"
    RETURNING_TO_BASE = "returning_to_base"
    RECHARGING = "recharging"
    CRASHED = "crashed"

@dataclass
class LidarData:
    f: float = 0.0
    fr: float = 0.0
    r: float = 0.0
    br: float = 0.0
    b: float = 0.0
    bl: float = 0.0
    l: float = 0.0
    fl: float = 0.0
    up: float = 0.0
    d: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'f': self.f, 'fr': self.fr, 'r': self.r, 'br': self.br,
            'b': self.b, 'bl': self.bl, 'l': self.l, 'fl': self.fl,
            'up': self.up, 'd': self.d
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'LidarData':
        return cls(**data)

@dataclass
class EngineState:
    fr: float = 0.0
    fl: float = 0.0
    br: float = 0.0
    bl: float = 0.0
    rf: float = 0.0
    rb: float = 0.0
    lf: float = 0.0
    lb: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'fr': self.fr, 'fl': self.fl, 'br': self.br, 'bl': self.bl,
            'rf': self.rf, 'rb': self.rb, 'lf': self.lf, 'lb': self.lb
        }

class Drone:
    def __init__(self, drone_id: int):
        self.id = drone_id
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        
        self.lidar_data = LidarData()
        self.engine_state = EngineState()
        
        self.state = DroneState.IDLE
        self.is_crashed = False
        self.is_recharged = True
        self.has_extinguisher = True
        self.can_drop_extinguisher = False
        self.sees_fire = False
        
        self.target_position: Optional[np.ndarray] = None
        self.current_path: List[np.ndarray] = []
        self.path_index = 0
        
        self.last_update_time = 0.0
        self.fuel_level = 1.0
        
    def update_from_simulator_data(self, data: Dict):
        self.position = np.array(data.get('droneVector', [0, 0, 0]))
        self.velocity = np.array(data.get('linearVelocity', [0, 0, 0]))
        self.angular_velocity = np.array(data.get('angularVelocity', [0, 0, 0]))
        self.rotation = np.array(data.get('droneAxisRotation', [0, 0, 0]))
        
        self.lidar_data = LidarData.from_dict(data.get('lidarInfo', {}))
        self.is_crashed = data.get('isDroneCrushed', False)
        self.sees_fire = data.get('isDroneSeeFire', False)
        self.can_drop_extinguisher = data.get('dropExtinguisher', False)
        self.is_recharged = data.get('isRecharged', True)
        
        if self.is_crashed:
            self.state = DroneState.CRASHED
            
    def set_target(self, target: np.ndarray):
        self.target_position = target.copy()
        self.state = DroneState.MOVING_TO_TARGET
        
    def set_path(self, path: List[np.ndarray]):
        self.current_path = path.copy()
        self.path_index = 0
        if path:
            self.target_position = path[0].copy()
            
    def get_next_waypoint(self) -> Optional[np.ndarray]:
        if self.path_index < len(self.current_path):
            return self.current_path[self.path_index]
        return None
        
    def advance_waypoint(self):
        self.path_index += 1
        if self.path_index < len(self.current_path):
            self.target_position = self.current_path[self.path_index].copy()
        else:
            self.target_position = None
            
    def is_at_target(self, threshold: float = 1.0) -> bool:
        if self.target_position is None:
            return True
        distance = np.linalg.norm(self.position - self.target_position)
        return distance < threshold
        
    def is_path_complete(self) -> bool:
        return self.path_index >= len(self.current_path)
        
    def get_obstacle_distances(self) -> Dict[str, float]:
        return self.lidar_data.to_dict()
        
    def has_obstacle_in_direction(self, direction: str, threshold: float = 2.0) -> bool:
        distances = self.get_obstacle_distances()
        return distances.get(direction, float('inf')) < threshold
        
    def calculate_engine_powers(self, target_velocity: np.ndarray) -> EngineState:
        error = target_velocity - self.velocity
        
        forward_power = np.clip(error[0] * 50, -100, 100)
        right_power = np.clip(error[1] * 50, -100, 100)
        up_power = np.clip(error[2] * 50, -100, 100)
        
        base_power = 50 + up_power
        
        engines = EngineState()
        engines.fr = base_power + forward_power + right_power
        engines.fl = base_power + forward_power - right_power
        engines.br = base_power - forward_power + right_power
        engines.bl = base_power - forward_power - right_power
        engines.rf = base_power + right_power
        engines.rb = base_power + right_power
        engines.lf = base_power - right_power
        engines.lb = base_power - right_power
        
        for attr in ['fr', 'fl', 'br', 'bl', 'rf', 'rb', 'lf', 'lb']:
            setattr(engines, attr, np.clip(getattr(engines, attr), 0, 100))
            
        return engines
        
    def should_return_to_base(self, base_position: np.ndarray) -> bool:
        if not self.is_recharged:
            return True
        distance_to_base = np.linalg.norm(self.position - base_position)
        return self.fuel_level < 0.3 and distance_to_base > 5.0
        
    def can_extinguish_fire(self, fire_position: np.ndarray, max_distance: float = 2.0) -> bool:
        if not self.has_extinguisher or not self.can_drop_extinguisher:
            return False
        distance = np.linalg.norm(self.position - fire_position)
        return distance <= max_distance
        
    def drop_extinguisher(self):
        if self.can_drop_extinguisher and self.has_extinguisher:
            self.has_extinguisher = False
            self.can_drop_extinguisher = True
            return True
        return False
        
    def recharge(self):
        self.is_recharged = True
        self.has_extinguisher = True
        self.fuel_level = 1.0
        self.state = DroneState.IDLE
        
    def to_command_dict(self) -> Dict:
        return {
            "id": self.id,
            "engines": self.engine_state.to_dict(),
            "dropExtinguisher": self.can_drop_extinguisher and self.has_extinguisher
        }