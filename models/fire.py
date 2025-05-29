import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import time

class FireState(Enum):
    ACTIVE = "active"
    BEING_EXTINGUISHED = "being_extinguished"
    EXTINGUISHED = "extinguished"

@dataclass
class Fire:
    position: np.ndarray
    intensity: float = 1.0
    discovery_time: float = 0.0
    state: FireState = FireState.ACTIVE
    assigned_drone_id: Optional[int] = None
    extinguish_progress: float = 0.0
    
    def __post_init__(self):
        if self.discovery_time == 0.0:
            self.discovery_time = time.time()
            
    @property
    def age(self) -> float:
        return time.time() - self.discovery_time
        
    @property
    def priority_score(self) -> float:
        age_factor = min(self.age / 60.0, 2.0)
        intensity_factor = self.intensity
        return age_factor * 0.3 + intensity_factor * 0.7
        
    def assign_drone(self, drone_id: int):
        self.assigned_drone_id = drone_id
        self.state = FireState.BEING_EXTINGUISHED
        
    def unassign_drone(self):
        self.assigned_drone_id = None
        if self.state == FireState.BEING_EXTINGUISHED:
            self.state = FireState.ACTIVE
            
    def extinguish(self):
        self.state = FireState.EXTINGUISHED
        self.extinguish_progress = 1.0
        
    def is_within_range(self, position: np.ndarray, max_range: float) -> bool:
        distance = np.linalg.norm(self.position - position)
        return distance <= max_range
        
    def distance_to(self, position: np.ndarray) -> float:
        return np.linalg.norm(self.position - position)
        
    @classmethod
    def from_position_list(cls, positions: list) -> list['Fire']:
        fires = []
        for i, pos in enumerate(positions):
            fire_pos = np.array(pos[:3]) if len(pos) >= 3 else np.array([pos[0], pos[1], 0])
            fires.append(cls(position=fire_pos, intensity=1.0))
        return fires