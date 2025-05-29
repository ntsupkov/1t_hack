import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import time

class MissionType(Enum):
    EXTINGUISH_FIRE = "extinguish_fire"
    PATROL = "patrol"
    RETURN_TO_BASE = "return_to_base"
    RECHARGE = "recharge"
    SCOUT = "scout"

class MissionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Mission:
    mission_id: str
    mission_type: MissionType
    drone_id: int
    target_position: np.ndarray
    priority: float = 1.0
    status: MissionStatus = MissionStatus.PENDING
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    waypoints: List[np.ndarray] = field(default_factory=list)
    fire_id: Optional[str] = None
    estimated_duration: float = 0.0
    fuel_required: float = 0.0
    
    def start(self):
        self.status = MissionStatus.IN_PROGRESS
        self.start_time = time.time()
        
    def complete(self):
        self.status = MissionStatus.COMPLETED
        self.completion_time = time.time()
        
    def fail(self):
        self.status = MissionStatus.FAILED
        self.completion_time = time.time()
        
    def cancel(self):
        self.status = MissionStatus.CANCELLED
        self.completion_time = time.time()
        
    @property
    def duration(self) -> Optional[float]:
        if self.start_time is None:
            return None
        end_time = self.completion_time or time.time()
        return end_time - self.start_time
        
    @property
    def is_active(self) -> bool:
        return self.status in [MissionStatus.PENDING, MissionStatus.IN_PROGRESS]
        
    @property
    def is_completed(self) -> bool:
        return self.status == MissionStatus.COMPLETED
        
    def estimate_completion_time(self, current_position: np.ndarray, speed: float = 5.0) -> float:
        if not self.waypoints:
            distance = np.linalg.norm(self.target_position - current_position)
            return distance / speed
            
        total_distance = 0.0
        prev_pos = current_position
        
        for waypoint in self.waypoints:
            total_distance += np.linalg.norm(waypoint - prev_pos)
            prev_pos = waypoint
            
        total_distance += np.linalg.norm(self.target_position - prev_pos)
        return total_distance / speed

class MissionPlanner:
    def __init__(self):
        self.missions: Dict[str, Mission] = {}
        self.mission_counter = 0
        
    def create_mission(self, mission_type: MissionType, drone_id: int, 
                      target_position: np.ndarray, **kwargs) -> Mission:
        self.mission_counter += 1
        mission_id = f"mission_{self.mission_counter}"
        
        mission = Mission(
            mission_id=mission_id,
            mission_type=mission_type,
            drone_id=drone_id,
            target_position=target_position,
            **kwargs
        )
        
        self.missions[mission_id] = mission
        return mission
        
    def get_active_missions(self) -> List[Mission]:
        return [m for m in self.missions.values() if m.is_active]
        
    def get_drone_missions(self, drone_id: int) -> List[Mission]:
        return [m for m in self.missions.values() if m.drone_id == drone_id and m.is_active]
        
    def cancel_drone_missions(self, drone_id: int):
        for mission in self.get_drone_missions(drone_id):
            mission.cancel()
            
    def get_mission_by_id(self, mission_id: str) -> Optional[Mission]:
        return self.missions.get(mission_id)
        
    def complete_mission(self, mission_id: str):
        mission = self.get_mission_by_id(mission_id)
        if mission:
            mission.complete()
            
    def prioritize_missions(self, missions: List[Mission]) -> List[Mission]:
        return sorted(missions, key=lambda m: (m.priority, m.created_time), reverse=True)