import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json

class Logger:
    def __init__(self, name: str = "DroneSwarm", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"drone_swarm_{int(time.time())}.log"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.metrics = {}
        self.start_time = time.time()
        
    def info(self, message: str):
        self.logger.info(message)
        
    def debug(self, message: str):
        self.logger.debug(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
        
    def critical(self, message: str):
        self.logger.critical(message)
        
    def log_drone_status(self, drone_id: int, status: Dict[str, Any]):
        self.debug(f"Drone {drone_id} status: {json.dumps(status, default=str)}")
        
    def log_mission_start(self, mission_id: str, drone_id: int, mission_type: str):
        self.info(f"Mission {mission_id} started: Drone {drone_id} - {mission_type}")
        
    def log_mission_complete(self, mission_id: str, duration: float):
        self.info(f"Mission {mission_id} completed in {duration:.2f}s")
        
    def log_fire_detected(self, fire_id: str, position: list):
        self.info(f"Fire {fire_id} detected at position {position}")
        
    def log_fire_extinguished(self, fire_id: str, drone_id: int):
        self.info(f"Fire {fire_id} extinguished by Drone {drone_id}")
        
    def log_collision_avoidance(self, drone_id: int, other_drone_id: int):
        self.warning(f"Collision avoidance: Drone {drone_id} avoiding Drone {other_drone_id}")
        
    def log_path_planning(self, drone_id: int, start: list, end: list, path_length: int):
        self.debug(f"Path planned for Drone {drone_id}: {start} -> {end} ({path_length} waypoints)")
        
    def log_recharge_needed(self, drone_id: int, fuel_level: float):
        self.warning(f"Drone {drone_id} needs recharge (fuel: {fuel_level:.2f})")
        
    def log_system_performance(self, metrics: Dict[str, float]):
        self.info(f"System performance: {json.dumps(metrics, indent=2)}")
        
    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'timestamp': time.time(),
            'value': value
        })
        
    def get_metric_average(self, name: str, time_window: Optional[float] = None) -> Optional[float]:
        if name not in self.metrics:
            return None
            
        current_time = time.time()
        values = []
        
        for entry in self.metrics[name]:
            if time_window is None or (current_time - entry['timestamp']) <= time_window:
                values.append(entry['value'])
                
        return sum(values) / len(values) if values else None
        
    def save_metrics(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"metrics_{int(time.time())}.json"
            
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        
        with open(metrics_dir / filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
            
    def get_uptime(self) -> float:
        return time.time() - self.start_time