import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.fig = None
        self.ax = None
        self.figsize = figsize
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        
    def setup_2d_plot(self, xlim: Tuple[float, float] = (-50, 50), 
                      ylim: Tuple[float, float] = (-50, 50)):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
    def setup_3d_plot(self, xlim: Tuple[float, float] = (-50, 50),
                      ylim: Tuple[float, float] = (-50, 50),
                      zlim: Tuple[float, float] = (0, 20)):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        
    def plot_drone_positions(self, drone_positions: Dict[int, np.ndarray], 
                           drone_states: Optional[Dict[int, str]] = None):
        if self.ax is None:
            self.setup_2d_plot()
            
        for drone_id, position in drone_positions.items():
            color = self.colors[drone_id % len(self.colors)]
            marker = 'o'
            
            if drone_states and drone_id in drone_states:
                state = drone_states[drone_id]
                if state == 'crashed':
                    marker = 'x'
                elif state == 'recharging':
                    marker = 's'
                elif state == 'extinguishing':
                    marker = '^'
                    
            self.ax.scatter(position[0], position[1], 
                          c=color, marker=marker, s=100, 
                          label=f'Drone {drone_id}')
                          
    def plot_fire_positions(self, fire_positions: List[np.ndarray],
                          fire_states: Optional[List[str]] = None):
        if self.ax is None:
            self.setup_2d_plot()
            
        for i, position in enumerate(fire_positions):
            color = 'red'
            marker = '*'
            size = 150
            
            if fire_states and i < len(fire_states):
                if fire_states[i] == 'extinguished':
                    color = 'gray'
                    size = 100
                elif fire_states[i] == 'being_extinguished':
                    color = 'orange'
                    
            self.ax.scatter(position[0], position[1], 
                          c=color, marker=marker, s=size, 
                          alpha=0.8)
                          
    def plot_base_position(self, base_position: np.ndarray):
        if self.ax is None:
            self.setup_2d_plot()
            
        self.ax.scatter(base_position[0], base_position[1], 
                       c='black', marker='H', s=200, 
                       label='Base')
                       
    def plot_drone_paths(self, drone_paths: Dict[int, List[np.ndarray]]):
        if self.ax is None:
            self.setup_2d_plot()
            
        for drone_id, path in drone_paths.items():
            if len(path) > 1:
                color = self.colors[drone_id % len(self.colors)]
                x_coords = [pos[0] for pos in path]
                y_coords = [pos[1] for pos in path]
                
                self.ax.plot(x_coords, y_coords, 
                           color=color, alpha=0.6, linewidth=2,
                           linestyle='--', label=f'Path {drone_id}')
                           
    def plot_obstacles(self, obstacle_positions: List[np.ndarray]):
        if self.ax is None:
            self.setup_2d_plot()
            
        for position in obstacle_positions:
            circle = patches.Circle((position[0], position[1]), 
                                  radius=1.0, color='brown', alpha=0.5)
            self.ax.add_patch(circle)
            
    def plot_detection_ranges(self, drone_positions: Dict[int, np.ndarray], 
                            detection_range: float = 8.0):
        if self.ax is None:
            self.setup_2d_plot()
            
        for drone_id, position in drone_positions.items():
            color = self.colors[drone_id % len(self.colors)]
            circle = patches.Circle((position[0], position[1]), 
                                  radius=detection_range, 
                                  color=color, alpha=0.1, 
                                  linestyle=':', linewidth=1)
            self.ax.add_patch(circle)
            
    def create_real_time_plot(self, world_state_callback, update_interval: int = 100):
        self.setup_2d_plot()
        
        def update_frame(frame):
            self.ax.clear()
            self.setup_2d_plot()
            
            world_data = world_state_callback()
            
            if 'drones' in world_data:
                self.plot_drone_positions(world_data['drones'], 
                                        world_data.get('drone_states'))
                                        
            if 'fires' in world_data:
                self.plot_fire_positions(world_data['fires'], 
                                       world_data.get('fire_states'))
                                       
            if 'base' in world_data:
                self.plot_base_position(world_data['base'])
                
            if 'paths' in world_data:
                self.plot_drone_paths(world_data['paths'])
                
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.set_title(f"Drone Swarm Status - Frame {frame}")
            
        animation = FuncAnimation(self.fig, update_frame, 
                                interval=update_interval, blit=False)
        return animation
        
    def plot_performance_metrics(self, metrics: Dict[str, List[float]], 
                               time_stamps: Optional[List[float]] = None):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
            
        for i, (metric_name, values) in enumerate(metrics.items()):
            x_data = time_stamps if time_stamps else range(len(values))
            axes[i].plot(x_data, values, marker='o', linewidth=2)
            axes[i].set_title(f'{metric_name}')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Time' if time_stamps else 'Step')
            
        plt.tight_layout()
        return fig
        
    def plot_mission_timeline(self, missions: List[Dict]):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        drone_ids = list(set(mission['drone_id'] for mission in missions))
        drone_ids.sort()
        
        y_positions = {drone_id: i for i, drone_id in enumerate(drone_ids)}
        
        for mission in missions:
            drone_id = mission['drone_id']
            start_time = mission.get('start_time', 0)
            end_time = mission.get('end_time', start_time + 10)
            duration = end_time - start_time
            
            y_pos = y_positions[drone_id]
            
            color_map = {
                'extinguish_fire': 'red',
                'patrol': 'blue',
                'return_to_base': 'green',
                'recharge': 'orange'
            }
            
            color = color_map.get(mission.get('type', 'unknown'), 'gray')
            
            ax.barh(y_pos, duration, left=start_time, 
                   height=0.8, color=color, alpha=0.7,
                   label=mission.get('type', 'unknown'))
                   
        ax.set_yticks(range(len(drone_ids)))
        ax.set_yticklabels([f'Drone {drone_id}' for drone_id in drone_ids])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Mission Timeline')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        return fig
        
    def save_plot(self, filename: str, dpi: int = 300):
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            
    def show(self):
        if self.fig:
            plt.show()
            
    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None