import numpy as np
from typing import List, Tuple, Optional
import math

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def lerp_vector(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)

def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2 - p1)

def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2[:2] - p1[:2])

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_angle = np.dot(normalize_vector(v1), normalize_vector(v2))
    cos_angle = clamp(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)

def rotate_vector_2d(vector: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return rotation_matrix @ vector[:2]

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = 0
        
    return x, y, z

def smooth_path(path: List[np.ndarray], smoothing_factor: float = 0.5) -> List[np.ndarray]:
    if len(path) < 3:
        return path
        
    smoothed_path = [path[0]]
    
    for i in range(1, len(path) - 1):
        prev_point = path[i-1]
        current_point = path[i]
        next_point = path[i+1]
        
        smoothed_point = lerp_vector(
            current_point,
            (prev_point + next_point) / 2,
            smoothing_factor
        )
        
        smoothed_path.append(smoothed_point)
    
    smoothed_path.append(path[-1])
    return smoothed_path

def calculate_trajectory(start: np.ndarray, end: np.ndarray, 
                        max_velocity: float, max_acceleration: float,
                        time_step: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    direction = normalize_vector(end - start)
    total_distance = distance_3d(start, end)
    
    acceleration_time = max_velocity / max_acceleration
    acceleration_distance = 0.5 * max_acceleration * acceleration_time**2
    
    if 2 * acceleration_distance >= total_distance:
        acceleration_time = math.sqrt(total_distance / max_acceleration)
        max_velocity_reached = max_acceleration * acceleration_time
        deceleration_time = acceleration_time
    else:
        max_velocity_reached = max_velocity
        constant_velocity_distance = total_distance - 2 * acceleration_distance
        constant_velocity_time = constant_velocity_distance / max_velocity
        deceleration_time = acceleration_time
    
    trajectory = []
    current_time = 0.0
    current_position = start.copy()
    current_velocity = np.zeros(3)
    
    while distance_3d(current_position, end) > 0.1:
        if current_time <= acceleration_time:
            velocity_magnitude = max_acceleration * current_time
        elif current_time <= acceleration_time + constant_velocity_time:
            velocity_magnitude = max_velocity_reached
        else:
            time_in_decel = current_time - acceleration_time - constant_velocity_time
            velocity_magnitude = max_velocity_reached - max_acceleration * time_in_decel
            velocity_magnitude = max(0, velocity_magnitude)
        
        current_velocity = direction * velocity_magnitude
        current_position += current_velocity * time_step
        
        trajectory.append((current_position.copy(), current_velocity.copy()))
        current_time += time_step
        
        if velocity_magnitude == 0:
            break
    
    return trajectory

def calculate_circle_intersection(center1: np.ndarray, radius1: float,
                                center2: np.ndarray, radius2: float) -> List[np.ndarray]:
    d = distance_2d(center1, center2)
    
    if d > radius1 + radius2 or d < abs(radius1 - radius2) or d == 0:
        return []
    
    a = (radius1**2 - radius2**2 + d**2) / (2 * d)
    h = math.sqrt(radius1**2 - a**2)
    
    p = center1[:2] + a * (center2[:2] - center1[:2]) / d
    
    intersection1 = np.array([
        p[0] + h * (center2[1] - center1[1]) / d,
        p[1] - h * (center2[0] - center1[0]) / d,
        0
    ])
    
    intersection2 = np.array([
        p[0] - h * (center2[1] - center1[1]) / d,
        p[1] + h * (center2[0] - center1[0]) / d,
        0
    ])
    
    return [intersection1, intersection2]

def point_in_polygon(point: np.ndarray, polygon: List[np.ndarray]) -> bool:
    x, y = point[:2]
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0][:2]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][:2]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def create_grid_points(bounds: Tuple[float, float, float, float], resolution: float) -> List[np.ndarray]:
    min_x, max_x, min_y, max_y = bounds
    points = []
    
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            points.append(np.array([x, y, 0]))
            y += resolution
        x += resolution
    
    return points

def calculate_centroid(points: List[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros(3)
    return np.mean(points, axis=0)

def find_nearest_point(target: np.ndarray, points: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    if not points:
        return -1, np.zeros(3)
    
    distances = [distance_3d(target, point) for point in points]
    min_index = np.argmin(distances)
    return min_index, points[min_index]

def generate_waypoints_around_point(center: np.ndarray, radius: float, count: int) -> List[np.ndarray]:
    waypoints = []
    angle_step = 2 * math.pi / count
    
    for i in range(count):
        angle = i * angle_step
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        waypoints.append(np.array([x, y, z]))
    
    return waypoints

def calculate_bounding_box(points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros(3), np.zeros(3)
    
    points_array = np.array(points)
    min_bounds = np.min(points_array, axis=0)
    max_bounds = np.max(points_array, axis=0)
    
    return min_bounds, max_bounds

def is_point_in_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    return distance_3d(point, center) <= radius

def calculate_velocity_from_positions(pos1: np.ndarray, pos2: np.ndarray, dt: float) -> np.ndarray:
    if dt == 0:
        return np.zeros(3)
    return (pos2 - pos1) / dt

def predict_future_position(current_pos: np.ndarray, velocity: np.ndarray, time: float) -> np.ndarray:
    return current_pos + velocity * time