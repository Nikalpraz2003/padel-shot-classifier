import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

from .utils import BoundingBox, calculate_distance

class ShotClassifier:
    """Classify padel shots based on movement patterns"""
    
    # Shot types
    SHOT_TYPES = ["forehand", "backhand", "serve", "smash", "volley", "unknown"]
    
    def __init__(self):
        self.ball_trajectory = []  # History of ball positions
        self.racket_trajectory = []  # History of racket positions
        self.max_history = 30  # Keep last 30 frames
    
    def add_observation(self, ball_bbox: Optional[BoundingBox], 
                       racket_bbox: Optional[BoundingBox]):
        """Add frame observation for trajectory analysis"""
        if ball_bbox:
            self.ball_trajectory.append(ball_bbox.center)
        if racket_bbox:
            self.racket_trajectory.append(racket_bbox.center)
        
        # Keep limited history
        if len(self.ball_trajectory) > self.max_history:
            self.ball_trajectory.pop(0)
        if len(self.racket_trajectory) > self.max_history:
            self.racket_trajectory.pop(0)
    
    def classify_shot(self) -> Tuple[str, float]:
        """
        Classify the shot based on trajectory
        
        Returns:
            Tuple of (shot_type, confidence)
        """
        if len(self.ball_trajectory) < 5 or len(self.racket_trajectory) < 5:
            return "unknown", 0.0
        
        # Calculate trajectories (recent movement)
        recent_frames = min(10, len(self.ball_trajectory))
        ball_movement = self._calculate_movement(self.ball_trajectory[-recent_frames:])
        racket_movement = self._calculate_movement(self.racket_trajectory[-recent_frames:])
        
        # Analyze patterns
        shot_type, confidence = self._analyze_patterns(ball_movement, racket_movement)
        
        return shot_type, confidence
    
    def _calculate_movement(self, trajectory: List[Tuple[float, float]]) -> Dict:
        """Calculate movement characteristics"""
        if len(trajectory) < 2:
            return {"speed": 0, "direction": 0, "vertical": 0, "horizontal": 0}
        
        # Total displacement
        total_dx = trajectory[-1][0] - trajectory[0][0]
        total_dy = trajectory[-1][1] - trajectory[0][1]
        
        # Speed (pixels per frame)
        speed = np.sqrt(total_dx**2 + total_dy**2) / len(trajectory)
        
        # Direction (angle in degrees)
        direction = np.arctan2(total_dy, total_dx) * 180 / np.pi
        
        # Vertical and horizontal components
        vertical = abs(total_dy)
        horizontal = abs(total_dx)
        
        return {
            "speed": speed,
            "direction": direction,
            "vertical": vertical,
            "horizontal": horizontal,
            "total_displacement": np.sqrt(total_dx**2 + total_dy**2)
        }
    
    def _analyze_patterns(self, ball_movement: Dict, racket_movement: Dict) -> Tuple[str, float]:
        """Analyze movement patterns to classify shot"""
        
        ball_speed = ball_movement["speed"]
        ball_vertical = ball_movement["vertical"]
        ball_horizontal = ball_movement["horizontal"]
        
        racket_speed = racket_movement["speed"]
        racket_vertical = racket_movement["vertical"]
        racket_horizontal = racket_movement["horizontal"]
        
        # Serve: High vertical movement, high speed
        if ball_vertical > ball_horizontal and ball_speed > 5:
            return "serve", 0.85
        
        # Smash: High upward then downward motion, very high speed
        if ball_speed > 8 and ball_vertical > 50:
            return "smash", 0.85
        
        # Volley: Low ball height, quick racket movement
        if racket_speed > 3 and ball_speed < 5:
            return "volley", 0.75
        
        # Forehand: Horizontal movement to the right (from player perspective)
        if ball_horizontal > ball_vertical * 1.5 and ball_movement["direction"] > -45:
            return "forehand", 0.80
        
        # Backhand: Horizontal movement to the left
        if ball_horizontal > ball_vertical * 1.5 and ball_movement["direction"] < -135:
            return "backhand", 0.80
        
        # Default
        return "unknown", 0.5
    
    def reset(self):
        """Reset trajectory history"""
        self.ball_trajectory = []
        self.racket_trajectory = []


class ShotDetector:
    """Detect when a shot is being taken"""
    
    def __init__(self, window_size: int = 15, motion_threshold: float = 5):
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.ball_positions = []
        self.racket_positions = []
        self.shot_detected = False
    
    def update(self, ball_bbox: Optional[BoundingBox], 
              racket_bbox: Optional[BoundingBox] = None) -> bool:
        """
        Update detector with new frame
        Can detect shots from ball OR racket motion
        
        Returns:
            True if a shot was detected in this frame
        """
        # Update ball positions if available
        if ball_bbox is not None:
            self.ball_positions.append(ball_bbox.center)
            if len(self.ball_positions) > self.window_size:
                self.ball_positions.pop(0)
        else:
            self.ball_positions = []
        
        # Update racket positions if available
        if racket_bbox is not None:
            self.racket_positions.append(racket_bbox.center)
            if len(self.racket_positions) > self.window_size:
                self.racket_positions.pop(0)
        else:
            self.racket_positions = []
        
        # Try to detect shot from ball motion (preferred)
        if len(self.ball_positions) >= self.window_size:
            if self._detect_ball_acceleration():
                return True
        
        # Fallback: detect from racket motion
        if len(self.racket_positions) >= self.window_size:
            if self._detect_racket_acceleration():
                return True
        
        return False
    
    def _detect_ball_acceleration(self) -> bool:
        """Detect shot from rapid ball acceleration"""
        first_half = self.ball_positions[:self.window_size // 2]
        second_half = self.ball_positions[self.window_size // 2:]
        
        # Calculate average speeds
        first_speed = self._calculate_avg_speed(first_half)
        second_speed = self._calculate_avg_speed(second_half)
        
        # Shot detected if there's significant acceleration
        if second_speed > first_speed * 1.5 and second_speed > self.motion_threshold:
            self.shot_detected = True
            return True
        
        return False
    
    def _detect_racket_acceleration(self) -> bool:
        """Detect shot from rapid racket acceleration"""
        first_half = self.racket_positions[:self.window_size // 2]
        second_half = self.racket_positions[self.window_size // 2:]
        
        # Calculate average speeds
        first_speed = self._calculate_avg_speed(first_half)
        second_speed = self._calculate_avg_speed(second_half)
        
        # Shot detected if racket moves rapidly
        # Lower threshold than ball since racket doesn't move as far
        if second_speed > first_speed * 1.3 and second_speed > (self.motion_threshold * 0.7):
            self.shot_detected = True
            return True
        
        return False
    
    def _calculate_avg_speed(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate average speed in a sequence of positions"""
        if len(positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(positions)):
            dist = calculate_distance(positions[i-1], positions[i])
            total_distance += dist
        
        return total_distance / (len(positions) - 1)
