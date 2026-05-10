"""
Object detection using YOLO
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

from .utils import BoundingBox

class ObjectDetector:
    """YOLO-based object detector for ball, racket, and players"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model to use (yolov8n, yolov8s, yolov8m, etc.)
        """
        self.model = YOLO(model_name)
        self.device = "cpu"  # Use "cuda" if GPU available
        
        # Get class names for debugging
        self.class_names = self.model.names
        logger.info(f"YOLO model loaded: {model_name}")
        logger.info(f"Available classes: {list(self.class_names.values())[:10]}...")  # Show first 10
    
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> Dict[str, List[BoundingBox]]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame
            conf: Confidence threshold
            
        Returns:
            Dictionary with detected objects by class
        """
        # Use lower confidence for ball detection (they're often small/hard to detect)
        ball_conf = max(0.2, conf - 0.2)  # At least 0.2 for balls
        
        # Run detection with confidence threshold
        results = self.model(frame, conf=min(conf, ball_conf), verbose=False)
        
        detections = {
            "ball": [],
            "racket": [],
            "person": [],
            "other": []
        }
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf_score = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Skip if confidence is too low for non-ball objects
                if class_id != 32 and conf_score < conf:
                    continue
                
                bbox = BoundingBox(
                    float(x1), float(y1), float(x2), float(y2),
                    confidence=conf_score, class_name=class_name
                )
                
                # Categorize detections
                # Sports ball: id=32 in COCO
                if class_id == 32 or "ball" in class_name.lower():
                    detections["ball"].append(bbox)
                    logger.debug(f"Ball detected: {class_name} (id={class_id}, conf={conf_score:.2f})")
                
                # Racket/Paddle detection: Tennis Racket is id=43
                elif class_id == 43 or class_id == 46 or "racket" in class_name.lower() or "bat" in class_name.lower():
                    detections["racket"].append(bbox)
                    logger.debug(f"Racket detected: {class_name} (id={class_id}, conf={conf_score:.2f})")
                
                # Person detection: id=0 in COCO
                elif class_id == 0 or "person" in class_name.lower():
                    detections["person"].append(bbox)
                    logger.debug(f"Person detected: {class_name} (id={class_id}, conf={conf_score:.2f})")
                
                # Everything else
                else:
                    detections["other"].append(bbox)
                    logger.debug(f"Other detected: {class_name} (id={class_id}, conf={conf_score:.2f})")
        
        return detections
    
    def detect_ball_by_color(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect balls using color-based approach (fallback for YOLO)
        Padel balls are typically yellow/green colored
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected ball bounding boxes
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for yellow/green (padel ball colors)
        # Yellow: H: 15-40, Saturation: 100-255, Value: 100-255
        # Green: H: 35-85, Saturation: 100-255, Value: 100-255
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        lower_green = np.array([35, 80, 100])
        upper_green = np.array([85, 255, 255])
        
        # Create masks
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_yellow, mask_green)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (balls should have a minimum size)
            # Balls in a 1920x1080 video are typically 10-100 pixels in area
            if area < 10 or area > 500:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (balls should be roughly circular)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Create bbox object
            bbox = BoundingBox(
                float(x), float(y), float(x + w), float(y + h),
                confidence=0.6,  # Color detection confidence
                class_name="ball_color"
            )
            balls.append(bbox)
        
        if balls and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Color-based ball detection found {len(balls)} potential balls")
        
        return balls
    
    def track(self, frame: np.ndarray, conf: float = 0.5) -> Dict[str, List[Dict]]:
        """
        Track objects across frames
        
        Args:
            frame: Input frame
            conf: Confidence threshold
            
        Returns:
            Dictionary with tracked objects and track IDs
        """
        results = self.model.track(frame, conf=conf, persist=True, verbose=False)
        
        tracks = {
            "ball": [],
            "racket": [],
            "person": [],
        }
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                track_id = int(box.id) if box.id is not None else -1
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                bbox = BoundingBox(
                    float(x1), float(y1), float(x2), float(y2),
                    confidence=conf, class_name=class_name
                )
                
                track_info = {
                    "bbox": bbox,
                    "track_id": track_id,
                    "class": class_name
                }
                
                # Categorize
                if "ball" in class_name.lower() or class_id == 32:
                    tracks["ball"].append(track_info)
                elif "racket" in class_name.lower() or "tennis" in class_name.lower():
                    tracks["racket"].append(track_info)
                elif "person" in class_name.lower() or class_id == 0:
                    tracks["person"].append(track_info)
        
        return tracks


class SimpleTracker:
    """Simple centroid-based tracker for maintaining object identities"""
    
    def __init__(self, max_distance: float = 50, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}  # {track_id: {"center": (x, y), "disappeared": count}}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
    
    def update(self, detections: List[BoundingBox]) -> Dict[int, BoundingBox]:
        """
        Update tracked objects
        
        Args:
            detections: List of detected bounding boxes
            
        Returns:
            Dictionary mapping track IDs to bounding boxes
        """
        if len(detections) == 0:
            # Increment disappeared count for all tracked objects
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["disappeared"] += 1
                if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.objects[obj_id]
            return {}
        
        # Get centers of current detections
        detection_centers = [det.center for det in detections]
        
        # Match detections to existing tracks
        used_detections = set()
        matched_tracks = {}
        
        for obj_id, obj_data in self.objects.items():
            distances = [
                (i, np.sqrt((detection_centers[i][0] - obj_data["center"][0])**2 + 
                           (detection_centers[i][1] - obj_data["center"][1])**2))
                for i in range(len(detection_centers))
            ]
            
            if distances:
                min_idx, min_dist = min(distances, key=lambda x: x[1])
                
                if min_dist < self.max_distance:
                    used_detections.add(min_idx)
                    self.objects[obj_id]["center"] = detection_centers[min_idx]
                    self.objects[obj_id]["disappeared"] = 0
                    matched_tracks[obj_id] = detections[min_idx]
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self.objects[self.next_id] = {
                    "center": detection.center,
                    "disappeared": 0
                }
                matched_tracks[self.next_id] = detection
                self.next_id += 1
        
        return matched_tracks
