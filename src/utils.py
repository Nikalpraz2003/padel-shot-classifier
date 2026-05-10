import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging
import imageio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video reading and frame extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.use_imageio = False
        self.imageio_reader = None
        self.frame_buffer = []
        self.current_frame_idx = 0
        
        # Try OpenCV first
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Fallback to imageio if OpenCV fails
        if self.frame_count == 0:
            logger.info("OpenCV failed to read video, falling back to imageio...")
            try:
                self.imageio_reader = imageio.get_reader(video_path)
                self.fps = self.imageio_reader.get_meta_data().get('fps', 30)
                
                # Try to get frame count, but imageio may return incorrect values
                try:
                    frame_count = len(self.imageio_reader)
                    # Check if it's a reasonable number (not max int)
                    if frame_count > 1000000:
                        self.frame_count = 999999  # Use a large arbitrary number
                        logger.info("Frame count unknown, will process until end")
                    else:
                        self.frame_count = frame_count
                except:
                    self.frame_count = 999999
                
                frame = self.imageio_reader.get_data(0)
                self.height, self.width = frame.shape[:2]
                self.use_imageio = True
                logger.info(f"Using imageio @ {self.fps} FPS ({self.width}x{self.height})")
            except Exception as e:
                logger.error(f"Both OpenCV and imageio failed: {e}")
        
        logger.info(f"Video loaded: {self.frame_count} frames @ {self.fps} FPS")
    
    def get_frame(self, frame_number: int = None) -> Tuple[bool, np.ndarray]:
        """Get specific frame or next frame"""
        if self.use_imageio:
            try:
                if frame_number is not None:
                    self.current_frame_idx = frame_number
                
                frame = self.imageio_reader.get_data(self.current_frame_idx)
                self.current_frame_idx += 1
                
                # Convert RGB to BGR for consistency with OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return True, frame
            except (IndexError, ValueError, EOFError, RuntimeError):
                # End of video reached or invalid frame index
                return False, None
            except Exception as e:
                logger.warning(f"Error reading frame {self.current_frame_idx}: {e}")
                return False, None
        else:
            if frame_number is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            return ret, frame
    
    def get_timestamp(self, frame_number: int) -> float:
        """Convert frame number to timestamp in seconds"""
        return frame_number / self.fps if self.fps > 0 else 0
    
    def close(self):
        """Release video capture"""
        if self.use_imageio and self.imageio_reader:
            self.imageio_reader.close()
        else:
            self.cap.release()


class BoundingBox:
    """Bounding box representation and utility"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 confidence: float = 0.0, class_name: str = "unknown"):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_name = class_name
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center of bounding box"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box"""
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union_area = self.area + other.area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
             thickness: int = 2, label: str = None):
        """Draw bounding box on frame"""
        cv2.rectangle(frame, (int(self.x1), int(self.y1)), 
                     (int(self.x2), int(self.y2)), color, thickness)
        
        if label:
            text = f"{label} ({self.confidence:.2f})"
            cv2.putText(frame, text, (int(self.x1), int(self.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_dominant_color(frame: np.ndarray, bbox: BoundingBox) -> Tuple[int, int, int]:
    """Get dominant color in a bounding box region"""
    region = frame[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
    if region.size == 0:
        return (0, 0, 0)
    
    pixels = region.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]
    return tuple(dominant)


def smooth_trajectory(points: List[Tuple[float, float]], 
                     window_size: int = 5) -> List[Tuple[float, float]]:
    """Smooth trajectory using moving average"""
    if len(points) < window_size:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        window = points[start:end]
        
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        smoothed.append((avg_x, avg_y))
    
    return smoothed


def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[float, float]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Draw trajectory line on frame"""
    for i in range(1, len(trajectory)):
        p1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
        p2 = (int(trajectory[i][0]), int(trajectory[i][1]))
        cv2.line(frame, p1, p2, color, thickness)
    
    return frame
