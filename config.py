"""
Configuration file for Padel Shot Classification System
Easily adjust parameters without modifying core code
"""

# ============================================================================
# VIDEO PROCESSING SETTINGS
# ============================================================================

# Frame skipping (1 = process all, 2 = every 2nd frame, etc.)
SKIP_FRAMES = 1

# Whether to save annotated output video
SAVE_OUTPUT_VIDEO = True

# Output directory
OUTPUT_DIR = "output"

# ============================================================================
# DETECTION SETTINGS
# ============================================================================

# YOLO model to use
# Options: yolov8n.pt (nano, fastest)
#          yolov8s.pt (small, balanced)
#          yolov8m.pt (medium, most accurate)
YOLO_MODEL = "yolov8n.pt"

# Confidence threshold for YOLO (0.0 to 1.0)
# Lower = more detections but more false positives
# Higher = fewer detections but higher quality
DETECTION_CONFIDENCE = 0.5

# ============================================================================
# TRACKING SETTINGS
# ============================================================================

# Maximum distance (pixels) to match detections across frames
TRACKER_MAX_DISTANCE = 50

# Maximum frames an object can disappear before removing track
TRACKER_MAX_DISAPPEARED = 30

# ============================================================================
# SHOT DETECTION SETTINGS
# ============================================================================

# Window size for motion analysis (frames)
SHOT_DETECTOR_WINDOW = 15

# Minimum motion threshold (pixels per frame)
# Higher = need faster ball movement to detect shot
SHOT_DETECTOR_MOTION_THRESHOLD = 5

# Minimum confidence to record a shot
MIN_SHOT_CONFIDENCE = 0.5

# ============================================================================
# CLASSIFICATION SETTINGS
# ============================================================================

# Trajectory history size (frames to keep)
CLASSIFIER_MAX_HISTORY = 30

# Serve detection: minimum vertical component
SERVE_MIN_VERTICAL = 30

# Serve detection: minimum speed
SERVE_MIN_SPEED = 5

# Smash detection: minimum speed
SMASH_MIN_SPEED = 8

# Smash detection: minimum vertical displacement
SMASH_MIN_VERTICAL = 50

# Volley detection: maximum ball speed
VOLLEY_MAX_SPEED = 5

# Forehand/Backhand: minimum horizontal bias
STROKE_MIN_RATIO = 1.5

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Generate JSON output
SAVE_JSON = True
JSON_FILENAME = "shots.json"

# Generate CSV output
SAVE_CSV = True
CSV_FILENAME = "shots.csv"

# Generate statistics
SAVE_STATISTICS = True
STATISTICS_FILENAME = "statistics.json"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Colors for bounding boxes (BGR format)
COLOR_BALL = (0, 255, 255)    # Cyan
COLOR_RACKET = (0, 165, 255)  # Orange
COLOR_PLAYER = (255, 0, 0)    # Red

# Bounding box thickness
BBOX_THICKNESS = 2

# Font size for labels
FONT_SIZE = 0.5

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Use GPU if available (requires CUDA)
USE_GPU = False

# Maximum workers for parallel processing
NUM_WORKERS = 4

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Print progress every N frames
PROGRESS_INTERVAL = 100

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Smooth trajectories using moving average
SMOOTH_TRAJECTORY = True
TRAJECTORY_SMOOTH_WINDOW = 5

# Apply temporal filtering to shots
TEMPORAL_FILTERING = True
MIN_FRAMES_BETWEEN_SHOTS = 10  # Minimum frames between consecutive shots

# ============================================================================
# EXAMPLE USAGE IN CODE
# ============================================================================

"""
from config import (
    SKIP_FRAMES,
    DETECTION_CONFIDENCE,
    OUTPUT_DIR,
    MIN_SHOT_CONFIDENCE
)

# In main code:
detections = detector.detect(frame, conf=DETECTION_CONFIDENCE)

if confidence > MIN_SHOT_CONFIDENCE:
    output_gen = OutputGenerator(OUTPUT_DIR)
    output_gen.add_shot(shot_record)
"""
