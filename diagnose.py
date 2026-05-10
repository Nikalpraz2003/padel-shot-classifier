"""
Diagnostic script to check what YOLO is detecting in the video
"""

import cv2
import sys
from pathlib import Path
from collections import defaultdict

# Windows compatibility
if sys.platform == 'win32':
    import os
    os.add_dll_directory(r'C:\Windows\System32')

from src.detector import ObjectDetector
from src.utils import VideoProcessor

def diagnose_detections(video_path: str, num_frames: int = 50):
    """Analyze first N frames to see what's being detected"""
    
    print("\n" + "="*70)
    print("OBJECT DETECTION DIAGNOSTIC")
    print("="*70 + "\n")
    
    # Load video and detector
    video_proc = VideoProcessor(video_path)
    detector = ObjectDetector("yolov8n.pt")
    
    print(f"Video: {video_path}")
    print(f"Resolution: {video_proc.width}x{video_proc.height} @ {video_proc.fps} FPS\n")
    
    # Track what we find
    found_classes = defaultdict(int)
    detection_counts = {"ball": 0, "racket": 0, "person": 0, "other": 0}
    frame_count = 0
    
    print(f"Analyzing first {num_frames} frames...\n")
    
    while frame_count < num_frames:
        ret, frame = video_proc.get_frame()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame, conf=0.3)  # Lower confidence to catch small objects
        
        # Count detections
        for det_type, bboxes in detections.items():
            if bboxes:
                detection_counts[det_type] += len(bboxes)
                if det_type != "other":
                    print(f"Frame {frame_count}: Found {len(bboxes)} {det_type}(s)")
        
        # Track detected class names
        for det_type, bboxes in detections.items():
            for bbox in bboxes:
                found_classes[bbox.class_name] += 1
        
        frame_count += 1
    
    video_proc.close()
    
    # Print results
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Frames analyzed: {frame_count}")
    print(f"\nDetection counts:")
    for det_type, count in detection_counts.items():
        if count > 0:
            avg = count / frame_count
            print(f"  {det_type:10} {count:4} detections ({avg:.1f} per frame)")
    
    if not any(detection_counts.values()):
        print("  WARNING: No objects detected!")
    
    print(f"\nAll detected classes:")
    for class_name, count in sorted(found_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:20} {count:3} times")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if detection_counts["ball"] == 0:
        print("❌ No balls detected - model may not see balls in this video")
    else:
        print(f"✓ Balls detected ({detection_counts['ball']} times)")
    
    if detection_counts["racket"] == 0:
        print("❌ No rackets detected - YOLO COCO doesn't include 'racket' class")
        print("   The detector looks for 'baseball bat' (closest match)")
    else:
        print(f"✓ Rackets detected ({detection_counts['racket']} times)")
    
    if detection_counts["person"] == 0:
        print("❌ No people detected - unusual for a sports video")
    else:
        print(f"✓ People detected ({detection_counts['person']} times)")
    
    print("\nNEXT STEPS:")
    if detection_counts["ball"] == 0 and detection_counts["person"] == 0:
        print("  - Video may not contain padel match content")
        print("  - Try a different video or check if file is corrupted")
    elif detection_counts["racket"] == 0:
        print("  - YOLOv8n doesn't have 'racket' class in COCO dataset")
        print("  - Options:")
        print("    1. Train custom YOLO model on padel data")
        print("    2. Use pose estimation to detect rackets from player poses")
        print("    3. Use ball trajectory alone (no racket detection)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/infernce_sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)
    
    diagnose_detections(video_path)
