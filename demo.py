
import cv2
import os
import sys
from pathlib import Path

if not hasattr(cv2, 'setNumThreads'):
    def dummy_set_threads(n):
        pass
    cv2.setNumThreads = dummy_set_threads

if sys.platform == 'win32':
    os.add_dll_directory(r'C:\Windows\System32')

from src.main import PadelShotAnalyzer

def demo_video_analysis(video_path: str):
    """
    Demo: Analyze a padel video
    
    Args:
        video_path: Path to padel match video
    """
    print("\n" + "="*60)
    print("PADEL SHOT CLASSIFICATION SYSTEM - DEMO")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = PadelShotAnalyzer(video_path, output_dir="output")
    
    # Process video
    print(f"Starting analysis for: {video_path}")
    print("This may take a few minutes depending on video length\n")
    
    # skip_frames=5 speeds up processing by 5x while maintaining accuracy
    stats = analyzer.process_video(save_video=True, skip_frames=5)
    
    print("\nAnalysis complete!")
    print(f"Output files saved to: {analyzer.output_dir}/")
    print("\nGenerated files:")
    print("  - shots.json (detailed shot information)")
    print("  - shots.csv (shot data in CSV format)")
    print("  - statistics.json (aggregate statistics)")
    print("  - output_with_detections.mp4 (video with annotations)")


def demo_image_analysis(image_path: str):
    """
    Demo: Analyze a single image
    """
    print("\n" + "="*60)
    print("PADEL SHOT CLASSIFICATION SYSTEM - IMAGE DEMO")
    print("="*60 + "\n")
    
    analyzer = PadelShotAnalyzer("dummy.mp4", output_dir="output")
    
    print(f"Analyzing image: {image_path}")
    detections = analyzer.process_image(image_path)
    
    print("\nDetections found:")
    print(f"  - Balls: {len(detections['ball'])}")
    print(f"  - Rackets: {len(detections['racket'])}")
    print(f"  - Players: {len(detections['person'])}")
    print(f"\nOutput saved to: output/detection_output.jpg")


if __name__ == "__main__":
    
    sample_video = "data/infernce_sample_video.mp4"
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else sample_video

    if Path(video_path).exists():
        demo_video_analysis(str(video_path))
    else:
        print(f"\nError: File not found at '{video_path}'")
        print("\n" + "="*60)
        print("USAGE EXAMPLES")
        print("="*60)
        print(f"1. Default: python demo.py")
        print(f"2. Custom:  python demo.py path/to/infernce_sample_video.mp4")
        print("\n" + "="*60 + "\n")
