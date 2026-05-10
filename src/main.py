"""
Main processing pipeline for padel shot classification
"""
import cv2
import numpy as np
import logging
from typing import Optional
from pathlib import Path
import imageio

from .detector import ObjectDetector, SimpleTracker
from .classifier import ShotClassifier, ShotDetector
from .output import OutputGenerator, ShotRecord
from .utils import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PadelShotAnalyzer:
    """Main pipeline for analyzing padel videos"""
    
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Initialize components
        self.video_processor = VideoProcessor(video_path)
        self.detector = ObjectDetector(model_name="yolov8n.pt")
        self.ball_tracker = SimpleTracker(max_distance=50)
        self.racket_tracker = SimpleTracker(max_distance=60)
        self.shot_classifier = ShotClassifier()
        self.shot_detector = ShotDetector()
        self.output_gen = OutputGenerator(output_dir)
        
        self.shot_count = 0
    
    def process_video(self, save_video: bool = True, skip_frames: int = 1) -> dict:
        """
        Process video and detect shots
        
        Args:
            save_video: Whether to save output video with visualizations
            skip_frames: Process every nth frame (for speed)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting video processing...")
        
        # Video output setup
        writer = None
        use_imageio = False
        
        if save_video:
            # Try different codecs - ordered by Windows compatibility
            # Using tuple format: (codec_string, extension, description)
            codec_options = [
                ('MJPG', '.avi', 'Motion JPEG'),      # Most compatible
                ('XVID', '.avi', 'MPEG-4'),           # MPEG-4  
                ('DIVX', '.avi', 'DivX'),             # DivX
                ('FFV1', '.avi', 'FFMpeg Lossless'), # FFMpeg
                ('WMV1', '.avi', 'Windows Media 7'),  # WMV
                ('H264', '.mp4', 'H.264'),            # H.264 (if available)
            ]
            
            out_path = None
            for codec_str, ext, description in codec_options:
                try:
                    out_path_codec = Path(self.output_dir) / f"output_with_detections{ext}"
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)
                    
                    writer = cv2.VideoWriter(
                        str(out_path_codec),
                        fourcc,
                        self.video_processor.fps,
                        (self.video_processor.width, self.video_processor.height)
                    )
                    
                    if writer.isOpened():
                        # Test write to verify it works
                        test_frame = np.zeros((self.video_processor.height, self.video_processor.width, 3), dtype=np.uint8)
                        if writer.write(test_frame):
                            logger.info(f"✓ Video writer ready: {description} ({codec_str}) -> {ext}")
                            out_path = out_path_codec
                            # Don't break yet - remove the test frame by re-opening
                            writer.release()
                            writer = cv2.VideoWriter(
                                str(out_path_codec),
                                fourcc,
                                self.video_processor.fps,
                                (self.video_processor.width, self.video_processor.height)
                            )
                            break
                        else:
                            logger.debug(f"✗ Codec {codec_str}: write test failed")
                            writer.release()
                            writer = None
                    else:
                        logger.debug(f"✗ Codec {codec_str}: failed to open")
                except Exception as e:
                    logger.debug(f"✗ Codec {codec_str} error: {e}")
                    if writer:
                        writer.release()
                    writer = None
            
            # If OpenCV failed, fall back to imageio
            if not writer or not writer.isOpened():
                logger.info("OpenCV codecs unavailable, trying imageio for video output...")
                try:
                    out_path = Path(self.output_dir) / "output_with_detections.mp4"
                    writer = imageio.get_writer(
                        str(out_path),
                        fps=self.video_processor.fps,
                        codec='libx264',
                        quality=7,  # Quality 0-10, lower is better quality
                        pixelformat='yuv420p'
                    )
                    use_imageio = True
                    logger.info(f"✓ Using imageio writer (libx264 H.264) -> output_with_detections.mp4")
                except Exception as e:
                    logger.warning(f"⚠ imageio writer also failed: {e}")
                    logger.warning("Video output disabled. Data files (JSON, CSV, statistics) will still be saved.")
                    save_video = False
                    writer = None
                    use_imageio = False
        
        frame_count = 0
        processed_frames = 0
        ball_detections = 0
        racket_detections = 0
        shot_attempts = 0
        
        while True:
            ret, frame = self.video_processor.get_frame()
            if not ret:
                break
            
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            processed_frames += 1
            
            # Ensure frame dimensions match writer expectations
            if save_video and writer:
                if frame.shape[:2] != (self.video_processor.height, self.video_processor.width):
                    frame = cv2.resize(frame, (self.video_processor.width, self.video_processor.height))
            
            # Get timestamp
            timestamp = self.video_processor.get_timestamp(frame_count)
            
            # Detect objects (use lower confidence to catch small balls)
            detections = self.detector.detect(frame, conf=0.3)
            
            # If no balls detected by YOLO, try color-based detection
            if not detections["ball"]:
                color_balls = self.detector.detect_ball_by_color(frame)
                detections["ball"].extend(color_balls)
            
            # Get best detections for each type
            ball_bbox = detections["ball"][0] if detections["ball"] else None
            racket_bbox = detections["racket"][0] if detections["racket"] else None
            
            if ball_bbox:
                ball_detections += 1
            if racket_bbox:
                racket_detections += 1
            
            # Update classifiers
            self.shot_classifier.add_observation(ball_bbox, racket_bbox)
            
            # Check if shot is being made (using ball or racket motion)
            if self.shot_detector.update(ball_bbox, racket_bbox):
                shot_attempts += 1
                shot_type, confidence = self.shot_classifier.classify_shot()
                
                if confidence > 0.5:
                    self.shot_count += 1
                    shot_record = ShotRecord(
                        shot_id=self.shot_count,
                        frame_number=frame_count,
                        timestamp=timestamp,
                        shot_type=shot_type,
                        confidence=confidence,
                        player_id=-1
                    )
                    self.output_gen.add_shot(shot_record)
                    logger.info(f"Shot {self.shot_count}: {shot_type} (confidence: {confidence:.2f}) at {timestamp:.2f}s")
                    
                    # Reset for next shot
                    self.shot_classifier.reset()
            
            # Draw detections on frame
            if save_video and writer:
                frame = self._draw_detections(frame, ball_bbox, racket_bbox, 
                                             detections["person"], timestamp)
                try:
                    if use_imageio:
                        # imageio expects RGB, convert from BGR
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame_rgb)
                    else:
                        # OpenCV expects BGR
                        success = writer.write(frame)
                        if not success:
                            logger.warning(f"Frame write returned False at frame {processed_frames}")
                except Exception as e:
                    logger.error(f"Error writing frame {processed_frames}: {e}")
                    save_video = False
            
            frame_count += 1
            
            if processed_frames % 100 == 0:
                detection_rate = (ball_detections / processed_frames * 100) if processed_frames > 0 else 0
                logger.info(f"Processed {processed_frames} frames | Shots: {self.shot_count} | Detection rate: {detection_rate:.1f}%")
        
        # Cleanup
        self.video_processor.close()
        if save_video and writer:
            if use_imageio:
                writer.close()
                logger.info(f"Video saved to output_with_detections.mp4 ({processed_frames} frames written)")
            else:
                writer.release()
                logger.info(f"Video saved to output_with_detections ({processed_frames} frames written)")
        
        # Log detection statistics
        logger.info("="*60)
        logger.info("DETECTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total frames processed: {processed_frames}")
        logger.info(f"Ball detections: {ball_detections} ({ball_detections/processed_frames*100:.1f}%)" if processed_frames > 0 else "Ball detections: 0")
        logger.info(f"Racket detections: {racket_detections} ({racket_detections/processed_frames*100:.1f}%)" if processed_frames > 0 else "Racket detections: 0")
        logger.info(f"Shot detection attempts: {shot_attempts}")
        logger.info(f"Confirmed shots: {self.shot_count}")
        logger.info("="*60)
        
        # Save results
        self.output_gen.save_json("shots.json")
        self.output_gen.save_csv("shots.csv")
        self.output_gen.save_statistics("statistics.json")
        self.output_gen.print_summary()
        
        return self.output_gen.get_statistics()
    
    def _draw_detections(self, frame: np.ndarray, ball_bbox: Optional[object], 
                        racket_bbox: Optional[object], players: list, 
                        timestamp: float) -> np.ndarray:
        """Draw detections on frame"""
        
        # Draw ball
        if ball_bbox:
            frame = ball_bbox.draw(frame, color=(0, 255, 255), label="Ball")
        
        # Draw racket
        if racket_bbox:
            frame = racket_bbox.draw(frame, color=(0, 165, 255), label="Racket")
        
        # Draw players
        for i, player in enumerate(players[:2]):  # Max 2 players
            frame = player.draw(frame, color=(255, 0, 0), label=f"Player {i+1}")
        
        # Draw timestamp
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw shot count
        cv2.putText(frame, f"Shots: {self.shot_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detections
        """
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return {}
        
        detections = self.detector.detect(frame, conf=0.5)
        
        # Draw detections
        output_frame = frame.copy()
        for class_name in ["ball", "racket", "person"]:
            for bbox in detections[class_name]:
                output_frame = bbox.draw(output_frame, label=class_name)
        
        # Save output
        output_path = Path(self.output_dir) / "detection_output.jpg"
        cv2.imwrite(str(output_path), output_frame)
        logger.info(f"Output saved to {output_path}")
        
        return detections


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.main <video_path> [--skip-frames N] [--save-video]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    skip_frames = 1
    save_video = True
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--skip-frames" and i + 2 < len(sys.argv):
            skip_frames = int(sys.argv[i + 3])
        elif arg == "--no-video":
            save_video = False
    
    analyzer = PadelShotAnalyzer(video_path)
    analyzer.process_video(save_video=save_video, skip_frames=skip_frames)


if __name__ == "__main__":
    main()
