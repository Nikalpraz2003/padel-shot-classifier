import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ShotRecord:
    """Represents a single shot record"""
    
    def __init__(self, shot_id: int, frame_number: int, timestamp: float, 
                 shot_type: str, confidence: float, player_id: int = -1):
        self.shot_id = shot_id
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.shot_type = shot_type
        self.confidence = confidence
        self.player_id = player_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "shot_id": self.shot_id,
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "shot_type": self.shot_type,
            "confidence": round(self.confidence, 3),
            "player_id": self.player_id
        }


class OutputGenerator:
    """Generate and save analysis results"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shots = []
    
    def add_shot(self, shot: ShotRecord):
        """Add a detected shot"""
        self.shots.append(shot)
    
    def save_json(self, filename: str = "shots.json"):
        """Save results as JSON"""
        data = {
            "metadata": {
                "total_shots": len(self.shots),
                "video_info": "Padel match analysis"
            },
            "shots": [shot.to_dict() for shot in self.shots]
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_csv(self, filename: str = "shots.csv"):
        """Save results as CSV"""
        output_path = self.output_dir / filename
        
        if not self.shots:
            logger.warning("No shots to save")
            return
        
        fieldnames = ["shot_id", "frame_number", "timestamp", "shot_type", "confidence", "player_id"]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for shot in self.shots:
                writer.writerow(shot.to_dict())
        
        logger.info(f"Results saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Generate statistics about detected shots"""
        if not self.shots:
            return {}
        
        shot_counts = {}
        for shot in self.shots:
            shot_counts[shot.shot_type] = shot_counts.get(shot.shot_type, 0) + 1
        
        avg_confidence = sum(s.confidence for s in self.shots) / len(self.shots)
        
        return {
            "total_shots": len(self.shots),
            "shot_types": shot_counts,
            "average_confidence": round(avg_confidence, 3),
            "duration_seconds": self.shots[-1].timestamp if self.shots else 0
        }
    
    def save_statistics(self, filename: str = "statistics.json"):
        """Save statistics"""
        stats = self.get_statistics()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {output_path}")
    
    def print_summary(self):
        """Print summary to console"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("PADEL SHOT ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total shots detected: {stats.get('total_shots', 0)}")
        print(f"Average confidence: {stats.get('average_confidence', 0):.3f}")
        print("\nShot breakdown:")
        for shot_type, count in stats.get('shot_types', {}).items():
            print(f"  - {shot_type}: {count}")
        print("="*50 + "\n")
