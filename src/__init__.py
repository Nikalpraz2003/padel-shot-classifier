"""
Padel Shot Classification System
"""

__version__ = "1.0.0"
__author__ = "AI/ML Intern"

from .main import PadelShotAnalyzer
from .detector import ObjectDetector
from .classifier import ShotClassifier
from .output import OutputGenerator

__all__ = [
    "PadelShotAnalyzer",
    "ObjectDetector",
    "ShotClassifier",
    "OutputGenerator"
]
