"""
YOLO Object Detection Service for Rooftop Analysis
Main service class that orchestrates all detection and analysis modules
"""

from .service import YOLODetectionService, get_yolo_service

__all__ = ['YOLODetectionService', 'get_yolo_service']

