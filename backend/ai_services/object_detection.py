import torch
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class ObjectDetectionService:
    """AI service for detecting rooftop obstacles using YOLOv11 (YOLOv8 removed)"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Priority order: l (large) → m (medium) → s (small) → n (nano)
        try:
            # Try yolo11l-seg (large) - most accurate
            self.model = YOLO("yolo11l-seg.pt")
            logger.info("✅ Loaded YOLOv11l-seg model (large, highest accuracy)")
        except Exception as e:
            logger.warning(f"Could not load yolo11l-seg, trying yolo11m-seg: {e}")
            try:
                # Try yolo11m-seg (medium) - recommended
                self.model = YOLO("yolo11m-seg.pt")
                logger.info("✅ Loaded YOLOv11m-seg model (medium, high accuracy, recommended)")
            except Exception as e2:
                logger.warning(f"Could not load yolo11m-seg, trying yolo11s-seg: {e2}")
                try:
                    # Try yolo11s-seg (small) - balanced
                    self.model = YOLO("yolo11s-seg.pt")
                    logger.info("✅ Loaded YOLOv11s-seg model (small, balanced)")
                except Exception as e3:
                    logger.warning(f"Could not load yolo11s-seg, trying yolo11n-seg: {e3}")
                    # Try yolo11n-seg (nano) - fastest
                    self.model = YOLO("yolo11n-seg.pt")
                    logger.info("✅ Loaded YOLOv11n-seg model (nano, fastest)")

        # Define rooftop obstacle classes to detect
        self.rooftop_obstacles = {
            "chimney": ["chimney"],
            "vent": ["vent", "ventilation", "exhaust"],
            "skylight": ["skylight", "roof window"],
            "hvac": ["hvac", "air conditioner", "ac unit", "heat pump"],
            "satellite": ["satellite dish", "dish"],
            "antenna": ["antenna", "tv antenna"],
            "solar_panel": ["solar panel", "panel"],  # Existing panels
            "pipe": ["pipe", "duct"],
            "tank": ["water tank", "tank"],
            "person": ["person"],
            "bird": ["bird"],
            "drone": ["drone"],
            "vehicle": ["car", "truck", "vehicle"],
            "tree": ["tree"],
            "building": ["building", "structure"],
            "power_line": ["power line", "wire", "cable"]
        }

        # Confidence threshold for detections (lowered for better recall)
        self.confidence_threshold = 0.25
        
        # IOU threshold for NMS (Non-Maximum Suppression)
        self.iou_threshold = 0.45

    def detect_obstacles(self, image_path: str) -> Dict:
        """Detect obstacles on the rooftop"""
        try:
            # Run YOLO inference with improved parameters
            results = self.model(
                image_path, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=640,  # Standard input size for better accuracy
                agnostic_nms=False,  # Class-aware NMS
                max_det=300  # Maximum detections per image
            )

            detected_objects = []
            obstacle_mask = None

            if results and len(results) > 0:
                result = results[0]

                # Get bounding boxes, masks, and class predictions
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    # Get class names
                    class_names = [result.names[int(cls_id)] for cls_id in class_ids]

                    # Process each detection
                    for i, (box, conf, class_name) in enumerate(zip(boxes, confidences, class_names)):
                        # Check if this is a rooftop obstacle
                        obstacle_type = self._classify_obstacle(class_name.lower())

                        if obstacle_type:
                            obj = {
                                "id": i,
                                "type": obstacle_type,
                                "confidence": round(float(conf), 3),
                                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                "area": int((box[2] - box[0]) * (box[3] - box[1])),
                                "yolo_class": class_name
                            }
                            detected_objects.append(obj)

                # Create obstacle mask if segmentation masks are available
                if hasattr(result, 'masks') and result.masks is not None:
                    obstacle_mask = self._create_obstacle_mask(result.masks, result.orig_shape)

            # Create visualization
            visualization = self._create_detection_visualization(image_path, detected_objects)

            # Calculate obstacle statistics
            stats = self._calculate_obstacle_stats(detected_objects)

            return {
                "detected_objects": detected_objects,
                "total_obstacles": len(detected_objects),
                "obstacle_types": list(set(obj["type"] for obj in detected_objects)),
                "detection_visualization_base64": visualization,
                "obstacle_mask_base64": obstacle_mask,
                "statistics": stats,
                "model_used": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv11-seg",
                "confidence_threshold": self.confidence_threshold,
                "processing_time_seconds": 0.0  # Will be calculated by caller
            }

        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return {
                "detected_objects": [],
                "total_obstacles": 0,
                "obstacle_types": [],
                "detection_visualization_base64": "",
                "obstacle_mask_base64": "",
                "statistics": {},
                "error": str(e),
                "model_used": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv11-seg",
                "confidence_threshold": self.confidence_threshold,
                "processing_time_seconds": 0.0
            }

    def _classify_obstacle(self, class_name: str) -> str:
        """Classify YOLO detection into rooftop obstacle categories"""
        class_name_lower = class_name.lower()

        for obstacle_type, keywords in self.rooftop_obstacles.items():
            if any(keyword in class_name_lower for keyword in keywords):
                return obstacle_type

        # Additional classification logic
        if "window" in class_name_lower or "door" in class_name_lower:
            return "skylight"  # Roof windows/doors
        elif "pole" in class_name_lower or "post" in class_name_lower:
            return "antenna"  # Communication poles
        elif "roof" in class_name_lower:
            return "structure"  # Roof structures

        return None  # Not a rooftop obstacle

    def _create_obstacle_mask(self, masks, original_shape: Tuple[int, int]) -> str:
        """Create a binary mask of all detected obstacles"""
        try:
            if masks is None:
                return ""

            # Get masks as numpy array
            mask_array = masks.data.cpu().numpy()

            # Combine all masks
            combined_mask = np.zeros(original_shape[:2], dtype=np.uint8)
            for mask in mask_array:
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
                combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))

            # Convert to base64
            _, buffer = cv2.imencode('.png', combined_mask * 255)
            mask_base64 = base64.b64encode(buffer).decode()

            return f"data:image/png;base64,{mask_base64}"

        except Exception as e:
            logger.warning(f"Error creating obstacle mask: {e}")
            return ""

    def _create_detection_visualization(self, image_path: str, detected_objects: List[Dict]) -> str:
        """Create visualization image with bounding boxes and labels"""
        try:
            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Colors for different obstacle types
            colors = {
                "chimney": (255, 0, 0),      # Red
                "vent": (0, 255, 0),         # Green
                "skylight": (0, 0, 255),     # Blue
                "hvac": (255, 255, 0),       # Yellow
                "satellite": (255, 0, 255),  # Magenta
                "antenna": (0, 255, 255),    # Cyan
                "tree": (128, 0, 128),       # Purple
                "person": (255, 165, 0),     # Orange
                "vehicle": (0, 128, 128),    # Teal
                "building": (128, 128, 0),   # Olive
                "power_line": (128, 0, 0),   # Maroon
                "other": (128, 128, 128)     # Gray
            }

            # Draw bounding boxes and labels
            for obj in detected_objects:
                bbox = obj["bbox"]
                x1, y1, x2, y2 = bbox

                # Get color for obstacle type
                color = colors.get(obj["type"], colors["other"])

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{obj['type']}: {obj['confidence']:.2f}"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2, cv2.LINE_AA)

            # Convert to base64
            pil_image = Image.fromarray(image)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error creating detection visualization: {e}")
            return ""

    def _calculate_obstacle_stats(self, detected_objects: List[Dict]) -> Dict:
        """Calculate statistics about detected obstacles"""
        if not detected_objects:
            return {
                "total_area_covered": 0,
                "obstacles_by_type": {},
                "high_confidence_obstacles": 0,
                "average_confidence": 0.0
            }

        # Group by type
        by_type = {}
        total_area = 0
        high_conf = 0
        confidences = []

        for obj in detected_objects:
            obj_type = obj["type"]
            if obj_type not in by_type:
                by_type[obj_type] = []
            by_type[obj_type].append(obj)

            total_area += obj["area"]
            confidences.append(obj["confidence"])
            if obj["confidence"] > 0.7:
                high_conf += 1

        return {
            "total_area_covered_pixels": total_area,
            "obstacles_by_type": {k: len(v) for k, v in by_type.items()},
            "high_confidence_obstacles": high_conf,
            "average_confidence": round(np.mean(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "min_confidence": round(min(confidences), 3)
        }
