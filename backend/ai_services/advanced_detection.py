"""
NextGen Advanced Object Detection Service
Uses test-time augmentation, multi-scale detection, and confidence fusion
"""
import torch
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdvancedDetectionService:
    """NextGen object detection with TTA and multi-scale analysis"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Advanced Detection: Using device: {self.device}")

        # Load multiple YOLO models for ensemble
        self.models = {}
        
        # Load YOLOv11 models only (no YOLOv8)
        # YOLOv11 uses 'yolo11' naming (not 'yolov11')
        # Priority order: l (large) → m (medium) → s (small) → n (nano)
        # Try larger models first for better accuracy in ensemble
        model_priority = [
            "yolo11l-seg.pt",  # Large (most accurate, ~88MB)
            "yolo11m-seg.pt",  # Medium (recommended, ~52MB)
            "yolo11s-seg.pt",  # Small (balanced, ~22MB)
            "yolo11n-seg.pt"   # Nano (fastest, ~6MB)
        ]
        
        for model_file in model_priority:
            try:
                self.models[model_file] = YOLO(model_file)
                logger.info(f"✅ Loaded {model_file} for ensemble")
                if len(self.models) >= 2:  # Load at least 2 models
                    break
            except Exception as e:
                logger.warning(f"Could not load {model_file}: {e}")
                continue

        # If no models loaded, try default YOLOv11
        if not self.models:
            try:
                self.models["yolo11n-seg.pt"] = YOLO("yolo11n-seg.pt")
                logger.info("✅ Loaded default YOLOv11n-seg")
            except Exception as e:
                logger.error(f"Failed to load YOLOv11 model: {e}")
                raise RuntimeError("Could not load any YOLOv11 model. Please ensure Ultralytics is up to date.")

        # Test-time augmentation settings
        self.tta_enabled = True
        self.tta_flips = [False, True]  # Original and horizontal flip
        self.tta_scales = [0.9, 1.0, 1.1]  # Different scales

        # Confidence thresholds
        self.base_confidence = 0.2  # Lower for better recall
        self.fusion_confidence = 0.3  # After ensemble fusion

        # Rooftop obstacle classes
        self.rooftop_obstacles = {
            "chimney": ["chimney", "smokestack"],
            "vent": ["vent", "ventilation", "exhaust", "flue"],
            "skylight": ["skylight", "roof window", "dome"],
            "hvac": ["hvac", "air conditioner", "ac unit", "heat pump", "condenser"],
            "satellite": ["satellite dish", "dish", "antenna dish"],
            "antenna": ["antenna", "tv antenna", "radio antenna"],
            "solar_panel": ["solar panel", "panel", "photovoltaic"],
            "pipe": ["pipe", "duct", "conduit"],
            "tank": ["water tank", "tank", "storage tank"],
            "tree": ["tree", "plant", "vegetation"],
            "person": ["person", "human"],
            "vehicle": ["car", "truck", "vehicle", "automobile"],
            "building": ["building", "structure", "shed"],
            "power_line": ["power line", "wire", "cable", "electrical"],
            "fence": ["fence", "railing", "barrier"]
        }

    def detect_obstacles_advanced(self, image_path: str) -> Dict:
        """Advanced detection with TTA and ensemble"""
        try:
            all_detections = []
            detection_weights = []

            # Test-time augmentation loop
            for flip in self.tta_flips:
                for scale in self.tta_scales:
                    # Load and transform image
                    image = Image.open(image_path).convert("RGB")
                    
                    if flip:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    if scale != 1.0:
                        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)

                    # Save temp image for YOLO
                    temp_path = image_path.replace('.jpg', f'_temp_{scale}_{flip}.jpg')
                    image.save(temp_path)

                    # Run detection with each model
                    for model_name, model in self.models.items():
                        try:
                            results = model(
                                temp_path,
                                conf=self.base_confidence,
                                iou=0.45,
                                imgsz=640,
                                agnostic_nms=False,
                                max_det=500,
                                verbose=False
                            )

                            if results and len(results) > 0:
                                result = results[0]
                                
                                # Transform detections back to original coordinates
                                detections = self._transform_detections(
                                    result, image_path, flip, scale
                                )
                                
                                all_detections.extend(detections)
                                # Weight by model size (larger = more weight)
                                weight = 1.0 if 'l' in model_name else (0.8 if 'm' in model_name else 0.6)
                                detection_weights.extend([weight] * len(detections))

                        except Exception as e:
                            logger.warning(f"Error in {model_name} detection: {e}")

                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Step 2: Fusion and NMS
            fused_detections = self._fuse_detections(all_detections, detection_weights, image_path)

            # Step 3: Classify and filter obstacles
            obstacle_detections = self._classify_obstacles(fused_detections)

            # Step 4: Create visualization
            visualization = self._create_advanced_visualization(image_path, obstacle_detections)

            # Step 5: Calculate statistics
            stats = self._calculate_advanced_stats(obstacle_detections)

            return {
                "detected_objects": obstacle_detections,
                "total_obstacles": len(obstacle_detections),
                "obstacle_types": list(set(obj["type"] for obj in obstacle_detections)),
                "detection_visualization_base64": visualization,
                "statistics": stats,
                "model_used": f"Ensemble-{len(self.models)}models-TTA-YOLOv11",
                "confidence_threshold": self.fusion_confidence,
                "processing_time_seconds": 0.0,
                "advanced_features": {
                    "tta_enabled": self.tta_enabled,
                    "models_used": list(self.models.keys()),
                    "detection_count_before_fusion": len(all_detections),
                    "detection_count_after_fusion": len(obstacle_detections)
                }
            }

        except Exception as e:
            logger.error(f"Error in advanced detection: {str(e)}")
            return self._create_error_response(str(e))

    def _transform_detections(self, result, original_path: str, flipped: bool, scale: float) -> List[Dict]:
        """Transform detections back to original image coordinates"""
        try:
            original_image = Image.open(original_path)
            orig_width, orig_height = original_image.size

            detections = []
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                class_names = [result.names[int(cls_id)] for cls_id in class_ids]

                for box, conf, class_name in zip(boxes, confidences, class_names):
                    x1, y1, x2, y2 = box

                    # Scale back
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale

                    # Flip back if needed
                    if flipped:
                        x1_new = orig_width - x2
                        x2_new = orig_width - x1
                        x1, x2 = x1_new, x2_new

                    # Clamp to image bounds
                    x1 = max(0, min(orig_width, int(x1)))
                    y1 = max(0, min(orig_height, int(y1)))
                    x2 = max(0, min(orig_width, int(x2)))
                    y2 = max(0, min(orig_height, int(y2)))

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class": class_name,
                        "class_id": int(class_ids[0])
                    })

            return detections

        except Exception as e:
            logger.warning(f"Error transforming detections: {e}")
            return []

    def _fuse_detections(self, detections: List[Dict], weights: List[float], image_path: str) -> List[Dict]:
        """Fuse multiple detections using weighted NMS"""
        try:
            if not detections:
                return []

            # Group by class and location
            detection_groups = defaultdict(list)

            for det, weight in zip(detections, weights):
                # Create a key based on class and approximate location
                x_center = (det["bbox"][0] + det["bbox"][2]) / 2
                y_center = (det["bbox"][1] + det["bbox"][3]) / 2
                
                # Group detections within 50 pixels
                key_x = int(x_center / 50)
                key_y = int(y_center / 50)
                key = (det["class"], key_x, key_y)
                
                detection_groups[key].append((det, weight))

            # Fuse each group
            fused = []
            for group_dets in detection_groups.values():
                if not group_dets:
                    continue

                # Weighted average of bounding boxes
                total_weight = sum(w for _, w in group_dets)
                if total_weight == 0:
                    continue

                # Weighted confidence
                weighted_conf = sum(d["confidence"] * w for d, w in group_dets) / total_weight

                # Weighted bbox
                weighted_bbox = [
                    sum(d["bbox"][0] * w for d, w in group_dets) / total_weight,
                    sum(d["bbox"][1] * w for d, w in group_dets) / total_weight,
                    sum(d["bbox"][2] * w for d, w in group_dets) / total_weight,
                    sum(d["bbox"][3] * w for d, w in group_dets) / total_weight
                ]

                # Only keep if confidence is high enough
                if weighted_conf >= self.fusion_confidence:
                    fused.append({
                        "bbox": [int(x) for x in weighted_bbox],
                        "confidence": weighted_conf,
                        "class": group_dets[0][0]["class"],
                        "class_id": group_dets[0][0].get("class_id", 0)
                    })

            # Final NMS to remove overlapping detections
            return self._apply_nms(fused, iou_threshold=0.5)

        except Exception as e:
            logger.warning(f"Error fusing detections: {e}")
            return detections[:100]  # Return first 100 if fusion fails

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            # Take highest confidence
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping
            detections = [
                d for d in detections
                if self._calculate_iou(current["bbox"], d["bbox"]) < iou_threshold
            ]

        return keep

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _classify_obstacles(self, detections: List[Dict]) -> List[Dict]:
        """Classify detections into rooftop obstacle categories"""
        obstacles = []

        for i, det in enumerate(detections):
            class_name = det["class"].lower()
            obstacle_type = None

            # Check against obstacle dictionary
            for obs_type, keywords in self.rooftop_obstacles.items():
                if any(keyword in class_name for keyword in keywords):
                    obstacle_type = obs_type
                    break

            if obstacle_type:
                obstacles.append({
                    "id": i,
                    "type": obstacle_type,
                    "confidence": round(det["confidence"], 3),
                    "bbox": det["bbox"],
                    "area": int((det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])),
                    "yolo_class": det["class"]
                })

        return obstacles

    def _create_advanced_visualization(self, image_path: str, detections: List[Dict]) -> str:
        """Create advanced visualization"""
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Color map for obstacle types
            colors = {
                "chimney": (255, 0, 0),
                "vent": (0, 255, 0),
                "skylight": (0, 0, 255),
                "hvac": (255, 255, 0),
                "satellite": (255, 0, 255),
                "antenna": (0, 255, 255),
                "tree": (128, 0, 128),
                "person": (255, 165, 0),
                "vehicle": (0, 128, 128),
                "other": (128, 128, 128)
            }

            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox
                color = colors.get(det["type"], colors["other"])

                # Draw bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                # Draw label with confidence
                label = f"{det['type']}: {det['confidence']:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert to base64
            pil_image = Image.fromarray(image)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return ""

    def _calculate_advanced_stats(self, detections: List[Dict]) -> Dict:
        """Calculate advanced statistics"""
        if not detections:
            return {
                "total_area_covered_pixels": 0,
                "obstacles_by_type": {},
                "average_confidence": 0.0,
                "detection_quality_score": 0.0
            }

        by_type = defaultdict(list)
        total_area = 0
        confidences = []

        for det in detections:
            by_type[det["type"]].append(det)
            total_area += det["area"]
            confidences.append(det["confidence"])

        # Detection quality score (0-1)
        quality_score = min(1.0, (
            0.5 * (len(detections) / 20.0) +  # More detections = better
            0.5 * np.mean(confidences)  # Higher confidence = better
        ))

        return {
            "total_area_covered_pixels": total_area,
            "obstacles_by_type": {k: len(v) for k, v in by_type.items()},
            "average_confidence": round(np.mean(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "detection_quality_score": round(quality_score, 3)
        }

    def _create_error_response(self, error_msg: str) -> Dict:
        """Create error response"""
        return {
            "detected_objects": [],
            "total_obstacles": 0,
            "obstacle_types": [],
            "detection_visualization_base64": "",
            "statistics": {},
            "error": error_msg,
            "model_used": "Ensemble-Failed",
            "confidence_threshold": self.fusion_confidence,
            "processing_time_seconds": 0.0
        }
