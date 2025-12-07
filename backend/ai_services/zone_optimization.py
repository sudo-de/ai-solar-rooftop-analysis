import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ZoneOptimizationService:
    """Service for optimizing solar panel placement zones by subtracting obstacles"""

    def __init__(self):
        self.safety_margin_pixels = 50  # Safety margin around obstacles
        self.min_zone_area_pixels = 5000  # Reduced minimum area for better zone detection
        self.zone_expansion_pixels = 20  # Expand zones slightly for better panel placement
        
        # Try to load intelligent refinement
        try:
            from ai_services.intelligent_zone_refinement import IntelligentZoneRefinement
            self.intelligent_refinement = IntelligentZoneRefinement()
            self.use_intelligent_refinement = True
            logger.info("✅ Intelligent zone refinement enabled")
        except ImportError:
            self.intelligent_refinement = None
            self.use_intelligent_refinement = False

    def optimize_zones(self, roof_segmentation_result: Dict,
                      object_detection_result: Dict,
                      image_path: str) -> Dict:
        """Create optimized solar panel placement zones by subtracting obstacles"""
        try:
            # Load original image to get dimensions
            original_image = Image.open(image_path)
            image_width, image_height = original_image.size

            # Create roof mask from segmentation
            roof_mask = self._create_roof_mask(roof_segmentation_result, (image_width, image_height))

            # Create obstacle mask from detections
            obstacle_mask = self._create_obstacle_mask(object_detection_result, (image_width, image_height))

            # Subtract obstacles from roof area
            clean_zone_mask = self._subtract_obstacles(roof_mask, obstacle_mask)

            # Find optimal solar zones (use intelligent refinement if available)
            if self.use_intelligent_refinement and self.intelligent_refinement:
                logger.info("Using intelligent zone refinement")
                roof_features = {
                    "roof_area_pixels": roof_segmentation_result.get("roof_area_pixels", 0),
                    "roof_percentage": roof_segmentation_result.get("roof_percentage", 0)
                }
                optimal_zones = self.intelligent_refinement.refine_zones_intelligent(
                    clean_zone_mask, 
                    (image_width, image_height),
                    roof_features
                )
            else:
                optimal_zones = self._find_optimal_zones(clean_zone_mask, image_path)

            # Create visualization
            visualization = self._create_zone_visualization(
                image_path, roof_mask, obstacle_mask, clean_zone_mask, optimal_zones
            )

            # Calculate zone statistics
            zone_stats = self._calculate_zone_statistics(optimal_zones, image_path)

            return {
                "clean_zones_found": len(optimal_zones),
                "total_clean_area_pixels": int(np.sum(clean_zone_mask)),
                "optimal_zones": optimal_zones,
                "zone_visualization_base64": visualization,
                "zone_statistics": zone_stats,
                "roof_coverage_percentage": round((np.sum(roof_mask) / (image_width * image_height)) * 100, 2),
                "obstacle_coverage_percentage": round((np.sum(obstacle_mask) / (image_width * image_height)) * 100, 2),
                "usable_roof_percentage": round((np.sum(clean_zone_mask) / np.sum(roof_mask)) * 100, 2) if np.sum(roof_mask) > 0 else 0,
                "processing_time_seconds": 0.0  # Will be calculated by caller
            }

        except Exception as e:
            logger.error(f"Error in zone optimization: {str(e)}")
            return {
                "clean_zones_found": 0,
                "total_clean_area_pixels": 0,
                "optimal_zones": [],
                "zone_visualization_base64": "",
                "zone_statistics": {},
                "error": str(e),
                "roof_coverage_percentage": 0.0,
                "obstacle_coverage_percentage": 0.0,
                "usable_roof_percentage": 0.0,
                "processing_time_seconds": 0.0
            }

    def _create_roof_mask(self, roof_result: Dict, image_size: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from roof segmentation result"""
        try:
            # If we have a segmented image, extract the mask
            if "segmented_image_base64" in roof_result and roof_result["segmented_image_base64"]:
                # Decode base64 image
                img_data = roof_result["segmented_image_base64"].split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))

                # Convert to grayscale and threshold to get binary mask
                img_gray = img.convert("L")
                img_array = np.array(img_gray)

                # Simple thresholding to get roof areas (assuming red tint from segmentation)
                if img.mode == "RGB":
                    img_rgb = np.array(img)
                    # Roof areas should be tinted red, so check red channel
                    roof_mask = img_rgb[:, :, 0] > 100  # Red channel threshold
                else:
                    roof_mask = img_array > 128

                return roof_mask.astype(np.uint8)
            else:
                # Fallback: create mask from roof outline coordinates
                mask = np.zeros(image_size[::-1], dtype=np.uint8)  # (height, width)

                if "roof_outline_coordinates" in roof_result and roof_result["roof_outline_coordinates"]:
                    # Convert coordinates to numpy array
                    points = np.array(roof_result["roof_outline_coordinates"], dtype=np.int32)
                    if len(points) > 2:
                        cv2.fillPoly(mask, [points], 255)

                return (mask > 0).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Error creating roof mask: {e}")
            # Return empty mask as fallback
            return np.zeros(image_size[::-1], dtype=np.uint8)

    def _create_obstacle_mask(self, detection_result: Dict, image_size: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from object detection results"""
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # (height, width)

        try:
            if "detected_objects" in detection_result:
                for obj in detection_result["detected_objects"]:
                    if "bbox" in obj:
                        x1, y1, x2, y2 = obj["bbox"]

                        # Add safety margin around obstacle
                        x1_safe = max(0, x1 - self.safety_margin_pixels)
                        y1_safe = max(0, y1 - self.safety_margin_pixels)
                        x2_safe = min(image_size[0], x2 + self.safety_margin_pixels)
                        y2_safe = min(image_size[1], y2 + self.safety_margin_pixels)

                        # Draw obstacle area on mask
                        cv2.rectangle(mask, (x1_safe, y1_safe), (x2_safe, y2_safe), 255, -1)

            return mask

        except Exception as e:
            logger.warning(f"Error creating obstacle mask: {e}")
            return mask

    def _subtract_obstacles(self, roof_mask: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
        """Subtract obstacle areas from roof areas to get clean zones"""
        # Clean zones = roof areas minus obstacle areas
        clean_zone_mask = roof_mask & (1 - obstacle_mask.astype(bool))
        
        # Apply morphological operations to smooth and expand zones slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_zone_mask = cv2.morphologyEx(clean_zone_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        clean_zone_mask = cv2.morphologyEx(clean_zone_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return clean_zone_mask.astype(np.uint8)

    def _find_optimal_zones(self, clean_zone_mask: np.ndarray, image_path: str) -> List[Dict]:
        """Find optimal rectangular zones for solar panel placement"""
        optimal_zones = []

        try:
            # Find contours in the clean zone mask
            contours, _ = cv2.findContours(clean_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and sort contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_zone_area_pixels:
                    valid_contours.append((contour, area))

            # Sort by area (largest first)
            valid_contours.sort(key=lambda x: x[1], reverse=True)

            # Take top zones (limit to prevent too many)
            for i, (contour, area) in enumerate(valid_contours[:10]):  # Max 10 zones
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate aspect ratio and suitability
                aspect_ratio = w / h if h > 0 else 0
                suitability_score = self._calculate_zone_suitability(w, h, area, aspect_ratio)

                zone = {
                    "id": i + 1,
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area_pixels": int(area),
                    "width_pixels": int(w),
                    "height_pixels": int(h),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "suitability_score": round(suitability_score, 2),
                    "orientation": "landscape" if w > h else "portrait",
                    "estimated_panels": self._estimate_panel_count(w, h)
                }

                optimal_zones.append(zone)

            return optimal_zones

        except Exception as e:
            logger.error(f"Error finding optimal zones: {e}")
            return []

    def _calculate_zone_suitability(self, width: float, height: float, area: float, aspect_ratio: float) -> float:
        """Calculate how suitable a zone is for solar panels"""
        # Ideal aspect ratio for solar panels is around 1.5-2.0 (landscape)
        ideal_aspect_ratio = 1.8
        aspect_score = 1.0 - min(abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio, 1.0)

        # Size score (larger is better, but not too large)
        size_score = min(area / 50000, 1.0)  # Ideal around 50,000 pixels

        # Shape regularity score
        regularity_score = 1.0 if 0.5 <= aspect_ratio <= 3.0 else 0.5

        # Combined score
        suitability = (aspect_score * 0.4 + size_score * 0.4 + regularity_score * 0.2)

        return suitability

    def _estimate_panel_count(self, width_pixels: float, height_pixels: float) -> Dict:
        """Estimate how many solar panels can fit in the zone"""
        # Assume standard solar panel dimensions (approximately 1m x 1.6m)
        # Convert pixels to approximate meters (rough estimation)
        pixels_per_meter = 100  # This would need calibration based on image resolution

        panel_width_m = 1.0  # meters
        panel_height_m = 1.6  # meters
        panel_area_m2 = panel_width_m * panel_height_m

        # Convert zone dimensions to meters
        zone_width_m = width_pixels / pixels_per_meter
        zone_height_m = height_pixels / pixels_per_meter
        zone_area_m2 = zone_width_m * zone_height_m

        # Calculate how many panels can fit
        panels_width = int(zone_width_m / panel_width_m)
        panels_height = int(zone_height_m / panel_height_m)
        total_panels = panels_width * panels_height

        # Account for spacing and orientation
        spacing_factor = 0.8  # 80% efficiency due to spacing
        total_panels = int(total_panels * spacing_factor)

        return {
            "estimated_count": max(1, total_panels),
            "panel_layout": f"{panels_width}x{panels_height}",
            "total_area_m2": round(zone_area_m2, 2),
            "panel_efficiency": round(spacing_factor, 2)
        }

    def _create_zone_visualization(self, image_path: str, roof_mask: np.ndarray,
                                 obstacle_mask: np.ndarray, clean_zone_mask: np.ndarray,
                                 optimal_zones: List[Dict]) -> str:
        """Create visualization showing all zones and obstacles"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create overlay image
            overlay = image.copy()

            # Add semi-transparent overlays
            # Roof areas (blue tint)
            roof_overlay = np.zeros_like(image)
            roof_overlay[roof_mask > 0] = [0, 0, 255]  # Blue for roof
            cv2.addWeighted(roof_overlay, 0.3, overlay, 1.0, 0, overlay)

            # Obstacle areas (red tint)
            obstacle_overlay = np.zeros_like(image)
            obstacle_overlay[obstacle_mask > 0] = [255, 0, 0]  # Red for obstacles
            cv2.addWeighted(obstacle_overlay, 0.5, overlay, 1.0, 0, overlay)

            # Clean zones (green tint)
            clean_overlay = np.zeros_like(image)
            clean_overlay[clean_zone_mask > 0] = [0, 255, 0]  # Green for clean zones
            cv2.addWeighted(clean_overlay, 0.2, overlay, 1.0, 0, overlay)

            # Draw optimal zone rectangles with enhanced visualization
            for zone in optimal_zones:
                x1, y1, x2, y2 = zone["bbox"]
                color = (0, 255, 255)  # Cyan for optimal zones (more visible)
                
                # Draw filled rectangle with transparency
                zone_overlay = overlay.copy()
                cv2.rectangle(zone_overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(zone_overlay, 0.15, overlay, 1.0, 0, overlay)
                
                # Draw border
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
                
                # Add zone label with background for better visibility
                label = f"Zone {zone['id']}: {zone['estimated_panels']['estimated_count']} panels"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(overlay, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 0, 0), -1)
                cv2.putText(overlay, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2, cv2.LINE_AA)
                
                # Add suitability score
                score_label = f"Suitability: {zone['suitability_score']:.1f}"
                cv2.putText(overlay, score_label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, color, 2, cv2.LINE_AA)

            # Add enhanced legend with background
            legend_y = 30
            legend_bg = np.zeros((150, 250, 3), dtype=np.uint8)
            legend_bg[:] = (0, 0, 0)  # Black background
            legend_bg = cv2.addWeighted(legend_bg, 0.7, np.ones((150, 250, 3), dtype=np.uint8) * 255, 0.3, 0)
            overlay[legend_y:legend_y+150, 10:260] = cv2.addWeighted(overlay[legend_y:legend_y+150, 10:260], 0.5, legend_bg, 0.5, 0)
            
            cv2.putText(overlay, "Legend:", (15, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "Blue: Roof Area", (15, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(overlay, "Red: Obstacles", (15, legend_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(overlay, "Green: Clean Zones", (15, legend_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, "Cyan: Optimal Zones", (15, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Convert to base64
            pil_image = Image.fromarray(overlay)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error creating zone visualization: {e}")
            return ""

    def _calculate_zone_statistics(self, optimal_zones: List[Dict], image_path: str) -> Dict:
        """Calculate statistics about the optimal zones"""
        if not optimal_zones:
            return {
                "total_zones": 0,
                "average_zone_area": 0,
                "total_estimated_panels": 0,
                "best_zone_suitability": 0.0,
                "zone_distribution": {}
            }

        total_area = sum(zone["area_pixels"] for zone in optimal_zones)
        total_panels = sum(zone["estimated_panels"]["estimated_count"] for zone in optimal_zones)
        avg_area = total_area / len(optimal_zones)
        best_suitability = max(zone["suitability_score"] for zone in optimal_zones)

        # Zone distribution by size
        distribution = {
            "small": len([z for z in optimal_zones if z["area_pixels"] < 20000]),
            "medium": len([z for z in optimal_zones if 20000 <= z["area_pixels"] < 50000]),
            "large": len([z for z in optimal_zones if z["area_pixels"] >= 50000])
        }

        return {
            "total_zones": len(optimal_zones),
            "average_zone_area_pixels": round(avg_area),
            "total_estimated_panels": total_panels,
            "best_zone_suitability": round(best_suitability, 2),
            "zone_distribution": distribution,
            "panels_per_zone_avg": round(total_panels / len(optimal_zones), 1)
        }
