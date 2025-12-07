"""
NextGen Intelligent Zone Refinement
Advanced algorithms for optimal solar zone identification
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from scipy.spatial import distance
import math

logger = logging.getLogger(__name__)

class IntelligentZoneRefinement:
    """NextGen zone refinement with intelligent algorithms"""

    def __init__(self):
        self.min_zone_area_pixels = 3000  # Even smaller for better detection
        self.optimal_aspect_ratio = 1.8  # Landscape panels
        self.min_panel_count = 2  # Minimum panels per zone

    def refine_zones_intelligent(self, 
                                 clean_zone_mask: np.ndarray,
                                 image_size: Tuple[int, int],
                                 roof_features: Optional[Dict] = None) -> List[Dict]:
        """Intelligent zone refinement with advanced algorithms"""
        try:
            # Step 1: Find all potential zones
            potential_zones = self._find_potential_zones(clean_zone_mask, image_size)

            # Step 2: Merge nearby zones intelligently
            merged_zones = self._intelligent_merge(potential_zones)

            # Step 3: Optimize zone boundaries
            optimized_zones = self._optimize_boundaries(merged_zones, clean_zone_mask)

            # Step 4: Rank zones by multiple criteria
            ranked_zones = self._rank_zones(optimized_zones, roof_features)

            # Step 5: Select optimal zones
            optimal_zones = self._select_optimal_zones(ranked_zones)

            return optimal_zones

        except Exception as e:
            logger.error(f"Error in intelligent zone refinement: {e}")
            return []

    def _find_potential_zones(self, mask: np.ndarray, image_size: Tuple[int, int]) -> List[Dict]:
        """Find all potential zones using advanced contour analysis"""
        try:
            # Use multiple methods to find zones
            zones = []

            # Method 1: Connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_zone_area_pixels:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]

                    zones.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "centroid": (int(centroids[i][0]), int(centroids[i][1])),
                        "method": "connected_component"
                    })

            # Method 2: Contour-based (for irregular shapes)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_zone_area_pixels:
                    x, y, w, h = cv2.boundingRect(contour)
                    zones.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "centroid": (x + w//2, y + h//2),
                        "contour": contour,
                        "method": "contour"
                    })

            return zones

        except Exception as e:
            logger.warning(f"Error finding potential zones: {e}")
            return []

    def _intelligent_merge(self, zones: List[Dict]) -> List[Dict]:
        """Intelligently merge nearby zones"""
        if len(zones) <= 1:
            return zones

        try:
            # Calculate zone centers
            centers = [zone["centroid"] for zone in zones]
            
            # Distance threshold (10% of image diagonal)
            image_diagonal = math.sqrt(zones[0]["bbox"][2]**2 + zones[0]["bbox"][3]**2)
            merge_threshold = image_diagonal * 0.1

            # Merge zones that are close together
            merged = []
            used = set()

            for i, zone1 in enumerate(zones):
                if i in used:
                    continue

                merged_zone = zone1.copy()
                merged_indices = [i]

                for j, zone2 in enumerate(zones[i+1:], i+1):
                    if j in used:
                        continue

                    dist = distance.euclidean(centers[i], centers[j])
                    if dist < merge_threshold:
                        # Merge zones
                        x1_min = min(merged_zone["bbox"][0], zone2["bbox"][0])
                        y1_min = min(merged_zone["bbox"][1], zone2["bbox"][1])
                        x2_max = max(merged_zone["bbox"][2], zone2["bbox"][2])
                        y2_max = max(merged_zone["bbox"][3], zone2["bbox"][3])

                        merged_zone["bbox"] = [x1_min, y1_min, x2_max, y2_max]
                        merged_zone["area"] = merged_zone.get("area", 0) + zone2.get("area", 0)
                        merged_zone["centroid"] = (
                            (merged_zone["centroid"][0] + zone2["centroid"][0]) // 2,
                            (merged_zone["centroid"][1] + zone2["centroid"][1]) // 2
                        )
                        merged_indices.append(j)

                for idx in merged_indices:
                    used.add(idx)

                merged.append(merged_zone)

            return merged

        except Exception as e:
            logger.warning(f"Error in intelligent merge: {e}")
            return zones

    def _optimize_boundaries(self, zones: List[Dict], mask: np.ndarray) -> List[Dict]:
        """Optimize zone boundaries to fit clean areas better"""
        optimized = []

        for zone in zones:
            x1, y1, x2, y2 = zone["bbox"]

            # Expand zone slightly to include nearby clean pixels
            expansion = 20
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(mask.shape[1], x2 + expansion)
            y2_exp = min(mask.shape[0], y2 + expansion)

            # Check if expanded area is still clean
            expanded_roi = mask[y1_exp:y2_exp, x1_exp:x2_exp]
            clean_ratio = np.sum(expanded_roi > 0) / expanded_roi.size if expanded_roi.size > 0 else 0

            if clean_ratio > 0.8:  # 80% clean
                zone["bbox"] = [x1_exp, y1_exp, x2_exp, y2_exp]
                zone["area"] = (x2_exp - x1_exp) * (y2_exp - y1_exp)

            optimized.append(zone)

        return optimized

    def _rank_zones(self, zones: List[Dict], roof_features: Optional[Dict] = None) -> List[Dict]:
        """Rank zones by multiple criteria"""
        ranked = []

        for i, zone in enumerate(zones):
            x1, y1, x2, y2 = zone["bbox"]
            width = x2 - x1
            height = y2 - y1
            area = zone.get("area", width * height)
            aspect_ratio = width / height if height > 0 else 0

            # Scoring criteria
            scores = {
                "size_score": min(1.0, area / 50000),  # Larger is better
                "aspect_score": 1.0 - min(abs(aspect_ratio - self.optimal_aspect_ratio) / self.optimal_aspect_ratio, 1.0),
                "regularity_score": 1.0 if 0.5 <= aspect_ratio <= 3.0 else 0.5,
                "position_score": self._calculate_position_score(zone, roof_features)
            }

            # Weighted total score
            total_score = (
                scores["size_score"] * 0.3 +
                scores["aspect_score"] * 0.3 +
                scores["regularity_score"] * 0.2 +
                scores["position_score"] * 0.2
            )

            zone["suitability_score"] = round(total_score, 3)
            zone["scores"] = scores
            zone["id"] = i + 1
            zone["width_pixels"] = width
            zone["height_pixels"] = height
            zone["aspect_ratio"] = round(aspect_ratio, 2)
            zone["orientation"] = "landscape" if width > height else "portrait"

            # Estimate panels
            zone["estimated_panels"] = self._estimate_panels_advanced(width, height)

            ranked.append(zone)

        # Sort by suitability score
        ranked.sort(key=lambda x: x["suitability_score"], reverse=True)

        return ranked

    def _calculate_position_score(self, zone: Dict, roof_features: Optional[Dict] = None) -> float:
        """Calculate score based on zone position"""
        # Prefer zones in center of roof (less edge effects)
        centroid = zone["centroid"]
        
        # Default score
        score = 0.7

        if roof_features:
            # If we have roof outline, prefer zones away from edges
            # This is simplified - in practice would check distance to roof edges
            score = 0.8

        return score

    def _estimate_panels_advanced(self, width_px: float, height_px: float) -> Dict:
        """Advanced panel estimation with better algorithms"""
        # Convert pixels to meters (rough estimate)
        pixels_per_meter = 150.0

        panel_width_m = 1.0
        panel_height_m = 1.6
        spacing = 0.1  # 10cm

        zone_width_m = width_px / pixels_per_meter
        zone_height_m = height_px / pixels_per_meter

        # Calculate optimal grid
        panels_per_row = max(1, int((zone_width_m + spacing) / (panel_width_m + spacing)))
        rows = max(1, int((zone_height_m + spacing) / (panel_height_m + spacing)))

        # Refine based on actual fit
        total_panels = 0
        for row in range(rows):
            for col in range(panels_per_row):
                x_pos = col * (panel_width_m + spacing)
                y_pos = row * (panel_height_m + spacing)
                
                if (x_pos + panel_width_m <= zone_width_m and 
                    y_pos + panel_height_m <= zone_height_m):
                    total_panels += 1

        return {
            "estimated_count": max(self.min_panel_count, total_panels),
            "panel_layout": f"{panels_per_row}x{rows}",
            "total_area_m2": round(zone_width_m * zone_height_m, 2),
            "panel_efficiency": round(total_panels / (panels_per_row * rows), 2) if (panels_per_row * rows) > 0 else 0
        }

    def _select_optimal_zones(self, ranked_zones: List[Dict], max_zones: int = 10) -> List[Dict]:
        """Select optimal zones using intelligent selection"""
        if not ranked_zones:
            return []

        # Filter by minimum quality
        quality_zones = [z for z in ranked_zones if z["suitability_score"] > 0.3]

        # Select top zones with diversity (not all in same area)
        selected = []
        used_centroids = []

        for zone in quality_zones[:max_zones * 2]:  # Consider more candidates
            if len(selected) >= max_zones:
                break

            centroid = zone["centroid"]
            
            # Check if zone is too close to already selected zones
            too_close = False
            for used_centroid in used_centroids:
                dist = distance.euclidean(centroid, used_centroid)
                if dist < 200:  # 200 pixels minimum distance
                    too_close = True
                    break

            if not too_close:
                selected.append(zone)
                used_centroids.append(centroid)

        # If not enough diverse zones, add top remaining
        while len(selected) < max_zones and len(selected) < len(quality_zones):
            for zone in quality_zones:
                if zone not in selected:
                    selected.append(zone)
                    break

        return selected[:max_zones]
