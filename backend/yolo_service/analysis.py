"""
Solar Panel Analysis Module
Handles hotspot detection and dirt accumulation analysis
"""

from typing import List, Dict
from PIL import Image
import numpy as np
import cv2


def detect_hotspots(image: Image.Image, solar_panels: List[Dict]) -> Dict:
    """
    Detect hotspots (overheating areas) in photovoltaic panels using YOLOv12 and thermal analysis
    
    Args:
        image: PIL Image object
        solar_panels: List of detected solar panels
    
    Returns:
        Dictionary with hotspot detection results
    """
    try:
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        hotspots = []
        hotspot_image = img_array.copy()
        
        # Process each detected solar panel with enhanced hotspot detection
        for panel_idx, panel in enumerate(solar_panels):
            bbox = panel.get("bbox", {})
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            
            # Ensure valid coordinates
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract panel region
            panel_region = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
            
            if panel_region is not None and panel_region.size > 0:
                # Enhanced statistics for photovoltaic hotspot detection
                mean_intensity = np.mean(panel_region)
                std_intensity = np.std(panel_region)
                median_intensity = np.median(panel_region)
                
                # Hotspots in photovoltaics appear as:
                # 1. Bright spots (high intensity) - thermal anomalies
                # 2. Areas with high variance - cell damage
                # 3. Statistical outliers in intensity distribution
                
                # Method 1: Bright spot detection (thermal hotspots)
                hotspot_threshold_bright = mean_intensity + (2.5 * std_intensity)
                hotspot_threshold_bright = min(255, max(220, hotspot_threshold_bright))
                
                # Method 2: Variance-based detection (damaged cells)
                # Calculate local variance
                kernel_size = 15
                if panel_region.shape[0] > kernel_size and panel_region.shape[1] > kernel_size:
                    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                    local_mean = cv2.filter2D(panel_region.astype(np.float32), -1, kernel)
                    local_variance = cv2.filter2D((panel_region.astype(np.float32) - local_mean) ** 2, -1, kernel)
                    variance_threshold = np.mean(local_variance) + (2 * np.std(local_variance))
                    
                    # Find high variance areas
                    high_variance_mask = local_variance > variance_threshold
                else:
                    high_variance_mask = np.zeros_like(panel_region, dtype=bool)
                
                # Find bright spots (thermal hotspots)
                _, bright_mask = cv2.threshold(panel_region, hotspot_threshold_bright, 255, cv2.THRESH_BINARY)
                
                # Combine both methods
                combined_mask = bright_mask.astype(bool) | high_variance_mask
                combined_mask = combined_mask.astype(np.uint8) * 255
                
                # Clean up noise
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours of hotspot areas
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 30:  # Minimum hotspot area (reduced for better detection)
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Ensure coordinates are within panel region
                        x = max(0, min(x, panel_region.shape[1] - 1))
                        y = max(0, min(y, panel_region.shape[0] - 1))
                        w = min(w, panel_region.shape[1] - x)
                        h = min(h, panel_region.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            # Get intensity statistics of hotspot
                            hotspot_region = panel_region[y:y+h, x:x+w]
                            hotspot_intensity = np.max(hotspot_region) if hotspot_region.size > 0 else mean_intensity
                            hotspot_mean = np.mean(hotspot_region) if hotspot_region.size > 0 else mean_intensity
                            
                            # Calculate severity based on intensity deviation
                            intensity_deviation = ((hotspot_mean - mean_intensity) / mean_intensity) * 100 if mean_intensity > 0 else 0
                            severity = min(100, max(0, 50 + intensity_deviation))  # Base 50% + deviation
                            
                            # Check if it's a high-variance area (damaged cell)
                            is_damaged_cell = False
                            if panel_region.shape[0] > kernel_size and panel_region.shape[1] > kernel_size:
                                if y+h <= local_variance.shape[0] and x+w <= local_variance.shape[1]:
                                    hotspot_variance = np.mean(local_variance[y:y+h, x:x+w])
                                    if hotspot_variance > variance_threshold:
                                        is_damaged_cell = True
                                        severity = min(100, severity + 20)  # Increase severity for damaged cells
                            
                            # Global coordinates
                            global_x = x1 + x
                            global_y = y1 + y
                            
                            hotspot_type = "thermal_hotspot" if not is_damaged_cell else "damaged_cell"
                            
                            hotspots.append({
                                "panel_id": panel_idx,
                                "hotspot_id": len(hotspots),
                                "type": hotspot_type,
                                "bbox": {
                                    "x1": float(global_x),
                                    "y1": float(global_y),
                                    "x2": float(global_x + w),
                                    "y2": float(global_y + h),
                                    "width": float(w),
                                    "height": float(h)
                                },
                                "area_pixels": float(area),
                                "intensity": float(hotspot_intensity),
                                "mean_intensity": float(hotspot_mean),
                                "severity": round(severity, 1),
                                "mean_panel_intensity": float(mean_intensity),
                                "intensity_deviation_percent": round(intensity_deviation, 1)
                            })
                            
                            # Draw hotspot on image with color coding
                            if severity > 70:
                                color = (0, 0, 255)  # Red for critical
                            elif severity > 50:
                                color = (0, 100, 255)  # Orange for moderate
                            else:
                                color = (0, 200, 255)  # Yellow for low
                            
                            cv2.rectangle(hotspot_image, 
                                        (global_x, global_y),
                                        (global_x + w, global_y + h),
                                        color, 4)
                            
                            label = f"{'🔥' if is_damaged_cell else '⚡'}{severity:.0f}%"
                            cv2.putText(hotspot_image, label,
                                      (global_x, global_y - 8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return {
            "hotspots": hotspots,
            "total_hotspots": len(hotspots),
            "hotspot_image": hotspot_image,
            "detection_method": "intensity_analysis"
        }
    except Exception as e:
        print(f"Hotspot detection error: {str(e)}")
        return {
            "hotspots": [],
            "total_hotspots": 0,
            "hotspot_image": np.array(image),
            "error": str(e)
        }


def detect_dirt_accumulation(image: Image.Image, solar_panels: List[Dict]) -> Dict:
    """
    Detect dirt accumulation and soiling on solar panels
    
    Args:
        image: PIL Image object
        solar_panels: List of detected solar panels
    
    Returns:
        Dictionary with dirt accumulation detection results
    """
    try:
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        dirt_patches = []
        dirt_image = img_array.copy()
        
        # Process each detected solar panel
        for panel in solar_panels:
            bbox = panel.get("bbox", {})
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            
            # Extract panel region
            panel_region = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
            
            if panel_region is not None and panel_region.size > 0:
                # Calculate statistics
                mean_intensity = np.mean(panel_region)
                std_intensity = np.std(panel_region)
                
                # Dirt appears as dark patches (low intensity)
                # Threshold: mean - 1.5*std (statistical outlier detection)
                dirt_threshold = mean_intensity - (1.5 * std_intensity)
                dirt_threshold = max(0, min(150, dirt_threshold))  # Clamp between 0-150
                
                # Find dark spots
                _, dark_mask = cv2.threshold(panel_region, dirt_threshold, 255, cv2.THRESH_BINARY_INV)
                
                # Morphological operations to clean up noise
                kernel = np.ones((5, 5), np.uint8)
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours of dark spots
                contours, _ = cv2.findContours(dark_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum dirt patch area
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Get intensity of dirt patch
                        dirt_intensity = np.min(panel_region[y:y+h, x:x+w]) if y+h <= panel_region.shape[0] and x+w <= panel_region.shape[1] else 0
                        
                        # Calculate soiling level (0-100%)
                        # Lower intensity = more dirt
                        soiling_level = min(100, ((mean_intensity - dirt_intensity) / mean_intensity) * 100) if mean_intensity > 0 else 0
                        
                        # Global coordinates
                        global_x = x1 + x
                        global_y = y1 + y
                        
                        dirt_patches.append({
                            "panel_id": len(dirt_patches),
                            "bbox": {
                                "x1": float(global_x),
                                "y1": float(global_y),
                                "x2": float(global_x + w),
                                "y2": float(global_y + h),
                                "width": float(w),
                                "height": float(h)
                            },
                            "area_pixels": float(area),
                            "intensity": float(dirt_intensity),
                            "soiling_level": round(soiling_level, 1),
                            "mean_panel_intensity": float(mean_intensity)
                        })
                        
                        # Draw dirt patch on image (brown/dark for dirt)
                        color = (42, 42, 165)  # Dark brown/blue
                        cv2.rectangle(dirt_image, 
                                    (global_x, global_y),
                                    (global_x + w, global_y + h),
                                    color, 3)
                        cv2.putText(dirt_image, f"Dirt {soiling_level:.0f}%",
                                  (global_x, global_y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return {
            "dirt_patches": dirt_patches,
            "total_dirt_patches": len(dirt_patches),
            "dirt_image": dirt_image,
            "detection_method": "intensity_analysis"
        }
    except Exception as e:
        print(f"Dirt accumulation detection error: {str(e)}")
        return {
            "dirt_patches": [],
            "total_dirt_patches": 0,
            "dirt_image": np.array(image),
            "error": str(e)
        }

