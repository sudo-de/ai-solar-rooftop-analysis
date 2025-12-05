"""
Area Calculations Module
Handles roof area, usable area, and obstruction calculations
"""

from typing import Dict
import numpy as np


def calculate_usable_roof_area(roof_area_pixels: float, obstruction_area_pixels: float, 
                               image_width: int, image_height: int) -> Dict:
    """
    Calculate usable roof area excluding obstructions
    
    Args:
        roof_area_pixels: Total roof area in pixels
        obstruction_area_pixels: Total obstruction area in pixels
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Dictionary with area calculations
    """
    # Estimate pixel to meter ratio
    # Assume average roof image shows ~20m x 15m area (300 m²)
    # This is a rough estimate and should be calibrated based on actual measurements
    image_area_pixels = image_width * image_height
    estimated_roof_size_m = 20  # meters (assumed)
    pixel_to_meter_ratio = estimated_roof_size_m / np.sqrt(image_area_pixels)
    
    # Calculate areas
    total_roof_area_m2 = roof_area_pixels * (pixel_to_meter_ratio ** 2)
    obstruction_area_m2 = obstruction_area_pixels * (pixel_to_meter_ratio ** 2)
    usable_roof_area_m2 = max(0, total_roof_area_m2 - obstruction_area_m2)
    
    # Calculate percentages
    obstruction_percentage = (obstruction_area_pixels / roof_area_pixels * 100) if roof_area_pixels > 0 else 0
    usable_percentage = 100 - obstruction_percentage
    
    return {
        "total_roof_area_m2": round(total_roof_area_m2, 2),
        "obstruction_area_m2": round(obstruction_area_m2, 2),
        "usable_roof_area_m2": round(usable_roof_area_m2, 2),
        "obstruction_percentage": round(obstruction_percentage, 1),
        "usable_percentage": round(usable_percentage, 1),
        "pixel_to_meter_ratio": round(pixel_to_meter_ratio, 6)
    }

