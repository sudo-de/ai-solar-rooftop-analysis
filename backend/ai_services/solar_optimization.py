import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

class SolarOptimizationService:
    """Service for optimizing solar panel placement and calculating energy output"""

    def __init__(self):
        # Multiple solar panel types for better optimization
        self.panel_specs = {
            "standard": {
                "width_m": 1.0,      # meters
                "height_m": 1.6,     # meters
                "area_m2": 1.6,      # square meters
                "power_w": 400,      # watts
                "efficiency": 0.22   # 22% efficiency
            },
            "compact": {
                "width_m": 0.8,      # Smaller for tight spaces
                "height_m": 1.3,
                "area_m2": 1.04,
                "power_w": 300,
                "efficiency": 0.21
            },
            "premium": {
                "width_m": 1.0,
                "height_m": 1.6,
                "area_m2": 1.6,
                "power_w": 450,      # Higher efficiency
                "efficiency": 0.245
            }
        }
        
        # Use standard as default
        self.default_panel = self.panel_specs["standard"]

        # Installation parameters (optimized)
        self.spacing_factor = 0.1  # 10cm spacing between panels
        self.row_spacing_factor = 0.15  # 15cm spacing between rows (reduced from 20cm)
        self.tilt_angle_deg = 25  # Optimal tilt angle for most locations
        self.min_zone_area_m2 = 2.0  # Minimum zone area for panels (2 m²)

    def optimize_solar_layout(self, zone_optimization_result: Dict,
                            image_path: str, location_data: Optional[Dict] = None) -> Dict:
        """Create optimal solar panel layout for the clean zones"""
        try:
            optimal_zones = zone_optimization_result.get("optimal_zones", [])

            if not optimal_zones:
                return self._create_empty_result()

            # Process each zone for solar panel layout
            solar_panels_3d = []
            total_energy_calculation = {
                "total_panels": 0,
                "total_power_kw": 0,
                "estimated_yearly_kwh": 0,
                "estimated_cost_savings": 0,
                "payback_period_years": 0
            }

            for zone in optimal_zones:
                zone_panels = self._optimize_zone_layout(zone, image_path)

                if zone_panels:
                    solar_panels_3d.extend(zone_panels)
                    total_energy_calculation["total_panels"] += len(zone_panels)

            # Calculate energy metrics with advanced calculations
            if total_energy_calculation["total_panels"] > 0:
                # Try to use advanced solar calculations if available
                try:
                    from backend.ai_services.advanced_solar_calculations import AdvancedSolarCalculations
                    advanced_calc = AdvancedSolarCalculations()
                    
                    # Calculate average panel type and specs
                    panel_types = [p.get("panel_type", "standard") for p in solar_panels_3d]
                    most_common_type = max(set(panel_types), key=panel_types.count) if panel_types else "standard"
                    
                    # Get average shading
                    avg_shading = sum(p.get("shading_factor", 0.0) for p in solar_panels_3d) / len(solar_panels_3d) if solar_panels_3d else 0.0
                    
                    advanced_energy = advanced_calc.calculate_advanced_energy(
                        total_energy_calculation["total_panels"],
                        panel_type=most_common_type,
                        location_data=location_data,
                        shading_factor=avg_shading
                    )
                    
                    # Merge advanced calculations
                    total_energy_calculation.update(advanced_energy)
                    total_energy_calculation["calculation_method"] = "advanced"
                except Exception as e:
                    logger.warning(f"Advanced calculations not available, using basic: {e}")
                    total_energy_calculation.update(self._calculate_energy_metrics(
                        total_energy_calculation["total_panels"], location_data
                    ))
                    total_energy_calculation["calculation_method"] = "basic"

            # Create 3D visualization
            visualization_3d = self._create_3d_visualization(image_path, solar_panels_3d)

            return {
                "solar_panels_3d": solar_panels_3d,
                "total_panels_placed": len(solar_panels_3d),
                "energy_calculation": total_energy_calculation,
                "layout_visualization_base64": visualization_3d,
                "optimization_parameters": {
                    "panel_specs": self.panel_specs,
                    "spacing_factor": self.spacing_factor,
                    "row_spacing_factor": self.row_spacing_factor,
                    "tilt_angle_deg": self.tilt_angle_deg,
                    "min_zone_area_m2": self.min_zone_area_m2,
                    "location_data": location_data or {},
                    "optimization_features": {
                        "smart_panel_selection": True,
                        "dual_orientation": True,
                        "improved_spacing": True,
                        "advanced_energy_calc": "calculation_method" in total_energy_calculation
                    }
                },
                "layout_efficiency": self._calculate_layout_efficiency(solar_panels_3d, optimal_zones),
                "processing_time_seconds": 0.0  # Will be calculated by caller
            }

        except Exception as e:
            logger.error(f"Error in solar optimization: {str(e)}")
            return self._create_empty_result(error=str(e))

    def _optimize_zone_layout(self, zone: Dict, image_path: str) -> List[Dict]:
        """Optimize solar panel layout within a specific zone with smart panel selection"""
        try:
            # Get zone dimensions in pixels
            x1, y1, x2, y2 = zone["bbox"]
            zone_width_px = x2 - x1
            zone_height_px = y2 - y1

            # Estimate pixels per meter (rough approximation)
            pixels_per_meter = self._estimate_pixels_per_meter(image_path)

            # Convert zone dimensions to meters
            zone_width_m = zone_width_px / pixels_per_meter
            zone_height_m = zone_height_px / pixels_per_meter
            zone_area_m2 = zone_width_m * zone_height_m

            # Skip if zone is too small
            if zone_area_m2 < self.min_zone_area_m2:
                return []

            # Smart panel type selection based on zone size and aspect ratio
            aspect_ratio = zone_width_m / zone_height_m if zone_height_m > 0 else 1.0
            panel_type = self._select_optimal_panel_type(zone_width_m, zone_height_m, aspect_ratio)
            panel_spec = self.panel_specs[panel_type]

            # Try both orientations (landscape and portrait) and pick the best
            layouts = []
            
            # Landscape orientation (default)
            layout_landscape = self._calculate_panel_grid(
                zone_width_m, zone_height_m,
                panel_spec["width_m"], panel_spec["height_m"]
            )
            if layout_landscape["panels"]:
                layouts.append(("landscape", layout_landscape, panel_spec))

            # Portrait orientation (if zone is tall)
            if aspect_ratio < 1.0:  # Tall zone
                layout_portrait = self._calculate_panel_grid(
                    zone_width_m, zone_height_m,
                    panel_spec["height_m"], panel_spec["width_m"]  # Swapped
                )
                if layout_portrait["panels"]:
                    layouts.append(("portrait", layout_portrait, panel_spec))

            # Pick the layout with most panels
            if not layouts:
                return []

            best_layout = max(layouts, key=lambda x: len(x[1]["panels"]))
            orientation, panel_layout, panel_spec = best_layout

            # Convert panel positions back to pixel coordinates
            solar_panels = []
            for panel_pos in panel_layout["panels"]:
                # Convert meters back to pixels
                panel_x_px = x1 + (panel_pos["x_m"] * pixels_per_meter)
                panel_y_px = y1 + (panel_pos["y_m"] * pixels_per_meter)
                
                # Use correct dimensions based on orientation
                if orientation == "landscape":
                    panel_width_px = panel_spec["width_m"] * pixels_per_meter
                    panel_height_px = panel_spec["height_m"] * pixels_per_meter
                else:
                    panel_width_px = panel_spec["height_m"] * pixels_per_meter
                    panel_height_px = panel_spec["width_m"] * pixels_per_meter

                panel = {
                    "id": f"panel_{len(solar_panels) + 1}",
                    "zone_id": str(zone.get("id", "unknown")),
                    "panel_type": str(panel_type),
                    "orientation": str(orientation),
                    "position_pixels": [float(panel_x_px), float(panel_y_px)],
                    "dimensions_pixels": [float(panel_width_px), float(panel_height_px)],
                    "position_meters": [float(panel_pos["x_m"]), float(panel_pos["y_m"])],
                    "dimensions_meters": [float(panel_spec["width_m"]), float(panel_spec["height_m"])] if orientation == "landscape" else [float(panel_spec["height_m"]), float(panel_spec["width_m"])],
                    "tilt_angle_deg": float(self.tilt_angle_deg),
                    "orientation_angle_deg": float(self._calculate_optimal_orientation(zone)),
                    "power_w": int(panel_spec["power_w"]),
                    "efficiency": float(panel_spec["efficiency"]),
                    "shading_factor": float(panel_pos.get("shading_factor", 0.0)),
                    "row": int(panel_pos.get("row", 0)),
                    "col": int(panel_pos.get("col", 0))
                }

                solar_panels.append(panel)

            return solar_panels

        except Exception as e:
            logger.error(f"Error optimizing zone layout: {e}")
            return []

    def _calculate_panel_grid(self, zone_width_m: float, zone_height_m: float,
                            panel_width_m: float, panel_height_m: float) -> Dict:
        """Calculate optimal grid layout for solar panels in a zone with improved spacing"""
        # Add spacing between panels
        panel_width_with_spacing = panel_width_m * (1 + self.spacing_factor)
        panel_height_with_spacing = panel_height_m * (1 + self.row_spacing_factor)

        # Calculate how many panels fit (with better edge handling)
        panels_per_row = max(1, int((zone_width_m + self.spacing_factor * panel_width_m) / panel_width_with_spacing))
        rows = max(1, int((zone_height_m + self.row_spacing_factor * panel_height_m) / panel_height_with_spacing))

        # Ensure we don't exceed zone bounds
        panels_per_row = min(panels_per_row, int(zone_width_m / panel_width_m))
        rows = min(rows, int(zone_height_m / panel_height_m))

        panels = []
        total_panel_area = 0

        for row in range(rows):
            for col in range(panels_per_row):
                # Calculate position with smart centering
                total_width_used = panels_per_row * panel_width_with_spacing - (self.spacing_factor * panel_width_m)
                total_height_used = rows * panel_height_with_spacing - (self.row_spacing_factor * panel_height_m)
                
                x_offset = max(0, (zone_width_m - total_width_used) / 2)
                y_offset = max(0, (zone_height_m - total_height_used) / 2)

                x_m = x_offset + (col * panel_width_with_spacing) + (panel_width_m / 2)
                y_m = y_offset + (row * panel_height_with_spacing) + (panel_height_m / 2)

                # Enhanced bounds checking
                panel_left = x_m - panel_width_m/2
                panel_right = x_m + panel_width_m/2
                panel_top = y_m - panel_height_m/2
                panel_bottom = y_m + panel_height_m/2

                if (panel_left >= 0 and panel_right <= zone_width_m and
                    panel_top >= 0 and panel_bottom <= zone_height_m):

                    shading = self._calculate_shading_factor(row, col, rows, panels_per_row)
                    
                    panel = {
                        "x_m": x_m,
                        "y_m": y_m,
                        "row": row,
                        "col": col,
                        "shading_factor": shading
                    }
                    panels.append(panel)
                    total_panel_area += panel_width_m * panel_height_m

        zone_area = zone_width_m * zone_height_m
        utilization = total_panel_area / zone_area if zone_area > 0 else 0

        return {
            "panels": panels,
            "grid_layout": f"{panels_per_row}x{rows}",
            "total_panels": len(panels),
            "zone_utilization": round(utilization, 3),
            "efficiency_score": round(utilization * (len(panels) / max(panels_per_row * rows, 1)), 3)
        }

    def _calculate_shading_factor(self, row: int, col: int, total_rows: int, total_cols: int) -> float:
        """Calculate shading factor based on panel position (edge panels have less shading)"""
        # Edge panels have less shading from neighboring panels
        edge_factor = 0.0
        if row == 0 or row == total_rows - 1:
            edge_factor += 0.1
        if col == 0 or col == total_cols - 1:
            edge_factor += 0.1

        return min(0.2, edge_factor)  # Max 20% shading reduction

    def _estimate_pixels_per_meter(self, image_path: str) -> float:
        """Estimate pixels per meter from image (rough approximation)"""
        try:
            # This is a rough estimate - in production, you'd use:
            # 1. GPS coordinates and altitude for accurate scaling
            # 2. Camera intrinsics/extrinsics
            # 3. Known reference objects in image

            # For now, assume typical drone/rooftop photo resolution
            # Average: ~100-200 pixels per meter for rooftop photos
            return 150.0

        except Exception:
            return 150.0  # Fallback value

    def _select_optimal_panel_type(self, width_m: float, height_m: float, aspect_ratio: float) -> str:
        """Select optimal panel type based on zone characteristics"""
        area_m2 = width_m * height_m
        
        # Use compact panels for small or irregular zones
        if area_m2 < 5.0 or aspect_ratio < 0.5 or aspect_ratio > 3.0:
            return "compact"
        
        # Use premium for large zones
        if area_m2 > 20.0:
            return "premium"
        
        # Standard for most cases
        return "standard"
    
    def _calculate_optimal_orientation(self, zone: Optional[Dict] = None) -> float:
        """Calculate optimal panel orientation angle (azimuth)"""
        # For most locations, south-facing is optimal (180°)
        # Could be enhanced with:
        # - GPS location data
        # - Zone orientation from image analysis
        # - Seasonal optimization
        return 180.0  # South-facing in degrees

    def _calculate_energy_metrics(self, total_panels: int, location_data: Optional[Dict] = None) -> Dict:
        """Calculate energy production and financial metrics"""
        # Base calculations
        total_power_w = total_panels * self.panel_specs["power_w"]
        total_power_kw = total_power_w / 1000

        # Location-based solar irradiance (kWh/m²/day)
        # Default: average for northern India
        daily_irradiance = location_data.get("daily_irradiance", 5.5) if location_data else 5.5

        # System efficiency factors
        system_efficiency = 0.85  # 85% overall system efficiency
        performance_ratio = 0.8   # 80% performance ratio

        # Calculate yearly energy production
        yearly_kwh = (total_power_kw * daily_irradiance * 365 *
                     system_efficiency * performance_ratio)

        # Financial calculations (India 2025 estimates)
        cost_per_kw = 40000  # ₹40,000 per kW
        total_system_cost = total_power_kw * cost_per_kw

        electricity_rate = 8.0  # ₹8 per kWh
        annual_savings = yearly_kwh * electricity_rate

        payback_period = total_system_cost / annual_savings if annual_savings > 0 else 0

        return {
            "total_power_kw": round(total_power_kw, 2),
            "estimated_yearly_kwh": round(yearly_kwh),
            "system_efficiency": system_efficiency,
            "performance_ratio": performance_ratio,
            "daily_irradiance_kwh_m2": daily_irradiance,
            "estimated_cost_savings_annual": round(annual_savings),
            "estimated_system_cost": round(total_system_cost),
            "payback_period_years": round(payback_period, 1),
            "roi_percentage": round((annual_savings / total_system_cost) * 100, 1) if total_system_cost > 0 else 0,
            "carbon_savings_kg_co2_year": round(yearly_kwh * 0.82)  # 0.82 kg CO2 per kWh
        }

    def _create_3d_visualization(self, image_path: str, solar_panels: List[Dict]) -> str:
        """Create 3D visualization of solar panel layout"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create overlay for solar panels
            overlay = image.copy()

            for panel in solar_panels:
                x, y = panel["position_pixels"]
                w, h = panel["dimensions_pixels"]

                # Draw panel rectangle
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                # Panel color (dark blue)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), -1)

                # Panel border
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Add panel ID
                cv2.putText(overlay, panel["id"], (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add statistics overlay
            stats_text = f"Solar Panels: {len(solar_panels)}"
            cv2.putText(overlay, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)

            if solar_panels:
                total_power = sum(p["power_w"] for p in solar_panels)
                power_text = f"Total Power: {total_power}W ({total_power/1000:.1f}kW)"
                cv2.putText(overlay, power_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert to base64
            pil_image = Image.fromarray(overlay)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            return ""

    def _calculate_layout_efficiency(self, solar_panels: List[Dict], optimal_zones: List[Dict]) -> Dict:
        """Calculate efficiency metrics for the solar layout"""
        if not solar_panels or not optimal_zones:
            return {"space_utilization": 0.0, "power_density": 0.0}

        # Calculate total panel area vs total zone area
        total_panel_area = len(solar_panels) * self.panel_specs["area_m2"]
        total_zone_area = sum(zone["area_pixels"] * (1/150)**2 for zone in optimal_zones)  # Convert pixels to m²

        space_utilization = total_panel_area / total_zone_area if total_zone_area > 0 else 0

        # Power density (kW per m² of roof area)
        total_power_kw = sum(p["power_w"] for p in solar_panels) / 1000
        power_density = total_power_kw / total_zone_area if total_zone_area > 0 else 0

        return {
            "space_utilization": round(space_utilization, 2),
            "power_density_kw_per_m2": round(power_density, 3),
            "panels_per_m2": round(len(solar_panels) / total_zone_area, 2) if total_zone_area > 0 else 0
        }

    def _create_empty_result(self, error: str = "") -> Dict:
        """Create empty result structure"""
        result = {
            "solar_panels_3d": [],
            "total_panels_placed": 0,
            "energy_calculation": {
                "total_power_kw": 0,
                "estimated_yearly_kwh": 0,
                "estimated_cost_savings_annual": 0,
                "estimated_system_cost": 0,
                "payback_period_years": 0
            },
            "layout_visualization_base64": "",
            "optimization_parameters": {},
            "layout_efficiency": {},
            "processing_time_seconds": 0.0
        }

        if error:
            result["error"] = error

        return result
