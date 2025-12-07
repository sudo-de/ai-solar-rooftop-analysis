import json
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate formatted reports from AI pipeline results"""

    def __init__(self):
        pass

    def generate_text_report(self, analysis_result: Dict) -> str:
        """Generate a formatted text report from analysis results"""
        try:
            roof_analysis = analysis_result.get("roof_analysis", {})
            ai_results = roof_analysis.get("ai_pipeline_results", {})
            
            report = []
            report.append("=" * 80)
            report.append("AI SOLAR ROOFTOP ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Roof Analysis Section
            roof_data = ai_results.get("roof_analysis", {})
            if roof_data:
                report.append("━" * 80)
                report.append("STEP 1: ROOF SEGMENTATION ANALYSIS")
                report.append("━" * 80)
                report.append(f"Model Used: {roof_data.get('model_used', 'N/A')}")
                report.append(f"Roof Detected: {'✅ Yes' if roof_data.get('roof_detected') else '❌ No'}")
                report.append(f"Roof Area: {roof_data.get('roof_area_pixels', 0):,} pixels")
                report.append(f"Roof Coverage: {roof_data.get('roof_percentage', 0):.2f}% of image")
                report.append(f"Roof Perimeter: {roof_data.get('roof_perimeter_pixels', 0):.2f} pixels")
                report.append(f"Confidence Score: {roof_data.get('confidence_score', 0):.3f}")
                report.append(f"Processing Time: {roof_data.get('processing_time_seconds', 0):.2f}s")
                
                # Advanced Features (NextGen SegFormer with Alpha)
                advanced_features = roof_data.get("advanced_features", {})
                if advanced_features:
                    report.append("")
                    report.append("🚀 NEXTGEN SEGFORMER FEATURES:")
                    fusion_method = advanced_features.get("fusion_method", "")
                    if fusion_method:
                        report.append(f"  Fusion Method: {fusion_method.replace('_', ' ').title()}")
                    
                    roof_shape = advanced_features.get("roof_shape", "")
                    if roof_shape:
                        report.append(f"  Roof Shape: {roof_shape}")
                    
                    complexity = advanced_features.get("roof_complexity", 0)
                    if complexity > 0:
                        report.append(f"  Complexity Score: {complexity:.3f}")
                    
                    edge_quality = advanced_features.get("edge_quality", 0)
                    if edge_quality > 0:
                        report.append(f"  Edge Quality: {edge_quality:.3f}")
                    
                    uncertainty = advanced_features.get("uncertainty_score", 0)
                    if uncertainty > 0:
                        report.append(f"  Uncertainty Score: {uncertainty:.3f}")
                
                report.append("")
            
            # Object Detection Section
            detection_data = ai_results.get("object_detection", {})
            if detection_data:
                report.append("━" * 80)
                report.append("STEP 2: OBJECT DETECTION ANALYSIS")
                report.append("━" * 80)
                report.append(f"Model Used: {detection_data.get('model_used', 'N/A')}")
                report.append(f"Total Obstacles Detected: {detection_data.get('total_obstacles', 0)}")
                
                obstacle_types = detection_data.get('obstacle_types', [])
                if obstacle_types:
                    report.append(f"Obstacle Types: {', '.join(obstacle_types)}")
                else:
                    report.append("Obstacle Types: None detected")
                
                detected_objects = detection_data.get('detected_objects', [])
                if detected_objects:
                    report.append("")
                    report.append("Detected Objects:")
                    for i, obj in enumerate(detected_objects[:10], 1):  # Show first 10
                        report.append(f"  {i}. {obj.get('type', 'unknown').upper()}")
                        report.append(f"     Confidence: {obj.get('confidence', 0):.2%}")
                        report.append(f"     Area: {obj.get('area', 0):,} pixels")
                        bbox = obj.get('bbox', [])
                        if bbox:
                            report.append(f"     Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                
                stats = detection_data.get('statistics', {})
                if stats:
                    report.append("")
                    report.append("Detection Statistics:")
                    report.append(f"  Total Area Covered: {stats.get('total_area_covered_pixels', 0):,} pixels")
                    report.append(f"  Average Confidence: {stats.get('average_confidence', 0):.3f}")
                    report.append(f"  High Confidence Detections: {stats.get('high_confidence_obstacles', 0)}")
                
                report.append(f"Processing Time: {detection_data.get('processing_time_seconds', 0):.2f}s")
                report.append("")
            
            # Zone Optimization Section
            zone_data = ai_results.get("zone_optimization", {})
            if zone_data:
                report.append("━" * 80)
                report.append("STEP 3: ZONE OPTIMIZATION ANALYSIS")
                report.append("━" * 80)
                report.append(f"Clean Zones Found: {zone_data.get('clean_zones_found', 0)}")
                report.append(f"Total Clean Area: {zone_data.get('total_clean_area_pixels', 0):,} pixels")
                report.append(f"Roof Coverage: {zone_data.get('roof_coverage_percentage', 0):.2f}%")
                report.append(f"Obstacle Coverage: {zone_data.get('obstacle_coverage_percentage', 0):.2f}%")
                report.append(f"Usable Roof Percentage: {zone_data.get('usable_roof_percentage', 0):.2f}%")
                
                optimal_zones = zone_data.get('optimal_zones', [])
                if optimal_zones:
                    report.append("")
                    report.append("Optimal Solar Zones:")
                    for zone in optimal_zones:
                        report.append(f"  Zone {zone.get('id', 'N/A')}:")
                        report.append(f"    Area: {zone.get('area_pixels', 0):,} pixels")
                        report.append(f"    Dimensions: {zone.get('width_pixels', 0)} × {zone.get('height_pixels', 0)} pixels")
                        report.append(f"    Aspect Ratio: {zone.get('aspect_ratio', 0):.2f}")
                        report.append(f"    Suitability Score: {zone.get('suitability_score', 0):.2f}/1.0")
                        report.append(f"    Orientation: {zone.get('orientation', 'N/A')}")
                        panels = zone.get('estimated_panels', {})
                        if panels:
                            report.append(f"    Estimated Panels: {panels.get('estimated_count', 0)}")
                            report.append(f"    Panel Layout: {panels.get('panel_layout', 'N/A')}")
                            report.append(f"    Total Area: {panels.get('total_area_m2', 0):.2f} m²")
                else:
                    report.append("")
                    report.append("⚠️  No optimal zones found. Roof may have too many obstacles.")
                
                zone_stats = zone_data.get('zone_statistics', {})
                if zone_stats:
                    report.append("")
                    report.append("Zone Statistics:")
                    report.append(f"  Total Zones: {zone_stats.get('total_zones', 0)}")
                    report.append(f"  Average Zone Area: {zone_stats.get('average_zone_area_pixels', 0):,} pixels")
                    report.append(f"  Total Estimated Panels: {zone_stats.get('total_estimated_panels', 0)}")
                    report.append(f"  Best Zone Suitability: {zone_stats.get('best_zone_suitability', 0):.2f}/1.0")
                
                report.append(f"Processing Time: {zone_data.get('processing_time_seconds', 0):.2f}s")
                report.append("")
            
            # Solar Optimization Section
            solar_data = ai_results.get("solar_optimization", {})
            if solar_data:
                report.append("━" * 80)
                report.append("STEP 4: SOLAR PANEL OPTIMIZATION ANALYSIS")
                report.append("━" * 80)
                report.append(f"Total Panels Placed: {solar_data.get('total_panels_placed', 0)}")
                
                energy_calc = solar_data.get('energy_calculation', {})
                if energy_calc:
                    report.append("")
                    report.append("ENERGY CALCULATIONS:")
                    report.append(f"  Total System Power: {energy_calc.get('total_power_kw', 0):.2f} kW")
                    report.append(f"  Estimated Yearly Energy: {energy_calc.get('estimated_yearly_kwh', 0):,} kWh/year")
                    report.append(f"  System Efficiency: {energy_calc.get('system_efficiency', 0):.0%}")
                    report.append(f"  Performance Ratio: {energy_calc.get('performance_ratio', 0):.0%}")
                    report.append(f"  Daily Irradiance: {energy_calc.get('daily_irradiance_kwh_m2', 0):.2f} kWh/m²/day")
                    report.append("")
                    report.append("FINANCIAL ANALYSIS:")
                    report.append(f"  Estimated System Cost: ₹{energy_calc.get('estimated_system_cost', 0):,}")
                    report.append(f"  Annual Cost Savings: ₹{energy_calc.get('estimated_cost_savings_annual', 0):,}")
                    report.append(f"  Payback Period: {energy_calc.get('payback_period_years', 0):.1f} years")
                    report.append(f"  ROI: {energy_calc.get('roi_percentage', 0):.1f}%")
                    report.append(f"  Carbon Savings: {energy_calc.get('carbon_savings_kg_co2_year', 0):,} kg CO₂/year")
                
                layout_eff = solar_data.get('layout_efficiency', {})
                if layout_eff:
                    report.append("")
                    report.append("LAYOUT EFFICIENCY:")
                    report.append(f"  Space Utilization: {layout_eff.get('space_utilization', 0):.0%}")
                    report.append(f"  Power Density: {layout_eff.get('power_density_kw_per_m2', 0):.3f} kW/m²")
                    report.append(f"  Panels per m²: {layout_eff.get('panels_per_m2', 0):.2f}")
                
                report.append(f"Processing Time: {solar_data.get('processing_time_seconds', 0):.2f}s")
                report.append("")
            
            # Processing Summary
            summary = ai_results.get("processing_summary", {})
            if summary:
                report.append("━" * 80)
                report.append("PROCESSING SUMMARY")
                report.append("━" * 80)
                report.append(f"Total Processing Time: {summary.get('total_time_seconds', 0):.2f} seconds")
                report.append(f"Pipeline Steps: {summary.get('pipeline_steps', 0)}")
                models = summary.get('ai_models_used', [])
                if models:
                    report.append(f"AI Models Used: {', '.join(models)}")
                report.append(f"Status: {summary.get('status', 'N/A')}")
                report.append("")
            
            # CAD Analysis Section (from main response)
            cad_analysis = roof_analysis.get("cad_analysis", {})
            if cad_analysis:
                report.append("━" * 80)
                report.append("CAD ANALYSIS")
                report.append("━" * 80)
                report.append(f"3D Surface Area: {cad_analysis.get('surface_area_3d', 0):.2f} m²")
                
                optimal_zones_cad = cad_analysis.get('optimal_zones', [])
                if optimal_zones_cad:
                    report.append(f"Optimal Zones: {len(optimal_zones_cad)}")
                else:
                    report.append("Optimal Zones: None")
                
                solar_panels_cad = cad_analysis.get('solar_panels_3d', [])
                if solar_panels_cad:
                    report.append(f"3D Solar Panels: {len(solar_panels_cad)}")
                else:
                    report.append("3D Solar Panels: None")
                
                structural = cad_analysis.get('structural_analysis', {})
                if structural:
                    report.append("")
                    report.append("STRUCTURAL ANALYSIS:")
                    report.append(f"  Safety Factor: {structural.get('safety_factor', 0):.2f}")
                    report.append(f"  Structural Integrity: {structural.get('structural_integrity', 'N/A')}")
                report.append("")
            
            # Overall Summary
            report.append("=" * 80)
            report.append("OVERALL SUMMARY")
            report.append("=" * 80)
            report.append(f"Roof Suitability Score: {roof_analysis.get('suitability_score', 0):.1f}/10")
            report.append(f"Surface Area: {roof_analysis.get('surface_area', 0):.2f} m²")
            report.append(f"Estimated Energy Production: {roof_analysis.get('estimated_energy', 0):,} kWh/year")
            report.append(f"Estimated System Cost: ₹{roof_analysis.get('estimated_cost', 0):,}")
            report.append(f"Payback Period: {roof_analysis.get('payback_period', 0):.1f} years")
            report.append("")
            report.append("=" * 80)
            report.append("End of Report")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"

    def generate_json_report(self, analysis_result: Dict) -> Dict:
        """Generate a structured JSON report"""
        try:
            roof_analysis = analysis_result.get("roof_analysis", {})
            ai_results = roof_analysis.get("ai_pipeline_results", {})
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "AI Solar Rooftop Analysis"
                },
                "roof_analysis": self._format_roof_analysis(ai_results.get("roof_analysis", {})),
                "object_detection": self._format_object_detection(ai_results.get("object_detection", {})),
                "zone_optimization": self._format_zone_optimization(ai_results.get("zone_optimization", {})),
                "solar_optimization": self._format_solar_optimization(ai_results.get("solar_optimization", {})),
                "cad_analysis": self._format_cad_analysis(roof_analysis.get("cad_analysis", {})),
                "summary": self._format_summary(roof_analysis, ai_results.get("processing_summary", {}))
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return {"error": str(e)}

    def _format_roof_analysis(self, data: Dict) -> Dict:
        result = {
            "model_used": data.get("model_used", "N/A"),
            "roof_detected": data.get("roof_detected", False),
            "roof_area_pixels": data.get("roof_area_pixels", 0),
            "roof_coverage_percentage": round(data.get("roof_percentage", 0), 2),
            "confidence_score": round(data.get("confidence_score", 0), 3),
            "processing_time_seconds": round(data.get("processing_time_seconds", 0), 2)
        }
        
        # Include advanced features if available
        advanced_features = data.get("advanced_features", {})
        if advanced_features:
            result["advanced_features"] = {
                "ensemble_models": advanced_features.get("ensemble_models", []),
                "scales_analyzed": advanced_features.get("scales_analyzed", []),
                "num_scales": advanced_features.get("num_scales", 0),
                "alpha_scales": advanced_features.get("alpha_scales", []),
                "alpha_blending": advanced_features.get("alpha_blending", False),
                "tta_enabled": advanced_features.get("tta_enabled", False),
                "fusion_method": advanced_features.get("fusion_method", ""),
                "roof_shape": advanced_features.get("roof_shape", ""),
                "roof_complexity": round(advanced_features.get("roof_complexity", 0), 3),
                "edge_quality": round(advanced_features.get("edge_quality", 0), 3),
                "uncertainty_score": round(advanced_features.get("uncertainty_score", 0), 3)
            }
        
        return result

    def _format_object_detection(self, data: Dict) -> Dict:
        return {
            "model_used": data.get("model_used", "N/A"),
            "total_obstacles": data.get("total_obstacles", 0),
            "obstacle_types": data.get("obstacle_types", []),
            "detected_objects": data.get("detected_objects", []),
            "statistics": data.get("statistics", {}),
            "processing_time_seconds": round(data.get("processing_time_seconds", 0), 2)
        }

    def _format_zone_optimization(self, data: Dict) -> Dict:
        return {
            "clean_zones_found": data.get("clean_zones_found", 0),
            "total_clean_area_pixels": data.get("total_clean_area_pixels", 0),
            "usable_roof_percentage": round(data.get("usable_roof_percentage", 0), 2),
            "optimal_zones": data.get("optimal_zones", []),
            "zone_statistics": data.get("zone_statistics", {}),
            "processing_time_seconds": round(data.get("processing_time_seconds", 0), 2)
        }

    def _format_solar_optimization(self, data: Dict) -> Dict:
        return {
            "total_panels_placed": data.get("total_panels_placed", 0),
            "energy_calculation": data.get("energy_calculation", {}),
            "layout_efficiency": data.get("layout_efficiency", {}),
            "processing_time_seconds": round(data.get("processing_time_seconds", 0), 2)
        }

    def _format_cad_analysis(self, data: Dict) -> Dict:
        return {
            "surface_area_3d_m2": round(data.get("surface_area_3d", 0), 2),
            "optimal_zones_count": len(data.get("optimal_zones", [])),
            "solar_panels_3d_count": len(data.get("solar_panels_3d", [])),
            "structural_analysis": data.get("structural_analysis", {})
        }

    def _format_summary(self, roof_analysis: Dict, processing_summary: Dict) -> Dict:
        return {
            "suitability_score": round(roof_analysis.get("suitability_score", 0), 1),
            "surface_area_m2": round(roof_analysis.get("surface_area", 0), 2),
            "estimated_energy_kwh_year": roof_analysis.get("estimated_energy", 0),
            "estimated_cost_inr": roof_analysis.get("estimated_cost", 0),
            "payback_period_years": round(roof_analysis.get("payback_period", 0), 1),
            "total_processing_time_seconds": round(processing_summary.get("total_time_seconds", 0), 2),
            "ai_models_used": processing_summary.get("ai_models_used", []),
            "status": processing_summary.get("status", "N/A")
        }
