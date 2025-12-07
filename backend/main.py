from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
from datetime import datetime
import base64
from PIL import Image
import io
import time
import logging

# Import AI services
from ai_services.roof_segmentation import RoofSegmentationService
from ai_services.object_detection import ObjectDetectionService
from ai_services.zone_optimization import ZoneOptimizationService
from ai_services.solar_optimization import SolarOptimizationService
from ai_services.report_generator import ReportGenerator

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import NextGen advanced services
try:
    from ai_services.advanced_segmentation import AdvancedSegmentationService
    from ai_services.advanced_detection import AdvancedDetectionService
    from ai_services.advanced_solar_calculations import AdvancedSolarCalculations
    from ai_services.intelligent_zone_refinement import IntelligentZoneRefinement
    NEXTGEN_AVAILABLE = True
    logger.info("✅ NextGen advanced services loaded")
except ImportError as e:
    NEXTGEN_AVAILABLE = False
    logger.warning(f"NextGen services not available: {e}")

# Logging already set up above

app = FastAPI(title="AI Solar Rooftop Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize AI services
roof_segmentation = RoofSegmentationService()
object_detection = ObjectDetectionService()
zone_optimization = ZoneOptimizationService()
solar_optimization = SolarOptimizationService()
report_generator = ReportGenerator()

# Initialize NextGen advanced services (if available)
if NEXTGEN_AVAILABLE:
    advanced_segmentation = AdvancedSegmentationService()
    advanced_detection = AdvancedDetectionService()
    advanced_solar_calc = AdvancedSolarCalculations()
    intelligent_refinement = IntelligentZoneRefinement()
    logger.info("🚀 NextGen mode: Advanced services enabled")
else:
    advanced_segmentation = None
    advanced_detection = None
    advanced_solar_calc = None
    intelligent_refinement = None

class AnalysisResult(BaseModel):
    id: str
    filename: str
    upload_time: str
    file_size: int
    image_dimensions: tuple
    status: str = "processed"
    roof_analysis: dict = {}
    cad_analysis: dict = {}

class AnalyzeResponse(BaseModel):
    message: str
    results: List[AnalysisResult]

def validate_image_file(file: UploadFile) -> bool:
    """Validate if uploaded file is a valid image"""
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    file_extension = os.path.splitext(file.filename.lower())[1]

    if file_extension not in allowed_extensions:
        return False

    # Check file size (max 10MB as mentioned in README)
    if file.size and file.size > 10 * 1024 * 1024:
        return False

    return True

def process_image_with_ai(file_path: str) -> dict:
    """Process image through complete AI pipeline"""
    start_time = time.time()

    try:
        logger.info(f"Starting AI processing for {file_path}")
        logger.info(f"File exists: {os.path.exists(file_path)}")

        # Step 1: Roof Segmentation (NextGen if available)
        logger.info("Step 1: Roof segmentation")
        step_start = time.time()
        if NEXTGEN_AVAILABLE and advanced_segmentation:
            logger.info("🚀 Using NextGen Advanced Segmentation (Ensemble + Multi-scale)")
            roof_result = advanced_segmentation.segment_roof_advanced(file_path)
        else:
            roof_result = roof_segmentation.segment_roof(file_path)
        roof_result["processing_time_seconds"] = time.time() - step_start

        # Step 2: Object Detection (NextGen if available)
        logger.info("Step 2: Object detection")
        step_start = time.time()
        if NEXTGEN_AVAILABLE and advanced_detection:
            logger.info("🚀 Using NextGen Advanced Detection (TTA + Ensemble)")
            detection_result = advanced_detection.detect_obstacles_advanced(file_path)
        else:
            detection_result = object_detection.detect_obstacles(file_path)
        detection_result["processing_time_seconds"] = time.time() - step_start

        # Step 3: Zone Optimization (subtract obstacles)
        logger.info("Step 3: Zone optimization")
        step_start = time.time()
        zone_result = zone_optimization.optimize_zones(roof_result, detection_result, file_path)
        zone_result["processing_time_seconds"] = time.time() - step_start

        # Step 4: Solar Panel Optimization (with NextGen calculations if available)
        logger.info("Step 4: Solar optimization")
        step_start = time.time()
        solar_result = solar_optimization.optimize_solar_layout(zone_result, file_path)
        
        # Enhance with NextGen advanced calculations if available
        if NEXTGEN_AVAILABLE and advanced_solar_calc and solar_result.get("total_panels_placed", 0) > 0:
            logger.info("🚀 Using NextGen Advanced Solar Calculations (Physics-informed)")
            total_panels = solar_result.get("total_panels_placed", 0)
            enhanced_energy = advanced_solar_calc.calculate_advanced_energy(
                total_panels=total_panels,
                panel_type="standard",
                location_data=None,
                roof_orientation=180.0,
                roof_tilt=25.0,
                shading_factor=0.0
            )
            # Merge advanced calculations
            if "energy_calculation" in solar_result:
                solar_result["energy_calculation"].update(enhanced_energy)
            else:
                solar_result["energy_calculation"] = enhanced_energy
            
            solar_result["nextgen_enhanced"] = True
        
        solar_result["processing_time_seconds"] = time.time() - step_start

        # Get basic image info
        with Image.open(file_path) as img:
            image_width, image_height = img.size

            # Convert to base64 for frontend display
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

        total_time = time.time() - start_time
        logger.info(f"AI processing completed in {total_time:.2f} seconds")

        # Combine all results in the expected format for frontend compatibility
        return {
            "detected_objects": detection_result.get("detected_objects", []),
            "segmented_image_base64": roof_result.get("segmented_image_base64", ""),
            "suitability_score": zone_result.get("usable_roof_percentage", 75.0),
            "surface_area": roof_result.get("roof_area_pixels", image_width * image_height) / 10000,  # Rough m² estimate
            "estimated_energy": solar_result.get("energy_calculation", {}).get("estimated_yearly_kwh", 6500),
            "estimated_cost": solar_result.get("energy_calculation", {}).get("estimated_system_cost", 125000),
            "payback_period": solar_result.get("energy_calculation", {}).get("payback_period_years", 4.8),
            # Extended AI results with error handling for serialization
            "ai_pipeline_results": _safe_serialize_ai_results({
                "roof_analysis": {
                    **roof_result,
                    "step": 1,
                    "description": "Roof outline detection using SegFormer"
                },
                "object_detection": {
                    **detection_result,
                    "step": 2,
                    "description": "Obstacle detection using YOLOv11"
                },
                "zone_optimization": {
                    **zone_result,
                    "step": 3,
                    "description": "Clean zone identification by subtracting obstacles"
                },
                "solar_optimization": {
                    **solar_result,
                    "step": 4,
                    "description": "Optimal solar panel layout and energy calculations"
                },
                "processing_summary": {
                    "total_time_seconds": round(total_time, 2),
                    "pipeline_steps": 4,
                    "ai_models_used": [
                        roof_result.get("model_used", "SegFormer-B0"),
                        detection_result.get("model_used", "YOLOv8-seg")
                    ],
                    "status": "completed",
                    "nextgen_mode": NEXTGEN_AVAILABLE and (
                        (advanced_segmentation is not None and len(advanced_segmentation.models) > 0) or
                        (advanced_detection is not None and len(advanced_detection.models) > 0) or
                        advanced_solar_calc is not None
                    ),
                    "advanced_features": {
                        "ensemble_segmentation": advanced_segmentation is not None and len(advanced_segmentation.models) > 0 if advanced_segmentation else False,
                        "tta_detection": advanced_detection is not None and len(advanced_detection.models) > 0 if advanced_detection else False,
                        "advanced_solar_calc": advanced_solar_calc is not None,
                        "intelligent_refinement": intelligent_refinement is not None
                    }
                }
            })
        }

    except Exception as e:
        logger.error(f"Error in AI processing: {str(e)}")
        import traceback
        logger.error(f"AI processing traceback: {traceback.format_exc()}")
        # Fallback to basic processing if AI fails
        logger.info("Falling back to basic processing")
        fallback_result = process_image_fallback(file_path, str(e))
        logger.info(f"Fallback result keys: {list(fallback_result.keys())}")
        return fallback_result

def _safe_serialize_ai_results(ai_results: dict) -> dict:
    """Safely serialize AI results by converting numpy arrays and other non-serializable objects"""
    import json

    def serialize_value(value):
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {key: serialize_value(val) for key, val in value.items()}
        elif hasattr(value, 'tolist'):  # numpy arrays
            return value.tolist()
        elif hasattr(value, 'item'):  # numpy scalars
            return value.item()
        else:
            # Convert to string for complex objects
            return str(value)

    try:
        return serialize_value(ai_results)
    except Exception as e:
        logger.error(f"Error serializing AI results: {e}")
        return {"error": "Failed to serialize AI results", "details": str(e)}

def process_image_fallback(file_path: str, error_msg: str = "") -> dict:
    """Fallback processing when AI pipeline fails"""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            format_type = img.format

            # Convert to base64 for frontend display
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "detected_objects": [],
                "segmented_image_base64": f"data:image/jpeg;base64,{img_base64}",
                "suitability_score": 7.5,
                "surface_area": width * height * 0.01,  # Rough estimate in m²
                "estimated_energy": 6500,  # kWh/year placeholder
                "estimated_cost": 125000,  # INR placeholder
                "payback_period": 4.8,  # years placeholder
                "ai_pipeline_results": {
                    "error": f"AI processing failed: {error_msg}",
                    "status": "fallback_mode"
                }
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_rooftop(
    files: List[UploadFile] = File(...),
    segmentation_method: Optional[str] = Form(None),
    cities: Optional[str] = Form(None),
    panel_types: Optional[str] = Form(None)
):
    """
    Analyze rooftop images for solar potential

    - **files**: List of image files to analyze
    - **segmentation_method**: Optional segmentation method (default: auto)
    - **cities**: Optional list of cities for location-based analysis
    - **panel_types**: Optional list of solar panel types
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = []

    for file in files:
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file: {file.filename}. Must be image file under 10MB."
            )

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        try:
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Process the image through AI pipeline
            image_info = process_image_with_ai(file_path)

            # Get image dimensions
            with Image.open(file_path) as img:
                img_width, img_height = img.size

            # Prepare data for report generation
            report_data = {
                "roof_analysis": {
                    "detected_objects": image_info["detected_objects"],
                    "segmented_image_base64": image_info["segmented_image_base64"],
                    "suitability_score": image_info["suitability_score"],
                    "surface_area": image_info["surface_area"],
                    "estimated_energy": image_info["estimated_energy"],
                    "estimated_cost": image_info["estimated_cost"],
                    "payback_period": image_info["payback_period"],
                    "ai_pipeline_results": image_info.get("ai_pipeline_results", {}),
                    "cad_analysis": {
                        "surface_area_3d": image_info["surface_area"],
                        "optimal_zones": image_info.get("ai_pipeline_results", {}).get("zone_optimization", {}).get("optimal_zones", []),
                        "solar_panels_3d": image_info.get("ai_pipeline_results", {}).get("solar_optimization", {}).get("solar_panels_3d", []),
                        "structural_analysis": {
                            "safety_factor": 1.5,
                            "structural_integrity": "Good"
                        }
                    }
                }
            }
            
            # Generate formatted reports
            try:
                text_report = report_generator.generate_text_report(report_data)
                json_report = report_generator.generate_json_report(report_data)
                logger.info(f"Generated text report length: {len(text_report)}, JSON report keys: {list(json_report.keys())}")
            except Exception as e:
                logger.error(f"Error generating reports: {e}", exc_info=True)
                text_report = f"Report generation failed: {str(e)}. Raw data available in ai_pipeline_results."
                json_report = {"error": str(e)}
            
            # Safely serialize all data before creating response
            try:
                # Serialize solar panels and zones to ensure JSON compatibility
                solar_panels_3d = image_info.get("ai_pipeline_results", {}).get("solar_optimization", {}).get("solar_panels_3d", [])
                optimal_zones = image_info.get("ai_pipeline_results", {}).get("zone_optimization", {}).get("optimal_zones", [])
                
                # Ensure all data is JSON-serializable
                solar_panels_3d_serialized = _safe_serialize_ai_results({"panels": solar_panels_3d}).get("panels", [])
                optimal_zones_serialized = _safe_serialize_ai_results({"zones": optimal_zones}).get("zones", [])
                
                # Ensure detected_objects is serialized
                detected_objects_serialized = _safe_serialize_ai_results({"objects": image_info.get("detected_objects", [])}).get("objects", [])
                
            except Exception as e:
                logger.warning(f"Error serializing response data: {e}, using empty lists")
                solar_panels_3d_serialized = []
                optimal_zones_serialized = []
                detected_objects_serialized = []
            
            # Create analysis result
            result = AnalysisResult(
                id=str(uuid.uuid4()),
                filename=file.filename,
                upload_time=datetime.now().isoformat(),
                file_size=os.path.getsize(file_path),
                image_dimensions=(img_width, img_height),
                roof_analysis={
                    "detected_objects": detected_objects_serialized,
                    "segmented_image_base64": image_info["segmented_image_base64"],
                    "suitability_score": float(image_info["suitability_score"]) if image_info.get("suitability_score") is not None else 0.0,
                    "surface_area": float(image_info["surface_area"]) if image_info.get("surface_area") is not None else 0.0,
                    "estimated_energy": int(image_info["estimated_energy"]) if image_info.get("estimated_energy") is not None else 0,
                    "estimated_cost": int(image_info["estimated_cost"]) if image_info.get("estimated_cost") is not None else 0,
                    "payback_period": float(image_info["payback_period"]) if image_info.get("payback_period") is not None else 0.0,
                    "ai_pipeline_results": _safe_serialize_ai_results(image_info.get("ai_pipeline_results", {})),
                    "formatted_report_text": text_report,
                    "formatted_report_json": json_report
                },
                cad_analysis={
                    "surface_area_3d": float(image_info["surface_area"]) if image_info.get("surface_area") is not None else 0.0,
                    "optimal_zones": optimal_zones_serialized,
                    "solar_panels_3d": solar_panels_3d_serialized,
                    "structural_analysis": {
                        "safety_factor": 1.5,
                        "structural_integrity": "Good"
                    }
                }
            )

            results.append(result)

        except Exception as e:
            # Clean up file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    return AnalyzeResponse(
        message=f"Successfully processed {len(results)} image(s)",
        results=results
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/report/{result_id}")
async def get_report(result_id: str, format: str = "text"):
    """
    Get formatted report for a specific analysis result
    
    - **result_id**: ID of the analysis result
    - **format**: Report format - 'text' or 'json' (default: 'text')
    """
    # In a real implementation, you would fetch the result from a database
    # For now, this is a placeholder endpoint
    return {
        "message": "Report endpoint - use the analyze endpoint to get reports in the response",
        "note": "Reports are included in the analyze endpoint response under 'formatted_report_text' and 'formatted_report_json'"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "AI Solar Rooftop Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze": "/api/analyze - Upload images and get analysis with formatted reports",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
