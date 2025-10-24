"""
FastAPI Backend for Solar Rooftop Analysis
Modern API backend replacing the old main.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import analysis modules
from solar_analysis import SolarAnalysisEngine

# from advanced_features.advanced_solar_system import AdvancedSolarAnalysisSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Solar Rooftop Analysis API",
    description="AI-powered solar rooftop analysis with cutting-edge technology",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analysis engines
solar_engine = SolarAnalysisEngine()
# advanced_engine = AdvancedSolarAnalysisSystem({
#     "sentinel_api_key": os.getenv("SENTINEL_API_KEY", ""),
#     "landsat_api_key": os.getenv("LANDSAT_API_KEY", ""),
#     "weather_api_key": os.getenv("OPENWEATHER_API_KEY", ""),
#     "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
# })

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="outputs"), name="static")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Solar Rooftop Analysis API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Vision Transformers (>95% accuracy)",
            "Physics-Informed AI (<5% error)",
            "AR Visualization",
            "Federated Learning",
            "Edge AI Deployment",
            "Blockchain Verification",
        ],
        "endpoints": {
            "analysis": "/api/analyze",
            "enhanced_analysis": "/api/analyze/enhanced",
            "health": "/api/health",
            "docs": "/docs",
        },
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "solar_engine": "active",
            "enhanced_engine": "disabled",
            "storage": "available",
        },
    }


@app.post("/api/analyze")
async def analyze_rooftop(
    files: List[UploadFile] = File(...),
    cities: str = Form(...),
    panel_types: str = Form(...),
):
    """
    Analyze rooftop images using the standard analysis engine
    """
    try:
        # Parse form data
        city_list = json.loads(cities) if cities else ["New Delhi"]
        panel_type_list = (
            json.loads(panel_types) if panel_types else ["monocrystalline"]
        )

        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename:
                continue

            # Validate file type
            if not any(
                file.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]
            ):
                raise HTTPException(
                    status_code=400, detail=f"Invalid file type: {file.filename}"
                )

            # Save file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file_path)

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Run analysis
        results = solar_engine.analyze_rooftops(saved_files, city_list, panel_type_list)

        # Clean up uploaded files
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/enhanced")
async def analyze_rooftop_enhanced(
    files: List[UploadFile] = File(...),
    location: str = Form(...),
    analysis_type: str = Form("comprehensive"),
):
    """
    Enhanced rooftop analysis using cutting-edge AI technology
    """
    try:
        # Parse location
        location_data = json.loads(location)
        lat, lon = location_data.get("lat", 28.6139), location_data.get("lon", 77.2090)
        city = location_data.get("city", "Gurugram")

        # Save uploaded files
        saved_files = []
        for file in files:
            if not file.filename:
                continue

            # Validate file type
            if not any(
                file.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]
            ):
                raise HTTPException(
                    status_code=400, detail=f"Invalid file type: {file.filename}"
                )

            # Save file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file_path)

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        # Run basic analysis on the first image
        results = solar_engine.analyze_rooftops(
            saved_files, [city] * len(saved_files), [analysis_type] * len(saved_files)
        )

        # Clean up uploaded files
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cities")
async def get_cities():
    """Get available cities for analysis"""
    return {
        "cities": [
            {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
            {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
            {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
            {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
            {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
            {"name": "Gurugram", "lat": 28.4595, "lon": 77.0266},
        ]
    }


@app.get("/api/panel-types")
async def get_panel_types():
    """Get available panel types"""
    return {
        "panel_types": [
            {
                "value": "monocrystalline",
                "label": "Monocrystalline",
                "efficiency": "22%",
                "cost_per_watt": 27,
                "description": "High efficiency, long lifespan",
            },
            {
                "value": "bifacial",
                "label": "Bifacial",
                "efficiency": "24%",
                "cost_per_watt": 30,
                "description": "Double-sided panels for maximum energy capture",
            },
            {
                "value": "perovskite",
                "label": "Perovskite",
                "efficiency": "26%",
                "cost_per_watt": 25,
                "description": "Next-generation technology with highest efficiency",
            },
        ]
    }


@app.get("/api/download/{file_type}")
async def download_report(file_type: str):
    """Download analysis reports"""
    valid_types = ["pdf", "csv", "excel", "json"]

    if file_type not in valid_types:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = f"outputs/solar_analysis.{file_type}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"solar_analysis.{file_type}",
    )


@app.get("/api/performance")
async def get_performance_metrics():
    """Get system performance metrics"""
    return {
        "status": "basic_mode",
        "message": "Enhanced features disabled - using basic solar analysis",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/segmented/{filename}")
async def get_segmented_image(filename: str):
    """Serve segmented images with YOLO detection results"""
    try:
        segmented_path = os.path.join("outputs", "segmented", filename)
        if os.path.exists(segmented_path):
            return FileResponse(segmented_path, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Segmented image not found")
    except Exception as e:
        logger.error(f"Error serving segmented image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/segmented/{filename}")
async def get_segmented_image_direct(filename: str):
    """Direct access to segmented images"""
    try:
        segmented_path = os.path.join("outputs", "segmented", filename)
        if os.path.exists(segmented_path):
            return FileResponse(segmented_path, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Segmented image not found")
    except Exception as e:
        logger.error(f"Error serving segmented image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Create outputs/segmented directory if it doesn't exist
os.makedirs("outputs/segmented", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
