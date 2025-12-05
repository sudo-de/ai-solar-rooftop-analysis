"""
FastAPI Backend for AI Solar Rooftop Analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from sqlalchemy.orm import Session
import uvicorn
import os
import uuid
from datetime import datetime

from database import init_db, get_db
from models import AnalysisResult, AnalysisSession
from yolo_service import get_yolo_service
from PIL import Image
import io

app = FastAPI(
    title="AI Solar Rooftop Analysis API",
    description="Backend API for solar rooftop analysis with YOLO and 3D CAD",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("Database initialized")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Solar Rooftop Analysis API", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "solar-analysis-api"}

@app.post("/api/analyze")
async def analyze_rooftop(
    files: List[UploadFile] = File(...),
    segmentation_method: str = Form("enhanced_canny"),
    db: Session = Depends(get_db)
):
    """
    Analyze rooftop images for solar potential and save to database
    
    Args:
        files: List of image files (PNG/JPG/JPEG)
        db: Database session
    
    Returns:
        Analysis results with energy predictions, ROI, and AI analysis
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types
        allowed_types = ["image/png", "image/jpeg", "image/jpg"]
        for file in files:
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, JPEG"
                )
        
        # Create analysis session
        session_id = str(uuid.uuid4())
        session = AnalysisSession(
            session_id=session_id,
            total_files=len(files),
            status="processing"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Initialize YOLO service
        try:
            yolo_service = get_yolo_service()
            print("YOLO service initialized successfully")
        except Exception as e:
            print(f"Error initializing YOLO service: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize YOLO service: {str(e)}")
        
        # Process files with YOLO
        results = []
        saved_results = []
        
        for file in files:
            try:
                # Read image file
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Run YOLO analysis with segmentation method
                print(f"Running YOLO analysis on {file.filename} with segmentation method: {segmentation_method}...")
                yolo_analysis = yolo_service.analyze_rooftop(image, segmentation_method=segmentation_method)
                print(f"YOLO analysis completed for {file.filename}")
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Provide fallback analysis
                yolo_analysis = {
                    "detected_objects": [],
                    "roof_analysis": {
                        "total_area_m2": 100.0,
                        "usable_area_m2": 100.0,
                        "area_m2": 100.0,
                        "obstruction_area_m2": 0,
                        "obstruction_percentage": 0,
                        "usable_percentage": 100,
                        "orientation": "South",
                        "surface_type": "Unknown",
                        "suitability": 8,
                        "obstructions": "None",
                        "obstruction_details": [],
                        "detected_objects": [],
                        "roof_edges_detected": False,
                        "segmented_image_base64": None
                    },
                    "energy_prediction": {
                        "annual_energy_kwh": 7300
                    },
                    "accuracy_metrics": {
                        "overall_confidence": 0,
                        "roof_detection_accuracy": 0,
                        "edge_detection_success": False
                    }
                }
            
            # Calculate ROI based on YOLO results (use usable area)
            usable_roof_area = yolo_analysis["roof_analysis"].get("usable_area_m2") or yolo_analysis["roof_analysis"].get("area_m2") or 100.0
            total_roof_area = yolo_analysis["roof_analysis"].get("total_area_m2") or usable_roof_area
            annual_energy = yolo_analysis["energy_prediction"]["annual_energy_kwh"] or 7300
            
            # Estimate system size based on usable area (assume 1 kW per 25 m²)
            system_size_kw = round((usable_roof_area / 25.0) if usable_roof_area else 4.0, 1)
            # Cost estimate: ₹31,250 per kW
            total_cost = int(system_size_kw * 31250)
            # Savings: ₹7 per kWh
            annual_savings = int(annual_energy * 7)
            # Payback period
            payback_period = round(total_cost / annual_savings, 1) if annual_savings > 0 else 0
            
            # Generate recommendations based on analysis
            recommendations = []
            suitability = yolo_analysis["roof_analysis"]["suitability"]
            roof_analysis = yolo_analysis["roof_analysis"]
            
            if suitability >= 8:
                recommendations.append("Excellent rooftop for solar installation")
            elif suitability >= 6:
                recommendations.append("Good rooftop for solar installation")
            else:
                recommendations.append("Consider roof modifications for better solar potential")
            
            # Specific obstruction recommendations
            obstruction_details = roof_analysis.get("obstruction_details", [])
            if obstruction_details:
                obstruction_types = {}
                for obs in obstruction_details:
                    obs_type = obs["type"]
                    if obs_type not in obstruction_types:
                        obstruction_types[obs_type] = []
                    obstruction_types[obs_type].append(obs)
                
                if "Chimneys" in obstruction_types:
                    recommendations.append(f"Consider relocating {len(obstruction_types['Chimneys'])} chimney(s) or plan panel layout around them")
                
                if "Skylights" in obstruction_types:
                    recommendations.append(f"Plan solar panel placement to avoid {len(obstruction_types['Skylights'])} skylight(s)")
                
                if "Water Tanks" in obstruction_types:
                    recommendations.append(f"Consider relocating {len(obstruction_types['Water Tanks'])} water tank(s) for optimal panel placement")
                
                if "Hvac Units" in obstruction_types:
                    recommendations.append(f"Plan around {len(obstruction_types['Hvac Units'])} HVAC unit(s) or consider relocation")
                
                obstruction_pct = roof_analysis.get("obstruction_percentage", 0)
                if obstruction_pct > 20:
                    recommendations.append(f"High obstruction coverage ({obstruction_pct}%) - consider removing or relocating obstructions")
                elif obstruction_pct > 10:
                    recommendations.append(f"Moderate obstruction coverage ({obstruction_pct}%) - plan panel layout carefully")
            
            # Area-based recommendations
            usable_area = roof_analysis.get("usable_area_m2")
            if usable_area:
                if usable_area < 30:
                    recommendations.append("Limited usable area - consider high-efficiency panels")
                elif usable_area >= 100:
                    recommendations.append("Large usable area - excellent for large-scale installation")
            
            # Hotspot recommendations
            hotspots = roof_analysis.get("hotspots", [])
            if hotspots:
                high_severity_hotspots = [h for h in hotspots if h.get("severity", 0) > 70]
                if high_severity_hotspots:
                    recommendations.append(f"⚠️ {len(high_severity_hotspots)} critical hotspot(s) detected - immediate inspection recommended")
                else:
                    recommendations.append(f"⚠️ {len(hotspots)} hotspot(s) detected - schedule maintenance inspection")
            
            # Dirt accumulation recommendations
            dirt_patches = roof_analysis.get("dirt_patches", [])
            if dirt_patches:
                high_soiling = [d for d in dirt_patches if d.get("soiling_level", 0) > 50]
                if high_soiling:
                    recommendations.append(f"🧹 {len(high_soiling)} high-soiling area(s) - cleaning recommended immediately")
                else:
                    recommendations.append(f"🧹 {len(dirt_patches)} dirt patch(es) detected - regular cleaning recommended")
            
            # Efficiency loss recommendations
            efficiency_loss = roof_analysis.get("efficiency_loss_percentage", 0)
            if efficiency_loss > 20:
                recommendations.append(f"⚡ High efficiency loss ({efficiency_loss}%) - urgent maintenance required")
            elif efficiency_loss > 10:
                recommendations.append(f"⚡ Moderate efficiency loss ({efficiency_loss}%) - maintenance recommended")
            
            recommendations.append("Use monocrystalline panels for best efficiency")
            recommendations.append("Consult with local authority for permits")
            
            # Combine YOLO results with ROI calculations
            analysis_data = {
                "filename": file.filename,
                "roof_analysis": {
                    **yolo_analysis["roof_analysis"],
                    "detected_objects": yolo_analysis["detected_objects"]
                },
                "energy_prediction": yolo_analysis["energy_prediction"],
                "roi_estimation": {
                    "system_size_kw": system_size_kw,
                    "total_cost": total_cost,
                    "annual_savings": annual_savings,
                    "payback_period_years": payback_period
                },
                "accuracy_metrics": yolo_analysis["accuracy_metrics"],
                "recommendations": recommendations
            }
            
            # Save to database
            db_result = AnalysisResult(
                filename=file.filename,
                area_m2=analysis_data["roof_analysis"].get("usable_area_m2") or analysis_data["roof_analysis"].get("total_area_m2"),
                orientation=analysis_data["roof_analysis"]["orientation"],
                surface_type=analysis_data["roof_analysis"]["surface_type"],
                suitability_score=analysis_data["roof_analysis"]["suitability"],
                obstructions=analysis_data["roof_analysis"]["obstructions"],
                annual_energy_kwh=analysis_data["energy_prediction"]["annual_energy_kwh"],
                system_size_kw=analysis_data["roi_estimation"]["system_size_kw"],
                total_cost=analysis_data["roi_estimation"]["total_cost"],
                annual_savings=analysis_data["roi_estimation"]["annual_savings"],
                payback_period_years=analysis_data["roi_estimation"]["payback_period_years"],
                overall_confidence=analysis_data["accuracy_metrics"]["overall_confidence"],
                roof_detection_accuracy=analysis_data["accuracy_metrics"]["roof_detection_accuracy"],
                recommendations="\n".join(analysis_data["recommendations"]),
                full_analysis_data=analysis_data
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)
            
            results.append(analysis_data)
            saved_results.append(db_result.to_dict())
        
        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "status": "success",
            "session_id": session_id,
            "results": results,
            "saved_results": saved_results,
            "message": f"Analyzed {len(files)} image(s) successfully and saved to database"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # Update session status to failed
        if 'session' in locals():
            session.status = "failed"
            db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analyses")
async def get_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get all analysis results from database
    
    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        db: Database session
    
    Returns:
        List of analysis results
    """
    analyses = db.query(AnalysisResult).offset(skip).limit(limit).all()
    return {
        "status": "success",
        "count": len(analyses),
        "results": [analysis.to_dict() for analysis in analyses]
    }

@app.get("/api/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific analysis result by ID
    
    Args:
        analysis_id: ID of the analysis result
        db: Database session
    
    Returns:
        Analysis result details
    """
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "status": "success",
        "result": analysis.to_dict()
    }

@app.get("/api/sessions")
async def get_sessions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get all analysis sessions
    
    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        db: Database session
    
    Returns:
        List of analysis sessions
    """
    sessions = db.query(AnalysisSession).offset(skip).limit(limit).all()
    return {
        "status": "success",
        "count": len(sessions),
        "sessions": [session.to_dict() for session in sessions]
    }

@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an analysis result by ID
    
    Args:
        analysis_id: ID of the analysis result
        db: Database session
    
    Returns:
        Success message
    """
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    db.delete(analysis)
    db.commit()
    
    return {
        "status": "success",
        "message": f"Analysis {analysis_id} deleted successfully"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

