"""
Database models for Solar Rooftop Analysis
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from database import Base

class AnalysisResult(Base):
    """
    Model for storing rooftop analysis results
    """
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(500), nullable=True)
    
    # Roof Analysis Data
    area_m2 = Column(Float, nullable=True)
    orientation = Column(String(50), nullable=True)
    surface_type = Column(String(50), nullable=True)
    suitability_score = Column(Integer, nullable=True)
    obstructions = Column(Text, nullable=True)
    
    # Energy Prediction
    annual_energy_kwh = Column(Float, nullable=True)
    
    # ROI Estimation
    system_size_kw = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    annual_savings = Column(Float, nullable=True)
    payback_period_years = Column(Float, nullable=True)
    
    # Accuracy Metrics
    overall_confidence = Column(Integer, nullable=True)
    roof_detection_accuracy = Column(Integer, nullable=True)
    
    # Full analysis data as JSON
    full_analysis_data = Column(JSON, nullable=True)
    
    # Recommendations
    recommendations = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self):
        """
        Convert model instance to dictionary
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "roof_analysis": {
                "area_m2": self.area_m2,
                "orientation": self.orientation,
                "surface_type": self.surface_type,
                "suitability": self.suitability_score,
                "obstructions": self.obstructions,
            },
            "energy_prediction": {
                "annual_energy_kwh": self.annual_energy_kwh,
            },
            "roi_estimation": {
                "system_size_kw": self.system_size_kw,
                "total_cost": self.total_cost,
                "annual_savings": self.annual_savings,
                "payback_period_years": self.payback_period_years,
            },
            "accuracy_metrics": {
                "overall_confidence": self.overall_confidence,
                "roof_detection_accuracy": self.roof_detection_accuracy,
            },
            "recommendations": self.recommendations.split("\n") if self.recommendations else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class AnalysisSession(Base):
    """
    Model for storing analysis sessions (multiple files analyzed together)
    """
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    total_files = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self):
        """
        Convert model instance to dictionary
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "total_files": self.total_files,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

