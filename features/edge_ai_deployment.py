"""
Edge AI Deployment for Real-Time Solar Analysis
Lightweight AI models for drone and IoT device deployment
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
from dataclasses import dataclass
import onnx
import onnxruntime as ort
from pathlib import Path

@dataclass
class EdgeDevice:
    """Edge device configuration"""
    device_id: str
    device_type: str  # "drone", "iot_sensor", "mobile"
    compute_capability: str  # "low", "medium", "high"
    memory_limit: int  # MB
    power_limit: float  # Watts
    network_bandwidth: int  # Mbps

@dataclass
class EdgeModel:
    """Edge-optimized model configuration"""
    model_name: str
    model_size: int  # MB
    inference_time: float  # seconds
    accuracy: float
    power_consumption: float  # Watts
    memory_usage: int  # MB

class EdgeModelOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_techniques = {
            "quantization": self._apply_quantization,
            "pruning": self._apply_pruning,
            "distillation": self._apply_distillation,
            "onnx_conversion": self._convert_to_onnx
        }
    
    def optimize_for_edge(self, model: nn.Module, device: EdgeDevice) -> EdgeModel:
        """Optimize model for specific edge device"""
        self.logger.info(f"Optimizing model for {device.device_type} device")
        
        # Apply optimization techniques based on device capabilities
        optimized_model = model
        
        if device.compute_capability == "low":
            # Aggressive optimization for low-power devices
            optimized_model = self._apply_quantization(optimized_model, "int8")
            optimized_model = self._apply_pruning(optimized_model, 0.5)  # 50% pruning
        elif device.compute_capability == "medium":
            # Moderate optimization
            optimized_model = self._apply_quantization(optimized_model, "int16")
            optimized_model = self._apply_pruning(optimized_model, 0.3)  # 30% pruning
        else:
            # Light optimization for high-power devices
            optimized_model = self._apply_quantization(optimized_model, "float16")
            optimized_model = self._apply_pruning(optimized_model, 0.1)  # 10% pruning
        
        # Convert to ONNX for cross-platform deployment
        onnx_model = self._convert_to_onnx(optimized_model)
        
        # Calculate model metrics
        model_metrics = self._calculate_model_metrics(onnx_model, device)
        
        return EdgeModel(
            model_name=f"edge_optimized_{device.device_type}",
            model_size=model_metrics["size_mb"],
            inference_time=model_metrics["inference_time"],
            accuracy=model_metrics["accuracy"],
            power_consumption=model_metrics["power_consumption"],
            memory_usage=model_metrics["memory_usage"]
        )
    
    def _apply_quantization(self, model: nn.Module, precision: str = "int8") -> nn.Module:
        """Apply quantization to reduce model size"""
        if precision == "int8":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif precision == "int16":
            # Static quantization
            model.eval()
            quantized_model = torch.quantization.quantize_static(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint16
            )
        else:  # float16
            # Half precision
            quantized_model = model.half()
        
        return quantized_model
    
    def _apply_pruning(self, model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size"""
        # Simple magnitude-based pruning
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Calculate pruning mask
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity)
                mask = torch.abs(weight) > threshold
                
                # Apply pruning
                module.weight.data *= mask.float()
        
        return model
    
    def _apply_distillation(self, teacher_model: nn.Module, 
                          student_model: nn.Module) -> nn.Module:
        """Apply knowledge distillation to create smaller model"""
        # Simplified distillation - in practice, use proper distillation training
        return student_model
    
    def _convert_to_onnx(self, model: nn.Module) -> str:
        """Convert PyTorch model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Export to ONNX
            onnx_path = "edge_model.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {e}")
            raise
    
    def _calculate_model_metrics(self, onnx_path: str, device: EdgeDevice) -> Dict[str, Any]:
        """Calculate model performance metrics"""
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Calculate model size
        model_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        
        # Simulate inference time based on device capabilities
        base_inference_time = 0.1  # seconds
        if device.compute_capability == "low":
            inference_time = base_inference_time * 3
        elif device.compute_capability == "medium":
            inference_time = base_inference_time * 1.5
        else:
            inference_time = base_inference_time
        
        # Calculate power consumption
        power_consumption = self._estimate_power_consumption(model_size, device)
        
        # Calculate memory usage
        memory_usage = int(model_size * 1.5)  # Model size + overhead
        
        return {
            "size_mb": model_size,
            "inference_time": inference_time,
            "accuracy": 0.92,  # Simulated accuracy
            "power_consumption": power_consumption,
            "memory_usage": memory_usage
        }
    
    def _estimate_power_consumption(self, model_size: float, device: EdgeDevice) -> float:
        """Estimate power consumption for model inference"""
        # Base power consumption
        base_power = 0.5  # Watts
        
        # Power scaling with model size
        size_factor = model_size / 10.0  # Normalize to 10MB
        
        # Device-specific power scaling
        if device.compute_capability == "low":
            power_multiplier = 0.5
        elif device.compute_capability == "medium":
            power_multiplier = 1.0
        else:
            power_multiplier = 2.0
        
        return base_power * size_factor * power_multiplier

class EdgeInferenceEngine:
    """Real-time inference engine for edge devices"""
    
    def __init__(self, model_path: str, device_config: EdgeDevice):
        self.model_path = model_path
        self.device_config = device_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ONNX runtime
        self.session = self._initialize_onnx_session()
        
        # Performance monitoring
        self.inference_times = []
        self.power_consumption = []
    
    def _initialize_onnx_session(self) -> ort.InferenceSession:
        """Initialize ONNX runtime session"""
        try:
            # Configure session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create session
            session = ort.InferenceSession(self.model_path, session_options)
            
            return session
            
        except Exception as e:
            self.logger.error(f"ONNX session initialization failed: {e}")
            raise
    
    def run_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run inference on edge device"""
        start_time = time.time()
        
        try:
            # Preprocess input
            processed_input = self._preprocess_input(input_data)
            
            # Run inference
            outputs = self.session.run(None, {"input": processed_input})
            
            # Postprocess output
            results = self._postprocess_output(outputs[0])
            
            # Record performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return {
                "results": results,
                "inference_time": inference_time,
                "device_id": self.device_config.device_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def _preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input for model inference"""
        # Resize to model input size
        if len(input_data.shape) == 3:
            input_data = cv2.resize(input_data, (224, 224))
            input_data = np.expand_dims(input_data, axis=0)
        
        # Normalize
        input_data = input_data.astype(np.float32) / 255.0
        
        # Transpose for ONNX format (NCHW)
        if input_data.shape[-1] == 3:
            input_data = np.transpose(input_data, (0, 3, 1, 2))
        
        return input_data
    
    def _postprocess_output(self, output: np.ndarray) -> Dict[str, Any]:
        """Postprocess model output"""
        # Apply softmax for classification
        probabilities = torch.softmax(torch.from_numpy(output), dim=1)
        probabilities = probabilities.numpy()
        
        # Get top predictions
        top_indices = np.argsort(probabilities[0])[-5:][::-1]
        top_probabilities = probabilities[0][top_indices]
        
        # Map to class names
        class_names = ["roof_surface", "obstruction_tree", "obstruction_building", 
                      "obstruction_chimney", "solar_panel_area"]
        
        results = {
            "predictions": [
                {"class": class_names[i], "confidence": float(prob)} 
                for i, prob in zip(top_indices, top_probabilities)
            ],
            "top_prediction": class_names[top_indices[0]],
            "confidence": float(top_probabilities[0])
        }
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for edge device"""
        if not self.inference_times:
            return {"status": "no_inferences"}
        
        return {
            "average_inference_time": np.mean(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "total_inferences": len(self.inference_times),
            "device_capability": self.device_config.compute_capability
        }

class DroneSolarAnalyzer:
    """Specialized solar analyzer for drone deployment"""
    
    def __init__(self, edge_model: EdgeModel):
        self.edge_model = edge_model
        self.logger = logging.getLogger(__name__)
        self.flight_data = []
        self.analysis_results = []
    
    def analyze_rooftop_drone(self, image_data: np.ndarray, 
                             gps_coords: Tuple[float, float],
                             altitude: float) -> Dict[str, Any]:
        """Analyze rooftop from drone imagery"""
        try:
            # Initialize edge inference engine
            device_config = EdgeDevice(
                device_id="drone_001",
                device_type="drone",
                compute_capability="medium",
                memory_limit=512,
                power_limit=50.0,
                network_bandwidth=10
            )
            
            inference_engine = EdgeInferenceEngine(self.edge_model.model_name, device_config)
            
            # Run analysis
            results = inference_engine.run_inference(image_data)
            
            # Add drone-specific metadata
            drone_analysis = {
                "gps_coordinates": gps_coords,
                "altitude_meters": altitude,
                "image_resolution": image_data.shape[:2],
                "analysis_results": results["results"],
                "inference_time": results["inference_time"],
                "timestamp": time.time()
            }
            
            # Store flight data
            self.flight_data.append(drone_analysis)
            
            return drone_analysis
            
        except Exception as e:
            self.logger.error(f"Drone analysis failed: {e}")
            raise
    
    def generate_flight_report(self) -> Dict[str, Any]:
        """Generate comprehensive flight report"""
        if not self.flight_data:
            return {"status": "no_flight_data"}
        
        # Aggregate analysis results
        total_rooftops = len(self.flight_data)
        avg_inference_time = np.mean([data["inference_time"] for data in self.flight_data])
        
        # Calculate coverage area
        gps_coords = [data["gps_coordinates"] for data in self.flight_data]
        coverage_area = self._calculate_coverage_area(gps_coords)
        
        return {
            "flight_summary": {
                "total_rooftops_analyzed": total_rooftops,
                "average_inference_time": avg_inference_time,
                "coverage_area_km2": coverage_area,
                "flight_duration": self.flight_data[-1]["timestamp"] - self.flight_data[0]["timestamp"]
            },
            "detailed_results": self.flight_data,
            "performance_metrics": self._calculate_performance_metrics()
        }
    
    def _calculate_coverage_area(self, gps_coords: List[Tuple[float, float]]) -> float:
        """Calculate coverage area from GPS coordinates"""
        if len(gps_coords) < 2:
            return 0.0
        
        # Simple bounding box calculation
        lats = [coord[0] for coord in gps_coords]
        lons = [coord[1] for coord in gps_coords]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Convert to approximate area (simplified)
        area_km2 = lat_range * lon_range * 111 * 111  # Rough conversion
        return area_km2
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for drone analysis"""
        return {
            "total_processing_time": sum(data["inference_time"] for data in self.flight_data),
            "rooftops_per_minute": len(self.flight_data) / (self.flight_data[-1]["timestamp"] - self.flight_data[0]["timestamp"]) * 60,
            "average_confidence": np.mean([data["analysis_results"]["confidence"] for data in self.flight_data]),
            "success_rate": 1.0  # Assuming all analyses succeeded
        }

class IoTEdgeAnalyzer:
    """IoT sensor-based edge analysis"""
    
    def __init__(self, edge_model: EdgeModel):
        self.edge_model = edge_model
        self.logger = logging.getLogger(__name__)
        self.sensor_data = []
    
    def analyze_with_iot_sensors(self, image_data: np.ndarray,
                                sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """Analyze rooftop with IoT sensor data"""
        try:
            # Configure IoT device
            device_config = EdgeDevice(
                device_id="iot_sensor_001",
                device_type="iot_sensor",
                compute_capability="low",
                memory_limit=128,
                power_limit=5.0,
                network_bandwidth=1
            )
            
            # Run edge inference
            inference_engine = EdgeInferenceEngine(self.edge_model.model_name, device_config)
            results = inference_engine.run_inference(image_data)
            
            # Integrate sensor data
            iot_analysis = {
                "image_analysis": results["results"],
                "sensor_data": sensor_readings,
                "integrated_analysis": self._integrate_sensor_data(results, sensor_readings),
                "timestamp": time.time()
            }
            
            self.sensor_data.append(iot_analysis)
            return iot_analysis
            
        except Exception as e:
            self.logger.error(f"IoT analysis failed: {e}")
            raise
    
    def _integrate_sensor_data(self, image_results: Dict, sensor_data: Dict) -> Dict[str, Any]:
        """Integrate IoT sensor data with image analysis"""
        # Combine image analysis with sensor readings
        integrated_results = {
            "roof_condition": self._assess_roof_condition(image_results, sensor_data),
            "solar_potential": self._calculate_solar_potential_with_sensors(image_results, sensor_data),
            "maintenance_needs": self._identify_maintenance_needs(sensor_data),
            "environmental_factors": self._analyze_environmental_factors(sensor_data)
        }
        
        return integrated_results
    
    def _assess_roof_condition(self, image_results: Dict, sensor_data: Dict) -> str:
        """Assess roof condition using image and sensor data"""
        # Combine image-based assessment with sensor data
        image_confidence = image_results.get("confidence", 0.5)
        temperature = sensor_data.get("temperature", 25.0)
        humidity = sensor_data.get("humidity", 50.0)
        
        # Simple scoring system
        condition_score = image_confidence * 0.7 + (1 - humidity/100) * 0.3
        
        if condition_score > 0.8:
            return "excellent"
        elif condition_score > 0.6:
            return "good"
        elif condition_score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_solar_potential_with_sensors(self, image_results: Dict, sensor_data: Dict) -> float:
        """Calculate solar potential using both image and sensor data"""
        # Base potential from image analysis
        base_potential = image_results.get("confidence", 0.5) * 1000  # kWh/year
        
        # Adjust for environmental factors
        temperature_factor = 1 - (sensor_data.get("temperature", 25) - 25) * 0.004
        humidity_factor = 1 - sensor_data.get("humidity", 50) * 0.001
        
        adjusted_potential = base_potential * temperature_factor * humidity_factor
        
        return max(0, adjusted_potential)
    
    def _identify_maintenance_needs(self, sensor_data: Dict) -> List[str]:
        """Identify maintenance needs from sensor data"""
        maintenance_needs = []
        
        if sensor_data.get("temperature", 25) > 40:
            maintenance_needs.append("thermal_stress")
        
        if sensor_data.get("humidity", 50) > 80:
            maintenance_needs.append("moisture_damage")
        
        if sensor_data.get("wind_speed", 0) > 15:
            maintenance_needs.append("structural_check")
        
        return maintenance_needs
    
    def _analyze_environmental_factors(self, sensor_data: Dict) -> Dict[str, Any]:
        """Analyze environmental factors affecting solar performance"""
        return {
            "temperature_impact": sensor_data.get("temperature", 25) - 25,
            "humidity_impact": sensor_data.get("humidity", 50) - 50,
            "wind_impact": sensor_data.get("wind_speed", 0),
            "air_quality": sensor_data.get("air_quality", "good")
        }
