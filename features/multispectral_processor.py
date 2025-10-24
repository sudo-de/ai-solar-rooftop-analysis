"""
Advanced Multispectral Satellite Data Processor
Integrates Sentinel-2, Landsat, and LiDAR data for 3D roof modeling with sub-meter precision
"""

import numpy as np
import rasterio
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN
import laspy
import open3d as o3d
from shapely.geometry import Polygon, Point
import pyproj
from rasterio.warp import calculate_default_transform, reproject, Resampling

@dataclass
class SatelliteData:
    """Container for multispectral satellite data with enhanced metadata"""
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    nir: np.ndarray
    swir1: np.ndarray
    swir2: np.ndarray
    coordinates: Tuple[float, float]
    resolution: float
    timestamp: datetime
    cloud_coverage: float
    sun_azimuth: float
    sun_elevation: float
    data_quality: str

@dataclass
class LiDARData:
    """Container for LiDAR point cloud data"""
    points: np.ndarray  # (N, 3) XYZ coordinates
    intensities: np.ndarray  # (N,) intensity values
    classifications: np.ndarray  # (N,) point classifications
    coordinates: Tuple[float, float]
    resolution: float
    timestamp: datetime
    density: float  # points per square meter

@dataclass
class Roof3DModel:
    """3D roof model with sub-meter precision"""
    vertices: np.ndarray  # (N, 3) 3D vertices
    faces: np.ndarray  # (M, 3) triangular faces
    normals: np.ndarray  # (M, 3) face normals
    surface_area: float
    volume: float
    height_map: np.ndarray
    slope_map: np.ndarray
    aspect_map: np.ndarray
    material_classification: Dict[str, float]
    structural_integrity: float

class MultispectralProcessor:
    """Process multispectral satellite imagery for enhanced roof analysis with LiDAR integration"""
    
    def __init__(self, sentinel_api_key: str, landsat_api_key: str, lidar_api_key: str = None):
        self.sentinel_api_key = sentinel_api_key
        self.landsat_api_key = landsat_api_key
        self.lidar_api_key = lidar_api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize coordinate transformation
        self.wgs84 = pyproj.CRS('EPSG:4326')
        self.utm = None  # Will be determined based on location
        
        # Spectral indices for material classification
        self.spectral_indices = {
            'ndvi': lambda nir, red: (nir - red) / (nir + red + 1e-8),
            'ndwi': lambda green, nir: (green - nir) / (green + nir + 1e-8),
            'ndbi': lambda swir1, nir: (swir1 - nir) / (swir1 + nir + 1e-8),
            'savi': lambda nir, red, l=0.5: ((nir - red) / (nir + red + l)) * (1 + l),
            'evi': lambda nir, red, blue: 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        }
    
    def fetch_sentinel2_data(self, lat: float, lon: float, date_range: Tuple[str, str]) -> SatelliteData:
        """Fetch Sentinel-2 multispectral data"""
        try:
            # Mock implementation - replace with actual Sentinel Hub API
            self.logger.info(f"Fetching Sentinel-2 data for {lat}, {lon}")
            
            # Simulate multispectral bands
            size = (512, 512)
            return SatelliteData(
                red=np.random.randint(0, 255, size, dtype=np.uint8),
                green=np.random.randint(0, 255, size, dtype=np.uint8),
                blue=np.random.randint(0, 255, size, dtype=np.uint8),
                nir=np.random.randint(0, 255, size, dtype=np.uint8),
                swir1=np.random.randint(0, 255, size, dtype=np.uint8),
                swir2=np.random.randint(0, 255, size, dtype=np.uint8),
                coordinates=(lat, lon),
                resolution=10.0,  # 10m resolution for Sentinel-2
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Sentinel-2 data fetch failed: {e}")
            raise
    
    def calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        return (nir - red) / (nir + red + 1e-8)
    
    def calculate_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index"""
        return (green - nir) / (green + nir + 1e-8)
    
    def detect_roof_materials(self, satellite_data: SatelliteData) -> Dict[str, float]:
        """Detect roof materials using hyperspectral analysis"""
        # Advanced material classification using spectral signatures
        ndvi = self.calculate_ndvi(satellite_data.nir, satellite_data.red)
        ndwi = self.calculate_ndwi(satellite_data.green, satellite_data.nir)
        
        # Material classification based on spectral indices
        materials = {
            "asphalt": np.mean(ndvi < 0.1) * 100,  # Low vegetation
            "tile": np.mean((ndvi > 0.1) & (ndvi < 0.3)) * 100,  # Medium vegetation
            "metal": np.mean(ndvi > 0.3) * 100,  # High vegetation (corrosion)
            "concrete": np.mean(ndwi > 0.1) * 100  # Water absorption
        }
        
        return materials
    
    def fetch_lidar_data(self, lat: float, lon: float, radius: float = 1000) -> LiDARData:
        """Fetch LiDAR point cloud data for the specified location"""
        try:
            self.logger.info(f"Fetching LiDAR data for {lat}, {lon}")
            
            # Determine UTM zone for coordinate transformation
            utm_zone = int((lon + 180) / 6) + 1
            self.utm = pyproj.CRS(f'EPSG:326{utm_zone:02d}')
            
            # Transform coordinates to UTM
            transformer = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True)
            utm_x, utm_y = transformer.transform(lon, lat)
            
            # Mock LiDAR data - replace with actual API calls
            # In practice, this would fetch from USGS, OpenTopography, or local LiDAR databases
            n_points = 10000
            points = np.random.rand(n_points, 3) * 100  # 100m x 100m area
            points[:, 0] += utm_x - 50  # Center around location
            points[:, 1] += utm_y - 50
            
            # Generate realistic height variations
            points[:, 2] = self._generate_realistic_heights(points[:, :2])
            
            # Generate intensities and classifications
            intensities = np.random.rand(n_points) * 255
            classifications = np.random.choice([1, 2, 6, 9], n_points, p=[0.1, 0.3, 0.4, 0.2])  # Ground, vegetation, building, water
            
            return LiDARData(
                points=points,
                intensities=intensities,
                classifications=classifications,
                coordinates=(lat, lon),
                resolution=0.5,  # 0.5m resolution
                timestamp=datetime.now(),
                density=n_points / (100 * 100)  # points per square meter
            )
            
        except Exception as e:
            self.logger.error(f"LiDAR data fetch failed: {e}")
            raise
    
    def _generate_realistic_heights(self, xy_points: np.ndarray) -> np.ndarray:
        """Generate realistic height variations for LiDAR points"""
        # Create a simple building height model
        center_x, center_y = np.mean(xy_points, axis=0)
        distances = np.sqrt((xy_points[:, 0] - center_x)**2 + (xy_points[:, 1] - center_y)**2)
        
        # Building heights decrease from center
        heights = np.maximum(0, 10 - distances * 0.1 + np.random.normal(0, 1, len(distances)))
        return heights
    
    def generate_3d_model(self, satellite_data: SatelliteData, lidar_data: Optional[LiDARData] = None) -> Roof3DModel:
        """Generate 3D roof model with sub-meter precision using LiDAR and multispectral data"""
        try:
            if lidar_data is not None:
                # Use LiDAR for precise 3D modeling
                return self._create_lidar_3d_model(lidar_data, satellite_data)
            else:
                # Fallback to satellite-based estimation
                return self._create_satellite_3d_model(satellite_data)
                
        except Exception as e:
            self.logger.error(f"3D model generation failed: {e}")
            raise
    
    def _create_lidar_3d_model(self, lidar_data: LiDARData, satellite_data: SatelliteData) -> Roof3DModel:
        """Create 3D model from LiDAR point cloud data"""
        try:
            # Filter building points (classification 6)
            building_mask = lidar_data.classifications == 6
            building_points = lidar_data.points[building_mask]
            
            if len(building_points) == 0:
                raise ValueError("No building points found in LiDAR data")
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(building_points)
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Create mesh using Poisson reconstruction
            mesh, _ = pcd.create_mesh_poisson(depth=9)
            
            # Extract vertices and faces
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # Calculate normals
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            
            # Calculate surface area and volume
            surface_area = mesh.get_surface_area()
            volume = mesh.get_volume()
            
            # Create height map
            height_map = self._create_height_map_from_points(building_points)
            
            # Calculate slope and aspect
            slope_map = self._calculate_slope_angles(height_map)
            aspect_map = self._calculate_aspect_directions(height_map)
            
            # Classify materials using spectral data
            material_classification = self._classify_roof_materials_advanced(satellite_data, height_map)
            
            # Assess structural integrity
            structural_integrity = self._assess_structural_integrity_3d(vertices, faces, normals)
            
            return Roof3DModel(
                vertices=vertices,
                faces=faces,
                normals=normals,
                surface_area=surface_area,
                volume=volume,
                height_map=height_map,
                slope_map=slope_map,
                aspect_map=aspect_map,
                material_classification=material_classification,
                structural_integrity=structural_integrity
            )
            
        except Exception as e:
            self.logger.error(f"LiDAR 3D model creation failed: {e}")
            raise
    
    def _create_satellite_3d_model(self, satellite_data: SatelliteData) -> Roof3DModel:
        """Create 3D model from satellite data only (fallback method)"""
        try:
            # Estimate height from spectral data
            height_map = self._estimate_height_from_spectral_advanced(satellite_data)
            
            # Create mesh from height map
            vertices, faces = self._height_map_to_mesh(height_map)
            
            # Calculate normals
            normals = self._calculate_face_normals(vertices, faces)
            
            # Calculate surface area and volume
            surface_area = self._calculate_surface_area_from_mesh(vertices, faces)
            volume = self._calculate_volume_from_mesh(vertices, faces)
            
            # Calculate slope and aspect
            slope_map = self._calculate_slope_angles(height_map)
            aspect_map = self._calculate_aspect_directions(height_map)
            
            # Classify materials
            material_classification = self._classify_roof_materials_advanced(satellite_data, height_map)
            
            # Assess structural integrity
            structural_integrity = self._assess_structural_integrity_2d(height_map)
            
            return Roof3DModel(
                vertices=vertices,
                faces=faces,
                normals=normals,
                surface_area=surface_area,
                volume=volume,
                height_map=height_map,
                slope_map=slope_map,
                aspect_map=aspect_map,
                material_classification=material_classification,
                structural_integrity=structural_integrity
            )
            
        except Exception as e:
            self.logger.error(f"Satellite 3D model creation failed: {e}")
            raise
    
    def _estimate_height_from_spectral(self, data: SatelliteData) -> np.ndarray:
        """Estimate building height from spectral data"""
        # Simplified height estimation
        return np.random.uniform(0, 20, data.red.shape)
    
    def _calculate_slope_angles(self, height_map: np.ndarray) -> np.ndarray:
        """Calculate slope angles from height map"""
        grad_y, grad_x = np.gradient(height_map)
        return np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    
    def _calculate_aspect_directions(self, height_map: np.ndarray) -> np.ndarray:
        """Calculate aspect directions from height map"""
        grad_y, grad_x = np.gradient(height_map)
        return np.arctan2(grad_y, grad_x)
    
    def _create_height_map_from_points(self, points: np.ndarray, resolution: float = 1.0) -> np.ndarray:
        """Create height map from LiDAR points"""
        # Create grid
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_size = int((x_max - x_min) / resolution) + 1
        y_size = int((y_max - y_min) / resolution) + 1
        
        height_map = np.zeros((y_size, x_size))
        
        # Interpolate heights
        for point in points:
            x_idx = int((point[0] - x_min) / resolution)
            y_idx = int((point[1] - y_min) / resolution)
            if 0 <= x_idx < x_size and 0 <= y_idx < y_size:
                height_map[y_idx, x_idx] = point[2]
        
        return height_map
    
    def _height_map_to_mesh(self, height_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert height map to 3D mesh"""
        h, w = height_map.shape
        
        # Create vertices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        vertices = np.column_stack([
            x.flatten(),
            y.flatten(),
            height_map.flatten()
        ])
        
        # Create faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                # Two triangles per quad
                v1 = i * w + j
                v2 = i * w + (j + 1)
                v3 = (i + 1) * w + j
                v4 = (i + 1) * w + (j + 1)
                
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        return vertices, np.array(faces)
    
    def _calculate_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate face normals for mesh"""
        normals = []
        for face in faces:
            v1, v2, v3 = vertices[face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
        return np.array(normals)
    
    def _calculate_surface_area_from_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate surface area from mesh"""
        total_area = 0
        for face in faces:
            v1, v2, v3 = vertices[face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            total_area += area
        return total_area
    
    def _calculate_volume_from_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate volume from mesh using divergence theorem"""
        volume = 0
        for face in faces:
            v1, v2, v3 = vertices[face]
            # Volume contribution of this triangle
            vol = np.dot(v1, np.cross(v2, v3)) / 6
            volume += vol
        return abs(volume)
    
    def _estimate_height_from_spectral_advanced(self, data: SatelliteData) -> np.ndarray:
        """Advanced height estimation from spectral data"""
        # Use multiple spectral indices for height estimation
        ndvi = self.spectral_indices['ndvi'](data.nir, data.red)
        ndbi = self.spectral_indices['ndbi'](data.swir1, data.nir)
        
        # Buildings typically have high NDBI and low NDVI
        building_mask = (ndbi > 0.1) & (ndvi < 0.3)
        
        # Estimate heights based on spectral characteristics
        heights = np.zeros_like(data.red, dtype=float)
        heights[building_mask] = 5 + ndbi[building_mask] * 20  # 5-25m range
        
        return heights
    
    def _classify_roof_materials_advanced(self, satellite_data: SatelliteData, height_map: np.ndarray) -> Dict[str, float]:
        """Advanced roof material classification using spectral and geometric data"""
        # Calculate all spectral indices
        ndvi = self.spectral_indices['ndvi'](satellite_data.nir, satellite_data.red)
        ndwi = self.spectral_indices['ndwi'](satellite_data.green, satellite_data.nir)
        ndbi = self.spectral_indices['ndbi'](satellite_data.swir1, satellite_data.nir)
        savi = self.spectral_indices['savi'](satellite_data.nir, satellite_data.red)
        evi = self.spectral_indices['evi'](satellite_data.nir, satellite_data.red, satellite_data.blue)
        
        # Material classification based on spectral signatures
        materials = {
            "asphalt": np.mean((ndvi < 0.1) & (ndbi > 0.1) & (ndwi < 0.1)) * 100,
            "tile": np.mean((ndvi > 0.1) & (ndvi < 0.3) & (ndbi > 0.05)) * 100,
            "metal": np.mean((ndvi > 0.3) & (ndbi > 0.2)) * 100,
            "concrete": np.mean((ndwi > 0.1) & (ndbi > 0.15)) * 100,
            "green_roof": np.mean((ndvi > 0.4) & (savi > 0.3)) * 100,
            "solar_panels": np.mean((evi > 0.2) & (ndbi > 0.1)) * 100
        }
        
        return materials
    
    def _assess_structural_integrity_3d(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> float:
        """Assess structural integrity from 3D mesh"""
        # Calculate surface smoothness
        normal_variance = np.var(normals, axis=0)
        smoothness = 1 / (1 + np.sum(normal_variance))
        
        # Calculate surface continuity
        face_areas = []
        for face in faces:
            v1, v2, v3 = vertices[face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            face_areas.append(area)
        
        area_variance = np.var(face_areas)
        continuity = 1 / (1 + area_variance)
        
        # Overall structural integrity
        integrity = (smoothness + continuity) / 2
        return float(np.clip(integrity, 0, 1))
    
    def _assess_structural_integrity_2d(self, height_map: np.ndarray) -> float:
        """Assess structural integrity from 2D height map"""
        # Calculate surface smoothness
        grad_x, grad_y = np.gradient(height_map)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        smoothness = 1 / (1 + np.mean(gradient_magnitude))
        
        # Calculate surface continuity
        laplacian = ndimage.laplace(height_map)
        continuity = 1 / (1 + np.var(laplacian))
        
        # Overall structural integrity
        integrity = (smoothness + continuity) / 2
        return float(np.clip(integrity, 0, 1))
    
    def _calculate_3d_surface_area(self, height_map: np.ndarray) -> float:
        """Calculate 3D surface area from height map"""
        grad_x, grad_y = np.gradient(height_map)
        surface_area = np.sum(np.sqrt(1 + grad_x**2 + grad_y**2))
        return float(surface_area)

class WeatherDataIntegration:
    """Integrate real-time weather and IoT sensor data"""
    
    def __init__(self, weather_api_key: str):
        self.weather_api_key = weather_api_key
        self.logger = logging.getLogger(__name__)
    
    def fetch_weather_data(self, lat: float, lon: float) -> Dict:
        """Fetch real-time weather data for microclimate analysis"""
        try:
            # Mock weather data - replace with actual API
            return {
                "temperature": 25.5,
                "humidity": 65.0,
                "wind_speed": 3.2,
                "cloud_cover": 0.3,
                "irradiance": 800.0,
                "timestamp": datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Weather data fetch failed: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Default weather data when API fails"""
        return {
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 2.0,
            "cloud_cover": 0.2,
            "irradiance": 600.0,
            "timestamp": datetime.now()
        }
    
    def calculate_shading_impact(self, weather_data: Dict, roof_geometry: Dict) -> float:
        """Calculate dynamic shading impact from weather conditions"""
        cloud_factor = 1 - weather_data["cloud_cover"]
        wind_factor = min(1.0, weather_data["wind_speed"] / 10.0)
        return cloud_factor * wind_factor
