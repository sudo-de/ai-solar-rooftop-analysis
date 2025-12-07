#!/usr/bin/env python3
"""Test script for AI pipeline components"""

import sys
import os
sys.path.append('backend')

from ai_services.roof_segmentation import RoofSegmentationService
from ai_services.object_detection import ObjectDetectionService
from ai_services.zone_optimization import ZoneOptimizationService
from ai_services.solar_optimization import SolarOptimizationService
import time

def test_ai_pipeline():
    """Test each component of the AI pipeline"""
    print("🧪 Testing AI Pipeline Components")
    print("=" * 50)

    # Test image path
    test_image = "outputs/segmented/test1_segmented.jpg"

    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"📸 Using test image: {test_image}")

    try:
        # Initialize services
        print("\n1️⃣ Initializing AI Services...")
        roof_seg = RoofSegmentationService()
        obj_det = ObjectDetectionService()
        zone_opt = ZoneOptimizationService()
        solar_opt = SolarOptimizationService()
        print("✅ Services initialized")

        # Test Step 1: Roof Segmentation
        print("\n2️⃣ Testing Roof Segmentation (SegFormer)...")
        start_time = time.time()
        roof_result = roof_seg.segment_roof(test_image)
        roof_time = time.time() - start_time
        print(".2f")
        print(f"   - Roof detected: {roof_result.get('roof_detected', False)}")
        print(f"   - Roof area: {roof_result.get('roof_area_pixels', 0)} pixels")
        print(f"   - Confidence: {roof_result.get('confidence_score', 0):.3f}")

        # Test Step 2: Object Detection
        print("\n3️⃣ Testing Object Detection (YOLOv11)...")
        start_time = time.time()
        detection_result = obj_det.detect_obstacles(test_image)
        detection_time = time.time() - start_time
        print(".2f")
        print(f"   - Objects detected: {detection_result.get('total_obstacles', 0)}")
        print(f"   - Object types: {detection_result.get('obstacle_types', [])}")

        # Test Step 3: Zone Optimization
        print("\n4️⃣ Testing Zone Optimization...")
        start_time = time.time()
        zone_result = zone_opt.optimize_zones(roof_result, detection_result, test_image)
        zone_time = time.time() - start_time
        print(".2f")
        print(f"   - Clean zones found: {zone_result.get('clean_zones_found', 0)}")
        print(f"   - Usable roof: {zone_result.get('usable_roof_percentage', 0):.1f}%")

        # Test Step 4: Solar Optimization
        print("\n5️⃣ Testing Solar Panel Optimization...")
        start_time = time.time()
        solar_result = solar_opt.optimize_solar_layout(zone_result, test_image)
        solar_time = time.time() - start_time
        print(".2f")
        energy = solar_result.get('energy_calculation', {})
        print(f"   - Panels placed: {solar_result.get('total_panels_placed', 0)}")
        print(f"   - Estimated energy: {energy.get('estimated_yearly_kwh', 0)} kWh/year")

        # Summary
        total_time = roof_time + detection_time + zone_time + solar_time
        print("\n🎉 AI Pipeline Test Summary")
        print("=" * 30)
        print(".2f")
        print(f"Status: ✅ All components working")
        print("\n📊 Processing Times:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        return True

    except Exception as e:
        print(f"❌ AI Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_pipeline()
    sys.exit(0 if success else 1)
