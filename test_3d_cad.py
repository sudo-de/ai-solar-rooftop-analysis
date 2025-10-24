#!/usr/bin/env python3
"""
Test script for 3D CAD analysis integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.solar_analysis import SolarAnalysisEngine

def test_3d_cad_analysis():
    """Test 3D CAD analysis integration"""
    print("🧪 Testing 3D CAD analysis integration...")
    
    # Initialize engine
    engine = SolarAnalysisEngine()
    test_image = 'test_rooftop.jpg'
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Analyzing {test_image} with 3D CAD...")
    
    try:
        # Run analysis
        results = engine.analyze_rooftops([test_image], ['Gurugram'], ['monocrystalline'])
        
        if results.get('results'):
            first_result = results['results'][0]
            cad_analysis = first_result.get('cad_analysis', {})
            
            print("✅ Analysis completed!")
            print(f"📊 3D Surface Area: {cad_analysis.get('surface_area_3d', 0):.2f} m²")
            print(f"📦 Volume: {cad_analysis.get('volume_3d', 0):.2f} m³")
            print(f"🎯 Optimal Zones: {len(cad_analysis.get('optimal_zones', []))}")
            print(f"🔋 3D Solar Panels: {len(cad_analysis.get('solar_panels_3d', []))}")
            print(f"🏗️ Safety Factor: {cad_analysis.get('structural_analysis', {}).get('safety_factor', 1.0)}")
            
            # Check installation plan
            install_plan = cad_analysis.get('installation_plan', {})
            print(f"📋 Total Panels: {install_plan.get('total_panels', 0)}")
            print(f"⚡ Total Power: {install_plan.get('total_power_kw', 0)} kW")
            print(f"💰 Total Cost: ${install_plan.get('total_cost', 0)}")
            
            # Check for structural issues
            structural = cad_analysis.get('structural_analysis', {})
            issues = structural.get('structural_issues', [])
            if issues:
                print(f"⚠️ Structural Issues: {len(issues)}")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ No structural issues detected")
            
            print("✅ 3D CAD analysis integrated successfully!")
            
        else:
            print("❌ Analysis failed - no results returned")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3d_cad_analysis()
