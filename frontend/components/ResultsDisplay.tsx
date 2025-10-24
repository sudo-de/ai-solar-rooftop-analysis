'use client';

import { useState } from 'react';
import { 
  ChartBarIcon, 
  EyeIcon,
  ShareIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';

interface ResultsDisplayProps {
  results: any;
}

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: ChartBarIcon },
    { id: 'details', label: 'Details', icon: EyeIcon },
    { id: 'segmentation', label: 'AI Analysis', icon: EyeIcon },
    { id: 'cad', label: '3D CAD', icon: EyeIcon },
    { id: 'downloads', label: 'Downloads', icon: ArrowDownTrayIcon },
  ];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Analysis Results</h3>
        <div className="flex space-x-2">
          <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
            <ShareIcon className="w-5 h-5" />
          </button>
          <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
            <ArrowDownTrayIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-8">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">
                {results?.energy_prediction?.annual_energy_kwh || '0'} kWh
              </div>
              <div className="text-sm text-blue-700">Annual Energy</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-orange-600">
                ₹{results?.roi_estimation?.annual_savings || '0'}
              </div>
              <div className="text-sm text-orange-700">Annual Savings</div>
            </div>
          </div>

          {/* System Details */}
          <div className="space-y-4">
            <h4 className="font-semibold text-gray-900">System Configuration</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">System Size:</span>
                <span className="ml-2 font-medium">{results?.roi_estimation?.system_size_kw || '0'} kW</span>
              </div>
              <div>
                <span className="text-gray-500">Payback Period:</span>
                <span className="ml-2 font-medium">{results?.roi_estimation?.payback_period_years || '0'} years</span>
              </div>
              <div>
                <span className="text-gray-500">Total Cost:</span>
                <span className="ml-2 font-medium">₹{results?.roi_estimation?.total_cost || '0'}</span>
              </div>
              <div>
                <span className="text-gray-500">Suitability:</span>
                <span className="ml-2 font-medium">{results?.roof_analysis?.suitability || '0'}/10</span>
              </div>
            </div>
          </div>

          {/* Accuracy Metrics */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Accuracy Metrics</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Overall Confidence:</span>
                <span className="font-medium text-green-600">
                  {results?.accuracy_metrics?.overall_confidence || '0'}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Roof Detection:</span>
                <span className="font-medium text-green-600">
                  {results?.accuracy_metrics?.roof_detection_accuracy || '0'}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-6">
          {/* Roof Analysis */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Roof Analysis</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Area:</span>
                <span className="ml-2 font-medium">{results?.roof_analysis?.area_m2 || '0'} m²</span>
              </div>
              <div>
                <span className="text-gray-500">Orientation:</span>
                <span className="ml-2 font-medium">{results?.roof_analysis?.orientation || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-500">Surface Type:</span>
                <span className="ml-2 font-medium">{results?.roof_analysis?.surface_type || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-500">Obstructions:</span>
                <span className="ml-2 font-medium">{results?.roof_analysis?.obstructions || 'None'}</span>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
            <div className="space-y-2">
              {results?.recommendations?.map((rec: string, index: number) => (
                <div key={index} className="flex items-start space-x-2 text-sm">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <span className="text-gray-700">{rec}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'segmentation' && (
        <div className="space-y-6">
          <h4 className="font-semibold text-gray-900">AI Object Detection Results</h4>
          
          {/* Segmented Image Display */}
          {results?.results?.[0]?.roof_analysis?.segmented_image_base64 ? (
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-3">YOLO Segmentation Visualization</h5>
                <div className="relative">
                  <img 
                    src={results.results[0].roof_analysis.segmented_image_base64}
                    alt="Segmented rooftop analysis"
                    className="w-full h-auto rounded-lg border border-gray-200"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                      const nextElement = e.currentTarget.nextElementSibling as HTMLElement;
                      if (nextElement) {
                        nextElement.style.display = 'block';
                      }
                    }}
                  />
                  <div className="hidden text-center py-8 text-gray-500">
                    <p>Segmented image will appear here after analysis</p>
                  </div>
                </div>
              </div>
              
              {/* Detected Objects */}
              {results?.results?.[0]?.roof_analysis?.detected_objects && (
                <div className="bg-blue-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-3">Detected Objects</h5>
                  <div className="grid grid-cols-2 gap-3">
                    {results.results[0].roof_analysis.detected_objects.map((obj: any, index: number) => (
                      <div key={index} className="flex items-center space-x-2 p-2 bg-white rounded border">
                        <div 
                          className="w-4 h-4 rounded"
                          style={{ backgroundColor: `rgb(${obj.color.join(',')})` }}
                        />
                        <span className="text-sm font-medium">{obj.label}</span>
                        <span className="text-xs text-gray-500">({(obj.confidence * 100).toFixed(1)}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Analysis Summary */}
              <div className="bg-green-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-3">AI Analysis Summary</h5>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Objects Detected:</span>
                    <span className="ml-2 font-medium">{results?.results?.[0]?.roof_analysis?.detected_objects?.length || 0}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Obstructions:</span>
                    <span className="ml-2 font-medium">{results?.results?.[0]?.roof_analysis?.obstructions || 'None'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Roof Suitability:</span>
                    <span className="ml-2 font-medium">{results?.results?.[0]?.roof_analysis?.suitability || 0}/10</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Surface Type:</span>
                    <span className="ml-2 font-medium">{results?.results?.[0]?.roof_analysis?.surface_type || 'Unknown'}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <EyeIcon className="w-8 h-8 text-gray-400" />
              </div>
              <p>Segmentation analysis will appear here after image processing</p>
              {/* Debug info */}
              {process.env.NODE_ENV === 'development' && (
                <div className="mt-4 text-xs text-gray-400">
                  <p>Debug: {results?.results?.[0]?.roof_analysis?.segmented_image_base64 ? 'Has base64 data' : 'No base64 data'}</p>
                  <p>Results: {results?.results ? 'Has results' : 'No results'}</p>
                  <p>Structure: {JSON.stringify(results?.results?.[0]?.roof_analysis, null, 2).substring(0, 200)}...</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'cad' && (
        <div className="space-y-6">
          <h4 className="font-semibold text-gray-900">3D CAD Roof Analysis</h4>
          
          {/* 3D Analysis Results */}
          {results?.results?.[0]?.cad_analysis && (
            <div className="space-y-6">
              {/* 3D Geometry */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-3">3D Roof Geometry</h5>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">3D Surface Area:</span>
                    <span className="ml-2 font-medium">{results.results[0].cad_analysis.surface_area_3d?.toFixed(2) || '0'} m²</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Volume:</span>
                    <span className="ml-2 font-medium">{results.results[0].cad_analysis.volume_3d?.toFixed(2) || '0'} m³</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Optimal Zones:</span>
                    <span className="ml-2 font-medium">{results.results[0].cad_analysis.optimal_zones?.length || '0'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Safety Factor:</span>
                    <span className="ml-2 font-medium">{results.results[0].cad_analysis.structural_analysis?.safety_factor || '1.0'}</span>
                  </div>
                </div>
              </div>

              {/* Solar Panel Layout */}
              {results.results[0].cad_analysis.solar_panels_3d && results.results[0].cad_analysis.solar_panels_3d.length > 0 && (
                <div className="bg-green-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-3">3D Solar Panel Layout</h5>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 mb-2">Panel Configuration:</p>
                      <div className="space-y-2">
                        {results.results[0].cad_analysis.solar_panels_3d.slice(0, 5).map((panel: any, index: number) => (
                          <div key={index} className="bg-white rounded p-2 text-xs">
                            <div className="font-medium">Panel {index + 1}</div>
                            <div>Position: ({panel.position[0]?.toFixed(1)}, {panel.position[1]?.toFixed(1)}, {panel.position[2]?.toFixed(1)})</div>
                            <div>Power: {panel.power_output?.toFixed(0)}W</div>
                            <div>Shading: {(panel.shading_factor * 100)?.toFixed(1)}%</div>
                          </div>
                        ))}
                        {results.results[0].cad_analysis.solar_panels_3d.length > 5 && (
                          <div className="text-xs text-gray-500">
                            +{results.results[0].cad_analysis.solar_panels_3d.length - 5} more panels
                          </div>
                        )}
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-2">Installation Plan:</p>
                      <div className="space-y-1 text-xs">
                        <div>Total Panels: {results.results[0].cad_analysis.installation_plan?.total_panels || '0'}</div>
                        <div>Total Power: {results.results[0].cad_analysis.installation_plan?.total_power_kw || '0'} kW</div>
                        <div>Total Cost: ${results.results[0].cad_analysis.installation_plan?.total_cost || '0'}</div>
                        <div>Timeline: {results.results[0].cad_analysis.installation_plan?.timeline?.total_days || '0'} days</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Optimal Zones */}
              {results.results[0].cad_analysis.optimal_zones && results.results[0].cad_analysis.optimal_zones.length > 0 && (
                <div className="bg-purple-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-3">Optimal Panel Zones</h5>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {results.results[0].cad_analysis.optimal_zones.map((zone: any, index: number) => (
                      <div key={index} className="bg-white rounded p-3 text-sm">
                        <div className="font-medium">Zone {zone.id + 1}</div>
                        <div className="text-gray-600">Area: {zone.area?.toFixed(1)} m²</div>
                        <div className="text-gray-600">Suitability: {(zone.suitability_score * 100)?.toFixed(1)}%</div>
                        <div className="text-gray-600">Recommended Panels: {zone.recommended_panels}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Structural Analysis */}
              <div className="bg-orange-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-3">Structural Analysis</h5>
                <div className="space-y-2 text-sm">
                  {results.results[0].cad_analysis.structural_analysis?.structural_issues && 
                   results.results[0].cad_analysis.structural_analysis.structural_issues.length > 0 ? (
                    <div>
                      <p className="text-red-600 font-medium mb-2">Structural Issues:</p>
                      <ul className="list-disc list-inside text-red-600">
                        {results.results[0].cad_analysis.structural_analysis.structural_issues.map((issue: string, index: number) => (
                          <li key={index}>{issue}</li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <div className="text-green-600">
                      ✅ Roof structure is suitable for solar installation
                    </div>
                  )}
                  
                  {results.results[0].cad_analysis.structural_analysis?.recommendations && (
                    <div>
                      <p className="text-gray-600 font-medium mb-2">Recommendations:</p>
                      <ul className="list-disc list-inside text-gray-600">
                        {results.results[0].cad_analysis.structural_analysis.recommendations.map((rec: string, index: number) => (
                          <li key={index}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>

              {/* CAD Export */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-3">CAD Export</h5>
                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">
                    Download OBJ
                  </button>
                  <button className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700">
                    Download STL
                  </button>
                  <button className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700">
                    Download JSON
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  3D models available for CAD software integration
                </p>
              </div>
            </div>
          )}
          
          {!results?.results?.[0]?.cad_analysis && (
            <div className="text-center py-8 text-gray-500">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <EyeIcon className="w-8 h-8 text-gray-400" />
              </div>
              <p>3D CAD analysis will appear here after processing</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'downloads' && (
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-900">Download Reports</h4>
          <div className="grid grid-cols-2 gap-4">
            <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors">
              <ArrowDownTrayIcon className="w-5 h-5 text-blue-600" />
              <span className="font-medium">PDF Report</span>
            </button>
            <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors">
              <ArrowDownTrayIcon className="w-5 h-5 text-blue-600" />
              <span className="font-medium">Excel Report</span>
            </button>
            <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors">
              <ArrowDownTrayIcon className="w-5 h-5 text-blue-600" />
              <span className="font-medium">CSV Data</span>
            </button>
            <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors">
              <ArrowDownTrayIcon className="w-5 h-5 text-blue-600" />
              <span className="font-medium">JSON Data</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
