import { useState, useEffect } from 'react'

interface ResultsDisplayProps {
  results: any
}

const ResultsDisplay = ({ results }: ResultsDisplayProps) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [results]) // Re-trigger animation when results change

  // Debug: Log the results structure
  useEffect(() => {
    if (results) {
      console.log('Results structure:', JSON.stringify(results, null, 2))
    }
  }, [results])

  if (!results || !results.results || results.results.length === 0) {
    console.log('No results to display:', { results })
    return null
  }

  const result = results.results[0]
  const roofAnalysis = result?.roof_analysis || {}
  const aiPipelineResults = roofAnalysis?.ai_pipeline_results || {}
  const roofAnalysisData = aiPipelineResults?.roof_analysis || {}
  // const objectDetectionData = aiPipelineResults?.object_detection || {}
  // const zoneOptimizationData = aiPipelineResults?.zone_optimization || {}
  // const solarOptimizationData = aiPipelineResults?.solar_optimization || {}

  // Extract advanced features
  const advancedFeatures = roofAnalysisData?.advanced_features || {}

  return (
    <div 
      id="results"
      className={`liquid-glass-strong rounded-2xl p-8 md:p-10 mb-8 transition-all duration-700 ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2 flex items-center">
          <span className="mr-3 text-4xl">✨</span>
          Analysis Results
        </h2>
        <p className="text-gray-400">
          {result?.filename && `File: ${result.filename}`}
          {result?.image_dimensions && ` • ${result.image_dimensions[0]}×${result.image_dimensions[1]} pixels`}
        </p>
      </div>

      {/* Segmented Image */}
      {roofAnalysis?.segmented_image_base64 && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">🏠</span>
            Roof Segmentation
          </h3>
          <div className="relative w-full overflow-hidden rounded-xl bg-black/50 p-2">
            <img 
              src={roofAnalysis.segmented_image_base64}
              alt="Roof segmentation"
              className="w-full h-auto rounded-xl border-2 border-green-500/50 shadow-2xl"
              style={{ maxHeight: '600px', objectFit: 'contain' }}
            />
          </div>
        </div>
      )}

      {/* NextGen SegFormer Alpha Features */}
      {advancedFeatures && Object.keys(advancedFeatures).length > 0 && (
        <div className="mb-8 liquid-glass rounded-2xl p-6 bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30">
          <h3 className="text-2xl font-bold text-purple-300 mb-6 flex items-center">
            <span className="mr-3 text-3xl">🚀</span>
            NextGen SegFormer Alpha Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {advancedFeatures.fusion_method && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Fusion Method</div>
                <div className="text-white font-semibold">
                  {String(advancedFeatures.fusion_method).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
              </div>
            )}
            {advancedFeatures.roof_shape && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Roof Shape</div>
                <div className="text-white font-semibold capitalize">
                  {String(advancedFeatures.roof_shape)}
                </div>
              </div>
            )}
            {typeof advancedFeatures.roof_complexity === 'number' && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Complexity Score</div>
                <div className="text-white font-semibold">
                  {advancedFeatures.roof_complexity.toFixed(3)}
                </div>
              </div>
            )}
            {typeof advancedFeatures.edge_quality === 'number' && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Edge Quality</div>
                <div className="text-white font-semibold">
                  {advancedFeatures.edge_quality.toFixed(3)}
                </div>
              </div>
            )}
            {typeof advancedFeatures.uncertainty_score === 'number' && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Uncertainty Score</div>
                <div className="text-white font-semibold">
                  {advancedFeatures.uncertainty_score.toFixed(3)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Roof Analysis Statistics */}
      {roofAnalysisData && Object.keys(roofAnalysisData).length > 0 && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">📊</span>
            Roof Analysis Statistics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {typeof roofAnalysisData.roof_area_pixels === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-green-400">
                  {roofAnalysisData.roof_area_pixels.toLocaleString()}
                </div>
                <div className="text-sm text-gray-400 mt-1">Roof Area (pixels)</div>
              </div>
            )}
            {typeof roofAnalysisData.roof_coverage_percentage === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {roofAnalysisData.roof_coverage_percentage.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400 mt-1">Roof Coverage</div>
              </div>
            )}
            {typeof roofAnalysisData.confidence_score === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {(roofAnalysisData.confidence_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400 mt-1">Confidence</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Detected Objects */}
      {roofAnalysis?.detected_objects && Array.isArray(roofAnalysis.detected_objects) && roofAnalysis.detected_objects.length > 0 && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">🔍</span>
            Detected Objects
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {roofAnalysis.detected_objects.map((obj: any, index: number) => (
              <div key={index} className="liquid-glass rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-semibold capitalize">
                    {obj.class || obj.name || 'Unknown'}
                  </span>
                  {typeof obj.confidence === 'number' && (
                    <span className="text-blue-400 text-sm">
                      {(obj.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {obj.bbox && (
                  <div className="text-xs text-gray-400">
                    Position: ({Math.round(obj.bbox[0])}, {Math.round(obj.bbox[1])})
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Solar Analysis Results */}
      {(roofAnalysis?.suitability_score || roofAnalysis?.surface_area || roofAnalysis?.estimated_energy) && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">☀️</span>
            Solar Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {typeof roofAnalysis.suitability_score === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-yellow-400">
                  {roofAnalysis.suitability_score.toFixed(1)}/10
                </div>
                <div className="text-sm text-gray-400 mt-1">Suitability Score</div>
              </div>
            )}
            {typeof roofAnalysis.surface_area === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-green-400">
                  {roofAnalysis.surface_area.toFixed(1)} m²
                </div>
                <div className="text-sm text-gray-400 mt-1">Surface Area</div>
              </div>
            )}
            {typeof roofAnalysis.estimated_energy === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {roofAnalysis.estimated_energy.toLocaleString()} kWh
                </div>
                <div className="text-sm text-gray-400 mt-1">Estimated Energy/Year</div>
              </div>
            )}
            {typeof roofAnalysis.estimated_cost === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-purple-400">
                  ₹{roofAnalysis.estimated_cost.toLocaleString()}
                </div>
                <div className="text-sm text-gray-400 mt-1">Estimated Cost</div>
              </div>
            )}
            {typeof roofAnalysis.payback_period === 'number' && (
              <div className="liquid-glass rounded-xl p-4 text-center md:col-span-2 lg:col-span-4">
                <div className="text-2xl font-bold text-orange-400">
                  {roofAnalysis.payback_period.toFixed(1)} years
                </div>
                <div className="text-sm text-gray-400 mt-1">Payback Period</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Formatted Report (if available) */}
      {roofAnalysis?.formatted_report_text && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">📄</span>
            Detailed Report
          </h3>
          <div className="liquid-glass rounded-xl p-4 max-h-96 overflow-y-auto">
            <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
              {roofAnalysis.formatted_report_text}
            </pre>
          </div>
        </div>
      )}

    </div>
  )
}

export default ResultsDisplay
