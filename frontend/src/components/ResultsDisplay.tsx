import { useState, useEffect } from 'react'

interface ResultsDisplayProps {
  results: any
}

const ResultsDisplay = ({ results }: ResultsDisplayProps) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [])

  if (!results || !results.results || results.results.length === 0) {
    return null
  }

  const result = results.results[0]

  return (
    <div 
      id="results"
      className={`liquid-glass-strong rounded-2xl p-8 md:p-10 mb-8 transition-all duration-700 ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      {/* Preprocessing Info */}
      {result?.roof_analysis?.preprocessed && (
        <div className="mb-6 liquid-glass rounded-xl p-4">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <span>✅</span>
            <span>Image Preprocessed</span>
            {result?.roof_analysis?.preprocessed_size && (
              <span className="text-gray-400">
                ({result.roof_analysis.preprocessed_size[0]}x{result.roof_analysis.preprocessed_size[1]})
              </span>
            )}
            {result?.roof_analysis?.crop_box && (
              <span className="text-gray-400">• Auto-cropped around building</span>
            )}
          </div>
        </div>
      )}
      
      {/* Roof Segmentation & Mask */}
      {result?.roof_analysis?.roof_segmentation?.success && (
        <div className="mb-8 liquid-glass rounded-2xl p-6">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <span className="mr-3 text-3xl">🏠</span>
            Roof Segmentation & Mask
          </h3>
          
          {/* Segmentation Method Info */}
          <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="liquid-glass rounded-xl p-4">
              <div className="text-sm text-gray-400 mb-2">Segmentation Method</div>
              <div className="flex items-center gap-2">
                {result.roof_analysis.roof_segmentation.method && (
                  <>
                    {['unet', 'deeplabv3plus', 'deeplabv3+', 'deeplab', 'hrnet'].includes(
                      result.roof_analysis.roof_segmentation.method.toLowerCase()
                    ) ? (
                      <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-lg text-xs font-semibold border border-purple-500/50">
                        🤖 Deep Learning
                      </span>
                    ) : (
                      <span className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-lg text-xs font-semibold border border-blue-500/50">
                        👁️ Computer Vision
                      </span>
                    )}
                    <span className="text-white font-semibold text-lg">
                      {result.roof_analysis.roof_segmentation.method
                        .replace('_', ' ')
                        .replace('deeplabv3plus', 'DeepLabv3+')
                        .replace('deeplabv3+', 'DeepLabv3+')
                        .replace('deeplab', 'DeepLabv3+')
                        .toUpperCase()}
                    </span>
                  </>
                )}
              </div>
              <div className="text-sm text-gray-400 mt-2">
                {result.roof_analysis.roof_segmentation.contours_found || 0} contours found
              </div>
            </div>
            
            {result?.roof_analysis?.roof_mask?.success && (
              <div className="liquid-glass rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-2">Roof Mask Statistics</div>
                <div className="text-white font-semibold text-lg">
                  {result.roof_analysis.roof_mask.mask_area_pixels?.toLocaleString() || 0} pixels
                </div>
                {result.roof_analysis.roof_mask.polygon && (
                  <div className="text-sm text-gray-400 mt-2">
                    {result.roof_analysis.roof_mask.polygon.length} polygon points
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Deep Learning Model Info */}
          {result.roof_analysis.roof_segmentation.method && 
           ['unet', 'deeplabv3plus', 'deeplabv3+', 'deeplab', 'hrnet'].includes(
             result.roof_analysis.roof_segmentation.method.toLowerCase()
           ) && (
            <div className="mb-6 liquid-glass rounded-xl p-4 bg-purple-500/10 border border-purple-500/30">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">🤖</span>
                <span className="text-lg font-bold text-purple-300">Deep Learning Model</span>
              </div>
              <div className="text-sm text-gray-300">
                {result.roof_analysis.roof_segmentation.method.toLowerCase() === 'unet' && (
                  <div>
                    <div className="font-semibold text-white mb-1">U-Net Architecture</div>
                    <div className="text-gray-400">Encoder-Decoder with skip connections • ResNet34 backbone</div>
                  </div>
                )}
                {['deeplabv3plus', 'deeplabv3+', 'deeplab'].includes(
                  result.roof_analysis.roof_segmentation.method.toLowerCase()
                ) && (
                  <div>
                    <div className="font-semibold text-white mb-1">DeepLabv3+ Architecture</div>
                    <div className="text-gray-400">Atrous convolution with ASPP • ResNet50 backbone</div>
                  </div>
                )}
                {result.roof_analysis.roof_segmentation.method.toLowerCase() === 'hrnet' && (
                  <div>
                    <div className="font-semibold text-white mb-1">HRNet Architecture</div>
                    <div className="text-gray-400">High-Resolution Network • Multi-resolution parallel branches</div>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Roof Mask Visualization */}
          {result?.roof_analysis?.roof_mask?.mask_base64 && (
            <div className="mb-4">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center">
                <span className="mr-2">🎭</span>
                Roof Mask Visualization
              </h4>
              <div className="relative w-full overflow-hidden rounded-xl bg-black/50 p-2">
                <img 
                  src={result.roof_analysis.roof_mask.mask_base64}
                  alt="Roof segmentation mask"
                  className="w-full h-auto rounded-xl border-2 border-green-500/50 shadow-2xl"
                  style={{ maxHeight: '500px', objectFit: 'contain' }}
                />
                <div className="absolute top-4 right-4 bg-black/80 text-white px-3 py-1 rounded-lg text-xs font-semibold">
                  Binary Mask
                </div>
              </div>
            </div>
          )}
          
          {/* Segmentation Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="liquid-glass rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-green-400">
                {result.roof_analysis.roof_segmentation.roof_area_pixels?.toLocaleString() || 0}
              </div>
              <div className="text-sm text-gray-400 mt-1">Roof Area (pixels)</div>
            </div>
            <div className="liquid-glass rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-blue-400">
                {result.roof_analysis.roof_segmentation.contours_found || 0}
              </div>
              <div className="text-sm text-gray-400 mt-1">Contours Detected</div>
            </div>
            <div className="liquid-glass rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-purple-400">
                {result.roof_analysis.roof_mask?.success ? '✅' : '❌'}
              </div>
              <div className="text-sm text-gray-400 mt-1">Mask Generated</div>
            </div>
          </div>
        </div>
      )}
      
    </div>
  )
}

export default ResultsDisplay
