import { useState, useRef } from 'react'
import { analyzeRooftop } from '../services/api'
import ImagePreview from './ImagePreview'

interface AnalysisFormProps {
  onAnalysisStart: () => void
  onAnalysisComplete: (results: any) => void
  isAnalyzing: boolean
}

const AnalysisForm = ({ onAnalysisStart, onAnalysisComplete, isAnalyzing }: AnalysisFormProps) => {
  const [files, setFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (selectedFiles: FileList | null) => {
    if (selectedFiles) {
      const fileArray = Array.from(selectedFiles)
      const validFiles: File[] = []
      const errors: string[] = []

      fileArray.forEach(file => {
        const isValidType = ['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)
        const isValidSize = file.size <= 10 * 1024 * 1024
        
        if (!isValidType) {
          errors.push(`${file.name}: Invalid file type`)
        } else if (!isValidSize) {
          errors.push(`${file.name}: File too large (max 10MB)`)
        } else {
          validFiles.push(file)
        }
      })
      
      if (errors.length > 0) {
        setError(errors.join(', '))
      } else {
        setError(null)
      }
      
      setFiles(prev => [...prev, ...validFiles])
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    handleFileChange(e.dataTransfer.files)
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
    setError(null)
  }

  const clearAllFiles = () => {
    setFiles([])
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (files.length === 0) {
      setError('Please select at least one image file')
      return
    }

    onAnalysisStart()
    setError(null)

    try {
      const results = await analyzeRooftop(files)
      console.log('Analysis results received:', results)
      onAnalysisComplete(results)
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (err: any) {
      console.error('Analysis error:', err)
      setError(err.response?.data?.detail || err.message || 'Failed to analyze rooftop. Please try again.')
      // Reset analyzing state on error
      onAnalysisComplete(null)
    }
  }

  return (
    <div 
      id="analyze" 
      className="liquid-glass-strong rounded-2xl p-8 md:p-10 mb-8 animate-fade-in liquid-glass-hover"
    >
      <div className="flex items-start space-x-4 mb-8">
        <div className="w-14 h-14 bg-gradient-to-br from-blue-500 via-blue-600 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg transform hover:rotate-6 transition-transform">
          <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <div className="flex-1">
          <h2 className="text-3xl font-bold text-white mb-2">Upload Rooftop Images</h2>
          <p className="text-gray-300 leading-relaxed">
            Upload one or more high-quality images of your rooftop for comprehensive AI-powered analysis
          </p>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Area */}
        <div>
          <label className="block text-sm font-semibold text-gray-300 mb-3">
            Select Images <span className="text-gray-500 font-normal">(PNG, JPG, JPEG - Max 10MB each)</span>
          </label>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
              isDragging
                ? 'border-blue-400/50 liquid-glass scale-[1.02]'
                : 'border-white/20 liquid-glass hover:border-white/30'
            } ${isAnalyzing ? 'opacity-50 pointer-events-none' : ''}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/png,image/jpeg,image/jpg"
              onChange={(e) => handleFileChange(e.target.files)}
              className="hidden"
              id="file-upload"
              disabled={isAnalyzing}
            />
            <div className="space-y-6">
              <div className="mx-auto w-20 h-20 bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 rounded-3xl flex items-center justify-center shadow-lg transform hover:scale-110 transition-transform">
                <svg className="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
                >
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Choose Files
                </label>
                <p className="mt-4 text-sm text-gray-400 font-medium">
                  or <span className="text-blue-400">drag and drop</span> images here
                </p>
                <p className="mt-2 text-xs text-gray-500">
                  Supports multiple files • Maximum 10MB per file
                </p>
              </div>
            </div>
          </div>

          {/* File Count and Clear Button */}
          {files.length > 0 && (
            <div className="mt-4 flex items-center justify-between">
              <p className="text-sm font-medium text-gray-300">
                <span className="text-blue-400 font-bold">{files.length}</span> file{files.length > 1 ? 's' : ''} selected
              </p>
              <button
                type="button"
                onClick={clearAllFiles}
                className="text-sm text-red-400 hover:text-red-300 font-medium hover:underline transition-colors"
              >
                Clear all
              </button>
            </div>
          )}
        </div>

        {/* Image Preview */}
        <ImagePreview files={files} onRemove={removeFile} />


        {/* Error Message */}
        {error && (
          <div className="liquid-glass border-l-4 border-red-400/50 text-red-200 px-5 py-4 rounded-xl animate-shake">
            <div className="flex items-start">
              <svg className="w-6 h-6 mr-3 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="flex-1">
                <p className="font-semibold mb-1">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isAnalyzing || files.length === 0}
          className="group w-full bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 text-white py-4 px-8 rounded-xl font-bold text-lg hover:from-blue-700 hover:via-purple-700 hover:to-blue-800 disabled:from-gray-400 disabled:via-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:scale-[1.02] disabled:transform-none disabled:hover:scale-100 flex items-center justify-center space-x-3 relative overflow-hidden"
        >
          <span className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></span>
          {isAnalyzing ? (
            <>
              <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Analyzing Rooftop...</span>
            </>
          ) : (
            <>
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Start AI Analysis</span>
            </>
          )}
        </button>
      </form>
    </div>
  )
}

export default AnalysisForm
