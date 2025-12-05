import { useEffect, useState } from 'react'

interface LoadingProgressProps {
  isAnalyzing: boolean
}

const LoadingProgress = ({ isAnalyzing }: LoadingProgressProps) => {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!isAnalyzing) {
      setProgress(0)
      return
    }

    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) return prev
        return prev + Math.random() * 15
      })
    }, 500)

    return () => clearInterval(interval)
  }, [isAnalyzing])

  if (!isAnalyzing) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50">
      <div className="h-1 bg-blue-200">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300 ease-out"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
      <div className="bg-blue-600 text-white py-3 px-6 shadow-lg">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="font-medium">Analyzing your rooftop...</span>
          </div>
          <span className="text-sm font-semibold">{Math.round(progress)}%</span>
        </div>
      </div>
    </div>
  )
}

export default LoadingProgress

