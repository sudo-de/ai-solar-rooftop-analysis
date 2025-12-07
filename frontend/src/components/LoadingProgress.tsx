import { useEffect, useState } from 'react'

interface LoadingProgressProps {
  isAnalyzing: boolean
}

interface Step {
  id: number
  label: string
  status: 'pending' | 'active' | 'completed'
}

const LoadingProgress = ({ isAnalyzing }: LoadingProgressProps) => {
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)

  const steps: Step[] = [
    { id: 1, label: 'Roof Segmentation', status: 'pending' },
    { id: 2, label: 'Object Detection', status: 'pending' },
    { id: 3, label: 'Zone Optimization', status: 'pending' },
    { id: 4, label: 'Solar Optimization', status: 'pending' }
  ]

  useEffect(() => {
    if (!isAnalyzing) {
      setProgress(0)
      setCurrentStep(0)
      return
    }

    // Simulate progress updates (slower for longer processing times)
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) return prev // Stop at 95% until actual completion
        // Slower progress increment for longer processing
        return prev + Math.random() * 2
      })
    }, 800) // Slower update interval

    // Update steps based on progress
    const stepInterval = setInterval(() => {
      if (progress < 25) {
        setCurrentStep(1)
      } else if (progress < 50) {
        setCurrentStep(2)
      } else if (progress < 75) {
        setCurrentStep(3)
      } else if (progress < 90) {
        setCurrentStep(4)
      } else {
        setCurrentStep(5) // All completed
      }
    }, 1000)

    return () => {
      clearInterval(progressInterval)
      clearInterval(stepInterval)
    }
  }, [isAnalyzing, progress])

  useEffect(() => {
    if (!isAnalyzing && progress === 0) {
      setCurrentStep(0)
    } else if (isAnalyzing && progress >= 100) {
      setCurrentStep(5)
    }
  }, [isAnalyzing, progress])

  if (!isAnalyzing) return null

  const getStepStatus = (stepId: number): 'pending' | 'active' | 'completed' => {
    if (stepId < currentStep) return 'completed'
    if (stepId === currentStep) return 'active'
    return 'pending'
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-blue-600 to-blue-700 shadow-xl">
      <div className="container mx-auto px-6 py-6">
        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-white font-bold text-lg">Analyzing your rooftop photos...</h3>
            <span className="text-white font-semibold text-lg">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-blue-800 rounded-full h-6 overflow-hidden shadow-inner">
        <div
              className="h-full bg-gradient-to-r from-blue-400 to-blue-300 transition-all duration-500 ease-out flex items-center justify-center"
          style={{ width: `${Math.min(progress, 100)}%` }}
            >
              <span className="text-xs font-semibold text-blue-900">
                {Math.round(progress)}%
              </span>
      </div>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {steps.map((step) => {
            const status = getStepStatus(step.id)
            return (
              <div
                key={step.id}
                className={`flex items-center space-x-2 p-3 rounded-lg transition-all duration-300 ${
                  status === 'active'
                    ? 'bg-blue-500 text-white shadow-lg'
                    : status === 'completed'
                    ? 'bg-green-500 text-white'
                    : 'bg-blue-800/50 text-blue-200'
                }`}
              >
                {status === 'completed' ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                ) : status === 'active' ? (
                  <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                      clipRule="evenodd"
                    />
                  </svg>
                )}
                <span className="text-sm font-medium">{step.label}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default LoadingProgress

