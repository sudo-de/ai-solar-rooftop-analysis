import { useState } from 'react'
import Header from './components/Header'
import Hero from './components/Hero/index.tsx'
import Features from './components/Features'
import AnalysisForm from './components/AnalysisForm'
import ResultsDisplay from './components/ResultsDisplay'
import Footer from './components/Footer'
import LoadingProgress from './components/LoadingProgress'
import Toast from './components/Toast'
import './App.css'

function App() {
  const [results, setResults] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null)

  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ message, type })
  }

  const handleAnalysisComplete = (analysisResults: any) => {
    setResults(analysisResults)
    setIsAnalyzing(false)
    showToast('✨ Analysis completed successfully!', 'success')
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
    setResults(null)
    showToast('🚀 Starting analysis...', 'info')
  }

  return (
    <div className="min-h-screen bg-black text-white relative">
      <LoadingProgress isAnalyzing={isAnalyzing} />
      <Header />
      <main className="pt-20 relative z-10">
        <Hero />
        <Features />
        <div className="container mx-auto px-4 py-16 max-w-6xl">
          <AnalysisForm 
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            isAnalyzing={isAnalyzing}
          />
          {results && <ResultsDisplay results={results} />}
      </div>
      </main>
      <Footer />
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
      </div>
  )
}

export default App
