'use client';

import { useState } from 'react';
import { Header } from '@/components/Header';
import Hero from '@/components/Hero';
import { AnalysisForm } from '@/components/AnalysisForm';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { Features } from '@/components/Features';
import { Footer } from '@/components/Footer';

export default function Home() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalysis = async (formData: FormData) => {
    setIsAnalyzing(true);
    try {
      // Send FormData to the API route
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData, // Send FormData directly
      });
      
      const results = await response.json();
      setAnalysisResults(results);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-orange-50">
      <Header />
      
      <main>
        <Hero />
        
        <section className="py-16 px-4">
          <div className="max-w-7xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-12 items-start">
              <div className="space-y-8">
                <div>
                  <h2 className="text-3xl font-bold text-gray-900 mb-4">
                    Analyze Your Rooftop
                  </h2>
                  <p className="text-lg text-gray-600">
                    Upload satellite images and get comprehensive solar potential analysis 
                    powered by cutting-edge AI technology.
                  </p>
                </div>
                
                <AnalysisForm 
                  onAnalyze={handleAnalysis}
                  isAnalyzing={isAnalyzing}
                />
              </div>
              
              <div className="space-y-8">
                {analysisResults ? (
                  <ResultsDisplay results={analysisResults} />
                ) : (
                  <div className="card">
                    <div className="text-center py-12">
                      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">
                        Ready for Analysis
                      </h3>
                      <p className="text-gray-600">
                        Upload your rooftop images to get started with our AI analysis.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
        
        <Features />
      </main>
      
      <Footer />
    </div>
  );
}
