import React from 'react';

export default function Hero() {
  return (
    <section className="relative py-20 px-4 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-orange-50" />
      <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%230ea5e9%22%20fill-opacity%3D%220.05%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%222%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-40" />
      
      <div className="relative max-w-7xl mx-auto text-center">
        <div className="space-y-8">
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
            <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M11.3 3.3a1 1 0 011.4 0l6 6 2.3 2.3a1 1 0 01-1.4 1.4L19 12.4V19a1 1 0 11-2 0v-5.6l-4.3 4.3a1 1 0 01-1.4-1.4l6-6z" clipRule="evenodd" />
            </svg>
            Powered by AI Technology
          </div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 leading-tight">
            Solar Rooftop Analysis
            <span className="block bg-gradient-to-r from-blue-600 to-orange-500 bg-clip-text text-transparent">Powered by AI</span>
          </h1>

          {/* Subheading */}
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Get comprehensive solar potential analysis with cutting-edge technology including 
            vision transformers, physics-informed AI, and AR visualization. 
            <span className="font-semibold text-blue-600">95%+ accuracy</span> guaranteed.
          </p>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">95%+</div>
              <div className="text-sm text-gray-600">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">&lt;5%</div>
              <div className="text-sm text-gray-600">Prediction Error</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">6x</div>
              <div className="text-sm text-gray-600">Faster Processing</div>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-blue-600 hover:bg-blue-700 text-white text-lg px-8 py-3 rounded-lg shadow-lg transition-colors">
              Start Analysis
            </button>
            <button className="bg-gray-200 hover:bg-gray-300 text-gray-800 text-lg px-8 py-3 rounded-lg transition-colors">
              View Demo
            </button>
          </div>

          {/* Trust Indicators */}
          <div className="pt-8 border-t border-gray-200">
            <p className="text-sm text-gray-500 mb-4">Trusted by leading organizations</p>
            <div className="flex justify-center items-center space-x-8 opacity-60">
              <div className="text-lg font-semibold text-gray-400">Google</div>
              <div className="text-lg font-semibold text-gray-400">Microsoft</div>
              <div className="text-lg font-semibold text-gray-400">Tesla</div>
              <div className="text-lg font-semibold text-gray-400">SolarCity</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
