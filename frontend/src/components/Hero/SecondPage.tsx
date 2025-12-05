import { useEffect, useRef, useState } from 'react'
import './SolarSystem.css'

const SecondPage = () => {
  const [isVisible, setIsVisible] = useState(false)
  const sectionRef = useRef<HTMLElement>(null)
  const leftRef = useRef<HTMLDivElement>(null)
  const centerRef = useRef<HTMLDivElement>(null)
  const rightRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          // Staggered animations for columns
          setTimeout(() => {
            if (leftRef.current) leftRef.current.style.opacity = '1'
            if (leftRef.current) leftRef.current.style.transform = 'translateX(0)'
          }, 200)
          setTimeout(() => {
            if (centerRef.current) centerRef.current.style.opacity = '1'
            if (centerRef.current) centerRef.current.style.transform = 'scale(1)'
          }, 400)
          setTimeout(() => {
            if (rightRef.current) rightRef.current.style.opacity = '1'
            if (rightRef.current) rightRef.current.style.transform = 'translateX(0)'
          }, 600)
        }
      },
      { threshold: 0.1 }
    )

    if (sectionRef.current) {
      observer.observe(sectionRef.current)
    }

    return () => {
      if (sectionRef.current) {
        observer.unobserve(sectionRef.current)
      }
    }
  }, [])

  return (
    <section 
      ref={sectionRef}
      className={`relative text-white min-h-screen flex items-center justify-center overflow-hidden transition-all duration-1000 ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-20'
      }`}
    >
      {/* Liquid Glass Overlay */}
      <div className="absolute inset-0 liquid-glass"></div>
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full py-12 sm:py-16 md:py-20">
        <div className="max-w-7xl mx-auto w-full">
          {/* Section Title - Top */}
          <div className="text-center mb-12 sm:mb-16 md:mb-20">
            <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-extrabold px-4 mb-3 bg-gradient-to-r from-yellow-300 via-white to-yellow-300 bg-clip-text text-transparent">
              Solar System Visualization
            </h2>
            <div className="w-24 h-1 bg-gradient-to-r from-transparent via-blue-400 to-transparent mx-auto rounded-full"></div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 sm:gap-10 lg:gap-12 items-center">
            {/* Left Side - Subtitle */}
            <div 
              ref={leftRef}
              className="lg:col-span-1 flex items-center justify-center lg:justify-start transition-all duration-700 ease-out opacity-0 lg:translate-x-[-30px] translate-x-0"
            >
              <div className="w-full max-w-md group">
                <div className="relative">
                  <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity duration-300"></div>
                  <p className="relative text-base sm:text-lg md:text-xl text-gray-200 leading-relaxed liquid-glass liquid-glass-strong rounded-2xl px-6 sm:px-8 md:px-10 py-6 sm:py-8 backdrop-blur-xl">
                    Discover your rooftop's solar potential with <span className="font-semibold text-white bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent">advanced AI analysis</span>. 
                    Get instant 3D modeling, precise energy predictions, and detailed ROI calculations—all in one powerful platform.
                  </p>
                </div>
              </div>
            </div>

            {/* Center - Solar System */}
            <div 
              ref={centerRef}
              className="lg:col-span-1 flex items-center justify-center transition-all duration-700 ease-out opacity-0 scale-95"
            >
              <div className="relative w-full max-w-lg">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
                <div className="relative solar-syst w-full flex items-center justify-center">
                  <div className="sun"></div>
                  <div className="mercury"></div>
                  <div className="venus"></div>
                  <div className="earth"></div>
                  <div className="mars"></div>
                  <div className="jupiter"></div>
                  <div className="saturn"></div>
                  <div className="uranus"></div>
                  <div className="neptune"></div>
                  <div className="pluto"></div>
                  <div className="asteroids-belt"></div>
                </div>
              </div>
            </div>

            {/* Right Side - Feature Badges */}
            <div 
              ref={rightRef}
              className="lg:col-span-1 flex items-center justify-center lg:justify-end transition-all duration-700 ease-out opacity-0 lg:translate-x-[30px] translate-x-0"
            >
              <div className="flex flex-col gap-4 sm:gap-5 w-full max-w-md">
                <div className="liquid-glass liquid-glass-hover rounded-xl px-5 sm:px-6 py-3 sm:py-4 w-full text-center lg:text-left group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-blue-500/20">
                  <span className="text-sm sm:text-base font-semibold flex items-center justify-center lg:justify-start gap-2">
                    <span className="text-xl sm:text-2xl group-hover:scale-110 transition-transform duration-300">⚡</span>
                    <span className="bg-gradient-to-r from-yellow-300 to-yellow-200 bg-clip-text text-transparent">Lightning Fast</span>
                  </span>
                </div>
                <div className="liquid-glass liquid-glass-hover rounded-xl px-5 sm:px-6 py-3 sm:py-4 w-full text-center lg:text-left group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-purple-500/20">
                  <span className="text-sm sm:text-base font-semibold flex items-center justify-center lg:justify-start gap-2">
                    <span className="text-xl sm:text-2xl group-hover:scale-110 transition-transform duration-300">🎯</span>
                    <span className="bg-gradient-to-r from-blue-300 to-blue-200 bg-clip-text text-transparent">Precision AI</span>
                  </span>
                </div>
                <div className="liquid-glass liquid-glass-hover rounded-xl px-5 sm:px-6 py-3 sm:py-4 w-full text-center lg:text-left group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-green-500/20">
                  <span className="text-sm sm:text-base font-semibold flex items-center justify-center lg:justify-start gap-2">
                    <span className="text-xl sm:text-2xl group-hover:scale-110 transition-transform duration-300">📈</span>
                    <span className="bg-gradient-to-r from-green-300 to-green-200 bg-clip-text text-transparent">Smart Insights</span>
                  </span>
                </div>
                <div className="liquid-glass liquid-glass-hover rounded-xl px-5 sm:px-6 py-3 sm:py-4 w-full text-center lg:text-left group cursor-pointer transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-amber-500/20">
                  <span className="text-sm sm:text-base font-semibold flex items-center justify-center lg:justify-start gap-2">
                    <span className="text-xl sm:text-2xl group-hover:scale-110 transition-transform duration-300">💎</span>
                    <span className="bg-gradient-to-r from-amber-300 to-amber-200 bg-clip-text text-transparent">Maximum ROI</span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default SecondPage
