import { useState, useEffect } from 'react'

const Header = () => {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled 
        ? 'liquid-glass-strong py-3 border-b border-white/20' 
        : 'liquid-glass py-4 border-b border-white/10'
    }`}>
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between">
          <a href="/" className="flex items-center space-x-3 group">
            <div>
              <h1 className={`font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-600 bg-clip-text text-transparent transition-all duration-300 ${
                scrolled ? 'text-xl' : 'text-2xl'
              }`}>
                AI Solar Analysis
              </h1>
              {!scrolled && (
                <p className="text-xs text-gray-400 -mt-1">Rooftop Analysis</p>
              )}
            </div>
          </a>
          <nav className="hidden md:flex items-center space-x-6">
            <a 
              href="#analyze" 
              className="text-gray-300 hover:text-blue-400 transition-colors font-medium relative group"
            >
              Analyze
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-400 group-hover:w-full transition-all duration-300"></span>
            </a>
            <a 
              href="#features" 
              className="text-gray-300 hover:text-blue-400 transition-colors font-medium relative group"
            >
              Features
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-400 group-hover:w-full transition-all duration-300"></span>
            </a>
            <a 
              href="#about" 
              className="text-gray-300 hover:text-blue-400 transition-colors font-medium relative group"
            >
              About
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-400 group-hover:w-full transition-all duration-300"></span>
            </a>
            <a
              href="#analyze"
              className="px-5 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-105"
            >
              Get Started
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header
