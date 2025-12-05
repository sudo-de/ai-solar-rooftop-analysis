import BlackHoleBackground from '../BlackHoleBackground'

const FirstPage = () => {
  return (
    <section className="relative text-white min-h-screen flex flex-col justify-between overflow-hidden">
      {/* Black Hole Background */}
      <BlackHoleBackground />
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full flex-1 flex flex-col justify-between py-8 sm:py-12 md:py-16">
        <div className="max-w-5xl mx-auto text-center w-full">
          {/* Main Heading - Top */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold leading-tight px-4">
            <span className="block">Transform Your</span>
            <span className="block bg-gradient-to-r from-yellow-300 via-white to-yellow-300 bg-clip-text text-transparent">
              Rooftop Into Power
            </span>
          </h1>
        </div>

        {/* CTA Button - Bottom */}
        <div className="max-w-5xl mx-auto text-center w-full">
          <a
            href="#analyze"
            className="inline-flex items-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-bold text-base sm:text-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-xl hover:shadow-2xl transform hover:scale-105 liquid-glass-hover"
          >
            <span className="mr-2">🚀</span>
            Start Your Analysis
          </a>
        </div>
      </div>
    </section>
  )
}

export default FirstPage

