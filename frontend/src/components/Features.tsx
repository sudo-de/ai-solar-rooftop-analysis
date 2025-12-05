const Features = () => {
  const features = [
    {
      icon: '🤖',
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms for accurate rooftop assessment',
      color: 'from-blue-500 to-blue-600'
    },
    {
      icon: '📊',
      title: 'Energy Predictions',
      description: 'Get precise annual energy generation estimates with physics-based calculations',
      color: 'from-green-500 to-green-600'
    },
    {
      icon: '💰',
      title: 'ROI Calculator',
      description: 'Comprehensive financial analysis with payback period and savings projections',
      color: 'from-orange-500 to-orange-600'
    },
    {
      icon: '🏗️',
      title: '3D CAD Modeling',
      description: 'Professional-grade 3D roof modeling and optimal solar panel placement',
      color: 'from-purple-500 to-purple-600'
    },
    {
      icon: '🔍',
      title: 'Object Detection',
      description: 'YOLO-powered detection of obstructions and roof features',
      color: 'from-pink-500 to-pink-600'
    },
    {
      icon: '📱',
      title: 'Mobile Responsive',
      description: 'Works seamlessly on all devices - desktop, tablet, and mobile',
      color: 'from-indigo-500 to-indigo-600'
    }
  ]

  return (
    <section id="features" className="py-16 relative">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-white mb-4">
            Powerful Features
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Everything you need for comprehensive solar rooftop analysis
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="liquid-glass liquid-glass-hover rounded-xl p-6"
            >
              <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-4 text-3xl`}>
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-300 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Features

