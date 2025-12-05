const Footer = () => {
  return (
    <footer className="liquid-glass text-white mt-16 border-t border-white/10">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <h3 className="text-lg font-semibold">AI Solar Analysis</h3>
            </div>
            <p className="text-gray-400 text-sm">
              Advanced AI-powered solar rooftop analysis platform with 3D CAD modeling and comprehensive energy predictions.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>
                <a href="#analyze" className="hover:text-white transition-colors">Analyze Rooftop</a>
              </li>
              <li>
                <a href="#about" className="hover:text-white transition-colors">About</a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Features</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>⚡ AI-Powered Analysis</li>
              <li>📊 Energy Predictions</li>
              <li>💰 ROI Calculations</li>
              <li>🤖 Object Detection</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
          <p>&copy; 2025 AI Solar Rooftop Analysis. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer
