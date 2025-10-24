#!/bin/bash
# Virtual Environment Setup Script for Solar Rooftop Analysis

echo "🚀 Setting up Solar Rooftop Analysis Environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest black flake8 mypy

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs outputs temp samples screenshots

# Set up environment variables
echo "🔐 Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << EOF
# API Keys (replace with your actual keys)
OPENWEATHER_API_KEY=your_openweather_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
SENTINEL_API_KEY=your_sentinel_key_here
LANDSAT_API_KEY=your_landsat_key_here

# Database
DATABASE_URL=sqlite:///solar_analysis.db

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Advanced Features
ENABLE_FEDERATED_LEARNING=True
ENABLE_EDGE_AI=True
ENABLE_BLOCKCHAIN=True
ENABLE_AR_VISUALIZATION=True
EOF
    echo "✅ Created .env file with default configuration"
else
    echo "⚠️  .env file already exists"
fi

echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
