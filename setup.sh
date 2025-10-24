#!/bin/bash
# Complete Setup Script for Solar Rooftop Analysis System

echo "🚀 Setting up Solar Rooftop Analysis System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p uploads outputs logs ssl

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

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check backend
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend health check failed"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🛠️  Management commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Update services: docker-compose pull && docker-compose up -d"
echo ""
echo "📚 Next steps:"
echo "   1. Update .env file with your API keys"
echo "   2. Visit http://localhost:3000 to start using the application"
echo "   3. Check http://localhost:8000/docs for API documentation"
