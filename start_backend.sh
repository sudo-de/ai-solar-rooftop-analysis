#!/bin/bash

# Start Backend Server (Python FastAPI)

echo "🚀 Starting Backend Server..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Navigate to backend directory
cd backend

# Install backend dependencies if needed
if [ ! -d "venv" ] || [ ! -f "../venv/bin/activate" ]; then
    echo "📦 Installing backend dependencies..."
    pip install -r requirements.txt
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "📦 Installing uvicorn..."
    pip install uvicorn[standard]
fi

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000

