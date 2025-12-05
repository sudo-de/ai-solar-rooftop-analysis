#!/bin/bash

# Start Frontend Server (React/Vite)

echo "🚀 Starting Frontend Server..."

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the Vite development server
echo "🌐 Starting Vite dev server on http://localhost:5173"
echo ""
npm run dev

