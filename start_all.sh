#!/bin/bash

# Start Both Backend and Frontend

echo "🚀 Starting AI Solar Rooftop Analysis Application..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start Backend in background
echo "📡 Starting Backend (FastAPI)..."
./start_backend.sh > backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   Logs: tail -f backend.log"
echo ""

# Wait a bit for backend to start
sleep 3

# Start Frontend in background
echo "🎨 Starting Frontend (React/Vite)..."
./start_frontend.sh > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   Logs: tail -f frontend.log"
echo ""

echo "✨ Application is running!"
echo ""
echo "📍 Access points:"
echo "   Frontend: http://localhost:5173"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for both processes
wait

