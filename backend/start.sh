#!/bin/bash
set -e

echo "Starting AI Solar Rooftop Analysis Backend..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Create necessary directories
mkdir -p uploads outputs logs

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found!"
    exit 1
fi

# Check if uvicorn is installed
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "ERROR: uvicorn not installed!"
    exit 1
fi

# Check if fastapi is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "ERROR: fastapi not installed!"
    exit 1
fi

echo "All checks passed. Starting uvicorn server..."

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

