#!/bin/bash

# Run SayCan Frontend
# This script starts the web interface for the SayCan project

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo "🚀 Starting SayCan Frontend"
echo "📁 Project directory: $PROJECT_DIR"
echo "🌐 Frontend directory: $FRONTEND_DIR"

# Activate virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "❌ Virtual environment not found at $PROJECT_DIR/.venv/"
    echo "Please create and activate your virtual environment first"
    exit 1
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
pip install -r "$FRONTEND_DIR/requirements.txt"

# Change to frontend directory
cd "$FRONTEND_DIR"

# Start the Flask server
echo "🌐 Starting Flask server on port 5002..."
echo "📋 Prerequisites:"
echo "   1. Ray cluster running (ray start --head --port=6379)"
echo "   2. Worker A running (python 'Distributed Inference/Ray Distributed/llama_worker_a.py')"
echo "   3. Worker B running (python 'Distributed Inference/Ray Distributed/llama_worker_b.py')"
echo "🔗 Open http://localhost:5002 in your browser"
echo ""

python app.py --port 5002 