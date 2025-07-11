#!/bin/bash

# Run SayCan Clean Test
# This script runs the clean, efficient SayCan implementation

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting SayCan Clean Test"
echo "📁 Project directory: $PROJECT_DIR"

# Activate virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
else
    echo "❌ Virtual environment not found at $PROJECT_DIR/.venv/"
    echo "Please create and activate your virtual environment first"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Run the clean SayCan test
echo "🤖 Running SayCan Clean Test..."
echo "📋 Prerequisites:"
echo "   1. Ray cluster running (ray start --head --port=6379)"
echo "   2. Worker A running (python 'Distributed Inference/Ray Distributed/llama_worker_a.py')"
echo "   3. Worker B running (python 'Distributed Inference/Ray Distributed/llama_worker_b.py')"
echo ""

python saycan_clean.py

echo "✅ SayCan Clean test completed!" 