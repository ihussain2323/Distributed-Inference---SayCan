#!/bin/bash
# Script to run Worker B from ANY directory

# Get the absolute path to the project directory
PROJECT_DIR="/Users/ibrahimhussain/Desktop/USC Github REU/Distributed-Inference---SayCan"

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Navigate to the Ray Distributed directory
cd "$PROJECT_DIR/Distributed Inference/Ray Distributed"

echo "🚀 Starting Worker B..."
echo "📁 Directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Virtual env: $VIRTUAL_ENV"

# Run Worker B
python3 llama_worker_b.py 