#!/bin/bash
# Script to start Ray from ANY directory

# Get the absolute path to the project directory
PROJECT_DIR="/Users/ibrahimhussain/Desktop/USC Github REU/Distributed-Inference---SayCan"

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Navigate to the project directory
cd "$PROJECT_DIR"

echo "🚀 Starting Ray cluster..."
ray start --head --port=6379

echo "✅ Ray cluster started"
echo "📊 Ray dashboard: http://localhost:8265"
echo "�� To stop: ray stop" 