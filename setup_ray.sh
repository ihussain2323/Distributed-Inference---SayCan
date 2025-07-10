#!/bin/bash

# Auto-setup script for Ray Distributed Inference
# Usage: source setup_ray.sh

echo "🚀 Setting up Ray Distributed Inference environment..."

# Deactivate any existing virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "🔄 Deactivating current virtual environment..."
    deactivate
fi

# Navigate to the project directory
cd "Distributed Inference/Ray Distributed"

# Activate the virtual environment
source ../../.venv311/bin/activate

echo "✅ Environment ready!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Virtual env: $VIRTUAL_ENV"

# Check if Ray cluster is running
if pgrep -f "ray start" > /dev/null; then
    echo "✅ Ray cluster already running"
else
    echo "🔧 Starting Ray cluster..."
    ray start --head --port=6379 2>/dev/null || echo "Ray cluster started or already running"
fi

echo ""
echo "🎯 Ready to run:"
echo "  python llama_worker_a.py    # Start Worker A"
echo "  python llama_worker_b.py    # Start Worker B" 
echo "  python llama_client.py      # Run client"
echo "" 