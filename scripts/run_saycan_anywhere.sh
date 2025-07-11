#!/bin/bash
# Script to run SayCan from ANY directory

# Get the absolute path to the project directory
PROJECT_DIR="/Users/ibrahimhussain/Desktop/USC Github REU/Distributed-Inference---SayCan"

# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Navigate to the project directory
cd "$PROJECT_DIR"

echo "🚀 Running SayCan test..."
echo "📁 Directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Virtual env: $VIRTUAL_ENV"

# Run SayCan test
python3 saycan-test.py 