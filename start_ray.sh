#!/bin/bash
# Start Ray cluster for distributed inference
source .venv/bin/activate

echo "🚀 Starting Ray cluster..."
ray start --head --port=6379

echo "✅ Ray cluster started"
echo "📊 Ray dashboard: http://localhost:8265"
echo "🛑 To stop: ray stop"
