#!/bin/bash
# Setup aliases to run commands from anywhere

PROJECT_DIR="/Users/ibrahimhussain/Desktop/USC Github REU/Distributed-Inference---SayCan"

echo "Setting up aliases for distributed inference commands..."
echo ""

# Create aliases
alias start-ray="$PROJECT_DIR/start_ray_anywhere.sh"
alias worker-a="$PROJECT_DIR/run_worker_a_anywhere.sh"
alias worker-b="$PROJECT_DIR/run_worker_b_anywhere.sh"
alias client="$PROJECT_DIR/run_client_anywhere.sh"
alias saycan="$PROJECT_DIR/run_saycan_anywhere.sh"

echo "✅ Aliases created! You can now use:"
echo "   start-ray    # Start Ray cluster"
echo "   worker-a     # Run Worker A"
echo "   worker-b     # Run Worker B"
echo "   client       # Run Client"
echo "   saycan       # Run SayCan test"
echo ""
echo "⚠️  Note: These aliases will only work in this terminal session."
echo "   To make them permanent, add them to your ~/.zshrc file."
echo ""
echo "Example usage:"
echo "   start-ray    # Start Ray from anywhere"
echo "   worker-a     # Run Worker A from anywhere" 