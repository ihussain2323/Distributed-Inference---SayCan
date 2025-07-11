# Distributed Inference with SayCan

## 🚀 Quick Start (Works from ANY directory!)

### Option 1: Use the anywhere scripts (recommended)
```bash
# Start Ray cluster (from anywhere)
/Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/start_ray_anywhere.sh

# Run everything (from anywhere):
/Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/run_worker_a_anywhere.sh    # Terminal 1
/Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/run_worker_b_anywhere.sh    # Terminal 2  
/Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/run_client_anywhere.sh      # Terminal 3

# Test SayCan (from anywhere)
/Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/run_saycan_anywhere.sh
```

### Option 2: Setup aliases (one-time setup)
```bash
# Run this once to setup aliases
source /Users/ibrahimhussain/Desktop/USC\ Github\ REU/Distributed-Inference---SayCan/setup_aliases.sh

# Then use simple commands from anywhere:
start-ray    # Start Ray cluster
worker-a     # Run Worker A
worker-b     # Run Worker B
client       # Run Client
saycan       # Run SayCan test
```

### Option 3: Traditional way (from project directory)
```bash
cd "Distributed-Inference---SayCan"
./start_ray.sh
./run_worker_a.sh    # Terminal 1
./run_worker_b.sh    # Terminal 2  
./run_client.sh      # Terminal 3
./run_saycan.sh
```

## 📁 Files

- `requirements.txt` - Dependencies
- `start_ray_anywhere.sh` - Start Ray from anywhere
- `run_worker_a_anywhere.sh` - Run Worker A from anywhere
- `run_worker_b_anywhere.sh` - Run Worker B from anywhere
- `run_client_anywhere.sh` - Run Client from anywhere
- `run_saycan_anywhere.sh` - Run SayCan from anywhere
- `setup_aliases.sh` - Setup simple aliases
- `saycan-test.py` - SayCan implementation
- `Distributed Inference/Ray Distributed/` - Main inference code

## 🎯 That's it!

No more directory navigation! Run commands from anywhere on your system! 🚀 