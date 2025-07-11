# Distributed Inference with SayCan

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Start Ray cluster
./start_ray.sh

# Run distributed inference
cd "Distributed Inference/Ray Distributed"
python llama_worker_a.py    # Terminal 1
python llama_worker_b.py    # Terminal 2
python llama_client.py      # Terminal 3

# Test SayCan
cd ../../
python saycan-test.py
```

## Files

- `requirements.txt` - Dependencies
- `activate_env.sh` - Activate environment
- `start_ray.sh` - Start Ray cluster
- `stop_ray.sh` - Stop Ray cluster
- `saycan-test.py` - SayCan implementation
- `Distributed Inference/Ray Distributed/` - Main inference code 