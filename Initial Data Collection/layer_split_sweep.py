from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, csv, os, signal
import gc
import psutil  # Add this if you want to monitor system resources

model_id = "meta-llama/Llama-3.2-11B-Vision"
tok = AutoTokenizer.from_pretrained(model_id)

# Prompts to test
prompts = [
    "What is the capital of the USA?",
    "Tell me about if robotics or AI is more important for the future of humanity",
    "Encrypt this message: 'Hello, world!' with caesar cipher.",
    "Write a poem about mountains and the ocean.",
    "Describe quantum mechanics easily.",
    "What is the capital of France?",
    "Explain machine learning in simple terms.",
    "Write a short story about a robot.",
    "What are the benefits of renewable energy?",
    "How does a computer work?"
]

# Load a temp model to get layer count
temp_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="mps",
    trust_remote_code=True
)
total_layers = len(temp_model.model.layers)
del temp_model

results = []
# Strategic CPU/GPU ratios to test:
# - 0 CPU: Pure GPU baseline
# - 5 CPU: Light CPU offload (early layers)
# - 8 CPU: Moderate CPU offload 
# - 12 CPU: Heavy CPU offload (late layers)
# - 15 CPU: Maximum safe CPU offload
sweep_steps = [0, 5, 8, 12, 15]

# Add timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Set timeout for each prompt (30 seconds)
signal.signal(signal.SIGALRM, timeout_handler)

for cpu_layers in sweep_steps:
    mps_layers = total_layers - cpu_layers
    print(f"\n=== Testing split: {cpu_layers} CPU / {mps_layers} MPS ===")

    # Device Map Building:
    device_map = {}
    for i in range(cpu_layers):
        device_map[f"model.layers.{i}"] = "cpu"
    for i in range(cpu_layers, total_layers):
        device_map[f"model.layers.{i}"] = "mps"

    # Always keep these on MPS
    device_map["model.embed_tokens"] = "mps"
    device_map["model.norm"] = "mps"
    device_map["lm_head"] = "mps"
    device_map["model.rope"] = "mps"
    device_map["model.rotary_emb"] = "mps"

    # Load model with split
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        offload_folder="offload"
    )

    # Run prompts
    for i, prompt in enumerate(prompts):
        try:
            signal.alarm(30)  # 30 second timeout
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            ) 
            latency = time.time() - start
            result = tok.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt {i+1} latency: {latency:.2f}s")
            signal.alarm(0)  # Cancel timeout
    
        except TimeoutError:
            print(f"Prompt {i+1} timed out after 30 seconds")
            latency = -1
            result = "TIMEOUT"
        except Exception as e:
            latency = -1
            result = f"ERROR: {e}"
            print(f"Prompt {i+1} error: {e}")
            signal.alarm(0)  # Cancel timeout

        results.append({
            "cpu_layers": cpu_layers,
            "mps_layers": mps_layers,
            "prompt_number": i+1,
            "prompt": prompt,
            "latency_seconds": latency,
            "response": result[:200]
        })

    # Free memory between sweeps
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()  # Force garbage collection

# Save results
csv_file = "layer_split_sweep_results.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nSweep complete! Results saved to {csv_file}") 