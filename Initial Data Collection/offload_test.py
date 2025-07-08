from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, csv, signal

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# Prompts to test (same three as before)
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

# Timeout handler (30 s cap per prompt)
class TimeoutError(Exception): pass
def timeout_handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, timeout_handler)

def main():
    # Load entire model on MPS, with offload
    print("Loading model with offload enabled…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",               # let 🤗 place layers on MPS/CPU as needed
        offload_folder="offload",        # spill CPU weights to disk
        offload_state_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        try:
            signal.alarm(30)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            latency = time.perf_counter() - start
            signal.alarm(0)
        except TimeoutError:
            latency, text = -1, "TIMEOUT"
        except Exception as e:
            latency, text = -1, f"ERROR: {e}"
            signal.alarm(0)

        print(f"Latency: {latency:.3f}s")
        results.append({
            "prompt_number": i+1,
            "prompt": prompt,
            "latency_seconds": latency
        })

    # Save to CSV
    with open("offload_results.csv","w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("\nOffload test complete! Results in offload_results.csv")

if __name__ == "__main__":
    main()