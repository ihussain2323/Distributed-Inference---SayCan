from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, csv


def main():
    
    #1. Load model and tokenizer
    print("Loading model...")
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    print(f"🚀 Loading tokenizer for {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    print("✅ Tokenizer loaded")

    #2. Create a simpler layer-split device map
    print("\nCreating layer-split device map...")

    # First load model temporarily to get layer count
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="mps",      
        trust_remote_code=True
    )
    
    total_layers = len(temp_model.model.layers)
    cpu_layers = total_layers // 2
    mps_layers = total_layers - cpu_layers

    print(f"Total layers: {total_layers}")
    print(f"CPU layers: {cpu_layers}")
    print(f"MPS layers: {mps_layers}")

    # Create a simpler device map - keep most components on MPS
    device_map = {}
    
    # Put first half of layers on CPU
    for i in range(cpu_layers):
        layer_name = f"model.layers.{i}"
        device_map[layer_name] = "cpu"
    
    # Put second half of layers on MPS
    for i in range(cpu_layers, total_layers):
        layer_name = f"model.layers.{i}"
        device_map[layer_name] = "mps"
    
    # Keep all other components on MPS to avoid device conflicts
    device_map["model.embed_tokens"] = "mps"
    device_map["model.norm"] = "mps"
    device_map["lm_head"] = "mps"
    
    # Add any missing components that might cause issues
    device_map["model.rope"] = "mps"  # Rotary position embeddings
    device_map["model.rotary_emb"] = "mps"  # Alternative name for rotary embeddings
            
    print(f"Device map created with {len(device_map)} components")

    #3. Load model with layer-split device map
    print("\nLoading model with layer-split device map...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        offload_folder="offload"  # Add offload folder for CPU components
    )
    print("✅ Layer-split model loaded")

    #4. Test each prompt and collect results
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
    
    #5. Test each prompt and collect results
    results = []
    print(f"\nTesting {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        start_time = time.time()
        
        try:
            # Tokenize and move to the model's device
            inputs = tok(prompt, return_tensors="pt")
            
            # Move inputs to the model's device (MPS for embedding layer)
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

            result = tok.decode(outputs[0], skip_special_tokens=True)
            end_time = time.time()
            latency = end_time - start_time

            print(f"Time: {latency:.3f} seconds")
            print(f"Response: {result[:100]}...")

            results.append({
                'prompt_number': i+1,
                'prompt': prompt,
                'latency_seconds': latency,
                'response': result
            })
            
        except Exception as e:
            print(f"Error on prompt {i+1}: {e}")
            results.append({
                'prompt_number': i+1,
                'prompt': prompt,
                'latency_seconds': -1,
                'response': f"ERROR: {str(e)}"
            })

    #6. Save results to CSV
    print(f"\nSaving results to CSV...")
    csv_filename = "layer_split_prototype_results.csv"

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['prompt_number','prompt','latency_seconds','response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")

    #7. Print summary
    successful_results = [r for r in results if r['latency_seconds'] > 0]
    if successful_results:
        print(f"\nSUMMARY:")
        print(f"Successful prompts: {len(successful_results)}/{len(results)}")
        print(f"Average latency: {sum(r['latency_seconds'] for r in successful_results)/len(successful_results):.3f} seconds")
        print(f"Total time: {sum(r['latency_seconds'] for r in successful_results):.3f} seconds")
    else:
        print(f"\nSUMMARY: No successful prompts completed")

if __name__ == "__main__":
    main()









