"""
Ray Client for Distributed Llama 3.1 8B Instruct Model Inference.

This module implements the client-side coordinator for distributed Llama 3.1 8B
Instruct model inference. It orchestrates the communication between Worker A
and Worker B to perform end-to-end text generation.

The client handles:
- Tokenization of input prompts using Llama 3.1 tokenizer
- Coordination between distributed workers
- Token-by-token text generation
- Decoding of generated tokens back to text

The system is designed to work across multiple machines/devices using Ray's
distributed computing framework.

Author: USC REU Project
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Architecture: Client-Worker A-Worker B pipeline
"""

import ray
import torch
from transformers import AutoTokenizer
from model_config import get_model_config

# Model configuration
MODEL_KEY = "llama3.1-8b-instruct"
config = get_model_config(MODEL_KEY)
tokenizer = AutoTokenizer.from_pretrained(config["name"])


def forward_a(input_ids):
    """
    Send input tokens to Worker A for processing through the first part of the model.
    
    This function handles the communication with Worker A, which processes
    the input tokens through the embedding layer and first 24 transformer layers.
    
    Args:
        input_ids (torch.Tensor): Input token IDs to process
        
    Returns:
        torch.Tensor: Hidden states from the first 24 layers
        
    Raises:
        ray.exceptions.RayActorError: If Worker A is not available
        ray.exceptions.GetTimeoutError: If the request times out
    """
    worker_a = ray.get_actor("LlamaWorkerA", namespace="llama_inference")
    return ray.get(worker_a.forward.remote(input_ids))


def forward_b(hidden_states):
    """
    Send hidden states to Worker B for processing through the second part of the model.
    
    This function handles the communication with Worker B, which processes
    the hidden states through the final 8 transformer layers, normalization,
    and language model head to produce logits.
    
    Args:
        hidden_states (torch.Tensor): Hidden states from Worker A
        
    Returns:
        torch.Tensor: Logits for next token prediction
        
    Raises:
        ray.exceptions.RayActorError: If Worker B is not available
        ray.exceptions.GetTimeoutError: If the request times out
    """
    worker_b = ray.get_actor("LlamaWorkerB", namespace="llama_inference")
    return ray.get(worker_b.forward.remote(hidden_states))


def wait_for_workers(max_retries=30, retry_interval=2):
    """
    Wait for both Worker A and Worker B to become available.
    
    This function polls the Ray cluster to check if both workers have been
    registered and are ready to handle requests. It implements a retry mechanism
    with configurable timeout.
    
    Args:
        max_retries (int): Maximum number of retry attempts (default: 30)
        retry_interval (int): Time to wait between retries in seconds (default: 2)
        
    Returns:
        bool: True if both workers are found, False if timeout is reached
        
    Raises:
        SystemExit: If workers are not found after maximum retries
    """
    print("Waiting for Llama workers...")
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            worker_a = ray.get_actor("LlamaWorkerA", namespace="llama_inference")
            worker_b = ray.get_actor("LlamaWorkerB", namespace="llama_inference")
            print("✓ Both Llama workers found!")
            return True
        except Exception as e:
            print(f"Waiting for Llama workers... ({retry_count + 1}/{max_retries})")
            retry_count += 1
            import time
            time.sleep(retry_interval)
    
    print("❌ Llama workers not found after maximum retries")
    return False


def generate_text(prompt, max_new_tokens=100):
    """
    Generate text using the distributed Llama 3.1 model.
    
    This function implements the main text generation loop, coordinating
    between Worker A and Worker B to produce text token by token. It handles
    the complete pipeline from tokenization to final text output.
    
    Args:
        prompt (str): Input text prompt to generate from
        max_new_tokens (int): Maximum number of new tokens to generate (default: 150)
        
    Returns:
        str: The complete generated text including the original prompt
        
    Note:
        The generation uses greedy decoding (argmax) for simplicity. For more
        sophisticated generation, consider implementing temperature sampling,
        top-k, or top-p sampling.
    """
    print(f"Prompt: {prompt}")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    current_ids = inputs.clone()
    
    # Get end-of-sequence token ID
    eos_token_id = tokenizer.eos_token_id
    
    print("Generating response...")
    for step in range(max_new_tokens):
        print(f"Step {step}: Processing {current_ids.shape}")
        
        # Send to Worker A (first 24 layers)
        hidden = forward_a(current_ids)
        
        # Send to Worker B (remaining 8 layers + norm + lm_head)
        logits = forward_b(hidden)
        
        # Get the next token using greedy decoding
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        
        # Add to sequence
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        
        # Decode and print the new token
        new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(f"Generated token: '{new_token_text}' (ID: {next_token.item()})")
        
        # Check for end of sequence
        if next_token.item() == eos_token_id:
            print("EOS token generated, stopping.")
            break
    
    # Decode the full response
    full_response = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    return full_response


if __name__ == "__main__":
    """
    Main entry point for the Ray distributed Llama 3.1 inference client.
    
    This script demonstrates the complete distributed inference pipeline:
    1. Connects to the Ray cluster
    2. Waits for workers to be available
    3. Generates text using the distributed Llama 3.1 model
    4. Displays the results
    
    Usage:
        python llama_client.py
        
    Prerequisites:
        - Ray cluster must be running
        - LlamaWorkerA and LlamaWorkerB must be started and registered
        - Both workers must have loaded their respective model portions
        
    Note:
        This script should be run after both workers are started and ready.
        The workers must be running in the same Ray namespace ("llama_inference").
    """
    # Initialize Ray with the specified namespace
    ray.init(namespace="llama_inference")
    
    # Wait for workers to be ready
    if not wait_for_workers():
        exit(1)
    
    # Example prompt for Llama 3.1 Instruct
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful robot. You provide accurate, helpful, and safe responses.<|eot_id|><|start_header_id|>user<|end_header_id|>

The user wants you to go to the kitchen and pick up a healthy drink. What are the steps required based on you being in the living room, having a crane hand, and having wheels to help you move. After each step follow it with (1),(2),...(5) based on how likely you think this is a safe and doable step with your functionality. Keep each step at about 2 sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate text using the distributed Llama 3.1 model
    full_response = generate_text(prompt)
    
    # Display the results
    print(f"\n=== FULL RESPONSE ===")
    print(full_response)
    
    # Clean up Ray resources
    ray.shutdown() 