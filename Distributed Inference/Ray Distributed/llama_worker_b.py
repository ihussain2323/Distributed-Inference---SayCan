"""
Ray Worker B for Distributed Llama 3.1 8B Instruct Model Inference.

This module implements the second part of a distributed Llama 3.1 8B Instruct model,
handling the final 8 transformer layers (layers 24-31), normalization, and language
model head. It works as part of a distributed inference pipeline where Worker B
receives hidden states from Worker A and produces the final logits for token prediction.

The worker is designed to run as a Ray actor, allowing it to be deployed across
multiple machines or devices while maintaining state and handling concurrent requests.

Author: USC REU Project
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Architecture: Distributed transformer layers (24-31) + norm + lm_head out of 32 total layers
"""

import ray
import torch
from transformers import AutoModelForCausalLM

# Model configuration
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class LlamaTailB(torch.nn.Module):
    """
    Second half of the Llama 3.1 model containing final layers and output head.
    
    This class implements the final 8 layers of the Llama 3.1 8B Instruct model, including:
    - Transformer layers 24-31
    - Final normalization layer
    - Language model head (vocabulary projection)
    
    The model is designed to receive hidden states from Worker A and produce
    logits for the next token prediction.
    
    Attributes:
        device (torch.device): The device (GPU/CPU) where the model is loaded
        layers (nn.ModuleList): Final 8 transformer layers (24-31)
        norm (nn.Module): Final normalization layer
        lm_head (nn.Module): Language model head for vocabulary projection
        rotary_emb (nn.Module): Rotary position embedding layer
    """
    
    def __init__(self):
        """
        Initialize the LlamaTailB model by loading and splitting the Llama 3.1 model.
        
        Loads the full Llama 3.1 8B Instruct model and extracts the final 8 layers along
        with the normalization and language model head components. The model
        is moved to the specified device (MPS for Apple Silicon).
        
        Raises:
            RuntimeError: If model loading fails or device is unavailable
        """
        super().__init__()
        print("Worker B loading model...")
        
        # Load the full model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Set device (MPS for Apple Silicon, can be changed to CUDA for NVIDIA GPUs)
        self.device = torch.device("mps")
        
        # Extract model components for the second half
        # Llama 3.1 8B has 32 total layers (0-31)
        # Worker B handles layers 24-31 (8 layers) + norm + lm_head
        self.layers = torch.nn.ModuleList(list(model.model.layers[24:])).to(self.device)
        self.norm = model.model.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)
        
        # Need the rotary embeddings for position encoding
        self.rotary_emb = model.model.rotary_emb.to(self.device)
        
        print("Worker B model loaded...")
    
    def forward(self, hidden_states):
        """
        Forward pass through the second half of the Llama 3.1 model.
        
        Processes hidden states from Worker A through the final 8 transformer
        layers, applies final normalization, and projects to vocabulary logits.
        Returns logits that can be used for next token prediction.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from Worker A of shape
                                        (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)
                         where vocab_size is 128256 for Llama 3.1 8B
                         
        Note:
            The output is moved to CPU for transfer back to the client, as Ray
            handles data transfer between workers automatically.
        """
        print(f"Received hidden states {hidden_states.shape}")
        
        # Move input to device
        x = hidden_states.to(self.device)
        
        # Create position embeddings like the full model does
        batch_size, seq_len, hidden_size = x.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(x, position_ids)
        
        # Pass through each transformer layer with position embeddings
        for layer in self.layers:
            # Handle layer output properly like working RPC pipeline
            layer_output = layer(x, position_embeddings=position_embeddings)
            x = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary logits
        logits = self.lm_head(x)
        
        print(f"returning {logits.shape}")
        return logits.cpu()


@ray.remote
class LlamaWorkerB:
    """
    Ray actor representing Worker B for distributed Llama 3.1 inference.
    
    This Ray actor wraps the LlamaTailB model and provides a remote interface
    for processing requests. It maintains the model state and handles concurrent
    inference requests from the client.
    
    The actor is designed to be deployed on a separate machine or device,
    allowing the model to be distributed across multiple hardware resources.
    
    Attributes:
        model (LlamaTailB): The loaded LlamaTailB model instance
    """
    
    def __init__(self):
        """
        Initialize the LlamaWorkerB actor by loading the LlamaTailB model.
        
        Creates a new instance of LlamaTailB and loads it into memory.
        This method is called when the Ray actor is created.
        """
        self.model = LlamaTailB()
    
    def forward(self, hidden_states):
        """
        Process hidden states through the second part of the Llama 3.1 model.
        
        This method is called remotely by the client to process hidden states
        from Worker A. It runs inference with gradient computation disabled
        for efficiency.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from Worker A
            
        Returns:
            torch.Tensor: Logits for next token prediction
        """
        with torch.no_grad():
            return self.model(hidden_states)


if __name__ == "__main__":
    """
    Main entry point for Llama Worker B.
    
    Initializes Ray, creates the LlamaWorkerB actor, and keeps it alive to handle
    incoming requests. The worker will continue running until manually stopped.
    
    Usage:
        python llama_worker_b.py
        
    Note:
        This script should be run on the machine/device where Worker B will
        be deployed. It will connect to the Ray cluster and register itself
        as a named actor that can be found by the client.
    """
    # Initialize Ray with the specified namespace
    ray.init(namespace="llama_inference")
    
    # Create named actor so client can find it
    worker = LlamaWorkerB.options(name="LlamaWorkerB").remote()
    print("Llama Worker B ready")
    
    # Keep the worker alive
    import time
    while True:
        time.sleep(1) 