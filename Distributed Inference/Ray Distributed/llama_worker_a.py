"""
Ray Worker A for Distributed Llama 3.1 8B Instruct Model Inference.

This module implements the first part of a distributed Llama 3.1 8B Instruct model,
handling the embedding layer and the first 24 transformer layers (layers 0-23).
It works as part of a distributed inference pipeline where Worker A processes
the initial layers and passes hidden states to Worker B for final processing.

The worker is designed to run as a Ray actor, allowing it to be deployed across
multiple machines or devices while maintaining state and handling concurrent requests.

Author: USC REU Project
Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Architecture: Distributed transformer layers (0-23) out of 32 total layers
"""

import ray
import torch
from transformers import AutoModelForCausalLM

# Model configuration
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class LlamaHeadA(torch.nn.Module):
    """
    First half of the Llama 3.1 model containing embedding and initial transformer layers.
    
    This class implements the first 24 layers of the Llama 3.1 8B Instruct model, including:
    - Token embeddings
    - Rotary position embeddings  
    - Transformer layers 0-23
    
    The model is designed to process input tokens and output hidden states that
    can be passed to the second half of the model (Worker B).
    
    Attributes:
        device (torch.device): The device (GPU/CPU) where the model is loaded
        embed (nn.Module): Token embedding layer
        layers (nn.ModuleList): First 24 transformer layers
        rotary_emb (nn.Module): Rotary position embedding layer
    """
    
    def __init__(self):
        """
        Initialize the LlamaHeadA model by loading and splitting the Llama 3.1 model.
        
        Loads the full Llama 3.1 8B Instruct model and extracts the first 24 layers along
        with the embedding and rotary position embedding components. The model
        is moved to the specified device (MPS for Apple Silicon).
        
        Raises:
            RuntimeError: If model loading fails or device is unavailable
        """
        super().__init__()
        print("Worker A loading model...")
        
        # Load the full model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Set device (MPS for Apple Silicon, can be changed to CUDA for NVIDIA GPUs)
        self.device = torch.device("mps")
        
        # Extract model components for the first half
        # Llama 3.1 8B has 32 total layers (0-31)
        # Worker A handles layers 0-23 (24 layers)
        self.embed = model.model.embed_tokens.to(self.device)
        self.layers = torch.nn.ModuleList(list(model.model.layers[:24])).to(self.device)
        
        # Need the rotary embeddings for position encoding
        self.rotary_emb = model.model.rotary_emb.to(self.device)
        
        print("Worker A model loaded...")
    
    def forward(self, input_ids):
        """
        Forward pass through the first half of the Llama 3.1 model.
        
        Processes input token IDs through the embedding layer and first 24
        transformer layers, applying rotary position embeddings at each layer.
        Returns hidden states that can be passed to the second half of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Hidden states of shape (batch_size, seq_len, hidden_size)
                         where hidden_size is 4096 for Llama 3.1 8B
                         
        Note:
            The output is moved to CPU for transfer to Worker B, as Ray handles
            data transfer between workers automatically.
        """
        print(f"Received input ids {input_ids.shape}")
        
        # Convert input to embeddings
        x = self.embed(input_ids.to(self.device))
        
        # Create position embeddings like the full model does
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(x, position_ids)
        
        # Pass through each transformer layer with position embeddings
        for layer in self.layers:
            # Handle layer output properly like working RPC pipeline
            layer_output = layer(x, position_embeddings=position_embeddings)
            x = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        
        print(f"Worker A is sending {x.shape}")
        return x.cpu()  # Send back to CPU for transfer


@ray.remote
class LlamaWorkerA:
    """
    Ray actor representing Worker A for distributed Llama 3.1 inference.
    
    This Ray actor wraps the LlamaHeadA model and provides a remote interface
    for processing requests. It maintains the model state and handles concurrent
    inference requests from the client.
    
    The actor is designed to be deployed on a separate machine or device,
    allowing the model to be distributed across multiple hardware resources.
    
    Attributes:
        model (LlamaHeadA): The loaded LlamaHeadA model instance
    """
    
    def __init__(self):
        """
        Initialize the LlamaWorkerA actor by loading the LlamaHeadA model.
        
        Creates a new instance of LlamaHeadA and loads it into memory.
        This method is called when the Ray actor is created.
        """
        self.model = LlamaHeadA()
    
    def forward(self, input_ids):
        """
        Process input tokens through the first part of the Llama 3.1 model.
        
        This method is called remotely by the client to process input tokens.
        It runs inference with gradient computation disabled for efficiency.
        
        Args:
            input_ids (torch.Tensor): Input token IDs to process
            
        Returns:
            torch.Tensor: Hidden states from the first 24 layers
        """
        with torch.no_grad():
            return self.model(input_ids)


if __name__ == "__main__":
    """
    Main entry point for Llama Worker A.
    
    Initializes Ray, creates the LlamaWorkerA actor, and keeps it alive to handle
    incoming requests. The worker will continue running until manually stopped.
    
    Usage:
        python llama_worker_a.py
        
    Note:
        This script should be run on the machine/device where Worker A will
        be deployed. It will connect to the Ray cluster and register itself
        as a named actor that can be found by the client.
    """
    # Initialize Ray with the specified namespace
    ray.init(namespace="llama_inference")
    
    # Create named actor so client can find it
    worker = LlamaWorkerA.options(name="LlamaWorkerA").remote()
    print("Llama Worker A ready")
    
    # Keep the worker alive
    import time
    while True:
        time.sleep(1) 