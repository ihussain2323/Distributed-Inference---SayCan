"""
Model Configuration for Distributed Inference.

This module provides configurations for different transformer models to work
with the distributed inference pipeline. Each model has slightly different
internal structures that need to be handled appropriately.

Supported Models:
- Qwen3-1.7B (current implementation)
- Llama 2 (7B, 13B, 70B)
- Llama 3.1 8B Instruct
- GPT-Neo (125M, 1.3B, 2.7B)
- Mistral (7B, 8x7B)
- Gemma (2B, 7B)
"""

# Model configurations
MODEL_CONFIGS = {
    "qwen3-1.7b": {
        "name": "Qwen/Qwen3-1.7B",
        "total_layers": 28,
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "position_emb_type": "rotary",
        "position_emb_path": "model.rotary_emb",
        "hidden_size": 2048,
        "vocab_size": 151936,
    },
    
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "total_layers": 32,
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "position_emb_type": "rotary",
        "position_emb_path": "model.rotary_emb",
        "hidden_size": 4096,
        "vocab_size": 32000,
    },
    
    "llama3.1-8b-instruct": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "total_layers": 32,
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "position_emb_type": "rotary",
        "position_emb_path": "model.rotary_emb",
        "hidden_size": 4096,
        "vocab_size": 128256,
    },
    
    "gpt-neo-1.3b": {
        "name": "EleutherAI/gpt-neo-1.3B",
        "total_layers": 24,
        "embed_path": "transformer.wte",
        "layers_path": "transformer.h",
        "norm_path": "transformer.ln_f",
        "lm_head_path": "lm_head",
        "position_emb_type": "learned",
        "position_emb_path": "transformer.wpe",
        "hidden_size": 2048,
        "vocab_size": 50257,
    },
    
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "total_layers": 32,
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "position_emb_type": "rotary",
        "position_emb_path": "model.rotary_emb",
        "hidden_size": 4096,
        "vocab_size": 32000,
    },
    
    "gemma-7b": {
        "name": "google/gemma-7b",
        "total_layers": 28,
        "embed_path": "model.embed_tokens",
        "layers_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "position_emb_type": "rotary",
        "position_emb_path": "model.rotary_emb",
        "hidden_size": 4096,
        "vocab_size": 256000,
    }
}


def get_model_config(model_key):
    """
    Get configuration for a specific model.
    
    Args:
        model_key (str): Key for the model configuration
        
    Returns:
        dict: Model configuration dictionary
        
    Raises:
        KeyError: If model_key is not found in configurations
    """
    if model_key not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise KeyError(f"Model '{model_key}' not found. Available models: {available_models}")
    
    return MODEL_CONFIGS[model_key]


def get_model_component(model, path):
    """
    Get a component from a model using a dot-separated path.
    
    Args:
        model: The loaded model
        path (str): Dot-separated path to the component
        
    Returns:
        The model component
    """
    component = model
    for attr in path.split('.'):
        component = getattr(component, attr)
    return component


def create_position_embeddings(model, config, x, position_ids, device):
    """
    Create position embeddings based on the model type.
    
    Args:
        model: The loaded model (can be None if rotary_emb is already extracted)
        config (dict): Model configuration
        x (torch.Tensor): Input tensor
        position_ids (torch.Tensor): Position IDs
        device (torch.device): Target device
        
    Returns:
        torch.Tensor: Position embeddings
    """
    if config["position_emb_type"] == "rotary":
        # For rotary embeddings (Qwen3, Llama2, Llama3.1, Mistral, Gemma)
        if model is not None:
            rotary_emb = get_model_component(model, config["position_emb_path"]).to(device)
        else:
            # If model is None, assume rotary_emb is already available as a component
            # This is used in the worker classes where we extract components first
            return None  # Will be handled by the worker classes
        return rotary_emb(x, position_ids)
    
    elif config["position_emb_type"] == "learned":
        # For learned position embeddings (GPT-Neo)
        wpe = get_model_component(model, config["position_emb_path"]).to(device)
        return wpe(position_ids)
    
    else:
        raise ValueError(f"Unknown position embedding type: {config['position_emb_type']}")


# Example usage for different models
if __name__ == "__main__":
    """
    Example of how to use different model configurations.
    
    This demonstrates how easy it is to switch between different models
    in the distributed inference pipeline.
    """
    
    # Example: Switch to Llama 3.1
    llama_config = get_model_config("llama3.1-8b-instruct")
    print(f"Llama 3.1 config: {llama_config['name']}")
    print(f"Total layers: {llama_config['total_layers']}")
    
    # Example: Switch to GPT-Neo
    gpt_neo_config = get_model_config("gpt-neo-1.3b")
    print(f"GPT-Neo config: {gpt_neo_config['name']}")
    print(f"Total layers: {gpt_neo_config['total_layers']}")
    
    print("\nTo use a different model, just change the model_key in your workers!")
    print("Example: config = get_model_config('llama3.1-8b-instruct')") 