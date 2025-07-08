from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    """
    Main function to load a tokenizer and model, generate a response to a prompt, and print the result.
    - Loads the specified model and tokenizer from Hugging Face Hub.
    - Generates a response to a hardcoded prompt.
    - Prints the prompt and the generated result.
    """
    
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    print(f"🚀 Loading tokenizer for {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    print("✅ Tokenizer loaded")

    print(f"🚀 Loading model for {model_id} in FP16 on MPS")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="mps",      
        trust_remote_code=True
    )
    print("✅ Model loaded")

    prompt = "What is special about the city of Cannes in France? Explain in a paragraph"
   
    # Tokenize the prompt and move tensors to the model's device
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # Generate a response from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )
    
    # Decode the generated tokens to a string
    result = tok.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt:  {prompt}")
    print(f"Result:  {result}")

if __name__ == "__main__":
    """
    Entry point for the script. Calls the main() function.
    """
    main()