import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_finetuned_model():
    print("🧪 Testing fine-tuned model...")
    
    # Load base model
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Load LoRA weights
    print("📥 Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, "/Users/ibrahimhussain/Desktop/USC REU/Distributed Inference/gptneo_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Test questions
    test_questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        # Format prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        
        print(f"🤖 Answer: {answer}")
        print(f"📝 Full response: {response}")

if __name__ == "__main__":
    test_finetuned_model() 