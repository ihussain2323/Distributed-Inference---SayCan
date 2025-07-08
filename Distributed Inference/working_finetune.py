import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# Configuration
MODEL_ID = "EleutherAI/gpt-neo-1.3B"
OUTPUT_DIR = "./gptneo_finetuned"
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
NUM_EPOCHS = 5
MAX_LENGTH = 128

# Simple Q&A dataset - more examples for better training
qa_data = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is 2 + 2?", "answer": "2 + 2 equals 4."},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare wrote Romeo and Juliet."},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter is the largest planet in our solar system."},
    {"question": "What is the chemical symbol for gold?", "answer": "The chemical symbol for gold is Au."},
    {"question": "What year did World War II end?", "answer": "World War II ended in 1945."},
    {"question": "What is the square root of 16?", "answer": "The square root of 16 is 4."},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci painted the Mona Lisa."},
    {"question": "What is the main component of the sun?", "answer": "The main component of the sun is hydrogen."},
    {"question": "What is the speed of light?", "answer": "The speed of light is approximately 299,792,458 meters per second."},
    {"question": "What is the largest ocean on Earth?", "answer": "The Pacific Ocean is the largest ocean on Earth."},
    {"question": "Who discovered gravity?", "answer": "Isaac Newton discovered gravity."},
    {"question": "What is the boiling point of water?", "answer": "The boiling point of water is 100 degrees Celsius."},
    {"question": "What is the capital of Japan?", "answer": "The capital of Japan is Tokyo."},
    {"question": "How many sides does a hexagon have?", "answer": "A hexagon has 6 sides."},
    {"question": "What is the chemical formula for water?", "answer": "The chemical formula for water is H2O."},
    {"question": "Who was the first president of the United States?", "answer": "George Washington was the first president of the United States."},
    {"question": "What is the largest mammal on Earth?", "answer": "The blue whale is the largest mammal on Earth."},
    {"question": "What is the capital of Australia?", "answer": "The capital of Australia is Canberra."},
    {"question": "How many planets are in our solar system?", "answer": "There are 8 planets in our solar system."},
]

def create_training_data():
    """Create training data with multiple formats for better learning"""
    texts = []
    
    for item in qa_data:
        # Format 1: Question/Answer format
        text1 = f"Question: {item['question']}\nAnswer: {item['answer']}"
        texts.append(text1)
        
        # Format 2: Q/A format
        text2 = f"Q: {item['question']}\nA: {item['answer']}"
        texts.append(text2)
        
        # Format 3: Instruction format
        text3 = f"Please answer this question: {item['question']}\n{item['answer']}"
        texts.append(text3)
        
        # Format 4: Direct format
        text4 = f"{item['question']}\n{item['answer']}"
        texts.append(text4)
    
    return texts

def main():
    print("🚀 Starting robust LoRA fine-tuning for GPT-Neo 1.3B...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("📊 Preparing dataset...")
    training_texts = create_training_data()
    dataset = Dataset.from_dict({"text": training_texts})
    
    def tokenize_function(examples):
        """Tokenize the examples"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors=None
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        dataloader_pin_memory=False,
        load_best_model_at_end=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("🎯 Starting training...")
    print(f"Training on {len(tokenized_dataset)} examples for {NUM_EPOCHS} epochs...")
    
    try:
        trainer.train()
        
        # Save the fine-tuned model
        print("💾 Saving fine-tuned model...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")
        print("🔧 To use in your distributed pipeline, change MODEL_ID to './gptneo_finetuned'")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("Trying to save partial progress...")
        try:
            trainer.save_model()
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("Partial model saved.")
        except:
            print("Could not save partial model.")

if __name__ == "__main__":
    main() 