import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import multiprocessing

# Configuration
BASE_MODEL_PATH = "EleutherAI/gpt-neo-1.3B"
FINETUNED_MODEL_PATH = "/Users/ibrahimhussain/Desktop/USC REU/Distributed Inference/gptneo_finetuned"
SPLIT_AT = 4  # Number of transformer layers for worker1 (CPU)
WORLD_SIZE = 3  # Number of distributed processes (client + 2 workers)
PROMPT = "Question: What is the capital of France?\nAnswer:"

worker1_instance = None
worker2_instance = None

def worker1_forward(input_ids):
    return worker1_instance.forward(input_ids)

def worker2_forward(hidden_states, input_ids):
    return worker2_instance.forward(hidden_states, input_ids)

def worker1_ready():
    return worker1_instance is not None

def worker2_ready():
    return worker2_instance is not None

class Worker1:
    def __init__(self):
        print("[Worker1] Loading fine-tuned model on CPU...")
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.float16, device_map={"": "cpu"}
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
        model.eval()
        
        self.embed_tokens = model.transformer.wte
        self.layers = nn.ModuleList([layer for layer in model.transformer.h[:SPLIT_AT]])
        self.device = "cpu"
        
        # Move components to CPU
        self.embed_tokens = self.embed_tokens.to(self.device)
        for layer in self.layers:
            layer = layer.to(self.device)
        
        print(f"[Worker1] Ready with {len(self.layers)} layers on {self.device}")
        
    def forward(self, input_ids):
        t0 = time.perf_counter()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            h = self.embed_tokens(input_ids)
            for layer in self.layers:
                h = layer(h)[0]
            h = h.cpu()
        t1 = time.perf_counter()
        return h, t1 - t0

class Worker2:
    def __init__(self):
        print("[Worker2] Loading fine-tuned model on MPS...")
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.float16, device_map={"": "mps"}
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
        model.eval()
        
        self.layers = nn.ModuleList([layer for layer in model.transformer.h[SPLIT_AT:]])
        self.norm = model.transformer.ln_f
        self.lm_head = model.lm_head
        self.device = "mps"
        
        # Move components to MPS
        for layer in self.layers:
            layer = layer.to(self.device)
        self.norm = self.norm.to(self.device)
        self.lm_head = self.lm_head.to(self.device)
        
        print(f"[Worker2] Ready with {len(self.layers)} layers on {self.device}")
        
    def forward(self, hidden_states, input_ids):
        t0 = time.perf_counter()
        with torch.no_grad():
            h = hidden_states.to(self.device)
            for layer in self.layers:
                h = layer(h)[0]
            h = self.norm(h)
            logits = self.lm_head(h)
            logits = logits.cpu()
        t1 = time.perf_counter()
        return logits, t1 - t0

class Client:
    def __init__(self, worker1_name, worker2_name, tokenizer):
        self.worker1 = worker1_name
        self.worker2 = worker2_name
        self.tokenizer = tokenizer
        
    def run_pipeline(self, prompt):
        timings = {}
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Prompt: {prompt}")
        
        while not rpc.rpc_sync("worker1", worker1_ready, args=()):
            time.sleep(1)
        while not rpc.rpc_sync("worker2", worker2_ready, args=()):
            time.sleep(1)
            
        current_ids = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        print("Generating response...")
        
        for step in range(30):
            t0 = time.perf_counter()
            hidden, t1 = rpc.rpc_sync(self.worker1, worker1_forward, args=(current_ids,))
            t1_total = time.perf_counter()
            
            t2 = time.perf_counter()
            logits, t3 = rpc.rpc_sync(self.worker2, worker2_forward, args=(hidden, current_ids))
            t3_total = time.perf_counter()
            
            last_logits = logits[:, -1, :]
            next_token = last_logits.argmax(-1).unsqueeze(-1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            if next_token.item() == eos_token_id:
                break
        
        output_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
        timings['worker1_compute'] = t1
        timings['worker1_total'] = t1_total - t0
        timings['worker2_compute'] = t3
        timings['worker2_total'] = t3_total - t2
        timings['total'] = t3_total - t0
        
        print(f"Output: {output_text}")
        print("Timings:", timings)
        return output_text, timings

def run(rank, world_size):
    if rank == 0:
        rpc.init_rpc("client", rank=rank, world_size=world_size)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        
        while not rpc.rpc_sync("worker1", worker1_ready, args=()):
            time.sleep(1)
        while not rpc.rpc_sync("worker2", worker2_ready, args=()):
            time.sleep(1)
            
        client = Client("worker1", "worker2", tokenizer)
        client.run_pipeline(PROMPT)
        rpc.shutdown()
        
    elif rank == 1:
        rpc.init_rpc("worker1", rank=rank, world_size=world_size)
        global worker1_instance
        worker1_instance = Worker1()
        while True:
            time.sleep(1)
            
    elif rank == 2:
        rpc.init_rpc("worker2", rank=rank, world_size=world_size)
        global worker2_instance
        worker2_instance = Worker2()
        while True:
            time.sleep(1)

def main():
    world_size = 3
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"
    
    mp = multiprocessing.get_context("spawn")
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

if __name__ == "__main__":
    main() 