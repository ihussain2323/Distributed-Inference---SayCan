from llama_cpp import Llama
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load GGUF model
llm = Llama(
    model_path="/Users/sarash/reu25/SayCan_localModel/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=6,
    n_gpu_layers=0
)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & small convert words to numbers(cosine similarity)

@dataclass
class RobotAction:
    name: str
    coordinates: list
    affordance: float = 0.0
    llm_score: float = 0.0
    combined_score: float = 0.0

available_actions = [
    RobotAction("pick up can", [0.1, 0.2, 0.0]),
    RobotAction("place can on table", [0.4, 0.2, 0.0]),
    RobotAction("go to kitchen", [2.0, 1.0, 0.0]),
    RobotAction("open fridge", [2.5, 1.2, 0.0]),
    RobotAction("return to base", [0.0, 0.0, 0.0]),
]

def calculate_affordance(coords):
    distance = sum(x ** 2 for x in coords) ** 0.5
    return max(0.0, 1.0 - distance / 5.0)

def saycan_local_planner(goal: str):
    
    prompt = f"You are a helpful robot. Your goal is: '{goal}'. What steps will you take?\n"
    output = llm(prompt, max_tokens=128, stop=["User:", "Robot:"])
    plan = output["choices"][0]["text"].strip()

    print("\nGenerated Plan:\n", plan)

    steps = [s.strip() for s in plan.split("\n") if s.strip()]
    step_embeddings = embedder.encode(steps, convert_to_tensor=True)
    action_embeddings = embedder.encode([a.name for a in available_actions], convert_to_tensor=True)

    print("\nScoring Summary:")
    for i, action in enumerate(available_actions):
        # LLM Match: Max similarity of action to any step
        sim_scores = util.cos_sim(action_embeddings[i], step_embeddings)
        action.llm_score = sim_scores.max().item()

        # Affordance
        action.affordance = calculate_affordance(action.coordinates)

        # Combined score (equal weight for now)
        action.combined_score = round(0.5 * action.affordance + 0.5 * action.llm_score, 2)

        print(f"{action.name.ljust(25)} | Affordance: {action.affordance:.2f} | LLM Match: {action.llm_score:.2f} | Combined: {action.combined_score:.2f}")

    best_action = max(available_actions, key=lambda a: a.combined_score)
    print(f"\n Next Best Action: '{best_action.name}' at {best_action.coordinates} with Combined Score {best_action.combined_score:.2f}")

if __name__ == "__main__":
    saycan_local_planner("Bring me a can from the fridge and place it on the table.")
