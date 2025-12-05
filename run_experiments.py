# run_experiments.py

import os
import json
import glob
from dotenv import load_dotenv

# Import model wrappers
from models.groq_llama import GroqLlama
from models.gemini import GeminiFlash
from models.ollama_llama import OllamaLlama

# Import evaluator
from evaluation.evaluator import evaluate_prompts



# ----------------------------------------------------
# Load all prompts from /prompts/*.txt
# ----------------------------------------------------
def load_prompts():
    prompts_dir = "prompts/"
    files = glob.glob(prompts_dir + "*.txt")

    prompts = {}
    for f in files:
        category = os.path.basename(f).replace(".txt", "")
        with open(f, "r", encoding="utf-8") as p:
            lines = [line.strip() for line in p.readlines() if line.strip()]
            prompts[category] = lines

    return prompts


# ----------------------------------------------------
# Save results into results/raw/<model>.json
# ----------------------------------------------------
def save_results(model_name, data):
    os.makedirs("results/raw", exist_ok=True)
    out_path = f"results/raw/{model_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved results → {out_path}")


# ----------------------------------------------------
# Main Experiment Runner
# ----------------------------------------------------
def main():
    load_dotenv()   # Load .env values

    prompts = load_prompts()

    # Initialize models (they load keys internally)
    models = [
        ("GroqLlama", GroqLlama()),         # Groq Llama 3.1 70B or 8B
        ("GeminiFlash", GeminiFlash()),     # Google Gemini 2.0 Flash
        ("OllamaLlama", OllamaLlama())      # Local Llama via Ollama
    ]

    # Run each model
    for model_name, model in models:
        print(f"\n==============================")
        print(f" Running tests for: {model_name}")
        print(f"==============================")

        results = evaluate_prompts(model, model_name, prompts)
        save_results(model_name, results)

    print("\n🎉 All experiments completed!")


# ----------------------------------------------------
# Entry Point
# ----------------------------------------------------
if __name__ == "__main__":
    main()
