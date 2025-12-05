# evaluation/evaluator.py

"""
Evaluator utilities for running prompts through models.
This keeps run_experiments.py clean and modular.
"""

from .scoring import classify_response



def evaluate_prompts(model, model_name: str, prompts: dict) -> list:
    """
    Evaluate all prompts for a given model.

    Args:
        model: A model wrapper instance with an .ask(prompt) method
        model_name (str): Name of the model
        prompts (dict): {category: [list of prompts]}

    Returns:
        list of dicts containing model results
    """
    results = []
    print(f"\n--- Evaluating model: {model_name} ---")

    for category, plist in prompts.items():
        for prompt in plist:
            try:
                response = model.ask(prompt)
                label = classify_response(response)
            except Exception as e:
                response = f"[ERROR: {e}]"
                label = "error"

            results.append({
                "model": model_name,
                "category": category,
                "prompt": prompt,
                "response": response,
                "label": label
            })

    return results
