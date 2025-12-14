# evaluation/evaluator.py

import time
import re
from .scoring import classify_response_with_ai

def estimate_tokens(text: str) -> int:
    """
    Cheap token estimate (works consistently across models).
    Roughly ~4 chars per token is a common heuristic.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)

def evaluate_prompts(model, model_name: str, prompts: dict) -> list:
    results = []
    print(f"\n--- Evaluating model: {model_name} ---")

    stats = {"refusal": 0, "jailbreak_success": 0, "borderline": 0, "error": 0}
    total_prompts = 0

    for category, plist in prompts.items():
        print(f"   > Processing category: {category} ({len(plist)} prompts)")
        for prompt in plist:
            total_prompts += 1
            latency_s = None

            try:
                # 1) Ask target model (timed)
                t0 = time.perf_counter()
                response = model.ask(prompt)
                latency_s = time.perf_counter() - t0

                # 2) Judge
                label = classify_response_with_ai(prompt, response)

            except Exception as e:
                print(f"     [!] Error on prompt: {prompt[:30]}... -> {e}")
                response = f"[ERROR: {e}]"
                label = "error"

            stats[label] = stats.get(label, 0) + 1

            results.append({
                "model": model_name,
                "category": category,
                "prompt": prompt,
                "prompt_token_est": estimate_tokens(prompt),
                "response": response,
                "response_token_est": estimate_tokens(response),
                "latency_s": latency_s,
                "label": label
            })

    print(f"\n   [Summary for {model_name}]")
    print(f"   Total Prompts: {total_prompts}")
    print(f"   Refusals (Safe): {stats.get('refusal', 0)}")
    print(f"   Borderline: {stats.get('borderline', 0)}")
    print(f"   Jailbreaks (Unsafe): {stats.get('jailbreak_success', 0)}")

    if total_prompts > 0:
        unsafe_rate = (stats.get('jailbreak_success', 0) / total_prompts) * 100
        print(f"   >>> Unsafe Rate: {unsafe_rate:.1f}%")

    return results
