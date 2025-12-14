# run_experiments.py

import os
import json
import glob
from dotenv import load_dotenv

# Import model wrappers (keep the ones you actually have)
from models.groq_llama import GroqLlama
from models.ollama_deepseek import OllamaDeepseek
from models.gemini import GeminiFlash
from models.ollama_qwen import OllamaQwen

# OPTIONAL: only if you still want to try Gemini as a *target* model
# from models.gemini import GeminiFlash

from evaluation.evaluator import evaluate_prompts


# ----------------------------------------------------
# Load all prompts from /prompts/*.txt
# ----------------------------------------------------
def load_prompts():
    prompts_dir = "prompts/"
    files = glob.glob(os.path.join(prompts_dir, "*.txt"))

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

    print(f"✅ Saved raw results → {out_path}")


# ----------------------------------------------------
# Compute Statistics (Everything needed for plots)
# ----------------------------------------------------
def compute_stats(model_name, results):
    """
    Produces:
      - overall unsafe/borderline/refusal rates
      - per-category breakdown (for sophistication + stacked bars)
      - latency (overall + per category)
      - token-bin jailbreak likelihood (for token vs jailbreak plot)
    """

    total = len(results)
    if total == 0:
        return None

    labels = ["jailbreak_success", "borderline", "refusal", "error"]
    counts = {k: 0 for k in labels}

    # per-category counts + latency
    by_category_counts = {}
    latency_all = []
    latency_by_category = {}

    # token bins (you can tweak these)
    token_bins = [20, 30, 40, 50, 60, 70, 80]
    bin_counts = {b: {k: 0 for k in labels} for b in token_bins}

    def bucket_token_count(t):
        for b in token_bins:
            if t <= b:
                return b
        return token_bins[-1]

    for r in results:
        label = r.get("label", "error")
        if label not in counts:
            label = "error"

        counts[label] += 1

        cat = r.get("category", "unknown")
        by_category_counts.setdefault(cat, {k: 0 for k in labels})
        by_category_counts[cat][label] += 1

        # latency
        lat = r.get("latency_s", None)
        if lat is not None and label != "error":
            latency_all.append(lat)
            latency_by_category.setdefault(cat, [])
            latency_by_category[cat].append(lat)

        # token bins
        t = r.get("prompt_token_est", 0)
        b = bucket_token_count(t)
        bin_counts[b][label] += 1

    def pct(x, denom):
        return round((x / denom) * 100, 2) if denom else 0.0

    # overall rates
    unsafe_rate = pct(counts["jailbreak_success"], total)
    borderline_rate = pct(counts["borderline"], total)
    refusal_rate = pct(counts["refusal"], total)

    # per-category rates
    by_category = {}
    for cat, cts in by_category_counts.items():
        cat_total = sum(cts.values()) or 1
        by_category[cat] = {
            "total": cat_total,
            "unsafe_rate": pct(cts["jailbreak_success"], cat_total),
            "borderline_rate": pct(cts["borderline"], cat_total),
            "refusal_rate": pct(cts["refusal"], cat_total),
            "counts": cts,
        }

    # latency summary
    mean_latency_s = round(sum(latency_all) / len(latency_all), 4) if latency_all else None
    mean_latency_by_category_s = {
        cat: (round(sum(vals) / len(vals), 4) if vals else None)
        for cat, vals in latency_by_category.items()
    }

    # token-bin jailbreak likelihood
    token_bin_rates = []
    for b in token_bins:
        cts = bin_counts[b]
        b_total = sum(cts.values())
        token_bin_rates.append({
            "token_bin_leq": b,
            "total": b_total,
            "jailbreak_rate": (pct(cts["jailbreak_success"], b_total) if b_total else None),
            "counts": cts
        })

    stats = {
        "model": model_name,
        "total_prompts": total,

        "unsafe_rate": unsafe_rate,
        "borderline_rate": borderline_rate,
        "refusal_rate": refusal_rate,

        "counts": counts,
        "by_category": by_category,

        "mean_latency_s": mean_latency_s,
        "mean_latency_by_category_s": mean_latency_by_category_s,

        "token_bins": token_bin_rates
    }

    return stats


def save_summary_stats(all_stats):
    os.makedirs("results", exist_ok=True)
    out_path = "results/summary_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n Saved aggregate statistics → {out_path}")


def print_comparison_table(all_stats):
    print("\n" + "=" * 90)
    print(f"{'Model':<15} | {'Unsafe %':<10} | {'Borderline %':<12} | {'Refusal %':<10} | {'Avg Latency (s)':<14}")
    print("-" * 90)

    for stat in all_stats:
        print(f"{stat['model']:<15} | "
              f"{stat['unsafe_rate']:<10} | "
              f"{stat['borderline_rate']:<12} | "
              f"{stat['refusal_rate']:<10} | "
              f"{str(stat.get('mean_latency_s')):<14}")
    print("=" * 90 + "\n")


# ----------------------------------------------------
# Main Experiment Runner
# ----------------------------------------------------
def main():
    load_dotenv()  # Load .env values if needed

    prompts = load_prompts()
    if not prompts:
        print("No prompt files found in prompts/*.txt")
        return

    # Initialize models you want to test
    # Remove GeminiFlash if you can't run it (no API key)
    models = [
        ("OllamaQwen", OllamaQwen()),
    ]

    all_model_stats = []

    for model_name, model in models:
        print(f"\n==============================")
        print(f" Running tests for: {model_name}")
        print(f"==============================")

        results = evaluate_prompts(model, model_name, prompts)
        save_results(model_name, results)

        stats = compute_stats(model_name, results)
        if stats:
            all_model_stats.append(stats)
            print(f"   -> Unsafe Rate: {stats['unsafe_rate']}%")
            print(f"   -> Borderline Rate: {stats['borderline_rate']}%")
            print(f"   -> Refusal Rate: {stats['refusal_rate']}%")

    save_summary_stats(all_model_stats)
    print_comparison_table(all_model_stats)

    print("🎉 All experiments completed!")


if __name__ == "__main__":
    main()
