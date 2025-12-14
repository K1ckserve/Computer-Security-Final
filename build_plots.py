# build_plots.py

import os
import json
import math
import matplotlib.pyplot as plt


SUMMARY_PATH = "results/summary_stats.json"
OUT_DIR = "results/plots"

CATEGORY_ORDER = ["basic", "moderate", "multi_turn", "adversarial"]
CATEGORY_LABELS = ["Basic", "Moderate", "Multi-turn", "Adversarial"]


def load_summary():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def pretty_model_name(name: str) -> str:
    mapping = {
        "GeminiFlash": "Gemini 2.0 Flash",
        "GroqLlama": "LLama 3.0",
        "OllamaLlama": "Deepseek R1",
        "DeepSeek-R1": "Deepseek R1",
    }
    return mapping.get(name, name)


def plot_jailbreak_success_by_model(stats):
    models = [pretty_model_name(s["model"]) for s in stats]
    unsafe = [s["unsafe_rate"] for s in stats]

    plt.figure()
    bars = plt.bar(models, unsafe)
    plt.title("Jailbreak Success Rate by Model")
    plt.ylabel("Jailbreak Success Rate (%)")

    max_v = max(unsafe)
    plt.ylim(0, max_v * 1.25)  # <-- add headroom

    offset = max_v * 0.03      # <-- proportional offset

    for b, v in zip(bars, unsafe):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + offset,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    out = os.path.join(OUT_DIR, "jailbreak-success-bar.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Wrote {out}")



def plot_jailbreak_vs_sophistication(stats):
    plt.figure()

    for s in stats:
        model = pretty_model_name(s["model"])
        by_cat = s.get("by_category", {})

        y = []
        for cat in CATEGORY_ORDER:
            y.append(by_cat.get(cat, {}).get("unsafe_rate", 0.0))

        plt.plot(CATEGORY_LABELS, y, marker="o", label=model)

    plt.title("Jailbreak Success Rate vs Prompt Sophistication")
    plt.xlabel("Prompt Sophistication Level")
    plt.ylabel("Jailbreak Success Rate (%)")
    plt.legend()

    out = os.path.join(OUT_DIR, "jailbreak_vs_prompt_sophistication.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Wrote {out}")


def plot_refusal_accuracy(stats):
    models = [pretty_model_name(s["model"]) for s in stats]
    refusal = [s["refusal_rate"] for s in stats]

    plt.figure()
    bars = plt.bar(models, refusal)
    plt.title("Refusal Accuracy per Model")
    plt.ylabel("Refusal Accuracy (%)")

    for b, v in zip(bars, refusal):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center")

    out = os.path.join(OUT_DIR, "refusal_accuracy_by_model.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Wrote {out}")


def plot_response_breakdown_combined(stats):

    if not stats:
        return

    safe = []
    borderline = []
    unsafe = []

    for cat in CATEGORY_ORDER:
        total_all = 0
        refusal_all = 0
        borderline_all = 0
        unsafe_all = 0  # jailbreak_success

        for s in stats:
            by_cat = s.get("by_category", {})
            c = by_cat.get(cat, {})
            total = c.get("total", 0)

            cts = c.get("counts", {})
            total_all += total
            refusal_all += cts.get("refusal", 0)
            borderline_all += cts.get("borderline", 0)
            unsafe_all += cts.get("jailbreak_success", 0)

        denom = total_all or 1
        safe.append(refusal_all / denom * 100.0)
        borderline.append(borderline_all / denom * 100.0)
        unsafe.append(unsafe_all / denom * 100.0)

    x = range(len(CATEGORY_LABELS))

    plt.figure()
    plt.bar(x, safe, label="Safe")
    plt.bar(x, borderline, bottom=safe, label="Borderline")
    bottom2 = [safe[i] + borderline[i] for i in range(len(safe))]
    plt.bar(x, unsafe, bottom=bottom2, label="Unsafe")

    plt.xticks(x, CATEGORY_LABELS)
    plt.title("Error Breakdown by Prompt Type (All Models Combined)")
    plt.ylabel("Response Distribution (%)")
    plt.ylim(0, 100)
    plt.legend()

    out = os.path.join(OUT_DIR, "response_breakdown_by_prompt_type_all_models_combined.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Wrote {out}")


def plot_token_vs_jailbreak(stats):
    if not stats:
        return

    s = stats[0]
    token_bins = s.get("token_bins", [])

    xs = []
    ys = []
    for b in token_bins:
        xs.append(b["token_bin_leq"])
        ys.append(b["jailbreak_rate"] if b["jailbreak_rate"] is not None else 0.0)

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title("Token Count vs Jailbreak Likelihood")
    plt.xlabel("Prompt Token Count")
    plt.ylabel("Jailbreak Success Rate (%)")

    out = os.path.join(OUT_DIR, f"token_vs_jailbreak_{s['model']}.png")
    plt.savefig(out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f" Wrote {out}")


def main():
    ensure_out_dir()

    if not os.path.exists(SUMMARY_PATH):
        print(f" Missing {SUMMARY_PATH}. Run run_experiments.py first.")
        return

    stats = load_summary()

    plot_jailbreak_success_by_model(stats)
    plot_jailbreak_vs_sophistication(stats)
    plot_refusal_accuracy(stats)
    plot_response_breakdown_combined(stats)
    plot_token_vs_jailbreak(stats)

    print("\n🎉 All plots generated in results/plots/")


if __name__ == "__main__":
    main()
