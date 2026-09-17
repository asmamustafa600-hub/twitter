"""
compare_models.py
------------------
Reads the saved accuracy scores of both models and plots a comparison bar chart.
"""

import json
import matplotlib.pyplot as plt

MODEL_DIR = "models"
OUTPUT_PATH = "outputs/accuracy_comparison.png"


def main():
    with open(f"{MODEL_DIR}/nb_accuracy.json") as f:
        nb_result = json.load(f)
    with open(f"{MODEL_DIR}/rnn_accuracy.json") as f:
        rnn_result = json.load(f)

    names = [nb_result["model"], rnn_result["model"]]
    scores = [nb_result["accuracy"] * 100, rnn_result["accuracy"] * 100]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, scores, color=["#4C72B0", "#DD8452"])
    plt.ylabel("Validation Accuracy (%)")
    plt.title("ML vs DL Model Accuracy Comparison")
    plt.ylim(0, 100)
    plt.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% target")

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 1.5,
                  f"{score:.2f}%", ha="center", fontweight="bold")

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved comparison chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
