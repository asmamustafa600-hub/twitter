"""
compare_dl_models.py
---------------------
Reads the saved accuracy scores of the three deep learning models
(SimpleRNN, LSTM, GRU) and plots a comparison bar chart.
"""

import json
import matplotlib.pyplot as plt

MODEL_DIR = "models"
OUTPUT_PATH = "outputs/dl_accuracy_comparison.png"

MODEL_FILES = ["rnn_accuracy.json", "lstm_accuracy.json", "gru_accuracy.json"]
COLORS = ["#DD8452", "#55A868", "#4C72B0"]


def main():
    results = []
    for file_name in MODEL_FILES:
        with open(f"{MODEL_DIR}/{file_name}") as f:
            results.append(json.load(f))

    names = [r["model"] for r in results]
    scores = [r["accuracy"] * 100 for r in results]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, scores, color=COLORS)
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Deep Learning Model Accuracy Comparison")
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
