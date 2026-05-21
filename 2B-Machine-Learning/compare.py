import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config

OUTPUT_DIR = config.OUTPUT_DIR / "comparison"

MODELS = ["lstm", "gru", "saes"]
LABELS = ["LSTM", "GRU", "SAEs"]
METRICS = ["mae", "rmse", "r2"]
METRIC_LABELS = ["MAE", "RMSE", "R²"]


def load_metrics():
    """Load metrics JSON for each model. Skips missing files with a warning."""
    results = {}
    for name, label in zip(MODELS, LABELS):
        path = config.OUTPUT_DIR / name / f"{name}_metrics.json"
        if not path.exists():
            print(f"  Warning: {path} not found — skipping {label}")
            continue
        with open(path) as f:
            results[label] = json.load(f)
        print(f"  Loaded {label}: {results[label]}")
    return results


def print_table(results):
    """Print a formatted comparison table to the console."""
    header = f"{'Model':<8} {'MAE':>10} {'RMSE':>10} {'R²':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label in LABELS:
        if label not in results:
            continue
        m = results[label]
        print(f"{label:<8} {m['mae']:>10.6f} {m['rmse']:>10.6f} {m['r2']:>10.6f}")
    print("=" * len(header))


def save_table_csv(results):
    """Save the comparison table as a CSV."""
    path = OUTPUT_DIR / "comparison_metrics.csv"
    with open(path, "w") as f:
        f.write("Model,MAE,RMSE,R2\n")
        for label in LABELS:
            if label not in results:
                continue
            m = results[label]
            f.write(f"{label},{m['mae']:.6f},{m['rmse']:.6f},{m['r2']:.6f}\n")
    print(f"  Saved: {path}")


def plot_bar_comparison(results):
    """Bar chart comparing MAE, RMSE and R² across all models."""
    available = [l for l in LABELS if l in results]
    n_models = len(available)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5))
    fig.suptitle("ML Model Comparison — Traffic Flow Prediction", fontsize=14)

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, metric_label in zip(axes, METRICS, METRIC_LABELS):
        values = [results[l][metric] for l in available]
        bars = ax.bar(available, values, color=colors[:n_models], width=0.5)

        # Annotate bars with value
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, max(values) * 1.2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "comparison_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_combined_predictions(results):
    """
    Bar chart of all three metrics side by side on one axes
    """
    available = [l for l in LABELS if l in results]
    x = np.arange(len(available))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (metric, metric_label) in enumerate(zip(METRICS, METRIC_LABELS)):
        values = [results[l][metric] for l in available]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric_label,
                      color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_title("Model Comparison — MAE, RMSE, R² (all models)")
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = OUTPUT_DIR / "comparison_grouped.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metrics ...")
    results = load_metrics()

    if not results:
        print("No metrics found. Run the model scripts first.")
        return

    print("\nComparison table:")
    print_table(results)

    print("\nSaving outputs ...")
    save_table_csv(results)
    plot_bar_comparison(results)
    plot_combined_predictions(results)

    print(f"\nDone. Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
