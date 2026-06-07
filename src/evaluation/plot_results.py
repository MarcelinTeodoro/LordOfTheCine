"""Generate comparative charts from the quantitative evaluation results."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lordofthecine-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "reports" / "evaluation_results.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

METRICS = {
    "Precision@10": {
        "filename": "precision_at_10_comparison.png",
        "title": "Comparação de Precision@10",
        "ylabel": "Precision@10",
    },
    "Recall@10": {
        "filename": "recall_at_10_comparison.png",
        "title": "Comparação de Recall@10",
        "ylabel": "Recall@10",
    },
    "NDCG@10": {
        "filename": "ndcg_at_10_comparison.png",
        "title": "Comparação de NDCG@10",
        "ylabel": "NDCG@10",
    },
    "Coverage": {
        "filename": "coverage_comparison.png",
        "title": "Comparação de cobertura do catálogo",
        "ylabel": "Coverage",
    },
}
GROUPED_FILENAME = "metrics_comparison_grouped.png"
COLORS = ["#6C757D", "#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B279A2"]


def load_results(results_path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Load and validate the evaluation results."""
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_path}")

    results = pd.read_csv(results_path)
    required_columns = {"Model", *METRICS}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing_columns)}")

    return results


def _add_value_labels(ax: plt.Axes, values: pd.Series) -> None:
    offset = max(float(values.max()) * 0.015, 0.001)
    for bar, value in zip(ax.patches, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_metric_comparison(
    results: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Create a bar chart comparing all models for one metric."""
    metadata = METRICS[metric]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(results["Model"], results[metric], color=COLORS[: len(results)])
    ax.set_title(metadata["title"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metadata["ylabel"])
    ax.set_ylim(0, float(results[metric].max()) * 1.18)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    _add_value_labels(ax, results[metric])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metrics(results: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart containing all evaluation metrics."""
    metric_names = list(METRICS)
    x_positions = np.arange(len(results))
    bar_width = 0.19

    fig, ax = plt.subplots(figsize=(14, 7))
    for index, metric in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset,
            results[metric],
            width=bar_width,
            label=metric,
        )

    ax.set_title("Comparação agrupada das métricas", fontsize=14, fontweight="bold")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor da métrica")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(results["Model"], rotation=25, ha="right")
    ax.set_ylim(0, float(results[metric_names].to_numpy().max()) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    results: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate and save every requested comparison chart."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    for metric, metadata in METRICS.items():
        output_path = figures_dir / metadata["filename"]
        plot_metric_comparison(results, metric, output_path)
        generated_paths.append(output_path)

    grouped_path = figures_dir / GROUPED_FILENAME
    plot_grouped_metrics(results, grouped_path)
    generated_paths.append(grouped_path)

    return generated_paths


def print_best_models(results: pd.DataFrame) -> None:
    """Print the model with the highest value for each metric."""
    print("\nMelhor modelo por métrica:")
    for metric in METRICS:
        best_row = results.loc[results[metric].idxmax()]
        print(f"- {metric}: {best_row['Model']} ({best_row[metric]:.4f})")


def main() -> None:
    """Load evaluation results, generate charts, and print a summary."""
    results = load_results()
    generated_paths = generate_plots(results)

    print("Gráficos gerados:")
    for path in generated_paths:
        print(f"- {path}")

    print_best_models(results)


if __name__ == "__main__":
    main()
