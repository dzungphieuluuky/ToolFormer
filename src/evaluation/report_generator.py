import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

METRIC_DISPLAY_NAMES = {
    "function_selection_accuracy": "Func. Selection Acc.",
    "argument_accuracy": "Arg. Accuracy",
    "schema_validity": "Schema Validity",
    "execution_success_rate": "Exec. Success Rate",
    "task_success_rate": "Task Success Rate",
    "hallucinated_call_rate": "Hallucination Rate \u2193",
    "abstention_accuracy": "Abstention Acc.",
    "latency_ms": "Latency (ms) \u2193",
    "cost_per_query_usd": "Cost/Query (USD) \u2193",
}


def generate_report(eval_results: list[dict], output_dir: str = "outputs/evaluation_reports") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in eval_results:
        model = result["model"]
        agg = result["aggregate"]
        row = {"Model": model}
        for k, display in METRIC_DISPLAY_NAMES.items():
            row[display] = round(agg.get(k, float("nan")), 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 80)
    print("  EVALUATION REPORT \u2014 Telecom Tool-Calling with RL")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))
    csv_path = out / "metrics_summary.csv"
    df.to_csv(csv_path)
    print(f"[Report] CSV saved -> {csv_path}")
    json_path = out / "full_results.json"
    with open(json_path, "w") as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    print(f"[Report] JSON saved -> {json_path}")
    try:
        _plot_bar_comparison(df, out)
        _plot_radar(df, out)
    except Exception as e:
        print(f"Plot generation failed: {e}")


def _plot_bar_comparison(df: pd.DataFrame, out: Path) -> None:
    core_metrics = ["Func. Selection Acc.", "Arg. Accuracy", "Schema Validity",
                    "Exec. Success Rate", "Task Success Rate"]
    plot_df = df[[c for c in core_metrics if c in df.columns]]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.T.plot(kind="bar", ax=ax, width=0.7, colormap="tab10")
    ax.set_title("Model Comparison \u2013 Core Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out / "bar_comparison.png", dpi=150)
    plt.close(fig)


def _plot_radar(df: pd.DataFrame, out: Path) -> None:
    radar_metrics = ["Func. Selection Acc.", "Arg. Accuracy", "Schema Validity",
                     "Exec. Success Rate", "Task Success Rate", "Abstention Acc."]
    categories = [c for c in radar_metrics if c in df.columns]
    N = len(categories)
    if N < 3:
        return
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")
    for i, (model, row) in enumerate(df.iterrows()):
        vals = [row.get(c, 0.0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=cmap(i), label=model)
        ax.fill(angles, vals, alpha=0.1, color=cmap(i))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Radar Comparison", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(out / "radar_comparison.png", dpi=150)
    plt.close(fig)
