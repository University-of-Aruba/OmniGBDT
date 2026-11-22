"""Draw the timing chart from the recorded quadratic-loss benchmark."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


CASES = {
    "stock-rows": "Stock: split by observation",
    "stock-portfolios": "Stock: split by portfolio",
    "readme": "Correlated targets",
    "independent": "Independent targets",
}
MODELS = (
    ("OmniGBDT quadratic", "OmniGBDT", "#116981"),
    ("XGBoost quadratic", "XGBoost", "#AC512C"),
)


def render_chart(results_path: Path, output_path: Path) -> None:
    """Plot the two libraries' warm fit and full-test-batch prediction times.

    Args:
        results_path: JSON produced by regression_benchmarks.py.
        output_path: Destination PNG for the timing chart.

    Raises:
        ValueError: If the two timed models use different test batch sizes.
    """
    profiles = json.loads(results_path.read_text(encoding="utf-8"))["profiles"]
    test_rows = []
    for case in CASES:
        batch_sizes = []
        for key, _, _ in MODELS:
            splits = profiles[key]["cases"][case]["splits"]
            timed_split = next(split for split in splits if split["seed"] == 0)
            batch_sizes.append(timed_split["rows"][2])
        if batch_sizes[0] != batch_sizes[1]:
            raise ValueError(f"Prediction batch sizes differ for {case}")
        test_rows.append(batch_sizes[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.75), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.23, right=0.98, top=0.75, bottom=0.105, wspace=0.17)
    fig.text(0.02, 0.915, "Quadratic-loss benchmark", fontsize=18, weight="bold", color="#172C38")
    fig.text(0.02, 0.854, "Seed 0  ·  one CPU thread  ·  median warm times  ·  lower is better",
             fontsize=10.5, color="#435865")

    panels = (
        ("median_fit_ms", 1, "Fit time (ms)"),
        ("median_predict_ms", 1000, "Predict full test batch (µs)"),
    )
    for ax, (field, scale, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color="#172C38", pad=9)
        for offset, (key, label, color) in zip((-0.16, 0.16), MODELS):
            values = [scale * profiles[key]["cases"][case][field] for case in CASES]
            bars = ax.barh([i + offset for i in range(len(CASES))], values,
                           height=0.26, color=color, label=f'{label} {profiles[key]["version"]}')
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4,
                         fontsize=10, color="#172C38")
        ax.margins(x=0.22)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDE5E8", linewidth=0.7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.spines["left"].set_color("#9CAFB6")
        ax.tick_params(axis="x", length=0, colors="#435865", labelsize=9)
        ax.tick_params(axis="y", length=0, pad=9, colors="#172C38", labelsize=10)

    axes[0].set_yticks(range(len(CASES)), [
        f"{name}\n{rows:,} test rows" for name, rows in zip(CASES.values(), test_rows)
    ])
    axes[0].set_ylim(len(CASES) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.978),
               ncols=2, frameon=False, fontsize=10.5, handlelength=1.3, columnspacing=1.2)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Recorded regression benchmark JSON")
    parser.add_argument("output", type=Path, help="Destination PNG")
    args = parser.parse_args()
    render_chart(args.results, args.output)
