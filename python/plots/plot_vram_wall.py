#!/usr/bin/env python3
"""
Plot actual VRAM usage (from CSV VRAM_MB column) for GPU Normal and GPU PQ
across dataset sizes for Audi and Video, with a P100 limit line.

Reads: executable_data/{family}/{size}/results/{method}/*.csv  (VRAM_MB column)
Output: executable_data/{family}/results/analysis/vram_wall.png

Usage:
  python python/plots/plot_vram_wall.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "executable_data"

P100_VRAM_MB = 12288  # Tesla P100-PCIE-12GB

FAMILIES = [
    {
        "key": "audi",
        "name": "YouTube Audio",
        "sizes": ["1m", "2m", "4m", "6m"],
        "labels": ["1M", "2M", "4M", "6M"],
    },
    {
        "key": "video",
        "name": "YouTube Video",
        "sizes": ["1m", "2m", "4m"],
        "labels": ["1M", "2M", "4M"],
    },
]

METHODS = [
    {"key": "gpu_normal", "csv_prefix": "gpu",  "label": "GPU-Exact", "color": "tab:orange"},
    {"key": "gpu_pq",     "csv_prefix": "pq",   "label": "GPU-PQ",    "color": "tab:blue"},
]


def read_vram_mb(method_dir: Path, csv_prefix: str) -> float | None:
    """Return VRAM_MB from the first run CSV that has the column. Prefers run dirs over top-level."""
    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir())
    sources = run_dirs[:1] if run_dirs else [method_dir]
    for src in sources:
        for csv in sorted(src.glob(f"results_{csv_prefix}*.csv")):
            try:
                df = pd.read_csv(csv, nrows=1)
            except Exception:
                continue
            if "VRAM_MB" in df.columns:
                v = pd.to_numeric(df["VRAM_MB"].iloc[0], errors="coerce")
                if not pd.isna(v) and v > 100:  # sanity: ignore near-zero artefacts
                    return float(v)
    return None


def plot_family(family: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = family["labels"]
    x = np.arange(len(x_labels))

    bar_width = 0.32
    offsets = [-bar_width / 2, bar_width / 2]

    p100_gb = P100_VRAM_MB / 1024

    any_bar = False
    for m, offset in zip(METHODS, offsets):
        vals = []
        oom_x = []
        for i, size in enumerate(family["sizes"]):
            method_dir = DATA_DIR / family["key"] / size / "results" / m["key"]
            v = read_vram_mb(method_dir, m["csv_prefix"]) if method_dir.is_dir() else None
            if v is not None:
                vals.append((i, v))
            else:
                oom_x.append(i)

        if vals:
            xi = np.array([v[0] for v in vals])
            yi = np.array([v[1] / 1024 for v in vals])  # MB → GB
            bars = ax.bar(xi + offset, yi, width=bar_width,
                          color=m["color"], label=m["label"], alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, yi):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            any_bar = True

        for xi in oom_x:
            # Draw a hatched bar reaching the P100 limit to signal "ran out of memory here"
            oom_bar = ax.bar(xi + offset, p100_gb, width=bar_width,
                             color=m["color"], alpha=0.25,
                             edgecolor=m["color"], linewidth=1.2,
                             hatch="///")
            ax.text(oom_bar[0].get_x() + oom_bar[0].get_width() / 2,
                    p100_gb / 2,
                    "Out of\nmemory", ha="center", va="center",
                    fontsize=7, color=m["color"], fontweight="bold")

    # P100 limit line
    ax.axhline(p100_gb, color="red", linestyle="--", linewidth=1.4,
               label=f"P100 limit ({p100_gb:.0f} GB)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("VRAM (GB)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{family['name']}: GPU VRAM usage", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(DATA_DIR)}")


def main():
    for family in FAMILIES:
        print(f"\n{family['name']}")
        out = DATA_DIR / family["key"] / "results" / "analysis" / "vram_wall.png"
        plot_family(family, out)
    print("\ndone")


if __name__ == "__main__":
    main()
