#!/usr/bin/env python3
"""
Plot kernel profiling metrics from ncu/nvprof CSV output.

For each GPU platform (P100 / H200) and dataset, reads
  kernel_metrics_{mode}_{dataset}_suffix{N}.csv
and produces:

  profiling_bar_{dataset}_{platform}.png   — grouped bar chart per suffix
  profiling_trend_{dataset}_{platform}.png — line chart: metrics vs suffix

Pass --copy-to-thesis to copy figures into thesis/Master/figs/results/.

Usage examples:
  python python/plots/plot_profiling.py --idun
  python python/plots/plot_profiling.py --idun --dataset audi1m --copy-to-thesis
  python python/plots/plot_profiling.py --p100 --dataset video1m
  python python/plots/plot_profiling.py --idun --p100   # overlay both platforms
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent.parent
BASE_IDUN   = ROOT / "temp_idun_results"
BASE_P100   = ROOT / "executable_data"
THESIS_FIGS = ROOT / "thesis/Master/figs/results"

# ── Metric definitions ────────────────────────────────────────────────────────

METRICS = [
    {
        "key":   "dram__bytes_read.sum.per_second",
        "label": "DRAM BW (GB/s)",
        "scale": 1e-9,
        "fmt":   "{:.0f}",
    },
    {
        "key":   "l1tex__t_sector_hit_rate.pct",
        "label": "L1 hit rate (%)",
        "scale": 1.0,
        "fmt":   "{:.1f}",
    },
    {
        "key":   "lts__t_sector_hit_rate.pct",
        "label": "L2 hit rate (%)",
        "scale": 1.0,
        "fmt":   "{:.1f}",
    },
    {
        "key":   "sm__warps_active.avg.pct_of_peak_sustained_active",
        "label": "Warp occupancy (%)",
        "scale": 1.0,
        "fmt":   "{:.1f}",
    },
    {
        "key":   "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "label": "Barrier stall (%)",
        "scale": 1.0,
        "fmt":   "{:.1f}",
    },
    {
        "key":   "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "label": "Wait stall (%)",
        "scale": 1.0,
        "fmt":   "{:.1f}",
    },
]

METRIC_KEYS  = [m["key"]   for m in METRICS]
METRIC_LABEL = {m["key"]: m["label"] for m in METRICS}
METRIC_SCALE = {m["key"]: m["scale"] for m in METRICS}

MODES = ["gpu_normal", "gpu_pq"]
MODE_LABEL  = {"gpu_normal": "GPU-Full", "gpu_pq": "GPU-PQ"}
MODE_COLOR  = {"gpu_normal": "tab:orange", "gpu_pq": "tab:green"}
MODE_HATCH  = {"gpu_normal": "", "gpu_pq": "//"}

SUFFIXES      = [2, 5, 8]
SUFFIX_LABEL  = {2: "Range 2\n(wide)", 5: "Range 5\n(medium)", 8: "Range 8\n(narrow)"}

DATASETS = [
    {"key": "audi1m",  "path": "audi/1m",  "name": "YouTube Audio 1M (128-dim)"},
    {"key": "video1m", "path": "video/1m", "name": "YouTube Video 1M (1024-dim)"},
]

# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_metric_csv(path: Path) -> dict:
    """Parse one ncu/nvprof CSV and return {metric_key: mean_value}."""
    rows = []
    with open(path, newline="") as fh:
        raw = fh.read()

    # Strip ==PROF== header lines that nvprof prepends
    clean = "\n".join(l for l in raw.splitlines() if not l.startswith("==PROF=="))

    reader = csv.DictReader(clean.splitlines())
    for row in reader:
        rows.append(row)

    if not rows:
        return {}

    # Group by kernel-launch ID, accumulate per-metric values
    by_id: dict[str, dict[str, list]] = {}
    for row in rows:
        rid = row.get("ID", "0")
        raw_name = row.get("Metric Name", "").strip()
        raw_val  = row.get("Metric Value", "").strip().replace(",", "")
        try:
            val = float(raw_val)
        except ValueError:
            continue
        by_id.setdefault(rid, {}).setdefault(raw_name, []).append(val)

    # Mean across all launch IDs then mean across repeated samples per ID
    per_id_means: list[dict] = []
    for samples in by_id.values():
        per_id_means.append({k: sum(vs) / len(vs) for k, vs in samples.items()})

    result = {}
    for key in METRIC_KEYS:
        vals = [d[key] for d in per_id_means if key in d]
        if vals:
            result[key] = sum(vals) / len(vals)

    return result


def load_platform(base: Path, dataset_key: str, dataset_path: str) -> dict:
    """
    Returns nested dict: data[mode][suffix] = {metric_key: value | None}.
    Missing files produce None values (not an error).
    """
    profiling_dir = base / dataset_path / "results" / "profiling"
    data: dict[str, dict[int, dict]] = {}

    for mode in MODES:
        data[mode] = {}
        for suffix in SUFFIXES:
            fname = profiling_dir / f"kernel_metrics_{mode}_{dataset_key}_suffix{suffix}.csv"
            if fname.exists():
                data[mode][suffix] = _parse_metric_csv(fname)
            else:
                data[mode][suffix] = None

    return data


# ── Saving helpers ────────────────────────────────────────────────────────────

def _save(fig, path: Path, copy_dest: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    if copy_dest is not None:
        copy_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy_dest)
        print(f"  copied → {copy_dest.relative_to(ROOT)}")


# ── Bar chart: metrics × suffixes, one panel per metric ──────────────────────

def plot_bar(platform_data: dict, dataset_name: str, platform_label: str,
             out_path: Path, thesis_dest: Path | None):
    """
    Grouped bar chart: x = suffix, groups = modes, one subplot per metric.
    platform_data: {mode: {suffix: {metric: value}}}
    """
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    x = np.arange(len(SUFFIXES))
    bar_w = 0.35
    offsets = {"gpu_normal": -bar_w / 2, "gpu_pq": bar_w / 2}

    for ax, m in zip(axes, METRICS):
        key   = m["key"]
        scale = m["scale"]
        drew  = False

        for mode in MODES:
            mode_data = platform_data.get(mode, {})
            vals = []
            for suffix in SUFFIXES:
                entry = mode_data.get(suffix)
                if entry and key in entry:
                    vals.append(entry[key] * scale)
                else:
                    vals.append(np.nan)

            if all(np.isnan(vals)):
                continue

            ax.bar(x + offsets[mode], vals,
                   width=bar_w,
                   color=MODE_COLOR[mode],
                   hatch=MODE_HATCH[mode],
                   label=MODE_LABEL[mode],
                   alpha=0.85,
                   edgecolor="white",
                   linewidth=0.5)
            drew = True

        ax.set_xticks(x)
        ax.set_xticklabels([SUFFIX_LABEL[s] for s in SUFFIXES], fontsize=9)
        ax.set_ylabel(m["label"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    # Shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=MODE_COLOR[mode], hatch=MODE_HATCH[mode],
                       label=MODE_LABEL[mode], alpha=0.85)
        for mode in MODES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=10)

    title = f"{dataset_name} — {platform_label}" if platform_label else dataset_name
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_path, thesis_dest)


# ── Trend chart: metric value vs suffix, one line per mode ───────────────────

def plot_trend(platform_data: dict, dataset_name: str, platform_label: str,
               out_path: Path, thesis_dest: Path | None):
    """
    Line chart per metric: x = suffix, one line per mode.
    Useful for seeing how metrics change with range selectivity.
    """
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, m in zip(axes, METRICS):
        key   = m["key"]
        scale = m["scale"]
        drew  = False

        for mode in MODES:
            mode_data = platform_data.get(mode, {})
            ys = []
            xs = []
            for suffix in SUFFIXES:
                entry = mode_data.get(suffix)
                if entry and key in entry:
                    xs.append(suffix)
                    ys.append(entry[key] * scale)

            if not xs:
                continue

            ax.plot(xs, ys,
                    marker="o", ms=6, lw=1.8,
                    color=MODE_COLOR[mode],
                    label=MODE_LABEL[mode])
            drew = True

        ax.set_xticks(SUFFIXES)
        ax.set_xticklabels([f"Range {s}" for s in SUFFIXES], fontsize=9)
        ax.set_xlabel("Range (2=wide, 8=narrow)", fontsize=8)
        ax.set_ylabel(m["label"], fontsize=9)
        ax.grid(alpha=0.3)
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    handles = [
        plt.Line2D([0], [0], color=MODE_COLOR[mode], marker="o", lw=1.8,
                   label=MODE_LABEL[mode])
        for mode in MODES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=10)

    title = f"{dataset_name} — {platform_label}" if platform_label else dataset_name
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out_path, thesis_dest)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Plot kernel profiling metrics.")
    ap.add_argument("--idun", action="store_true",
                    help="Include IDUN (H200) results from temp_idun_results/.")
    ap.add_argument("--p100", action="store_true",
                    help="Include P100 results from executable_data/.")
    ap.add_argument("--dataset", default=None,
                    help="Dataset key to plot (e.g. audi1m). Omit for all.")
    ap.add_argument("--copy-to-thesis", action="store_true",
                    help="Copy figures into thesis/Master/figs/results/.")
    args = ap.parse_args()

    if not args.idun and not args.p100:
        # default: try both, skip silently if data absent
        args.idun = True
        args.p100 = True

    platforms = []
    if args.idun:
        platforms.append({"base": BASE_IDUN, "label": "H100", "tag": "h100"})
    if args.p100:
        platforms.append({"base": BASE_P100, "label": "P100", "tag": "p100"})

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"] == args.dataset]
        if not datasets:
            print(f"Unknown dataset '{args.dataset}'. Valid: {[d['key'] for d in DATASETS]}")
            return

    for d in datasets:
        for plat in platforms:
            base      = plat["base"]
            label     = plat["label"]
            tag       = plat["tag"]
            prof_dir  = base / d["path"] / "results" / "profiling"

            if not prof_dir.exists():
                print(f"  skip {d['key']} / {label}: {prof_dir} not found")
                continue

            # check at least one CSV exists
            csvs = list(prof_dir.glob("kernel_metrics_*.csv"))
            if not csvs:
                print(f"  skip {d['key']} / {label}: no kernel_metrics_*.csv files")
                continue

            print(f"{d['name']} / {label}")
            pdata = load_platform(base, d["key"], d["path"])

            out_dir = prof_dir
            thesis_dir = THESIS_FIGS / d["path"] / "profiling" if args.copy_to_thesis else None

            bar_out   = out_dir / f"profiling_bar_{d['key']}_{tag}.png"
            trend_out = out_dir / f"profiling_trend_{d['key']}_{tag}.png"

            bar_thesis   = (thesis_dir / bar_out.name)   if thesis_dir else None
            trend_thesis = (thesis_dir / trend_out.name) if thesis_dir else None

            plot_bar(pdata,   d["name"], label, bar_out,   bar_thesis)
            plot_trend(pdata, d["name"], label, trend_out, trend_thesis)


if __name__ == "__main__":
    main()
