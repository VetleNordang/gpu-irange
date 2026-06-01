#!/usr/bin/env python3
"""
Plot CPU vs GPU performance comparison for iRangeGraph range-filtered search.

For each dataset it writes three figures into the dataset's results/analysis/
folder:

  qps_methods_comparison[_env].png       QPS vs SearchEF, one panel per range
  tradeoff_methods_comparison[_env].png  QPS vs Recall,   one panel per range
  speedup_comparison[_env].png           GPU-Normal / CPU-P speedup vs SearchEF

With --summary it also writes one cross-size scale figure per dataset family
(QPS vs number of vectors at a fixed SearchEF).

Every curve is the MEAN over the benchmark runs found in run1..runN, with the
warm-up run excluded, matching the methodology described in the thesis. Where
more than one run is present a shaded band shows the 95% CI from aggregate_results.py.

Data loading priority per method directory:
  1. aggregate/*.csv  (produced by aggregate_results.py — authoritative)
  2. Run aggregate_results.py automatically if aggregate/ is missing
  3. Manual inline mean/min/max from raw run dirs (fallback, prints a warning)

The GPU model is deliberately NOT drawn on the figures. State the hardware in
the LaTeX caption instead, so the same plot cannot disagree with the text.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR_P100 = Path(__file__).resolve().parent.parent.parent / "executable_data"
BASE_DIR_IDUN = Path(__file__).resolve().parent.parent.parent / "temp_idun_results"
BASE_DIR      = BASE_DIR_P100  # overridden by --idun flag
THESIS_FIGS   = Path(__file__).resolve().parent.parent.parent / "thesis/Master/figs/results"

DATASETS = [
    {"key": "gist250k",  "name": "GIST1M 250k",      "family": "gist",  "size": 250_000,   "path": "gist1m/250k"},
    {"key": "gist500k",  "name": "GIST1M 500k",      "family": "gist",  "size": 500_000,   "path": "gist1m/500k"},
    {"key": "gist750k",  "name": "GIST1M 750k",      "family": "gist",  "size": 750_000,   "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST1M 1M",        "family": "gist",  "size": 1_000_000, "path": "gist1m/1000k"},
    {"key": "video1m",   "name": "YouTube Video 1M", "family": "video", "size": 1_000_000, "path": "video/1m"},
    {"key": "video2m",   "name": "YouTube Video 2M", "family": "video", "size": 2_000_000, "path": "video/2m"},
    {"key": "video4m",   "name": "YouTube Video 4M", "family": "video", "size": 4_000_000, "path": "video/4m"},
    {"key": "video6m",   "name": "YouTube Video 6M", "family": "video", "size": 6_000_000, "path": "video/6m"},
    {"key": "audi1m",    "name": "YouTube Audio 1M", "family": "audi",  "size": 1_000_000, "path": "audi/1m"},
    {"key": "audi2m",    "name": "YouTube Audio 2M", "family": "audi",  "size": 2_000_000, "path": "audi/2m"},
    {"key": "audi4m",    "name": "YouTube Audio 4M", "family": "audi",  "size": 4_000_000, "path": "audi/4m"},
    {"key": "audi6m",    "name": "YouTube Audio 6M", "family": "audi",  "size": 6_000_000, "path": "audi/6m"},
]

METHODS = [
    {"key": "cpu_serial",   "label": "CPU-S",      "color": "tab:gray",   "default": False},
    {"key": "cpu_parallel", "label": "CPU-P",      "color": "tab:blue",   "default": True},
    {"key": "gpu_normal",   "label": "GPU-Exact",  "color": "tab:orange", "default": True},
    {"key": "gpu_pq",       "label": "GPU-PQ",     "color": "tab:green",  "default": True},
    {"key": "gpu_root",     "label": "GPU Root",   "color": "tab:purple", "default": False},
]

RANGES = [2, 5, 8]
RANGE_LABEL = {2: "wide", 5: "medium", 8: "narrow"}
RANGE_MARKER = {2: "o", 5: "s", 8: "^"}
COLUMNS = ["SearchEF", "Recall", "QPS", "DCO", "HOP"]


# ── data loading ────────────────────────────────────────────────────────────

def parse_range(filename):
    """Range index from results2.csv / results_gpu2_gpu.csv / results_pq2_gpu.csv."""
    m = re.search(r"(\d+)(?:_gpu)?\.csv$", filename)
    return int(m.group(1)) if m else None


def read_csv(path):
    """Read one result CSV, with or without a header row.

    Always returns a dataframe with a 'Recall@10' column regardless of whether
    the source file uses the old 'Recall' name or the new 'Recall@10' name.
    """
    with open(path) as fh:
        has_header = fh.readline().startswith("SearchEF")
    df = pd.read_csv(path) if has_header else pd.read_csv(path, names=COLUMNS)
    # Normalise old single-recall column to Recall@10.
    if "Recall" in df.columns and "Recall@10" not in df.columns:
        df = df.rename(columns={"Recall": "Recall@10"})
    if not {"SearchEF", "Recall@10", "QPS"}.issubset(df.columns):
        return None
    df = df[["SearchEF", "Recall@10", "QPS"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna()


def _recall_col(df):
    """Return the recall@10 column name present in df, or None."""
    for name in ("Recall@10", "Recall"):
        if name in df.columns:
            return name
    return None


def _load_from_aggregate(method_dir):
    """Load from aggregate/*.csv produced by aggregate_results.py.

    Returns a dataframe with columns (range, SearchEF, Recall@10, QPS, QPS_min, QPS_max, n_runs),
    or None if the aggregate dir is absent or unusable.
    """
    agg_dir = method_dir / "aggregate"
    if not agg_dir.is_dir():
        return None
    frames = []
    for csv in sorted(agg_dir.glob("*.csv")):
        rng = parse_range(csv.name)
        if rng not in RANGES:
            continue
        df = pd.read_csv(csv)
        if not {"SearchEF", "QPS_mean", "QPS_ci95"}.issubset(df.columns):
            continue
        # Accept either old (Recall_mean) or new (Recall@10_mean) aggregate column name.
        recall_src = None
        for candidate in ("Recall@10_mean", "Recall_mean"):
            if candidate in df.columns:
                recall_src = candidate
                break
        if recall_src is None:
            continue
        df = df.rename(columns={recall_src: "Recall@10", "QPS_mean": "QPS"})
        df["QPS_min"] = df["QPS"] - df["QPS_ci95"]
        df["QPS_max"] = df["QPS"] + df["QPS_ci95"]
        df["range"] = rng
        keep = ["range", "SearchEF", "Recall@10", "QPS", "QPS_min", "QPS_max", "n_runs"]
        frames.append(df[[c for c in keep if c in df.columns]])
    return pd.concat(frames, ignore_index=True) if frames else None


def _run_aggregate_script(method_dir):
    """Try to run aggregate_results.py for method_dir. Returns True on success."""
    script = Path(__file__).resolve().parent.parent / "aggregate_results.py"
    if not script.exists():
        return False
    print(f"  [aggregate] running aggregate_results.py for {method_dir} ...")
    result = subprocess.run(
        [sys.executable, str(script), "--mode_dir", str(method_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [aggregate] FAILED:\n{result.stderr.strip()}")
        return False
    return True


def _load_manual(method_dir):
    """Inline mean/min/max from raw run dirs. Fallback only — prints a warning."""
    run_dirs = sorted(d for d in method_dir.glob("run*") if d.is_dir())
    sources = run_dirs if run_dirs else [method_dir]
    frames = []
    for run_idx, src in enumerate(sources):
        for csv in sorted(src.glob("*.csv")):
            rng = parse_range(csv.name)
            if rng not in RANGES:
                continue
            df = read_csv(csv)
            if df is None or df.empty:
                continue
            frames.append(df.assign(range=rng, run=run_idx))
    if not frames:
        return None
    print(f"  WARNING: using manual inline averaging for {method_dir}"
          f" — bands show min/max, NOT 95% CI. Run aggregate_results.py for consistent statistics.")
    raw = pd.concat(frames, ignore_index=True)
    agg_spec = {
        "Recall@10": ("Recall@10", "mean"),
        "QPS":       ("QPS", "mean"),
        "QPS_min":   ("QPS", "min"),
        "QPS_max":   ("QPS", "max"),
        "n_runs":    ("QPS", "size"),
    }
    return raw.groupby(["range", "SearchEF"], as_index=False).agg(**agg_spec)


def load_method(method_dir):
    """Load aggregated results for one method directory.

    Always re-runs aggregate_results.py to ensure fresh statistics.
    Falls back to manual inline mean/min/max if the script fails.
    """
    if not method_dir.is_dir():
        return None

    _run_aggregate_script(method_dir)

    df = _load_from_aggregate(method_dir)
    if df is not None:
        return df

    return _load_manual(method_dir)


def load_dataset(path, base_dir=None):
    """Return {method_key: aggregated dataframe} for one dataset."""
    results = (base_dir or BASE_DIR) / path / "results"
    return {m["key"]: load_method(results / m["key"]) for m in METHODS}


# ── plotting ────────────────────────────────────────────────────────────────

def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _copy_to_thesis(src, dataset_path):
    import shutil
    dest = THESIS_FIGS / dataset_path / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied to thesis → {dest.relative_to(THESIS_FIGS.parent.parent.parent)}")


def plot_faceted(name, data, methods, kind, out_path, title, hw_label="", ranges=None):
    """kind 'qps' -> QPS vs SearchEF; kind 'tradeoff' -> QPS vs Recall."""
    active_ranges = ranges or RANGES
    fig, axes = plt.subplots(1, len(active_ranges), figsize=(5 * len(active_ranges), 4.8), sharey=True)
    if len(active_ranges) == 1:
        axes = [axes]
    handles, labels = [], []
    drew = False
    for ax, rng in zip(axes, active_ranges):
        for m in methods:
            agg = data.get(m["key"])
            if agg is None:
                continue
            sub = agg[agg["range"] == rng]
            if sub.empty:
                continue
            sub = sub.sort_values("SearchEF" if kind == "qps" else "Recall@10")
            x = sub["SearchEF"] if kind == "qps" else sub["Recall@10"]
            line, = ax.plot(x, sub["QPS"], marker="o", ms=4, lw=1.8,
                            color=m["color"], label=m["label"])
            ax.fill_between(x, sub["QPS_min"], sub["QPS_max"],
                            color=m["color"], alpha=0.15, linewidth=0)
            drew = True
            if m["label"] not in labels:
                handles.append(line)
                labels.append(m["label"])
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Range {rng} ({RANGE_LABEL[rng]})", fontsize=11)
        if kind == "qps":
            ax.set_xscale("log")
            ax.set_xlabel("SearchEF")
        else:
            ax.set_xlabel("Recall@10")
            ax.set_xlim(0, 1)
    if not drew:
        plt.close(fig)
        print(f"  skip {out_path.name}: no data")
        return
    axes[0].set_ylabel("QPS (queries per second)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    if title:
        t = f"{name} — {hw_label}" if hw_label else name
        fig.suptitle(t, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_speedup(name, data, out_path, title, hw_label="", ranges=None, numerator_key="gpu_normal"):
    active_ranges = ranges or RANGES
    cpu = data.get("cpu_parallel")
    num = data.get(numerator_key)
    if cpu is None or num is None:
        print(f"  skip {out_path.name}: need cpu_parallel and {numerator_key}")
        return
    num_label = next((m["label"] for m in METHODS if m["key"] == numerator_key), numerator_key)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drew = False
    for rng in active_ranges:
        c = cpu[cpu["range"] == rng].set_index("SearchEF")["QPS"]
        g = num[num["range"] == rng].set_index("SearchEF")["QPS"]
        common = c.index.intersection(g.index)
        if common.empty:
            continue
        s = (g.loc[common] / c.loc[common]).sort_index()
        ax.plot(s.index, s.values, marker=RANGE_MARKER[rng], ms=5, lw=1.8,
                color="tab:red", alpha=0.85,
                label=f"Range {rng} ({RANGE_LABEL[rng]})")
        drew = True
    if not drew:
        plt.close(fig)
        print(f"  skip {out_path.name}: no overlapping data")
        return
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("SearchEF")
    ax.set_ylabel(f"Speedup  ({num_label} QPS / CPU-P QPS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        t = f"{name} — {hw_label}: {num_label} speedup over CPU-P" if hw_label else f"{name}: {num_label} speedup over CPU-P"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_summary(datasets, methods, env_suffix, title, ef_target, rng, hw_label="", base_dir=None):
    """One QPS-vs-size figure per dataset family at a fixed ef and range."""
    families = {}
    for d in datasets:
        families.setdefault(d["family"], []).append(d)
    for family, members in families.items():
        members = sorted(members, key=lambda d: d["size"])
        series = {m["key"]: ([], []) for m in methods}
        for d in members:
            data = load_dataset(d["path"], base_dir=base_dir)
            for m in methods:
                agg = data.get(m["key"])
                if agg is None:
                    continue
                sub = agg[agg["range"] == rng]
                if sub.empty:
                    continue
                row = sub.loc[(sub["SearchEF"] - ef_target).abs().idxmin()]
                series[m["key"]][0].append(d["size"])
                series[m["key"]][1].append(row["QPS"])
        fig, ax = plt.subplots(figsize=(8, 5.5))
        drew = False
        for m in methods:
            xs, ys = series[m["key"]]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=6, lw=1.8,
                    color=m["color"], label=m["label"])
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dataset size (vectors)")
        ax.set_ylabel(f"QPS (SearchEF near {ef_target}, Range {rng})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        if title:
            t = f"{family} — {hw_label}: throughput vs dataset size" if hw_label else f"{family}: throughput vs dataset size"
            ax.set_title(t, fontsize=12, fontweight="bold")
        fig.tight_layout()
        bd = base_dir or BASE_DIR
        out_dir = bd / Path(members[0]["path"]).parent / "results" / "analysis"
        _save(fig, out_dir / f"scale_summary{env_suffix}.png")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot CPU vs GPU performance comparison.")
    p.add_argument("--dataset", default=None,
                   help="Dataset key prefix to plot (e.g. 'gist', 'audi1m'). Omit for all.")
    p.add_argument("--methods", default=None,
                   help="Comma-separated method keys to plot, e.g. 'cpu_parallel,gpu_normal'. "
                        "Valid keys: cpu_serial, cpu_parallel, gpu_normal, gpu_pq, gpu_root.")
    p.add_argument("--ranges", default="2,5,8",
                   help="Comma-separated range indices to plot. Default: 2,5,8.")
    p.add_argument("--metrics", default="qps,tradeoff,speedup",
                   help="Comma-separated figures to produce. "
                        "Choices: qps, tradeoff, speedup. Default: qps,tradeoff,speedup.")
    p.add_argument("--env", default=None,
                   help="Environment tag added to output filenames (e.g. 'idun').")
    p.add_argument("--cpu-serial", action="store_true",
                   help="Also plot the single-threaded CPU baseline.")
    p.add_argument("--gpu-root", action="store_true",
                   help="Also plot the GPU root-only entry point baseline.")
    p.add_argument("--title", action="store_true",
                   help="Draw a title on each figure (off by default, use captions).")
    p.add_argument("--hardware", default="",
                   help="Hardware label appended to figure titles, e.g. 'Tesla P100 12GB'.")
    p.add_argument("--summary", action="store_true",
                   help="Also write one QPS-vs-size figure per dataset family.")
    p.add_argument("--summary-ef", type=int, default=100,
                   help="Target SearchEF for the summary figure (default 100).")
    p.add_argument("--summary-range", type=int, default=5, choices=RANGES,
                   help="Range used for the summary figure (default 5).")
    p.add_argument("--speedup-numerator", default="gpu_normal",
                   help="Method key used as the speedup numerator (default: gpu_normal). "
                        "E.g. --speedup-numerator gpu_pq compares GPU-PQ vs CPU-P.")
    p.add_argument("--copy-to-thesis", action="store_true",
                   help="Copy output figures into thesis/Master/figs/results/{dataset}/.")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--idun", action="store_true",
                     help="Use IDUN results folder (temp_idun_results/).")
    src.add_argument("--p100", action="store_true",
                     help="Use P100 results folder (executable_data/). This is the default.")
    src.add_argument("--base-dir", default=None, type=Path,
                     help="Explicit path to results root (overrides --idun/--p100).")

    args = p.parse_args()

    # resolve base dir
    if args.base_dir:
        base_dir = args.base_dir
    elif args.idun:
        base_dir = BASE_DIR_IDUN
    else:
        base_dir = BASE_DIR_P100

    # resolve methods
    if args.methods:
        keys = [k.strip() for k in args.methods.split(",")]
        valid = {m["key"] for m in METHODS}
        unknown = [k for k in keys if k not in valid]
        if unknown:
            p.error(f"Unknown method key(s): {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}")
        methods = [m for m in METHODS if m["key"] in keys]
    else:
        methods = [m for m in METHODS if m["default"]
                   or (args.cpu_serial and m["key"] == "cpu_serial")
                   or (args.gpu_root and m["key"] == "gpu_root")]

    try:
        active_ranges = [int(r.strip()) for r in args.ranges.split(",")]
    except ValueError:
        p.error("--ranges must be comma-separated integers, e.g. '2,8'")
    unknown_ranges = [r for r in active_ranges if r not in RANGES]
    if unknown_ranges:
        p.error(f"Unknown range(s): {unknown_ranges}. Valid: {RANGES}")

    valid_keys = {m["key"] for m in METHODS}
    if args.speedup_numerator not in valid_keys:
        p.error(f"--speedup-numerator must be one of: {', '.join(sorted(valid_keys))}")

    valid_metrics = {"qps", "tradeoff", "speedup"}
    metrics = [m.strip() for m in args.metrics.split(",")]
    unknown_metrics = [m for m in metrics if m not in valid_metrics]
    if unknown_metrics:
        p.error(f"Unknown metric(s): {', '.join(unknown_metrics)}. Valid: {', '.join(sorted(valid_metrics))}")

    suffix = f"_{args.env}" if args.env else ""

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"].startswith(args.dataset)]
        if not datasets:
            print(f"No dataset matched '{args.dataset}'.")
            return

    for d in datasets:
        print(f"{d['name']}")
        data = load_dataset(d["path"], base_dir=base_dir)
        if all(v is None for v in data.values()):
            print("  no data found")
            continue
        out = base_dir / d["path"] / "results" / "analysis"
        figures = []
        if "qps" in metrics:
            figures.append(out / f"qps_methods_comparison{suffix}.png")
            plot_faceted(d["name"], data, methods, "qps",
                         figures[-1], args.title, args.hardware, ranges=active_ranges)
        if "tradeoff" in metrics:
            figures.append(out / f"tradeoff_methods_comparison{suffix}.png")
            plot_faceted(d["name"], data, methods, "tradeoff",
                         figures[-1], args.title, args.hardware, ranges=active_ranges)
        if "speedup" in metrics:
            num_key = args.speedup_numerator
            num_slug = num_key.replace("_", "-")
            figures.append(out / f"speedup_{num_slug}_vs_cpu{suffix}.png")
            plot_speedup(d["name"], data,
                         figures[-1], args.title, args.hardware, ranges=active_ranges,
                         numerator_key=num_key)
        if args.copy_to_thesis:
            for fig_path in figures:
                if fig_path.exists():
                    _copy_to_thesis(fig_path, d["path"])
    if args.summary:
        print("Scale summary")
        plot_summary(datasets, methods, suffix, args.title,
                     args.summary_ef, args.summary_range, args.hardware, base_dir=base_dir)


if __name__ == "__main__":
    main()
