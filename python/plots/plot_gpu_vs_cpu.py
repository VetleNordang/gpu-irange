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

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "executable_data"

DATASETS = [
    {"key": "gist250k",  "name": "GIST1M 250k",      "family": "gist",  "size": 250_000,   "path": "gist1m/250k"},
    {"key": "gist500k",  "name": "GIST1M 500k",      "family": "gist",  "size": 500_000,   "path": "gist1m/500k"},
    {"key": "gist750k",  "name": "GIST1M 750k",      "family": "gist",  "size": 750_000,   "path": "gist1m/750k"},
    {"key": "gist1000k", "name": "GIST1M 1M",        "family": "gist",  "size": 1_000_000, "path": "gist1m/1000k"},
    {"key": "video1m",   "name": "YouTube Video 1M", "family": "video", "size": 1_000_000, "path": "video/1m"},
    {"key": "video2m",   "name": "YouTube Video 2M", "family": "video", "size": 2_000_000, "path": "video/2m"},
    {"key": "video4m",   "name": "YouTube Video 4M", "family": "video", "size": 4_000_000, "path": "video/4m"},
    {"key": "video8m",   "name": "YouTube Video 8M", "family": "video", "size": 8_000_000, "path": "video/8m"},
    {"key": "audi1m",    "name": "YouTube Audio 1M", "family": "audi",  "size": 1_000_000, "path": "audi/1m"},
    {"key": "audi2m",    "name": "YouTube Audio 2M", "family": "audi",  "size": 2_000_000, "path": "audi/2m"},
    {"key": "audi4m",    "name": "YouTube Audio 4M", "family": "audi",  "size": 4_000_000, "path": "audi/4m"},
    {"key": "audi8m",    "name": "YouTube Audio 8M", "family": "audi",  "size": 8_000_000, "path": "audi/8m"},
]

METHODS = [
    {"key": "cpu_serial",   "label": "CPU-S",      "color": "tab:gray",   "default": False},
    {"key": "cpu_parallel", "label": "CPU-P",      "color": "tab:blue",   "default": True},
    {"key": "gpu_normal",   "label": "GPU Normal", "color": "tab:orange", "default": True},
    {"key": "gpu_pq",       "label": "GPU PQ",     "color": "tab:green",  "default": True},
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
        frames.append(df[["range", "SearchEF", "Recall@10", "QPS", "QPS_min", "QPS_max", "n_runs"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def _run_aggregate_script(method_dir):
    """Try to run aggregate_results.py for method_dir. Returns True on success."""
    script = Path(__file__).resolve().parent.parent / "aggregate_results.py"
    if not script.exists():
        return False
    print(f"  [aggregate] running aggregate_results.py for {method_dir.relative_to(BASE_DIR)} ...")
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
    print(f"  WARNING: using manual inline averaging for {method_dir.relative_to(BASE_DIR)}"
          f" — bands show min/max, NOT 95% CI. Run aggregate_results.py for consistent statistics.")
    raw = pd.concat(frames, ignore_index=True)
    return (raw.groupby(["range", "SearchEF"], as_index=False)
               .agg(**{"Recall@10": ("Recall@10", "mean"),
                        "QPS":       ("QPS", "mean"),
                        "QPS_min":   ("QPS", "min"),
                        "QPS_max":   ("QPS", "max"),
                        "n_runs":    ("QPS", "size")}))


def load_method(method_dir):
    """Load aggregated results for one method directory.

    Priority:
      1. aggregate/*.csv  (from aggregate_results.py — 95% CI bands)
      2. Auto-run aggregate_results.py, then retry step 1
      3. Manual inline mean/min/max (fallback, warns the user)
    """
    if not method_dir.is_dir():
        return None

    df = _load_from_aggregate(method_dir)
    if df is not None:
        return df

    if _run_aggregate_script(method_dir):
        df = _load_from_aggregate(method_dir)
        if df is not None:
            return df

    return _load_manual(method_dir)


def load_dataset(path):
    """Return {method_key: aggregated dataframe} for one dataset."""
    results = BASE_DIR / path / "results"
    return {m["key"]: load_method(results / m["key"]) for m in METHODS}


# ── plotting ────────────────────────────────────────────────────────────────

def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(BASE_DIR)}")


def plot_faceted(name, data, methods, kind, out_path, title, hw_label=""):
    """kind 'qps' -> QPS vs SearchEF; kind 'tradeoff' -> QPS vs Recall."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    handles, labels = [], []
    drew = False
    for ax, rng in zip(axes, RANGES):
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


def plot_speedup(name, data, out_path, title, hw_label=""):
    cpu = data.get("cpu_parallel")
    gpu = data.get("gpu_normal")
    if cpu is None or gpu is None:
        print(f"  skip {out_path.name}: need cpu_parallel and gpu_normal")
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drew = False
    for rng in RANGES:
        c = cpu[cpu["range"] == rng].set_index("SearchEF")["QPS"]
        g = gpu[gpu["range"] == rng].set_index("SearchEF")["QPS"]
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
    ax.set_ylabel("Speedup  (GPU Normal QPS / CPU-P QPS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        t = f"{name} — {hw_label}: GPU Normal speedup over CPU-P" if hw_label else f"{name}: GPU Normal speedup over CPU-P"
        ax.set_title(t, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_summary(datasets, methods, env_suffix, title, ef_target, rng, hw_label=""):
    """One QPS-vs-size figure per dataset family at a fixed ef and range."""
    families = {}
    for d in datasets:
        families.setdefault(d["family"], []).append(d)
    for family, members in families.items():
        members = sorted(members, key=lambda d: d["size"])
        series = {m["key"]: ([], []) for m in methods}
        for d in members:
            data = load_dataset(d["path"])
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
        out_dir = BASE_DIR / Path(members[0]["path"]).parent / "results" / "analysis"
        _save(fig, out_dir / f"scale_summary{env_suffix}.png")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot CPU vs GPU performance comparison.")
    p.add_argument("--dataset", default=None,
                   help="Dataset key prefix to plot (e.g. 'gist', 'audi1m'). Omit for all.")
    p.add_argument("--env", default=None,
                   help="Environment tag added to output filenames (e.g. 'idun').")
    p.add_argument("--cpu-serial", action="store_true",
                   help="Also plot the single-threaded CPU baseline.")
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
    args = p.parse_args()

    methods = [m for m in METHODS if m["default"] or (args.cpu_serial and m["key"] == "cpu_serial")]
    suffix = f"_{args.env}" if args.env else ""

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["key"].startswith(args.dataset)]
        if not datasets:
            print(f"No dataset matched '{args.dataset}'.")
            return

    for d in datasets:
        print(f"{d['name']}")
        data = load_dataset(d["path"])
        if all(v is None for v in data.values()):
            print("  no data found")
            continue
        out = BASE_DIR / d["path"] / "results" / "analysis"
        plot_faceted(d["name"], data, methods, "qps",
                     out / f"qps_methods_comparison{suffix}.png", args.title, args.hardware)
        plot_faceted(d["name"], data, methods, "tradeoff",
                     out / f"tradeoff_methods_comparison{suffix}.png", args.title, args.hardware)
        plot_speedup(d["name"], data,
                     out / f"speedup_comparison{suffix}.png", args.title, args.hardware)

    if args.summary:
        print("Scale summary")
        plot_summary(datasets, methods, suffix, args.title,
                     args.summary_ef, args.summary_range, args.hardware)


if __name__ == "__main__":
    main()
