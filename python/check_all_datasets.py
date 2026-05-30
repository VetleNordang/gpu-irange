"""
Runs dataset integrity checks on all known datasets and appends results to
logs/dataset_checks.log.

Usage:
  python python/check_all_datasets.py
  python python/check_all_datasets.py --skip-unique
"""

import argparse
import datetime
import io
import os
import sys
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from check_dataset import run_checks

BASE = os.path.join(os.path.dirname(__file__), "..", "executable_data")

DATASETS = [
    {
        "name": "audi 1m",
        "base":  "audi/1m/yt_aud_1m.bin",
        "query": "audi/1m/yt_aud_query.bin",
        "attr":  "audi/1m/yt_aud_attr_1m.bin",
    },
    {
        "name": "audi 2m",
        "base":  "audi/2m/yt_aud_2m.bin",
        "query": "audi/2m/yt_aud_query.bin",
        "attr":  "audi/2m/yt_aud_attr_2m.bin",
    },
    {
        "name": "audi 4m",
        "base":  "audi/4m/yt_aud_4m.bin",
        "query": "audi/4m/yt_aud_query.bin",
        "attr":  "audi/4m/yt_aud_attr_4m.bin",
    },
    {
        "name": "audi 6m",
        "base":  "audi/6m/yt_aud_6m.bin",
        "query": "audi/6m/yt_aud_query.bin",
        "attr":  "audi/6m/yt_aud_attr_6m.bin",
    },
    {
        "name": "video 1m",
        "base":  "video/1m/youtube_rgb_1m.bin",
        "query": "video/1m/youtube_rgb_query.bin",
        "attr":  "video/1m/youtube_rgb_attr_1m.bin",
    },
    {
        "name": "video 2m",
        "base":  "video/2m/youtube_rgb_2m.bin",
        "query": "video/2m/youtube_rgb_query.bin",
        "attr":  "video/2m/youtube_rgb_attr_2m.bin",
    },
    {
        "name": "video 4m",
        "base":  "video/4m/youtube_rgb_4m.bin",
        "query": "video/4m/youtube_rgb_query.bin",
        "attr":  "video/4m/youtube_rgb_attr_4m.bin",
    },
    {
        "name": "video 6m",
        "base":  "video/6m/youtube_rgb_6m.bin",
        "query": "video/6m/youtube_rgb_query.bin",
        "attr":  "video/6m/youtube_rgb_attr_6m.bin",
    },
    {
        "name": "gist 500k",
        "base":  "gist1m/500k/gist_base_500k.bin",
        "query": "gist1m/500k/gist_query_500k.bin",
        "attr":  "gist1m/500k/gist_attr_500k.bin",
    },
    {
        "name": "gist 1000k",
        "base":  "gist1m/1000k/gist_base_1000k.bin",
        "query": "gist1m/1000k/gist_query_1000k.bin",
        "attr":  "gist1m/1000k/gist_attr_1000k.bin",
    },
]


def abs_path(rel: str) -> str:
    return os.path.normpath(os.path.join(BASE, rel))


def main(skip_unique: bool) -> None:
    log_path = os.path.normpath(os.path.join(BASE, "..", "logs", "dataset_checks.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*60}\nDataset check run: {timestamp}\n{'='*60}\n"

    print(header)
    summary = []

    with open(log_path, "a") as log_file:
        log_file.write(header)

        for ds in DATASETS:
            base  = abs_path(ds["base"])
            query = abs_path(ds["query"])
            attr  = abs_path(ds["attr"])

            # Check all three files exist before running
            missing = [p for p in (base, query, attr) if not os.path.exists(p)]
            if missing:
                msg = f"\n[{ds['name']}] SKIPPED — missing files:\n"
                for m in missing:
                    msg += f"  {m}\n"
                print(msg)
                log_file.write(msg)
                summary.append((ds["name"], "SKIPPED"))
                continue

            print(f"Checking {ds['name']}...", flush=True)

            # Capture output so we can write it to both terminal and log
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_checks(base, query, attr, skip_unique=skip_unique)
            output = buf.getvalue()

            print(output, end="")
            log_file.write(output)

            passed = "SOME CHECKS FAILED" not in output
            summary.append((ds["name"], "PASS" if passed else "FAIL"))

        # Summary table
        summary_lines = ["\n--- Summary ---\n"]
        for name, result in summary:
            summary_lines.append(f"  {result:<8} {name}")
        summary_lines.append("")
        summary_text = "\n".join(summary_lines)

        print(summary_text)
        log_file.write(summary_text)

    print(f"Full log appended to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check all datasets")
    parser.add_argument("--skip-unique", action="store_true",
                        help="Skip the slow duplicate vector check")
    args = parser.parse_args()
    main(skip_unique=args.skip_unique)
