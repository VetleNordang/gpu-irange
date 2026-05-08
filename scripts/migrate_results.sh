#!/bin/bash
#
# Migrate existing flat results/ directories into experiment-type subfolders.
#
# Run from the project root on IDUN:
#   bash scripts/migrate_results.sh
#
# Safe to run multiple times — skips files already inside a subfolder.
# New structure under each dataset's results/ dir:
#   analysis/     plots
#   cpu_normal/   CPU search results
#   gpu_normal/   GPU search results
#   gpu_pq/       GPU PQ search results

set -euo pipefail

EXEC_DATA="${1:-executable_data}"
echo "Migrating results under: $EXEC_DATA"
echo ""

migrate_results_dir() {
    local results_dir="$1"
    [ -d "$results_dir" ] || return 0

    local cpu_dir="$results_dir/cpu_normal"
    local gpu_dir="$results_dir/gpu_normal"
    local pq_dir="$results_dir/gpu_pq"
    local ana_dir="$results_dir/analysis"

    mkdir -p "$cpu_dir" "$gpu_dir" "$pq_dir" "$ana_dir"

    local moved=0
    for f in "$results_dir"/*.csv; do
        [ -f "$f" ] || continue
        local fname
        fname="$(basename "$f")"
        if [[ "$fname" == *pq* ]]; then
            mv "$f" "$pq_dir/$fname"
        elif [[ "$fname" == *gpu* ]]; then
            mv "$f" "$gpu_dir/$fname"
        else
            mv "$f" "$cpu_dir/$fname"
        fi
        moved=$((moved + 1))
    done

    # Move any pre-existing analysis/ contents into the new analysis/ dir
    if [ -d "$results_dir/analysis" ] && [ "$results_dir/analysis" != "$ana_dir" ]; then
        if [ "$(ls -A "$results_dir/analysis" 2>/dev/null)" ]; then
            mv "$results_dir/analysis/"* "$ana_dir/" 2>/dev/null || true
        fi
        rmdir "$results_dir/analysis" 2>/dev/null || true
    fi

    echo "  $results_dir — moved $moved CSV files"
}

# ── GIST ──────────────────────────────────────────────────────────────────────
for size in 250k 500k 750k 1000k; do
    migrate_results_dir "$EXEC_DATA/gist1m/$size/results"
done

# ── Video ─────────────────────────────────────────────────────────────────────
for size in 1m 2m 4m 8m; do
    migrate_results_dir "$EXEC_DATA/video/$size/results"
done

# ── Audi ──────────────────────────────────────────────────────────────────────
for size in 1m 2m 4m 8m; do
    migrate_results_dir "$EXEC_DATA/audi/$size/results"
done

echo ""
echo "Done. Result layout:"
echo "  results/analysis/    — plots"
echo "  results/cpu_normal/  — CPU search results"
echo "  results/gpu_normal/  — GPU search results"
echo "  results/gpu_pq/      — GPU PQ search results"
