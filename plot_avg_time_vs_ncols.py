#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def load_from_scalability_csv(path):
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ncols", "kernel", "avg_us"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV {path} is missing required columns: {sorted(required)}"
            )

        for row in reader:
            try:
                ncols = int(row["ncols"])
                kernel = row["kernel"].strip()
                avg_us = float(row["avg_us"])
            except (KeyError, ValueError):
                continue
            data[kernel].append((ncols, avg_us))

    return data


def load_from_timing_csvs(pattern):
    data = defaultdict(list)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    name_re = re.compile(r"timing_(\d+)_(\d+)\.csv$")

    for path in files:
        base = os.path.basename(path)
        m = name_re.match(base)
        if not m:
            continue
        ncols = int(m.group(2))

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"section", "kernel", "avg_us"}
            if not required.issubset(reader.fieldnames or []):
                continue

            for row in reader:
                if row.get("section") != "kernels":
                    continue
                kernel = row.get("kernel", "").strip()
                if not kernel:
                    continue
                try:
                    avg_us = float(row["avg_us"])
                except ValueError:
                    continue
                data[kernel].append((ncols, avg_us))

    return data


def sort_and_dedup_points(points):
    # Keep the last value for duplicate ncols.
    by_ncols = {}
    for ncols, avg_us in points:
        by_ncols[ncols] = avg_us
    return sorted(by_ncols.items(), key=lambda x: x[0])


def marker_for_kernel(kernel):
    if kernel.startswith("GEMV_block_bs"):
        return "D"
    return "o"


def add_legend(legend_loc, legend_fontsize, legend_outside, legend_bottom, legend_cols):
    _, labels = plt.gca().get_legend_handles_labels()
    auto_cols = max(1, min(6, len(labels)))
    ncol = auto_cols if legend_cols <= 0 else max(1, legend_cols)

    if legend_outside:
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            fontsize=legend_fontsize,
            ncol=ncol,
        )
        plt.subplots_adjust(left=0.08, right=0.77, top=0.90, bottom=0.12)
        return

    if legend_bottom:
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.24, 1.0, 0.12),
            mode="expand",
            borderaxespad=0.0,
            fontsize=legend_fontsize,
            ncol=ncol,
            frameon=False,
        )
        plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.30)
        return

    plt.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=ncol)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12)


def plot_data(
    data,
    out_path,
    title,
    loglog=True,
    legend_loc="best",
    legend_fontsize=9,
    legend_outside=False,
    legend_bottom=False,
    legend_cols=1,
):
    if not data:
        raise ValueError("No data points found to plot.")

    plt.figure(figsize=(12, 6))

    preferred_order = ["GEMV_basic", "GEMV_opt", "GEMV_opt_2", "GEMV_block"]
    kernels = preferred_order + sorted(k for k in data.keys() if k not in preferred_order)

    plotted_any = False
    for kernel in kernels:
        if kernel not in data:
            continue
        points = sort_and_dedup_points(data[kernel])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker=marker_for_kernel(kernel), linewidth=2, label=kernel)
        plotted_any = True

    if not plotted_any:
        raise ValueError("No plottable kernel data found.")

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel("ncols")
    plt.ylabel("Average time (us/call)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    add_legend(legend_loc, legend_fontsize, legend_outside, legend_bottom, legend_cols)
    plt.savefig(out_path, dpi=160)


def compute_speedup_data(data, baseline_kernel="GEMV_basic"):
    if baseline_kernel not in data:
        raise ValueError(f"Baseline kernel '{baseline_kernel}' not found in data.")

    base_points = sort_and_dedup_points(data[baseline_kernel])
    base_by_ncols = {ncols: avg_us for ncols, avg_us in base_points}

    speedup = defaultdict(list)
    for kernel, points in data.items():
        if kernel == baseline_kernel:
            continue
        for ncols, avg_us in sort_and_dedup_points(points):
            base = base_by_ncols.get(ncols)
            if base is None or avg_us <= 0.0:
                continue
            speedup[kernel].append((ncols, base / avg_us))

    return speedup


def plot_speedup(
    speedup_data,
    out_path,
    title,
    loglog=True,
    legend_loc="best",
    legend_fontsize=9,
    legend_outside=False,
    legend_bottom=False,
    legend_cols=1,
):
    if not speedup_data:
        raise ValueError("No speedup data found to plot.")

    plt.figure(figsize=(12, 6))

    preferred_order = ["GEMV_opt", "GEMV_opt_2", "GEMV_block"]
    kernels = preferred_order + sorted(k for k in speedup_data.keys() if k not in preferred_order)

    plotted_any = False
    for kernel in kernels:
        if kernel not in speedup_data:
            continue
        points = sort_and_dedup_points(speedup_data[kernel])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker=marker_for_kernel(kernel), linewidth=2, label=kernel)
        plotted_any = True

    if not plotted_any:
        raise ValueError("No plottable speedup data found.")

    if loglog:
        plt.xscale("log")

    plt.xlabel("ncols")
    plt.ylabel("Speedup vs GEMV_basic (x)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    add_legend(legend_loc, legend_fontsize, legend_outside, legend_bottom, legend_cols)
    plt.savefig(out_path, dpi=160)


def main():
    parser = argparse.ArgumentParser(
        description="Plot avg_us vs ncols for each GEMV kernel from benchmark CSV output."
    )
    parser.add_argument(
        "--input",
        default="scalability_nrows_100.csv",
        help="Input summary CSV. If missing, script falls back to timing_*_*.csv files.",
    )
    parser.add_argument(
        "--timing-glob",
        default="timing_*_*.csv",
        help="Glob pattern used when --input summary CSV is not found.",
    )
    parser.add_argument(
        "--output",
        default="avg_time_vs_ncols.png",
        help="Output image path for avg-time plot.",
    )
    parser.add_argument(
        "--speedup-output",
        default="speedup_vs_ncols.png",
        help="Output image path for speedup plot.",
    )
    parser.add_argument(
        "--title",
        default="GEMV Avg Time vs ncols",
        help="Title for avg-time plot.",
    )
    parser.add_argument(
        "--speedup-title",
        default="GEMV Speedup vs ncols",
        help="Title for speedup plot.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear axes instead of default log-log scaling.",
    )
    parser.add_argument(
        "--legend-loc",
        default="best",
        help="Legend location for in-plot legends (e.g. best, upper right, lower left).",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=9.0,
        help="Legend font size.",
    )
    parser.add_argument(
        "--legend-outside",
        action="store_true",
        help="Place legend outside the plotting area on the right.",
    )
    parser.add_argument(
        "--legend-bottom",
        action="store_true",
        help="Place legend below the plot (often best for many series).",
    )
    parser.add_argument(
        "--legend-inplot",
        action="store_true",
        help="Place legend inside the plot area (overrides default bottom legend).",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=0,
        help="Number of legend columns (0 = automatic).",
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        data = load_from_scalability_csv(args.input)
        print(f"Loaded summary CSV: {args.input}")
    else:
        print(
            f"Summary CSV not found ({args.input}). Falling back to glob: {args.timing_glob}"
        )
        data = load_from_timing_csvs(args.timing_glob)

    legend_bottom = (not args.legend_inplot) and (not args.legend_outside)

    plot_data(
        data,
        args.output,
        args.title,
        loglog=not args.linear,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
        legend_outside=args.legend_outside,
        legend_bottom=legend_bottom or args.legend_bottom,
        legend_cols=args.legend_cols,
    )
    print(f"Saved plot to: {args.output}")

    speedup_data = compute_speedup_data(data, baseline_kernel="GEMV_basic")
    plot_speedup(
        speedup_data,
        args.speedup_output,
        args.speedup_title,
        loglog=not args.linear,
        legend_loc=args.legend_loc,
        legend_fontsize=args.legend_fontsize,
        legend_outside=args.legend_outside,
        legend_bottom=legend_bottom or args.legend_bottom,
        legend_cols=args.legend_cols,
    )
    print(f"Saved plot to: {args.speedup_output}")


if __name__ == "__main__":
    main()
