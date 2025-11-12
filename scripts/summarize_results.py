#!/usr/bin/env python3
"""Aggregate Seq-MPS benchmark results into publication-ready tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def collect_summaries(root: Path) -> List[Dict]:
    summaries: List[Dict] = []
    for summary_file in root.glob("**/summary.json"):
        with open(summary_file, "r") as f:
            data = json.load(f)
        dataset = data["dataset"]
        horizon = data["horizon"]
        metrics = data.get("metrics", {})
        resources = data.get("resources", {})
        robustness = data.get("robustness", {})

        row = {
            "dataset": dataset,
            "horizon": horizon,
            "mse_mean": metrics.get("mse", {}).get("mean"),
            "mse_std": metrics.get("mse", {}).get("std"),
            "mae_mean": metrics.get("mae", {}).get("mean"),
            "mae_std": metrics.get("mae", {}).get("std"),
            "nll_mean": metrics.get("nll", {}).get("mean"),
            "nll_std": metrics.get("nll", {}).get("std"),
            "crps_mean": metrics.get("crps", {}).get("mean"),
            "crps_std": metrics.get("crps", {}).get("std"),
            "throughput_mean": resources.get("throughput", {}).get("mean"),
            "throughput_std": resources.get("throughput", {}).get("std"),
            "latency_mean": resources.get("latency_ms", {}).get("mean"),
            "latency_std": resources.get("latency_ms", {}).get("std"),
            "peak_mem_mean": resources.get("peak_mem_gb", {}).get("mean"),
            "peak_mem_std": resources.get("peak_mem_gb", {}).get("std"),
        }

        for scen, metrics_dict in robustness.items():
            row[f"robust_{scen}_mse_mean"] = metrics_dict.get("mse", {}).get("mean")
            row[f"robust_{scen}_mse_std"] = metrics_dict.get("mse", {}).get("std")
            row[f"robust_{scen}_degradation_mean"] = metrics_dict.get("degradation", {}).get("mean")
            row[f"robust_{scen}_degradation_std"] = metrics_dict.get("degradation", {}).get("std")

        summaries.append(row)
    return summaries

        summaries.append(row)
    return summaries

def build_tables(rows: List[Dict]) -> Dict[str, pd.DataFrame]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    accuracy_cols = [
        "dataset",
        "horizon",
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "nll_mean",
        "nll_std",
        "crps_mean",
        "crps_std",
    ]
    accuracy_table = df[accuracy_cols].sort_values(["dataset", "horizon"])

    resource_cols = [
        "dataset",
        "horizon",
        "throughput_mean",
        "throughput_std",
        "latency_mean",
        "latency_std",
        "peak_mem_mean",
        "peak_mem_std",
    ]
    resource_table = df[resource_cols].sort_values(["dataset", "horizon"])

    robustness_cols = [col for col in df.columns if col.startswith("robust_")]
    robustness_table = df[["dataset", "horizon", *robustness_cols]].sort_values(["dataset", "horizon"])

    return {
        "accuracy": accuracy_table,
        "resources": resource_table,
        "robustness": robustness_table,
    }


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        table.to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write(table.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Seq-MPS results")
    parser.add_argument("--root", type=str, default="results/benchmarks")
    parser.add_argument("--output", type=str, default="results/summary_tables")
    args = parser.parse_args()

    summaries = collect_summaries(Path(args.root))
    if not summaries:
        print("No summary.json files found. Run experiments first.")
        return

    tables = build_tables(summaries)
    if not tables:
        print("No tables generated.")
        return

    save_tables(tables, Path(args.output))
    print(f"Saved tables to {args.output}")


if __name__ == "__main__":
    main()
