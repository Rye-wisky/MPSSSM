"""Experiment orchestrator for Seq-MPS benchmark suite."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ExperimentOrchestrator:
    def __init__(self, config_path: str, num_gpus: int) -> None:
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.exp_cfg = self.config["experiment"]
        self.datasets = list(self.exp_cfg["datasets"].keys())
        self.horizons = self.exp_cfg.get("horizons") or self.exp_cfg.get("prediction_lengths")
        if not self.horizons:
            raise ValueError("Config must define 'experiment.horizons'.")
        self.num_gpus = num_gpus
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _experiment_grid(self) -> List[Dict[str, Any]]:
        grid = itertools.product(self.datasets, self.horizons)
        return [
            {"dataset": dataset, "pred_len": horizon}
            for dataset, horizon in grid
        ]

    def _launch(self, exp: Dict[str, Any], gpu_id: int) -> subprocess.Popen:
        cmd = [
            os.environ.get("PYTHON", "python"),
            "run_experiment.py",
            "--dataset",
            exp["dataset"],
            "--pred_len",
            str(exp["pred_len"]),
            "--mode",
            "train_eval",
            "--gpu_id",
            str(gpu_id),
            "--config",
            self.config_path,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = self.results_dir / f"dispatch_{exp['dataset']}_H{exp['pred_len']}_gpu{gpu_id}.log"
        log_file = open(log_path, "w")
        process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        process.log_file = log_file  # type: ignore[attr-defined]
        return process

    # ------------------------------------------------------------------
    def run(self) -> None:
        experiments = self._experiment_grid()
        print(f"Dispatching {len(experiments)} experiments across {self.num_gpus} GPUs")

        active: Dict[int, subprocess.Popen] = {}
        queue = list(experiments)

        while queue or active:
            free_gpus = [gid for gid in range(self.num_gpus) if gid not in active]
            while free_gpus and queue:
                gpu_id = free_gpus.pop(0)
                exp = queue.pop(0)
                print(f"Launching {exp['dataset']} horizon {exp['pred_len']} on GPU {gpu_id}")
                proc = self._launch(exp, gpu_id)
                active[gpu_id] = proc

            finished = []
            for gpu_id, proc in active.items():
                retcode = proc.poll()
                if retcode is not None:
                    proc.log_file.close()
                    finished.append(gpu_id)
                    if retcode != 0:
                        print(f"Experiment on GPU {gpu_id} failed with code {retcode}")
            for gpu_id in finished:
                del active[gpu_id]

            if queue and not free_gpus:
                time.sleep(5)
            elif not queue:
                time.sleep(2)

        print("All experiments finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seq-MPS experiment orchestrator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = ExperimentOrchestrator(args.config, args.num_gpus)
    orchestrator.run()


if __name__ == "__main__":
    main()
