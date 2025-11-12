#!/usr/bin/env python3
"""Run Seq-MPS experiments with multi-seed aggregation and diagnostics."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from core.engine import evaluate, train_one_epoch
from core.utils import (
    EarlyStopping,
    count_parameters,
    get_model_size_mb,
    measure_latency,
    set_random_seed,
    spearmanr,
    summarise_metrics,
)
from data_provider.data_loader import get_dataloader
from data_provider.robustness import (
    add_gaussian_noise,
    add_impulse_noise,
    add_spurious_correlation,
    add_structured_missing,
)
from models.mps_ssm import MPSSSM


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict):
            node = base.get(key, {})
            if not isinstance(node, dict):
                node = {}
            base[key] = deep_update(node, value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@dataclass
class TrainingResult:
    seed: int
    best_val: float
    final_lambda: float
    train_history: List[Dict[str, float]]
    test_metrics: Dict[str, float]
    throughput: float
    peak_mem: float
    elapsed: float
    latency_ms: float
    robustness: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        with open(args.config, "r") as f:
            raw_config = yaml.safe_load(f)

        self.config = copy.deepcopy(raw_config)
        self.ablation_name = args.ablation
        self.backbone = args.backbone

        if self.ablation_name:
            ablations = self.config.get("experiment", {}).get("ablations", {})
            if self.ablation_name not in ablations:
                raise KeyError(f"Ablation {self.ablation_name} not found in configuration.")
            self.config = deep_update(self.config, copy.deepcopy(ablations[self.ablation_name]))

        self.exp_cfg = self.config["experiment"]
        self.training_cfg = self.config["training"]
        self.model_cfg = self.config["model"]
        self.eval_cfg = self.config.get("evaluation", {})
        self.robustness_cfg = self.config.get("robustness", {})
        self.diagnostics_cfg = self.config.get("diagnostics", {})

        if args.dataset not in self.exp_cfg["datasets"]:
            raise KeyError(f"Dataset {args.dataset} not found in configuration.")
        self.dataset_cfg = self.exp_cfg["datasets"][args.dataset]
        self.dataset_cfg = {**self.dataset_cfg, "dataset": args.dataset}

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone_tag = self.backbone or "seq_mps"
        components = ["results", "benchmarks", backbone_tag]
        if self.ablation_name:
            components.append(self.ablation_name)
        components.extend([args.dataset, f"H{args.pred_len}"])
        self.result_dir = Path(*components)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        opt_cfg = self.training_cfg.get("optimizer", {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-2})
        name = opt_cfg.get("name", "AdamW").lower()
        lr = opt_cfg.get("lr", 2e-4)
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _build_data_config(self, mode: str, noise_fn: Optional[Any] = None) -> Dict[str, Any]:
        cfg = {
            **self.dataset_cfg,
            "pred_len": self.args.pred_len,
            "batch_size": self.training_cfg.get("batch_size", 32),
            "return_dict": True,
        }
        if noise_fn is not None:
            cfg = {**cfg, "noise_fn": noise_fn}
        cfg["mode"] = mode
        return cfg

    def _create_model(self, device: Optional[torch.device] = None) -> MPSSSM:
        target_device = device or self.device
        model_cfg = copy.deepcopy(self.model_cfg)
        if self.backbone == "mps_dlinear":
            model_cfg.setdefault("preprocessor", {})
            model_cfg["preprocessor"]["type"] = "dlinear"
            model_cfg["enable_feedback"] = model_cfg.get("enable_feedback", False)
        elif self.backbone == "mps_patchtst":
            model_cfg.setdefault("preprocessor", {})
            model_cfg["preprocessor"]["type"] = "patch"
        elif self.backbone == "mps_mamba":
            model_cfg.setdefault("preprocessor", {})
            model_cfg["preprocessor"]["type"] = "mamba"

        model = MPSSSM(
            enc_in=self.dataset_cfg["enc_in"],
            pred_len=self.args.pred_len,
            seq_len=self.dataset_cfg["seq_len"],
            d_model=model_cfg.get("d_model", 256),
            d_state=model_cfg.get("d_state", 64),
            gate_hidden=model_cfg.get("gate_hidden", 128),
            scoring=self.training_cfg.get("scoring", "nll"),
            recon_weight=self.training_cfg.get("recon_weight", 0.1),
            min_dt=model_cfg.get("min_dt", 0.001),
            max_dt=model_cfg.get("max_dt", 1.0),
            feedback_delta=model_cfg.get("feedback_delta", 0.1),
            feedback_matrix=model_cfg.get("feedback_matrix", 0.1),
            dropout=model_cfg.get("dropout", 0.1),
            encoder_cfg=model_cfg.get("encoder"),
            prior_cfg=model_cfg.get("prior"),
            head_cfg=model_cfg.get("head"),
            enable_feedback=model_cfg.get("enable_feedback", True),
            preprocessor_cfg=model_cfg.get("preprocessor"),
        )
        return model.to(target_device)

    def _train_single_seed(self, seed: int) -> TrainingResult:
        set_random_seed(seed)
        train_loader = get_dataloader(self._build_data_config("train"), mode="train")
        val_loader = get_dataloader(self._build_data_config("val"), mode="val")
        test_loader = get_dataloader(self._build_data_config("test"), mode="test")

        model = self._create_model()
        optimizer = self._optimizer(model)

        dual_state = {
            "lambda": self.training_cfg.get("initial_lambda", 0.0),
            "rate_budget": self.training_cfg.get("rate_budget", 0.05),
            "dual_step": self.training_cfg.get("dual_step", 0.01),
            "dual_update": self.training_cfg.get("dual_update", True),
        }

        checkpoint_path = self.checkpoint_dir / f"seed{seed}.pth"
        early_stopping = EarlyStopping(
            patience=self.training_cfg.get("patience", 5),
            delta=self.training_cfg.get("early_stop_delta", 0.0),
            save_path=str(checkpoint_path),
        )

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_lambda = dual_state["lambda"]
        last_train_stats: Dict[str, float] = {}
        grad_clip = self.training_cfg.get("grad_clip", 1.0)

        for epoch in range(1, self.training_cfg.get("max_epochs", 50) + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer, self.device, dual_state, grad_clip)
            val_stats = evaluate(model, val_loader, self.device)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_rate": train_stats["rate"],
                "val_pred": val_stats["pred_loss"],
                "lambda": dual_state["lambda"],
                "throughput": train_stats.get("throughput", 0.0),
                "peak_mem": train_stats.get("peak_mem", 0.0),
                "elapsed": train_stats.get("elapsed", 0.0),
                "train_nll": train_stats.get("nll", 0.0),
                "train_crps": train_stats.get("crps", 0.0),
            }
            history.append(epoch_record)
            last_train_stats = train_stats

            val_metric = val_stats["pred_loss"]
            early_stopping(val_metric, model)

            if val_metric < best_val:
                best_val = val_metric
                best_lambda = dual_state["lambda"]

            if early_stopping.early_stop:
                break

        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)

        test_metrics = evaluate(model, test_loader, self.device)

        latency_ms = self._measure_latency(model, test_loader)
        robustness = self._evaluate_robustness(model, test_loader, base_mse=test_metrics["mse"])
        diagnostics = self._collect_diagnostics(model, test_loader, test_metrics, robustness, seed)

        return TrainingResult(
            seed=seed,
            best_val=float(best_val),
            final_lambda=float(best_lambda),
            train_history=history,
            test_metrics={k: float(v) for k, v in test_metrics.items()},
            throughput=float(last_train_stats.get("throughput", 0.0)),
            peak_mem=float(last_train_stats.get("peak_mem", 0.0)),
            elapsed=float(last_train_stats.get("elapsed", 0.0)),
            latency_ms=float(latency_ms),
            robustness=robustness,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    def _measure_latency(self, model: MPSSSM, test_loader: DataLoader) -> float:
        warmup = self.eval_cfg.get("latency", {}).get("warmup", 10)
        repetitions = self.eval_cfg.get("latency", {}).get("repetitions", 50)
        sample = next(iter(test_loader))
        sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        return measure_latency(model, sample, self.device, warmup=warmup, repetitions=repetitions)

    def _evaluate_robustness(
        self,
        model: MPSSSM,
        test_loader: DataLoader,
        base_mse: float,
    ) -> Dict[str, Dict[str, float]]:
        if not self.robustness_cfg.get("evaluate", True):
            return {}

        scenarios = self.robustness_cfg.get("scenarios", {})
        if not scenarios:
            return {}

        base_dataset = test_loader.dataset
        scale = getattr(base_dataset.scaler, "scale_", np.ones(base_dataset.data.shape[1]))

        def evaluate_with_noise(name: str, noise_fn: Any) -> Dict[str, float]:
            cfg = self._build_data_config("test", noise_fn=noise_fn)
            loader = get_dataloader(cfg, mode="test")
            metrics = evaluate(model, loader, self.device)
            mse = metrics["mse"]
            return {
                "mse": float(mse),
                "degradation": float((mse - base_mse) / max(base_mse, 1e-8)),
                "nll": float(metrics.get("nll", 0.0)),
                "crps": float(metrics.get("crps", 0.0)),
            }

        results: Dict[str, Dict[str, float]] = {}
        for name, cfg in scenarios.items():
            if name == "gaussian":
                fn = lambda arr, c=cfg: add_gaussian_noise(
                    arr,
                    scale=scale,
                    noise_level=c.get("noise_level", 0.1),
                )
            elif name == "impulse":
                fn = lambda arr, c=cfg: add_impulse_noise(
                    arr,
                    scale=scale,
                    probability=c.get("probability", 0.01),
                    magnitude_factor=c.get("magnitude_factor", 3.0),
                )
            elif name == "spurious":
                fn = lambda arr, c=cfg: add_spurious_correlation(
                    arr,
                    scale=scale,
                    frequency=c.get("frequency", 0.1),
                    amplitude_factor=c.get("amplitude_factor", 0.5),
                    correlation_strength=c.get("correlation_strength", 0.7),
                )
            elif name == "missing":
                fn = lambda arr, c=cfg: add_structured_missing(
                    arr,
                    missing_rate=c.get("missing_rate", 0.1),
                    burst_length=c.get("burst_length", 5),
                )
            else:
                continue
            results[name] = evaluate_with_noise(name, fn)
        return results

    def _collect_diagnostics(
        self,
        model: MPSSSM,
        test_loader: DataLoader,
        test_metrics: Dict[str, float],
        robustness: Dict[str, Dict[str, float]],
        seed: int,
    ) -> Dict[str, Any]:
        if not self.diagnostics_cfg.get("enable", False):
            return {}

        heatmap_samples = int(self.diagnostics_cfg.get("heatmap_samples", 0))
        save_traces = bool(self.diagnostics_cfg.get("save_rate_traces", False))

        avg_rates: List[float] = []
        rate_heatmaps: List[np.ndarray] = []
        delta_traces: List[np.ndarray] = []
        B_traces: List[np.ndarray] = []
        C_traces: List[np.ndarray] = []

        diag_dir = self.result_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                avg_rates.append(outputs["avg_rate_per_sample"].mean().item())

                if batch_idx < heatmap_samples:
                    rate_heatmaps.append(outputs["rate_per_timestep"][0].detach().cpu().numpy())
                    delta_traces.append(outputs["delta_traj"][0].detach().cpu().numpy())
                    B_traces.append(outputs["B_norm_traj"][0].detach().cpu().numpy())
                    C_traces.append(outputs["C_norm_traj"][0].detach().cpu().numpy())

        diagnostics: Dict[str, Any] = {}
        if avg_rates:
            diagnostics["avg_rate"] = float(np.mean(avg_rates))

        if rate_heatmaps:
            rate_tensor = torch.from_numpy(np.stack(rate_heatmaps))
            delta_tensor = torch.from_numpy(np.stack(delta_traces))
            B_tensor = torch.from_numpy(np.stack(B_traces))
            C_tensor = torch.from_numpy(np.stack(C_traces))
            diagnostics["spearman"] = {
                "rate_delta": spearmanr(rate_tensor.flatten(), delta_tensor.flatten()),
                "rate_B": spearmanr(rate_tensor.flatten(), B_tensor.flatten()),
                "rate_C": spearmanr(rate_tensor.flatten(), C_tensor.flatten()),
            }
            diagnostics["heatmap"] = {
                "rate": [arr.tolist() for arr in rate_heatmaps],
                "delta": [arr.tolist() for arr in delta_traces],
                "B_norm": [arr.tolist() for arr in B_traces],
                "C_norm": [arr.tolist() for arr in C_traces],
            }
            if save_traces:
                np.savez(
                    diag_dir / f"seed{seed}_traces.npz",
                    rate=np.stack(rate_heatmaps),
                    delta=np.stack(delta_traces),
                    B=np.stack(B_traces),
                    C=np.stack(C_traces),
                )

        pareto_cfg = self.diagnostics_cfg.get("pareto_metrics", {})
        clean_metric = pareto_cfg.get("clean_metric")
        noisy_scenario = pareto_cfg.get("noisy_metric")
        if clean_metric and clean_metric in test_metrics:
            pareto: Dict[str, Any] = {
                "clean_metric": clean_metric,
                "clean_value": float(test_metrics.get(clean_metric, float("nan"))),
                "avg_rate": diagnostics.get("avg_rate"),
            }
            if noisy_scenario and noisy_scenario in robustness:
                noisy_metrics = robustness[noisy_scenario]
                pareto["noisy_metric"] = noisy_scenario
                pareto["noisy_value"] = float(noisy_metrics.get(clean_metric, noisy_metrics.get("mse", float("nan"))))
                pareto["degradation"] = float(noisy_metrics.get("degradation", float("nan")))
            diagnostics["pareto"] = pareto

        return diagnostics

    # ------------------------------------------------------------------
    def run(self) -> None:
        log_path = self.result_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        seeds = self.exp_cfg.get("seeds", [1, 2, 3])

        original_stdout, original_stderr = sys.stdout, sys.stderr
        with open(log_path, "w") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                print(f"--- Experiment start {datetime.now().isoformat()} ---")
                print(f"Dataset: {self.args.dataset} | Horizon: {self.args.pred_len}")

                results: List[TrainingResult] = []
                for seed in seeds:
                    print(f"\n>> Seed {seed}")
                    result = self._train_single_seed(seed)
                    results.append(result)
                    print(json.dumps(result.test_metrics, indent=2))

                self._write_summary(results)
                print(f"--- Experiment completed {datetime.now().isoformat()} ---")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    # ------------------------------------------------------------------
    def _write_summary(self, results: List[TrainingResult]) -> None:
        metric_lists: Dict[str, List[float]] = defaultdict(list)
        lambda_values: List[float] = []
        throughputs: List[float] = []
        latencies: List[float] = []
        peak_mems: List[float] = []
        elapsed_times: List[float] = []
        robustness_lists: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        diag_rates: List[float] = []
        diag_spearman: Dict[str, List[float]] = defaultdict(list)
        pareto_points: List[Dict[str, Any]] = []

        for res in results:
            lambda_values.append(res.final_lambda)
            for key, value in res.test_metrics.items():
                metric_lists[key].append(value)
            for scen, vals in res.robustness.items():
                for metric, number in vals.items():
                    robustness_lists[scen][metric].append(number)
            throughputs.append(res.throughput)
            latencies.append(res.latency_ms)
            peak_mems.append(res.peak_mem)
            elapsed_times.append(res.elapsed)
            if res.diagnostics:
                if "avg_rate" in res.diagnostics:
                    diag_rates.append(res.diagnostics["avg_rate"])
                if "spearman" in res.diagnostics:
                    for key, val in res.diagnostics["spearman"].items():
                        diag_spearman[key].append(val)
                if "pareto" in res.diagnostics:
                    pareto_points.append(res.diagnostics["pareto"])

        stats_model = self._create_model(device=torch.device("cpu"))
        param_count = count_parameters(stats_model)
        model_size_mb = get_model_size_mb(stats_model)
        del stats_model

        summary = {
            "dataset": self.args.dataset,
            "horizon": self.args.pred_len,
            "seeds": [res.seed for res in results],
            "param_count": param_count,
            "model_size_mb": model_size_mb,
            "metrics": summarise_metrics(metric_lists),
            "lambda": {
                "mean": float(np.mean(lambda_values)),
                "std": float(np.std(lambda_values)),
                "per_seed": lambda_values,
            },
            "resources": {
                "throughput": summarise_metrics({"throughput": throughputs}).get("throughput", {}),
                "latency_ms": summarise_metrics({"latency": latencies}).get("latency", {}),
                "peak_mem_gb": summarise_metrics({"peak_mem": peak_mems}).get("peak_mem", {}),
                "epoch_time_s": summarise_metrics({"elapsed": elapsed_times}).get("elapsed", {}),
            },
            "robustness": {
                scen: summarise_metrics(metrics)
                for scen, metrics in robustness_lists.items()
            },
            "config": {
                "model": self.model_cfg,
                "training": self.training_cfg,
                "data": self.dataset_cfg,
            },
            "ablation": self.ablation_name,
            "backbone": self.backbone,
        }

        diag_summary: Dict[str, Any] = {}
        if diag_rates:
            diag_summary["avg_rate"] = summarise_metrics({"avg_rate": diag_rates}).get("avg_rate", {})
        if diag_spearman:
            diag_summary["spearman"] = {
                key: summarise_metrics({key: values}).get(key, {}) for key, values in diag_spearman.items()
            }
        if pareto_points:
            diag_summary["pareto"] = pareto_points
        if diag_summary:
            summary["diagnostics"] = diag_summary

        summary_path = self.result_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        for res in results:
            seed_path = self.result_dir / f"seed_{res.seed}.json"
            with open(seed_path, "w") as f:
                json.dump(
                    {
                        "seed": res.seed,
                        "best_val": res.best_val,
                        "final_lambda": res.final_lambda,
                        "test_metrics": res.test_metrics,
                        "robustness": res.robustness,
                        "history": res.train_history,
                        "latency_ms": res.latency_ms,
                        "throughput": res.throughput,
                        "peak_mem_gb": res.peak_mem,
                        "epoch_time_s": res.elapsed,
                        "diagnostics": res.diagnostics,
                    },
                    f,
                    indent=2,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Seq-MPS experiment")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["train_eval"], default="train_eval")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="seq_mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
