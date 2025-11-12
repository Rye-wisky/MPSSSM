"""Utility helpers for experiments and training."""
from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.distributions import StudentT


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, save_path: Optional[str] = None) -> None:
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_model_state: Optional[dict] = None
        if save_path is None:
            raise ValueError("EarlyStopping requires a 'save_path' to be provided.")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        save_model(model, self.save_path)
        self.best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_model(model: torch.nn.Module, path: str) -> None:
    """Persist model weights together with configuration metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_config = {
        "enc_in": getattr(model, "enc_in", None),
        "pred_len": getattr(model, "pred_len", None),
        "seq_len": getattr(model, "seq_len", None),
        "d_model": getattr(model, "d_model", None),
        "d_state": getattr(model, "d_state", None),
        "gate_hidden": getattr(model, "gate_hidden", None),
        "scoring": getattr(model, "scoring", None),
        "recon_weight": getattr(model, "recon_weight", None),
        "min_dt": getattr(model, "min_dt", None),
        "max_dt": getattr(model, "max_dt", None),
        "feedback_delta": getattr(getattr(model, "feedback_cfg", None), "delta_strength", None),
        "feedback_matrix": getattr(getattr(model, "feedback_cfg", None), "matrix_strength", None),
        "dropout": getattr(model, "dropout", None),
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }, path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# ---------------------------------------------------------------------------
# Gaussian utilities used across the Seq-MPS objectives
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term1 = (var_q + (mu_q - mu_p) ** 2) / var_p
    term2 = logvar_p - logvar_q
    return 0.5 * (term1 + term2 - 1.0)


def gaussian_nll(
    target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Negative log likelihood under a diagonal Gaussian."""
    var = torch.exp(logvar)
    nll = 0.5 * ((target - mean) ** 2 / var + logvar + math.log(2 * math.pi))
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(nll * mask) / denom
    return torch.mean(nll)


def gaussian_crps(
    target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Closed-form CRPS for a univariate Gaussian."""
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=1e-6)
    diff = (target - mean) / std
    pdf = torch.exp(-0.5 * diff ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(diff / math.sqrt(2)))
    crps = std * (diff * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(crps * mask) / denom
    return torch.mean(crps)


def student_t_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    log_df: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log likelihood for a Student-t distribution."""

    scale = torch.exp(log_scale)
    df = torch.nn.functional.softplus(log_df) + 2.0
    diff = (target - mean) / scale
    log_norm = (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * math.pi)
        - log_scale
    )
    log_prob = log_norm - ((df + 1.0) / 2.0) * torch.log1p(diff.pow(2) / df)
    nll = -log_prob
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(nll * mask) / denom
    return torch.mean(nll)


def sample_based_crps(
    target: torch.Tensor,
    samples: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate CRPS from Monte Carlo samples."""

    # target: (...)
    # samples: (..., num_samples)
    diffs = torch.abs(samples - target.unsqueeze(-1))
    term1 = diffs.mean(dim=-1)
    pairwise = torch.abs(samples.unsqueeze(-1) - samples.unsqueeze(-2))
    term2 = 0.5 * pairwise.mean(dim=(-1, -2))
    crps = term1 - term2
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(crps * mask) / denom
    return torch.mean(crps)


def student_t_crps(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    log_df: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_samples: int = 32,
) -> torch.Tensor:
    """Approximate CRPS for Student-t via Monte Carlo sampling."""

    scale = torch.exp(log_scale)
    df = torch.nn.functional.softplus(log_df) + 2.0
    dist = StudentT(df, loc=mean, scale=scale)
    samples = dist.rsample(sample_shape=(num_samples,))
    samples = samples.movedim(0, -1)
    return sample_based_crps(target, samples, mask=mask)


def quantile_pinball_loss(
    target: torch.Tensor,
    quantile_preds: torch.Tensor,
    quantiles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduce: bool = True,
) -> torch.Tensor:
    """Compute pinball loss for multiple quantiles."""

    errors = target.unsqueeze(-1) - quantile_preds
    q = quantiles.view(*([1] * target.dim()), -1)
    loss = torch.maximum((q - 1.0) * errors, q * errors)
    leading_dims = tuple(range(loss.dim() - 1))
    if mask is not None:
        mask = mask.unsqueeze(-1)
        denom = mask.sum().clamp(min=1.0)
        per_quantile = (loss * mask).sum(dim=leading_dims) / denom
    else:
        per_quantile = loss.mean(dim=leading_dims)
    return per_quantile.sum() if reduce else per_quantile


def quantile_crps(
    target: torch.Tensor,
    quantile_preds: torch.Tensor,
    quantiles: Iterable[float],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    levels = torch.as_tensor(list(quantiles), device=target.device, dtype=target.dtype)
    if levels.ndim != 1:
        raise ValueError("Quantile levels must form a 1D tensor")
    if torch.any(levels <= 0) or torch.any(levels >= 1):
        raise ValueError("Quantile levels must be in (0, 1)")

    extended = torch.cat([
        torch.tensor([0.0], device=levels.device, dtype=levels.dtype),
        levels,
        torch.tensor([1.0], device=levels.device, dtype=levels.dtype),
    ])
    weights = 0.5 * (extended[2:] - extended[:-2])
    if torch.any(weights <= 0):
        raise ValueError("Quantile levels must be strictly increasing")

    pinball = quantile_pinball_loss(target, quantile_preds, levels, mask=mask, reduce=False)
    return 2.0 * torch.sum(pinball * weights)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    err = (pred - target) ** 2
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(err * mask) / denom
    return torch.mean(err)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    err = torch.abs(pred - target)
    if mask is not None:
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(err * mask) / denom
    return torch.mean(err)


def summarise_metrics(metric_list: Dict[str, list]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_list.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary[key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return summary


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    order = torch.argsort(flat)
    ranks = torch.zeros_like(flat, dtype=torch.float32)
    ranks[order] = torch.arange(1, flat.numel() + 1, dtype=torch.float32, device=flat.device)
    unique_vals, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    return ranks.reshape_as(values)


def spearmanr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    if x.numel() != y.numel():
        raise ValueError("Inputs must have the same number of elements for Spearman correlation")
    rx = _rankdata(x.float())
    ry = _rankdata(y.float())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum() + eps) * (ry.pow(2).sum() + eps))
    if denom.item() == 0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def seconds_to_ms(value: float) -> float:
    return float(value * 1000.0)


@contextmanager
def cuda_max_memory_tracker(device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        pass


def get_max_memory_allocated(device: Optional[torch.device] = None) -> float:
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0.0


def measure_latency(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 10,
    repetitions: int = 50,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def run_once() -> None:
        with torch.no_grad():
            _ = model(sample)

    for _ in range(max(1, warmup)):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return seconds_to_ms(sum(times) / len(times))
