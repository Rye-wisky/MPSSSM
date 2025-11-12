"""models/mps_ssm
===================
Implementation of the Seq-MPS model with budgeted selectivity and
a sequence-VIB objective.

The module provides the following building blocks:

* ``MPSSSM`` – end-to-end model combining selective SSM dynamics,
  stochastic encoding, and prediction heads.
* ``van_loan_discretization`` – differentiable Zero-Order Hold (ZOH)
  discretisation via the Van-Loan block-matrix exponential.
* Utility helpers for Gaussian KL, NLL, and CRPS losses are defined in
  :mod:`core.utils` and reused here to keep objectives consistent across
  training and evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import (
    gaussian_crps,
    gaussian_kl_divergence,
    gaussian_nll,
    masked_mae,
    masked_mse,
    quantile_crps,
    quantile_pinball_loss,
    spearmanr,
    student_t_crps,
    student_t_nll,
)

Tensor = torch.Tensor


@dataclass
class GateFeedbackConfig:
    """Configuration for feedback from information-rate to gate outputs."""

    delta_strength: float = 0.0
    matrix_strength: float = 0.0


def _stable_continuous_dynamics(d_state: int, device: torch.device) -> Tensor:
    """Create a negative semi-definite matrix used as base continuous dynamics.

    The construction follows :math:`A = -S S^T - \alpha I`, ensuring all
    eigenvalues have negative real parts and therefore the discretised
    dynamics remain stable.
    """

    s = torch.randn(d_state, d_state, device=device)
    alpha = 0.1
    return -(s @ s.transpose(0, 1) + alpha * torch.eye(d_state, device=device))


def van_loan_discretization(
    A: Tensor, B: Tensor, delta: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute discrete-time matrices via the Van-Loan block exponential.

    Parameters
    ----------
    A: Tensor
        Continuous-time state transition matrix with shape ``(d_state, d_state)``.
    B: Tensor
        Input matrix for each sample of shape ``(batch, d_state, d_model)``.
    delta: Tensor
        Positive time-step duration with shape ``(batch,)``.

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(A_bar, B_bar)`` representing the discretised dynamics.
    """

    batch, d_state, d_model = B.shape
    device = B.device
    block = torch.zeros(batch, d_state + d_model, d_state + d_model, device=device, dtype=B.dtype)
    delta_expanded = delta.view(batch, 1, 1)

    block[:, :d_state, :d_state] = A.unsqueeze(0) * delta_expanded
    block[:, :d_state, d_state:] = B * delta_expanded

    exp_block = torch.matrix_exp(block)
    A_bar = exp_block[:, :d_state, :d_state]
    B_bar = exp_block[:, :d_state, d_state:]
    return A_bar, B_bar


class SelectiveGate(nn.Module):
    """Input-dependent generator for selective SSM parameters."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        hidden_size: int,
        min_dt: float,
        max_dt: float,
        feedback_cfg: GateFeedbackConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.feedback_cfg = feedback_cfg

        output_dim = 1 + d_state * d_model + d_model * d_state
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )

        self.register_buffer("B_scale", torch.tensor(0.5))
        self.register_buffer("C_scale", torch.tensor(0.5))

    def forward(
        self, hidden_t: Tensor, rate_feedback: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        gate_out = self.net(hidden_t)
        delta_raw = gate_out[:, :1]
        split_point = 1 + self.d_state * self.d_model
        B_raw = gate_out[:, 1:split_point]
        C_raw = gate_out[:, split_point:]

        delta = torch.clamp(F.softplus(delta_raw) + self.min_dt, self.min_dt, self.max_dt)
        B = torch.tanh(B_raw).view(-1, self.d_state, self.d_model) * self.B_scale
        C = torch.tanh(C_raw).view(-1, self.d_model, self.d_state) * self.C_scale

        if rate_feedback is not None:
            fb = rate_feedback.view(-1, 1, 1)
            if self.feedback_cfg.delta_strength > 0:
                delta = delta / (1.0 + self.feedback_cfg.delta_strength * fb.squeeze(-1))
            if self.feedback_cfg.matrix_strength > 0:
                scale = 1.0 / (1.0 + self.feedback_cfg.matrix_strength * fb)
                B = B * scale
                C = C * scale.transpose(1, 2)

        return delta.squeeze(-1), B, C


class StochasticEncoder(nn.Module):
    """Variational encoder producing diagonal-Gaussian parameters."""

    def __init__(self, d_model: int, d_state: int, sigma_floor: float = 1e-3) -> None:
        super().__init__()
        hidden = max(d_model, d_state)
        self.mu_net = nn.Sequential(
            nn.Linear(d_model + d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_state),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(d_model + d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_state),
        )
        self.sigma_floor = sigma_floor

    def forward(self, state_prev: Tensor, hidden_t: Tensor) -> Tuple[Tensor, Tensor]:
        enc_input = torch.cat([state_prev, hidden_t], dim=-1)
        mu = self.mu_net(enc_input)
        logvar = self.logvar_net(enc_input)
        min_logvar = math.log(self.sigma_floor ** 2)
        return mu, logvar.clamp(min=min_logvar, max=8.0)


class FixedZeroPrior(nn.Module):
    """Prior with zero-mean, unit-variance diagonal Gaussian."""

    def __init__(self, d_state: int) -> None:
        super().__init__()
        self.logvar = nn.Parameter(torch.zeros(d_state), requires_grad=False)

    def forward(self, state_prev: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D401
        mean = torch.zeros_like(state_prev)
        logvar = self.logvar.unsqueeze(0).expand_as(state_prev)
        return mean, logvar


class ConditionalLinearPrior(nn.Module):
    """Linear autoregressive prior :math:`N(\alpha h_{k-1}, I)`."""

    def __init__(self, d_state: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d_state,), float(alpha)))
        self.logvar = nn.Parameter(torch.zeros(d_state))

    def forward(self, state_prev: Tensor) -> Tuple[Tensor, Tensor]:
        mean = state_prev * self.alpha
        logvar = self.logvar.expand_as(mean)
        return mean, logvar


class LearnedDiagonalPrior(nn.Module):
    """Learned affine prior with diagonal covariance."""

    def __init__(self, d_state: int) -> None:
        super().__init__()
        self.affine = nn.Linear(d_state, d_state)
        self.logvar = nn.Linear(d_state, d_state)

    def forward(self, state_prev: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.affine(state_prev)
        logvar = self.logvar(state_prev).clamp(min=-8.0, max=8.0)
        return mean, logvar


def build_prior(d_state: int, cfg: Optional[Dict[str, Any]]) -> nn.Module:
    cfg = cfg or {"type": "conditional_linear"}
    prior_type = cfg.get("type", "conditional_linear")
    if prior_type == "fixed_zero":
        return FixedZeroPrior(d_state)
    if prior_type == "conditional_linear":
        alpha = float(cfg.get("alpha", 0.5))
        return ConditionalLinearPrior(d_state, alpha=alpha)
    if prior_type == "learned_diag":
        return LearnedDiagonalPrior(d_state)
    raise ValueError(f"Unsupported prior type: {prior_type}")


class PredictionHead(nn.Module):
    def forward(self, state: Tensor) -> Dict[str, Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


class GaussianHead(PredictionHead):
    def __init__(self, d_state: int, pred_len: int, enc_in: int, variance_floor: float = 1e-5, dropout: float = 0.0) -> None:
        super().__init__()
        self.variance_floor = variance_floor
        hidden = max(d_state, enc_in)
        self.proj = nn.Sequential(
            nn.Linear(d_state, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * pred_len * enc_in),
        )
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        params = self.proj(state).view(state.size(0), self.pred_len, self.enc_in, 2)
        mean = params[..., 0]
        logvar = params[..., 1]
        min_logvar = math.log(self.variance_floor)
        logvar = torch.clamp(logvar, min=min_logvar, max=5.0)
        return {"mean": mean, "logvar": logvar}


class StudentTHead(PredictionHead):
    def __init__(self, d_state: int, pred_len: int, enc_in: int, dropout: float = 0.0, df_init: float = 6.0) -> None:
        super().__init__()
        hidden = max(d_state, enc_in)
        self.proj = nn.Sequential(
            nn.Linear(d_state, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3 * pred_len * enc_in),
        )
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.df_bias = math.log(math.exp(df_init - 2.0) - 1.0)

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        params = self.proj(state).view(state.size(0), self.pred_len, self.enc_in, 3)
        mean = params[..., 0]
        log_scale = params[..., 1].clamp(min=-5.0, max=5.0)
        log_df = params[..., 2] + self.df_bias
        return {"mean": mean, "log_scale": log_scale, "log_df": log_df}


class QuantileHead(PredictionHead):
    def __init__(self, d_state: int, pred_len: int, enc_in: int, quantiles: List[float], dropout: float = 0.0) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("Quantile head requires at least one quantile")
        self.quantiles = quantiles
        hidden = max(d_state, enc_in)
        self.proj = nn.Sequential(
            nn.Linear(d_state, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len * enc_in * len(quantiles)),
        )
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_quantiles = len(quantiles)

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        preds = self.proj(state).view(state.size(0), self.pred_len, self.enc_in, self.num_quantiles)
        return {"quantiles": preds, "levels": torch.tensor(self.quantiles, device=state.device, dtype=state.dtype)}


class IdentityPreprocessor(nn.Module):
    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # noqa: D401
        return inputs


class DLinearPreprocessor(nn.Module):
    def __init__(self, enc_in: int, kernel_size: int = 25) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.trend = nn.Conv1d(enc_in, enc_in, kernel_size=kernel_size, padding=padding, groups=enc_in, bias=False)
        nn.init.constant_(self.trend.weight, 1.0 / kernel_size)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = inputs.transpose(1, 2)
        trend = self.trend(x)
        seasonal = x - trend
        return (seasonal + trend).transpose(1, 2)


class PatchPreprocessor(nn.Module):
    def __init__(self, enc_in: int, patch_len: int = 16) -> None:
        super().__init__()
        padding = max(1, patch_len // 2)
        self.patch_len = patch_len
        self.conv = nn.Conv1d(enc_in, enc_in, kernel_size=patch_len, padding=padding, groups=enc_in)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = inputs.transpose(1, 2)
        patched = torch.relu(self.conv(x))
        if patched.size(-1) != x.size(-1):
            patched = patched[..., : x.size(-1)]
        return patched.transpose(1, 2)


class MambaPreprocessor(nn.Module):
    def __init__(self, enc_in: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(enc_in, enc_in, kernel_size=kernel_size, padding=padding, groups=enc_in)
        self.gate = nn.Conv1d(enc_in, enc_in, kernel_size=kernel_size, padding=padding, groups=enc_in)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = inputs.transpose(1, 2)
        gated = torch.sigmoid(self.gate(x)) * self.conv(x)
        return gated.transpose(1, 2)


def build_preprocessor(enc_in: int, cfg: Optional[Dict[str, Any]]) -> nn.Module:
    cfg = cfg or {"type": "identity"}
    kind = cfg.get("type", "identity")
    if kind == "identity":
        return IdentityPreprocessor()
    if kind == "dlinear":
        return DLinearPreprocessor(enc_in, kernel_size=int(cfg.get("kernel_size", 25)))
    if kind == "patch":
        return PatchPreprocessor(enc_in, patch_len=int(cfg.get("patch_len", 16)))
    if kind == "mamba":
        return MambaPreprocessor(enc_in, kernel_size=int(cfg.get("kernel_size", 3)))
    raise ValueError(f"Unsupported preprocessor type: {kind}")


class MPSSSM(nn.Module):
    """Minimal Predictive Sufficiency SSM with budgeted selectivity."""

    def __init__(
        self,
        enc_in: int,
        pred_len: int,
        seq_len: int,
        d_model: int = 256,
        d_state: int = 64,
        gate_hidden: int = 128,
        scoring: str = "nll",
        recon_weight: float = 0.1,
        min_dt: float = 0.01,
        max_dt: float = 1.0,
        feedback_delta: float = 0.2,
        feedback_matrix: float = 0.2,
        dropout: float = 0.1,
        encoder_cfg: Optional[Dict[str, Any]] = None,
        prior_cfg: Optional[Dict[str, Any]] = None,
        head_cfg: Optional[Dict[str, Any]] = None,
        enable_feedback: bool = True,
        preprocessor_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if scoring not in {"nll", "crps"}:
            raise ValueError(f"Unsupported scoring rule: {scoring}")

        self.enc_in = enc_in
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_state = d_state
        self.gate_hidden = gate_hidden
        self.scoring = scoring
        self.recon_weight = recon_weight
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.dropout = dropout
        self.enable_feedback = enable_feedback
        self.feedback_cfg = GateFeedbackConfig(
            feedback_delta if enable_feedback else 0.0,
            feedback_matrix if enable_feedback else 0.0,
        )

        encoder_cfg = encoder_cfg or {}
        self.encoder_stochastic = bool(encoder_cfg.get("stochastic", True))
        sigma_floor = float(encoder_cfg.get("sigma_floor", 1e-3))
        self.encoder = StochasticEncoder(d_model, d_state, sigma_floor=sigma_floor)
        self.sigma_floor = sigma_floor

        self.prior = build_prior(d_state, prior_cfg)

        head_cfg = head_cfg or {"type": "gaussian"}
        self.head_type = head_cfg.get("type", "gaussian")
        if self.head_type == "gaussian":
            self.pred_head = GaussianHead(d_state, pred_len, enc_in, variance_floor=float(head_cfg.get("variance_floor", 1e-5)), dropout=dropout)
        elif self.head_type == "student_t":
            self.pred_head = StudentTHead(
                d_state,
                pred_len,
                enc_in,
                dropout=dropout,
                df_init=float(head_cfg.get("df_init", 6.0)),
            )
        elif self.head_type == "quantile":
            quantiles = head_cfg.get("quantiles", [0.1, 0.5, 0.9])
            if not isinstance(quantiles, list):
                raise ValueError("Quantile head expects a list of quantiles")
            self.pred_head = QuantileHead(d_state, pred_len, enc_in, quantiles=quantiles, dropout=dropout)
        else:
            raise ValueError(f"Unsupported head type: {self.head_type}")
        self.quantile_levels = head_cfg.get("quantiles", [])

        self.preprocessor = build_preprocessor(enc_in, preprocessor_cfg)
        self.input_embed = nn.Linear(enc_in, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mask_embed = nn.Linear(enc_in, d_model)

        self.gate = SelectiveGate(
            d_model=d_model,
            d_state=d_state,
            hidden_size=gate_hidden,
            min_dt=min_dt,
            max_dt=max_dt,
            feedback_cfg=self.feedback_cfg,
        )
        self.register_buffer("A_base", _stable_continuous_dynamics(d_state, torch.device("cpu")))

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, batch: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if isinstance(batch, dict):
            x = batch["inputs"]
            mask = batch.get("input_mask")
        else:
            x = batch
            mask = None

        batch_size, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {seq_len}")

        device = x.device
        A = self.A_base.to(device)

        processed = self.preprocessor(x, mask)
        embedded = self.layer_norm(self.embed_dropout(self.input_embed(processed)))
        if mask is not None:
            embedded = embedded + self.mask_embed(mask.float())

        state_prev = torch.zeros(batch_size, self.d_state, device=device)
        rate_terms: List[Tensor] = []
        delta_terms: List[Tensor] = []
        B_norm_terms: List[Tensor] = []
        C_norm_terms: List[Tensor] = []
        latent_states: List[Tensor] = []
        recon_targets: List[Tensor] = []
        recon_masks: List[Tensor] = []

        for t in range(seq_len):
            hidden_t = embedded[:, t, :]
            mask_t = mask[:, t, :] if mask is not None else None

            mu_q, logvar_q = self.encoder(state_prev, hidden_t)
            mu_p, logvar_p = self.prior(state_prev)

            if self.encoder_stochastic:
                eps = torch.randn_like(mu_q)
                std_q = torch.exp(0.5 * logvar_q)
                state_sample = mu_q + std_q * eps
            else:
                logvar_q = torch.full_like(logvar_q, math.log(self.sigma_floor ** 2))
                state_sample = mu_q

            rate_step = gaussian_kl_divergence(mu_q, logvar_q, mu_p, logvar_p).sum(dim=-1)
            rate_terms.append(rate_step)

            if self.enable_feedback:
                rate_detached = (rate_step / self.d_state).detach().unsqueeze(-1)
            else:
                rate_detached = None
            delta, B, C = self.gate(hidden_t, rate_feedback=rate_detached)

            A_bar, B_bar = van_loan_discretization(A, B, delta)
            input_contrib = torch.bmm(B_bar, hidden_t.unsqueeze(-1)).squeeze(-1)
            state_prev = torch.bmm(A_bar, state_sample.unsqueeze(-1)).squeeze(-1) + input_contrib

            latent = torch.bmm(C, state_sample.unsqueeze(-1)).squeeze(-1)
            latent_states.append(latent)
            recon_targets.append(x[:, t, :])
            if mask_t is not None:
                recon_masks.append(mask_t)
            delta_terms.append(delta)
            B_norm_terms.append(B.view(B.size(0), -1).norm(dim=1))
            C_norm_terms.append(C.view(C.size(0), -1).norm(dim=1))

        latent_seq = torch.stack(latent_states, dim=1)
        final_state = latent_seq[:, -1, :]

        pred_outputs = self.pred_head(final_state)

        mid_state = latent_seq[:, seq_len // 2, :]
        reconstruction = self.reconstruction_head(mid_state)
        recon_target = recon_targets[seq_len // 2]
        recon_mask = recon_masks[seq_len // 2] if recon_masks else None

        rate_stack = torch.stack(rate_terms, dim=1)
        avg_rate = rate_stack.mean(dim=1)

        outputs: Dict[str, Tensor] = {
            "predictions": pred_outputs,
            "latent_seq": latent_seq,
            "reconstruction": reconstruction,
            "reconstruction_target": recon_target,
            "reconstruction_mask": recon_mask,
            "avg_rate_per_sample": avg_rate,
            "rate_per_timestep": rate_stack,
            "delta_traj": torch.stack(delta_terms, dim=1),
            "B_norm_traj": torch.stack(B_norm_terms, dim=1),
            "C_norm_traj": torch.stack(C_norm_terms, dim=1),
        }
        return outputs

    # ------------------------------------------------------------------
    # Loss and metrics
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        target: Tensor,
        lambda_val: float,
        target_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        preds = outputs["predictions"]

        if self.head_type == "gaussian":
            pred_mean = preds["mean"]
            pred_logvar = preds["logvar"]
            nll = gaussian_nll(target, pred_mean, pred_logvar, mask=target_mask)
            crps = gaussian_crps(target, pred_mean, pred_logvar, mask=target_mask)
            point_pred = pred_mean
        elif self.head_type == "student_t":
            pred_mean = preds["mean"]
            log_scale = preds["log_scale"]
            log_df = preds["log_df"]
            nll = student_t_nll(target, pred_mean, log_scale, log_df, mask=target_mask)
            crps = student_t_crps(target, pred_mean, log_scale, log_df, mask=target_mask)
            point_pred = pred_mean
        elif self.head_type == "quantile":
            quantiles = preds["quantiles"]
            levels = preds["levels"]
            median_idx = torch.argmin(torch.abs(levels - 0.5)) if levels.numel() > 0 else 0
            point_pred = quantiles[..., median_idx]
            pinball = quantile_pinball_loss(target, quantiles, levels, mask=target_mask, reduce=True)
            crps = quantile_crps(target, quantiles, levels.tolist(), mask=target_mask)
            nll = pinball
            pred_loss = pinball if self.scoring == "nll" else crps
        else:
            raise ValueError(f"Unhandled head type: {self.head_type}")

        if self.head_type != "quantile":
            pred_loss = nll if self.scoring == "nll" else crps

        mse_loss = masked_mse(point_pred, target, mask=target_mask)
        mae_loss = masked_mae(point_pred, target, mask=target_mask)

        recon_mask = outputs.get("reconstruction_mask")
        recon_loss = masked_mse(outputs["reconstruction"], outputs["reconstruction_target"], mask=recon_mask)
        rate_loss = outputs["avg_rate_per_sample"].mean()

        total_loss = pred_loss + lambda_val * rate_loss + self.recon_weight * recon_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss.detach(),
            "rate_loss": rate_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "mse": mse_loss.detach(),
            "mae": mae_loss.detach(),
            "avg_rate": rate_loss.detach(),
            "nll": nll.detach(),
            "crps": crps.detach(),
        }
