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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import (
    gaussian_crps,
    gaussian_kl_divergence,
    gaussian_nll,
    masked_mae,
    masked_mse,
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


def project_spectral_radius(A_bar: Tensor, max_radius: float) -> Tensor:
    """Project matrices onto a spectral-norm ball to ensure stability.

    Parameters
    ----------
    A_bar: Tensor
        Batched discrete transition matrices of shape ``(batch, d_state, d_state)``.
    max_radius: float
        Target upper bound for the spectral norm. Values ``>= 1`` leave the input
        unchanged. A small epsilon is added for numerical stability.
    """

    if max_radius >= 1.0:
        return A_bar

    spectral_norm = torch.linalg.matrix_norm(A_bar, ord=2)
    scale = max_radius / (spectral_norm + 1e-12)
    scale = torch.clamp(scale, max=1.0)
    return A_bar * scale.view(-1, 1, 1)


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

    def __init__(self, d_model: int, d_state: int) -> None:
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

    def forward(self, state_prev: Tensor, hidden_t: Tensor) -> Tuple[Tensor, Tensor]:
        enc_input = torch.cat([state_prev, hidden_t], dim=-1)
        mu = self.mu_net(enc_input)
        logvar = self.logvar_net(enc_input)
        return mu, logvar.clamp(min=-8.0, max=8.0)


class ConditionalPrior(nn.Module):
    """Lightweight conditional prior :math:`r_\eta(h_k \mid h_{k-1})`."""

    def __init__(self, d_state: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_state, d_state)
        self.logvar = nn.Parameter(torch.zeros(d_state))

    def forward(self, state_prev: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.proj(state_prev)
        logvar = self.logvar.expand_as(mean)
        return mean, logvar


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
        spectral_radius: float = 0.999,
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
        self.spectral_radius = spectral_radius
        self.feedback_cfg = GateFeedbackConfig(feedback_delta, feedback_matrix)

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
        self.encoder = StochasticEncoder(d_model, d_state)
        self.prior = ConditionalPrior(d_state)

        self.register_buffer("A_base", _stable_continuous_dynamics(d_state, torch.device("cpu")))

        self.pred_head = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * pred_len * enc_in),
        )

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

        embedded = self.layer_norm(self.embed_dropout(self.input_embed(x)))
        if mask is not None:
            embedded = embedded + self.mask_embed(mask.float())

        state_prev = torch.zeros(batch_size, self.d_state, device=device)
        rate_terms: List[Tensor] = []
        delta_terms: List[Tensor] = []
        latent_states: List[Tensor] = []
        recon_targets: List[Tensor] = []
        recon_masks: List[Tensor] = []

        for t in range(seq_len):
            hidden_t = embedded[:, t, :]
            mask_t = mask[:, t, :] if mask is not None else None

            mu_q, logvar_q = self.encoder(state_prev, hidden_t)
            mu_p, logvar_p = self.prior(state_prev)

            eps = torch.randn_like(mu_q)
            std_q = torch.exp(0.5 * logvar_q)
            state_sample = mu_q + std_q * eps

            rate_step = gaussian_kl_divergence(mu_q, logvar_q, mu_p, logvar_p).sum(dim=-1)
            rate_terms.append(rate_step)

            rate_detached = (rate_step / self.d_state).detach().unsqueeze(-1)
            delta, B, C = self.gate(hidden_t, rate_feedback=rate_detached)

            A_bar, B_bar = van_loan_discretization(A, B, delta)
            A_bar = project_spectral_radius(A_bar, self.spectral_radius)
            input_contrib = torch.bmm(B_bar, hidden_t.unsqueeze(-1)).squeeze(-1)
            state_prev = torch.bmm(A_bar, state_sample.unsqueeze(-1)).squeeze(-1) + input_contrib

            latent = torch.bmm(C, state_sample.unsqueeze(-1)).squeeze(-1)
            latent_states.append(latent)
            recon_targets.append(x[:, t, :])
            if mask_t is not None:
                recon_masks.append(mask_t)
            delta_terms.append(delta)

        latent_seq = torch.stack(latent_states, dim=1)
        final_state = latent_seq[:, -1, :]

        pred_params = self.pred_head(final_state)
        pred_params = pred_params.view(batch_size, self.pred_len, self.enc_in, 2)
        pred_mean = pred_params[..., 0]
        pred_logvar = pred_params[..., 1].clamp(min=-10.0, max=5.0)

        mid_state = latent_seq[:, seq_len // 2, :]
        reconstruction = self.reconstruction_head(mid_state)
        recon_target = recon_targets[seq_len // 2]
        recon_mask = recon_masks[seq_len // 2] if recon_masks else None

        rate_stack = torch.stack(rate_terms, dim=1)
        avg_rate = rate_stack.mean(dim=1)

        outputs: Dict[str, Tensor] = {
            "pred_mean": pred_mean,
            "pred_logvar": pred_logvar,
            "latent_seq": latent_seq,
            "reconstruction": reconstruction,
            "reconstruction_target": recon_target,
            "reconstruction_mask": recon_mask,
            "avg_rate_per_sample": avg_rate,
            "rate_per_timestep": rate_stack,
            "delta_traj": torch.stack(delta_terms, dim=1),
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
        pred_mean = outputs["pred_mean"]
        pred_logvar = outputs["pred_logvar"]

        nll = gaussian_nll(target, pred_mean, pred_logvar, mask=target_mask)
        crps = gaussian_crps(target, pred_mean, pred_logvar, mask=target_mask)
        pred_loss = nll if self.scoring == "nll" else crps

        mse_loss = masked_mse(pred_mean, target, mask=target_mask)
        mae_loss = masked_mae(pred_mean, target, mask=target_mask)

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
