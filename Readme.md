# Sequence-VIB: Seq-MPS with Budgeted Selectivity

Sequence-VIB implements the **Seq-MPS** architecture introduced in the refreshed theory (Sections 1–5) and the experimental protocol described in Sections 6–11. The repository now exposes a unified benchmark pipeline that covers multi-seed training, ablation suites, cross-backbone adapters, robustness diagnostics, resource accounting, and result summarisation.

## 1. Environment & Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch ≥ 2.0.0 with CUDA is recommended. Mixed precision is supported out of the box.

## 2. Data Preparation

Download the eight LTSF benchmarks into the `data/` directory. Example for the ETT family:

```bash
mkdir -p data/ETT-small
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm2.csv
```

Replicate the procedure for `weather`, `electricity`, `traffic`, and `exchange` datasets following their public sources. The loaders perform per-dataset z-score normalisation (training-split only), forward/backward filling, optional detrending, and missing-mask generation.

## 3. Configuration Schema

All experiments are configured through [`configs/seq_mps_suite.yaml`](configs/seq_mps_suite.yaml). Key sections:

- `experiment`: datasets, horizons, and random seeds. Each dataset block specifies `data_path`, `seq_len`, `enc_in`, split ratios, missing-value policy, and optional detrending window. The optional `ablations` dictionary captures per-study overrides (e.g., deterministic encoders, fixed-λ baselines, alternative priors, distribution heads).
- `training`: optimiser, dual-rate budget (`rate_budget`), dual ascent step (`dual_step`), scoring rule (NLL or CRPS), patience, gradient clipping, and dual-update toggles.
- `model`: architectural widths, feedback strengths, stochastic encoder settings, prior/head/preprocessor selections, and Van-Loan discretisation limits. `preprocessor.type` controls the backbone adapter (`identity`, `dlinear`, `patch`, `mamba`).
- `evaluation`: latency probe configuration (warm-up + repetitions).
- `robustness`: Gaussian, impulse, spurious, and structured-missing scenarios for Section 9 diagnostics.
- `diagnostics`: per-seed heatmap capture, correlation reporting, and Pareto-front settings used in Sections 9–10.

Modify the YAML to adapt horizons, seeds, or hardware settings.

## 4. Running the Benchmark Suite

The orchestrator dispatches dataset × horizon jobs across the available GPUs. Seq-MPS automatically performs multi-seed training, validation-driven early stopping, dual-λ adaptation, test evaluation, latency probing, robustness sweeps, and diagnostic logging.

```bash
python main.py --config configs/seq_mps_suite.yaml --num_gpus 2
```

To inspect individual studies (e.g., ablations or backbone swaps) without the orchestrator, use `run_experiment.py` with the new CLI overrides:

```bash
# Rate-only ablation with deterministic encoder on ETTh1@336
python run_experiment.py --dataset ETTh1 --pred_len 336 \
  --config configs/seq_mps_suite.yaml --ablation deterministic_encoder --backbone seq_mps

# Patch-style backbone + Student-t head on Electricity@96
python run_experiment.py --dataset electricity --pred_len 96 \
  --config configs/seq_mps_suite.yaml --ablation student_t_head --backbone mps_patchtst
```

Each job writes logs under `results/benchmarks/<backbone>/<ablation?>/<dataset>/H<horizon>/`. Per-seed artefacts include:

- `seed_<k>.json`: validation trace, final λ, latency, throughput, robustness metrics, and diagnostics summary.
- `checkpoints/seed<k>.pth`: best checkpoint (state dict + configuration snapshot).
- `diagnostics/seed<k>_traces.npz` (optional): KL-rate/Δₖ/‖Bₖ‖/‖Cₖ‖ trajectories saved when `diagnostics.save_rate_traces=true`.
- `summary.json`: aggregated mean/std tables over seeds, resource statistics, robustness summaries, and diagnostic aggregates (average rates, gate correlations, Pareto tuples).

## 5. Inspecting Results

Summarise the entire suite (Sections 8–11 tables) via:

```bash
python scripts/summarize_results.py --root results/benchmarks --output results/summary_tables
```

This produces Markdown/CSV tables for accuracy, resource usage, robustness degradation, and diagnostic statistics (average information rates, rate–gate correlations, Pareto fronts), ready to drop into the paper’s Section 8–10 figures/tables.

## 6. Architecture Highlights

- **Selective SSM Backbone**: Input-dependent Δₖ, Bₖ, Cₖ via a gated network, stabilised with Van-Loan discretisation and spectral projection. Optional preprocessors (`identity`/`dlinear`/`patch`/`mamba`) implement Section 10 backbone adapters.
- **Seq-VIB Objective**: Diagonal-Gaussian encoder + conditional prior yields per-step KL rates bounding I(U₁:ₖ; Hₖ). Student-t and quantile heads are available to probe alternative strictly proper scoring rules.
- **Budgeted Selectivity**: Dual ascent on the KL-rate constraint plus stop-gradient feedback shrink Δₖ, Bₖ, Cₖ proportionally to information cost. Dual updates can be frozen to reproduce fixed-λ baselines.
- **Diagnostics**: Training loop logs rate, λ, throughput, and memory. Evaluation returns MSE/MAE/NLL/CRPS; robustness injects Gaussian, impulse, spurious, and structured-missing perturbations. Diagnostics persist KL-rate heatmaps, gate norms, Spearman correlations, and clean/noisy Pareto tuples.

## 7. Mapping to Paper Sections

| Section | Repository Support |
|---------|--------------------|
| 6. Experimental Setup | `configs/seq_mps_suite.yaml` encodes datasets, splits, metrics, hardware logging, and fairness defaults. |
| 7. Accuracy Results | `summary.json` + `results/summary_tables/accuracy.{csv,md}` provide mean±std with significance-ready numbers. |
| 8. Robustness & Shift | `summary.json['robustness']` captures ΔMSE and degradation for each noise type. |
| 9. Ablations & Diagnostics | `experiment.ablations` + `diagnostics` configure Rate-only vs Rate+Reconstruct, prior/head/encoder variants, and export KL heatmaps + correlations under `diagnostics/`. |
| 10. Extensibility | `model.preprocessor.type` selects Seq-MPS / MPS-DLinear / MPS-PatchTST / MPS-Mamba adapters; results aggregate under `results/benchmarks/<backbone>/...`. |

## 8. Quick Start Cheat Sheet

```bash
# Single dataset / horizon (override orchestrator)
python run_experiment.py --dataset ETTh1 --pred_len 336 --config configs/seq_mps_suite.yaml --gpu_id 0

# With ablation and backbone overrides
python run_experiment.py --dataset Traffic --pred_len 96 --config configs/seq_mps_suite.yaml \
  --ablation rate_only --backbone mps_dlinear

# After runs, summarise tables
python scripts/summarize_results.py
```

Results reside in `results/benchmarks/<backbone>/<ablation?>/`, grouped by dataset and horizon. Each directory contains raw per-seed logs, diagnostics traces, and aggregated statistics to ease reproducibility.

## 9. License

This project remains under the MIT License (see `LICENSE`).

For questions or reproduction notes, open an issue or contact the maintainers.
