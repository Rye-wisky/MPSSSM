# Sequence-VIB: Seq-MPS with Budgeted Selectivity

Sequence-VIB implements the **Seq-MPS** architecture introduced in the refreshed theory (Sections 1–5) and the experimental protocol described in Sections 6–11. The repository now exposes a unified benchmark pipeline that covers multi-seed training, robustness diagnostics, resource accounting, and result summarisation.

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

- `experiment`: datasets, horizons, and random seeds. Each dataset block specifies `data_path`, `seq_len`, `enc_in`, split ratios, missing-value policy, and optional detrending window.
- `training`: optimiser, dual-rate budget (`rate_budget`), dual ascent step (`dual_step`), scoring rule (NLL or CRPS), patience, and gradient clipping.
- `model`: architectural widths, feedback strengths, and Van-Loan discretisation limits.
- `evaluation`: latency probe configuration (warm-up + repetitions).
- `robustness`: Gaussian, impulse, spurious, and structured-missing scenarios for Section 9 diagnostics.

Modify the YAML to adapt horizons, seeds, or hardware settings.

## 4. Running the Benchmark Suite

The orchestrator dispatches dataset × horizon jobs across the available GPUs. Seq-MPS automatically performs multi-seed training, validation-driven early stopping, dual-λ adaptation, test evaluation, latency probing, and robustness sweeps.

```bash
python main.py --config configs/seq_mps_suite.yaml --num_gpus 2
```

Each job writes logs under `results/benchmarks/<dataset>/H<horizon>/`. Per-seed artefacts include:

- `seed_<k>.json`: validation trace, final λ, latency, throughput, and robustness metrics.
- `checkpoints/seed<k>.pth`: best checkpoint (state dict + configuration snapshot).
- `summary.json`: aggregated mean/std tables over seeds, resource statistics, and robustness summaries.

## 5. Inspecting Results

Summarise the entire suite (Sections 8–11 tables) via:

```bash
python scripts/summarize_results.py --root results/benchmarks --output results/summary_tables
```

This produces Markdown/CSV tables for accuracy, resource usage, and robustness degradation, ready to drop into the paper’s Section 8–10 figures/tables.

## 6. Architecture Highlights

- **Selective SSM Backbone**: Input-dependent Δₖ, Bₖ, Cₖ via a gated network, stabilised with Van-Loan discretisation and spectral projection.
- **Seq-VIB Objective**: Diagonal-Gaussian encoder + conditional prior yields per-step KL rates bounding I(U₁:ₖ; Hₖ). NLL/CRPS scoring enforces predictive sufficiency.
- **Budgeted Selectivity**: Dual ascent on the KL-rate constraint plus stop-gradient feedback shrink Δₖ, Bₖ, Cₖ proportionally to information cost.
- **Diagnostics**: Training loop logs rate, λ, throughput, and memory. Evaluation returns MSE/MAE/NLL/CRPS; robustness injects Gaussian, impulse, spurious, and structured-missing perturbations.

## 7. Mapping to Paper Sections

| Section | Repository Support |
|---------|--------------------|
| 6. Experimental Setup | `configs/seq_mps_suite.yaml` encodes datasets, splits, metrics, hardware logging, and fairness defaults. |
| 7. Accuracy Results | `summary.json` + `results/summary_tables/accuracy.{csv,md}` provide mean±std with significance-ready numbers. |
| 8. Robustness & Shift | `summary.json['robustness']` captures ΔMSE and degradation for each noise type. |
| 9. Ablations | Extend `experiment.seeds` / `model` / `training` blocks; per-seed logs record KL heatmaps, λ trajectories, and rate feedback. |
| 10. Extensibility | Swap `model` parameters or integrate alternative backbones by following `models/mps_ssm.py` interface. |

## 8. Quick Start Cheat Sheet

```bash
# Single dataset / horizon (override orchestrator)
python run_experiment.py --dataset ETTh1 --pred_len 336 --config configs/seq_mps_suite.yaml --gpu_id 0

# After runs, summarise tables
python scripts/summarize_results.py
```

Results reside in `results/benchmarks/`, grouped by dataset and horizon. Each directory contains both raw per-seed logs and aggregated statistics to ease reproducibility.

## 9. License

This project remains under the MIT License (see `LICENSE`).

For questions or reproduction notes, open an issue or contact the maintainers.
