# Continuous HRV Estimation from PPG via State-Space Modeling

Official implementation for **Continuous Heart Rate Variability Estimation From PPG via State-Space Modeling** by Berken Utku Demirel and Christian Holz.

The method estimates ECG-aligned inter-beat intervals (IBIs) from wearable photoplethysmography (PPG), then refines the estimates with a causal, learnable state-space model (SSM). Motion (IMU) and temperature are used as auxiliary signals when available. The result is a continuous IBI sequence from which standard HRV measures, including SDNN, RMSSD, and LF/HF power, are computed.

## Highlights

- Two-stage model: a 1-D UNet predicts IBI mean and uncertainty; a learnable AR(2) state-space model refines the sequence.
- Multimodal support for PPG, 3-axis IMU, and temperature; a PPG-only path is used for BIDMC.
- Subject-level cross-validation on PPG-DaLiA, WildPPG, and BIDMC.
- Reproducible evaluation outputs for IBI accuracy, time-domain HRV (SDNN/RMSSD), and frequency-domain HRV (LF/HF Bland–Altman statistics).
- Traditional foot-to-foot and optional NeuroKit2 peak-detection baselines, plus a CNN–LSTM (DCL) learning baseline.

## Paper results at a glance

The following are the reported IBI results (mean ± standard deviation across subjects/folds, in ms). `r` is Pearson correlation with ECG-derived IBI.

| Dataset | Method | MAE ↓ | RMSE ↓ | r ↑ |
| --- | --- | ---: | ---: | ---: |
| DaLiA | Ours | 60.0 ± 7.3 | 83.8 ± 8.4 | 0.81 ± 0.11 |
| WildPPG | Ours | 89.4 ± 17.1 | 113.5 ± 25.6 | 0.60 ± 0.11 |
| BIDMC | Ours | 8.65 ± 3.12 | 13.2 ± 10.4 | 0.91 ± 0.10 |

For full SDNN, RMSSD, and frequency-domain results, see the manuscript. Example experiment artifacts—checkpoints, per-subject arrays, and fold summaries—are in `results/`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Main training/evaluation entry point. Runs all cross-validation folds for the selected dataset. |
| `trainer.py` | Model construction, two-stage training, inference, quality gating, and HRV evaluation. |
| `models/models_nc.py` | UNet and DCL model definitions. |
| `models/SSM.py` | Learnable AR(2) state-space head and causal-Transformer ablation head. |
| `data_preprocess/` | Dataset loaders and subject-level split logic. |
| `one_time_wild.py` | One-off converter from a WildPPG MATLAB segment file to per-subject NPZ shards. |
| `trad_eval.py` | Traditional signal-processing baseline (foot-to-foot or NeuroKit2 peaks). |
| `results/aggregate_folds.py` | Aggregates per-fold CSV files into overall summaries. |
| `Figures/to_plot.py` | Produces per-subject correlation and Bland–Altman SVG plots. |
| `results/` | Included checkpoints and evaluation artifacts. |

## Setup

The training entry point is designed for Linux with an NVIDIA GPU and a CUDA-enabled PyTorch installation. It uses `torchrun` even for one GPU because distributed initialization is part of `main.py`.

Create an environment (Python 3.8+), install the CUDA-enabled PyTorch build appropriate for your system from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m pip install numpy scipy pandas matplotlib seaborn scikit-learn fitlog mat73
```

`neurokit2` is only needed for the optional Elgendi/NeuroKit peak-detection baseline:

```bash
python -m pip install neurokit2
```

Run commands from the repository root. A quick environment check is:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data preparation

The datasets are **not included** in this repository. Please download and use them according to their respective licenses:

- [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia)
- [WildPPG](https://github.com/eth-siplab/WildPPG)
- [BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/)

The loaders consume pre-segmented `seg_*.npz` files, not the raw dataset downloads. All signals are represented as overlapping 8-second windows with a 2-second hop; PPG/IMU windows are sampled at 128 Hz in the experiment pipeline.

### NPZ schema

For **DaLiA** and **WildPPG**, each subject shard must contain:

```text
ppg         (n_windows, 1024)      PPG windows
imu         (n_windows, 1024, 3)   accelerometer windows
temp        (n_windows,) object    per-window temperature arrays
ibi_ecg_ms  (n_windows,) object    per-window ECG IBI arrays, in milliseconds
```

For **BIDMC**, the loader requires only:

```text
ppg         (n_windows, 2048)      PPG windows
ibi_ecg_ms  (n_windows,) object    per-window ECG IBI arrays, in milliseconds
```

The supplied `one_time_wild.py` converts a MATLAB file containing `segs` with these fields into WildPPG NPZ shards. Before running it, set `SRC` and `OUT_DIR` at the top of the file.

### Point the loaders at your files

Update the hard-coded `NPZ_DIR` near the top of these files:

```text
data_preprocess/data_prep_dalia.py
data_preprocess/data_prep_wildppg.py
```

For BIDMC, provide the directory without modifying source code:

```bash
export BIDMC_NPZ_DIR=/path/to/bidmc_npz
```

Each directory must contain files named `seg_*.npz`. The DaLiA/WildPPG fold map in the current loaders uses five held-out subject groups: `0–2`, `3–5`, `6–8`, `9–11`, and `12–14`; therefore keep the sort order of your shards consistent with this indexing.

## Train and evaluate

Every `main.py` run trains and evaluates all folds: five folds for DaLiA/WildPPG and ten for BIDMC. Results are written below `results/<model_tag>/fold_<k>/ratio_<keep_ratio>/`.

### Proposed UNet + SSM model

```bash
# WildPPG
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset wildppg --backbone unet --batch_size 128 --lr 5e-4 \
  --n_epoch 60 --keep_ratio 0.80

# PPG-DaLiA
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset dalia --backbone unet --batch_size 128 --lr 5e-4 \
  --n_epoch 60 --keep_ratio 0.80

# BIDMC
BIDMC_NPZ_DIR=/path/to/bidmc_npz \
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset bidmc --backbone unet --batch_size 128 --lr 5e-4 \
  --n_epoch 60 --keep_ratio 0.80
```

The UNet is trained first; its weights are then frozen while the SSM head is trained for 15 epochs. For the multimodal datasets the default sequential chunk length is eight 2-second updates. `--keep_ratio 0.80` retains the highest-quality 80% of intervals at evaluation, matching the paper setting.

### Ablations and baselines

```bash
# Stage-1 UNet only (no state-space refinement)
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset dalia --backbone unet --batch_size 128 --no_ssm

# Causal Transformer sequence-head ablation
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset wildppg --backbone unet --batch_size 128 --stage2_head ctf

# CNN–LSTM (DCL) baseline, evaluated without the SSM
torchrun --standalone --nproc_per_node=1 main.py \
  --dataset wildppg --backbone DCL --batch_size 128 --no_ssm
```

To run on multiple GPUs, change `--nproc_per_node` to the number of local GPUs. Make sure the per-GPU batch size and worker count fit available memory.

## Traditional PPG baseline

`trad_eval.py` implements 0.5–8 Hz filtering, PPG event detection, IBI cleaning, and top-SNR window selection. It currently reads `NPZ_DIR` at the top of the script; update that path before use.

```bash
# Foot-to-foot baseline (default)
python trad_eval.py --out_dir results/traditional_bidmc --detector scipy_feet

# NeuroKit2 peak detector
python trad_eval.py --out_dir results/traditional_bidmc_neurokit --detector nk_peaks
```

## Summarize and visualize results

Aggregate a full cross-validation experiment:

```bash
python results/aggregate_folds.py \
  --root results/unet_dalia_cuda0_bs128_sw1024 \
  --keep_ratio 0.80
```

This creates `fold_summary.csv` and `overall_summary.csv` in the corresponding `ratio_0.80/` directory.

Generate per-subject IBI correlation and Bland–Altman plots from a fold directory:

```bash
python Figures/to_plot.py \
  --results_dir results/unet_dalia_cuda0_bs128_sw1024/fold_0/ratio_0.80 \
  --out_dir Figures/Dalia/fold_0/ratio_0.80
```

The plotting script can optionally trim residual outliers for display only; use `--outlier_mode none` to plot all valid points.

## Output files

For each fold and quality ratio:

- `subject_<id>.npz`: estimated and ECG IBI sequences, masks, and per-subject metrics.
- `summary_fold<k>_r<ratio>.csv`: one-row-per-subject metrics.
- `fold_summary.csv` / `overall_summary.csv`: aggregate reports created by `aggregate_folds.py`.
- `<model_tag>.pt`: first-stage model checkpoint; `<model_tag>_ssm_last.pt`: final SSM head checkpoint.

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{11456553,
  author={Demirel, Berken Utku and Holz, Christian},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={Continuous Heart Rate Variability Estimation From PPG via State-Space Modeling}, 
  year={2026},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/TBME.2026.3678004}}
```

## License and data use

No repository license file is currently included. The individual datasets are distributed separately and remain subject to their original terms and licenses.
