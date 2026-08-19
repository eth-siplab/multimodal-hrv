import torch
import torch.nn as nn
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import os
import pickle as cp
from models.models_nc import *
from models.SSM import SSMHeadAR2
from models.SSM import CausalTransformerHead
from typing import Tuple, Optional, Dict
from data_preprocess import data_prep_wildppg
from data_preprocess import data_prep_dalia
from data_preprocess import data_prep_bidmc
from copy import deepcopy
from utils import _hrv_windowed, _hrv_freq_windowed, _corr_pairs, _ba_stats_log10, _ba_log10_to_percent, _compute_ba_pct
from units import DELTA, LOW_RR, HIGH_RR

def setup_dataloaders(args):
    if args.dataset == 'wildppg':
        args.n_feature = 1
        args.len_sw = 1024
        train_loaders, val_loader, test_loader = data_prep_wildppg.prep_wild(args)
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 1024
        train_loaders, val_loader, test_loader = data_prep_dalia.prep_dalia(args)       
    if args.dataset == 'bidmc':
        args.n_feature = 1
        args.len_sw = 2048
        train_loaders, val_loader, test_loader = data_prep_bidmc.prep_bidmc(args) 
    return train_loaders, val_loader, test_loader

def setup_ssm_dataloaders(args):
    if args.dataset == 'wildppg':
        args.n_feature = 1
        args.len_sw = 1024
        seq_train_loader, ordered_test = data_prep_wildppg.prep_wild(args)
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 1024
        seq_train_loader, ordered_test = data_prep_dalia.prep_dalia(args)
    if args.dataset == 'bidmc':
        args.n_feature = 1
        args.len_sw = 2048
        seq_train_loader, ordered_test = data_prep_bidmc.prep_bidmc(args)

    # Tag dataset so iter_subject_ordered can choose the right stacking funcs
    if isinstance(ordered_test, dict):
        ordered_test["_dataset"] = args.dataset

    return seq_train_loader, ordered_test        

def build_model(args, classifier=False, backbone=False):
    # set up backbone network
    if args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == 'unet':
        if args.dataset == 'wildppg' or args.dataset == 'dalia':
            model = UNET_1D_simp(input_dim=args.n_feature, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args)
        elif args.dataset == 'bidmc':
            model = UNET_1D_simp_PPGOnly(input_dim=args.n_feature, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args)
    elif args.backbone == 'resnet':
        model = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=2, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, output_dim=args.out_dim, backbone=True)
    elif args.backbone == 'DCL':
        if args.dataset == 'wildppg' or args.dataset == 'dalia':
            model = DeepConvLSTM(conv_kernels=64, kernel_size=5, LSTM_units=128)
        elif args.dataset == 'bidmc':
            model = DeepConvLSTM_PPGOnly(conv_kernels=64, kernel_size=5, LSTM_units=128)

    # set up linear classfier
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier
    else:
        return model

def setup_SSM(args, DEVICE):
    head_kind = getattr(args, "stage2_head", "ssm").lower()

    if head_kind == "ctf":
        head = CausalTransformerHead(
            feat_dim=args.ss_feat_dim,
            d_model=getattr(args, "ctf_d_model", 128),
            nhead=getattr(args, "ctf_nhead", 4),
            num_layers=getattr(args, "ctf_layers", 2),
            dropout=getattr(args, "ctf_dropout", 0.1),
        ).to(DEVICE)
    else:
        head = SSMHeadAR2(
            feat_dim=args.ss_feat_dim,
            hidden=args.ssm_hidden,
            use_bias=True,
            learn_R_scale=True,
        ).to(DEVICE)

    if torch.distributed.is_initialized():
        # keep your existing DDP-wrapping logic (currently stubbed in your file)
        pass

    return head

@torch.no_grad()
def forward_unet_seq(frozen_unet, windows):
    """
    Vectorized over time: one UNet call on [B*T] windows.
    """
    B, T, C, W = windows.shape
    # Split channels
    ppg   = windows[:, :, 0:1, :]                       # [B,T,1,W]
    imu   = windows[:, :, 1:4, :]                       # [B,T,3,W]
    tempc = windows[:, :, 4, :].mean(dim=-1, keepdim=True)  # [B,T,1]

    # Flatten time into batch
    ppg_bt   = ppg.reshape(B*T, 1, W)       # [B*T,1,W]
    imu_bt   = imu.reshape(B*T, 3, W)       # [B*T,3,W]
    temp_bt  = tempc.reshape(B*T, 1)        # [B*T,1]

    m_bt, s_bt = frozen_unet(ppg_bt, imu_bt, temp_bt)  if temp_bt.mean() > 0 else frozen_unet(ppg_bt)  # bidmc has no temp/imu, so skip those channels
    m_seq = m_bt.view(B, T)
    s_seq = s_bt.view(B, T)
    return m_seq, s_seq

@torch.inference_mode()  # memory-leaner than no_grad()
def forward_unet_seq_chunked(
    frozen_unet,
    windows: torch.Tensor,          # [B, T, C=5, W] (CPU or CUDA)
    device: torch.device,
    chunk_T: int = 1024,            # time-chunk length (tune: 512..4096)
    use_amp: bool = True            # enable autocast on CUDA for memory speed
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    UNet(ppg, imu, temp) over a long sequence by slicing the time axis.

    Returns
    -------
    m_seq  : [B, T]  UNet IBI predictions (on `device`)
    s_seq  : [B, T]  UNet log-variance (on `device`)
    """
    B, T, C, W = windows.shape

    # Ensure we don't move the *entire* long tensor to GPU at once.
    windows_cpu = windows if windows.device.type == "cpu" else windows.cpu()

    # Pre-allocate outputs directly on the target device (tiny compared to inputs)
    m_out = torch.empty(B, T, device=device)
    s_out = torch.empty(B, T, device=device)

    # Process in time-chunks
    for t0 in range(0, T, chunk_T):
        t1 = min(T, t0 + chunk_T)
        Tc = t1 - t0

        # Slice the chunk 
        chunk = windows_cpu[:, t0:t1]          # [B, Tc, 5, W]

        # Split channels
        ppg   = chunk[:, :, 0:1, :]            # [B,Tc,1,W]
        imu   = chunk[:, :, 1:4, :]            # [B,Tc,3,W]
        tempc = chunk[:, :, 4, :].mean(-1, keepdim=True)  # [B,Tc,1]

        # Flatten time into batch, then move just this chunk to GPU
        ppg_bt  = ppg.reshape(B*Tc, 1, W).to(device, non_blocking=True)
        imu_bt  = imu.reshape(B*Tc, 3, W).to(device, non_blocking=True)
        temp_bt = tempc.reshape(B*Tc, 1).to(device, non_blocking=True)

        # Run the UNet once for this chunk (optional AMP on CUDA)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                mu_bt, logvar_bt = frozen_unet(ppg_bt, imu_bt, temp_bt)  if temp_bt.mean() > 0 else frozen_unet(ppg_bt)  # bidmc has no temp/imu
        else:
            mu_bt, logvar_bt = frozen_unet(ppg_bt, imu_bt, temp_bt) if temp_bt.mean() > 0 else frozen_unet(ppg_bt)  # bidmc has no temp/imu

        # Reshape back to [B,Tc] and write into the pre-allocated tensors
        m_out[:, t0:t1] = mu_bt.view(B, Tc)
        s_out[:, t0:t1] = logvar_bt.view(B, Tc)

        # Free chunk tensors ASAP
        del chunk, ppg, imu, tempc, ppg_bt, imu_bt, temp_bt, mu_bt, logvar_bt
        if device.type == "cuda":
            torch.cuda.synchronize()  # safer timing when memory is tight

    return m_out, s_out    

def train_stage2_epoch(loader, unet_frozen, ssm_head, opt, device, epoch=0, warmup_epochs=3):
    unet_frozen.eval()
    ssm_head.train()

    tot_loss = 0.0
    count = 0

    warmup_w = 1.0 if epoch < warmup_epochs else 0.0

    for windows, feats, rr_ecg, mask in loader:
        windows = windows.to(device)
        feats   = feats.to(device)
        rr_ecg  = rr_ecg.to(device)
        mask    = mask.to(device)

        with torch.no_grad():
            m_seq, logvar = forward_unet_seq(unet_frozen, windows)

        rr_hat, dpat_hat, ex = ssm_head(m_seq, logvar, feats, mask=mask, warmup_w=warmup_w)

        valid_mask = (mask > 0) & torch.isfinite(rr_ecg) & torch.isfinite(rr_hat)
        if valid_mask.sum() == 0:
            continue

        # 1) accuracy
        L_rr = F.l1_loss(rr_hat[valid_mask], rr_ecg[valid_mask])

        # 2) innovation consistency (SSM-only)
        L_innov = rr_hat.new_tensor(0.0)
        if isinstance(ex, dict) and ("innov" in ex) and ("S" in ex):
            innov = ex["innov"]
            S = ex["S"]
            ok = valid_mask & torch.isfinite(innov) & torch.isfinite(S) & (S > 0)
            if ok.sum() > 0:
                innov_n = innov[ok] / torch.sqrt(S[ok] + 1e-6)
                L_innov = (innov_n.pow(2) - 1.0).abs().mean()

        # 3) bounds
        L_bounds = (F.relu(300.0 - rr_hat[valid_mask]).mean()+ F.relu(rr_hat[valid_mask] - 1500.0).mean())

        # 4) dpat regularization (keep for both heads; for CTF it's just "delta")
        L_dpat = dpat_hat[valid_mask].pow(2).mean() * 0.001

        # Total (innov term auto-zero for CTF)
        loss = L_rr + 0.05 * L_innov + 0.5 * L_bounds + L_dpat

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ssm_head.parameters(), 1.0)
        opt.step()

        tot_loss += float(loss.item())
        count += 1

    return tot_loss / max(count, 1)

@torch.no_grad()
def val_stage2_epoch(seq_loader, unet_frozen, ssm_head, device):
    unet_frozen.eval()
    ssm_head.eval()
    tot = cnt = 0.0, 0
    for windows, feats, rr_ecg, mask in seq_loader:
        windows = windows.to(device); feats = feats.to(device)
        rr_ecg  = rr_ecg.to(device);  mask  = mask.to(device)
        m_seq, logvar = forward_unet_seq(unet_frozen, windows)
        rr_hat, dpat_hat, ex = ssm_head(m_seq, logvar, feats, mask=mask)
        loss = F.huber_loss(rr_hat[mask>0], rr_ecg[mask>0], delta=0.02)
        tot += float(loss.item()); cnt += 1
    return tot/cnt if cnt else 0.0

@torch.no_grad()
def eval_subjectwise(
    unet_frozen: torch.nn.Module,
    ssm_head: torch.nn.Module,
    ordered_test: dict,
    DEVICE: torch.device,
    unet_chunk_T: int = 1024, # was 2048
    unet_amp: bool = True,
    quality_mask: bool = False,              # enable/disable gating
    keep_ratio: float = 0.80,            # used when quality_tau is None
    quality_weights=(1.0, 0.5, 0.5, 0.0),    # w for (z, sqrtR, |dRR|, imu)
    quality_tau: float = None,               # optional fixed threshold; None => auto 80%
    step_sec: float = 2.0,
) -> Dict[int, dict]:
    """
    Subject-wise evaluation with optional quality gating.
    Returns per-subject rr_hat/rr_ecg plus metrics computed on kept frames.
    """
    results: Dict[int, dict] = {}
    unet_frozen.eval()
    ssm_head.eval()

    for sid, windows, feats, rr_lab, mask in iter_subject_ordered(ordered_test):
        # 1) UNet over long sequence (chunked, memory-safe)
        m_seq, logvar = forward_unet_seq_chunked(
            unet_frozen,
            windows,            # stays on CPU; helper slices & ships chunks
            device=DEVICE,
            chunk_T=unet_chunk_T,
            use_amp=unet_amp
        )  # [1,T], [1,T] on DEVICE

        # 2) SSM head
        feats_d = feats.to(DEVICE)
        mask_d  = mask.to(DEVICE)
        rr_hat, _, ex = ssm_head(m_seq, logvar, feats_d, mask=mask_d)  # [1,T], extras dict
        # _qc(ex, rr_hat) # debug if Kalman is behaving
        # 3) to numpy
        rr_hat_np = rr_hat.squeeze(0).detach().cpu().numpy()                # [T]
        rr_ecg_np = rr_lab.squeeze(0).detach().cpu().numpy()                # [T]
        base_mask_np = mask.squeeze(0).detach().cpu().numpy().astype(bool)  # [T]

        # 4) quality gating (optional)
        if quality_mask:
            # If you have IMU per step and want it in the score, pass it here
            keep_q, score, scales, tau = make_quality_mask(
                rr_hat, ex, imu_feat=None, w=quality_weights, scales=None, tau=quality_tau, keep_ratio=keep_ratio
            )  # keep_q is a numpy bool array [T]
            keep_q = np.asarray(keep_q, dtype=bool)
            final_keep = base_mask_np & keep_q
            coverage = float(final_keep.mean())
            q_tau = float(tau)
        else:
            final_keep = base_mask_np
            coverage = float(final_keep.mean())
            q_tau = float("nan")

        # Guard: if nothing kept, avoid crashes
        if not np.any(final_keep):
            metrics = {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "r": np.nan}
            hrv_hat = {"SDNN": np.nan, "RMSSD": np.nan}
            hrv_ecg = {"SDNN": np.nan, "RMSSD": np.nan}
            hrv_win_hat = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m","SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
            hrv_win_ecg = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m","SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
            freq_win_hat = {"LF5m_vals": [], "HF5m_vals": [], "LF10m_vals": [], "HF10m_vals": [],
                            "LF5m_idx": [], "HF5m_idx": [], "LF10m_idx": [], "HF10m_idx": []}
            freq_win_ecg = {"LF5m_vals": [], "HF5m_vals": [], "LF10m_vals": [], "HF10m_vals": [],
                            "LF5m_idx": [], "HF5m_idx": [], "LF10m_idx": [], "HF10m_idx": []}
        else:
            # 5) metrics on kept frames only
            metrics = _ibi_pair_metrics(rr_hat_np[final_keep], rr_ecg_np[final_keep], np.ones(final_keep.sum(), bool))
            hrv_hat = _hrv_basic(rr_hat_np[final_keep])
            hrv_ecg = _hrv_basic(rr_ecg_np[final_keep])
            # 5–10 min window HRV (means across windows)
            hrv_win_hat = _hrv_windowed(rr_hat_np, final_keep, step_sec=step_sec, win_minutes=(5, 10))
            hrv_win_ecg = _hrv_windowed(rr_ecg_np, final_keep, step_sec=step_sec, win_minutes=(5, 10))      
            # 5–10 min window HRV (frequency: LF/HF per-window + means)
            freq_win_hat = _hrv_freq_windowed(
                rr_hat_np, final_keep, step_sec=step_sec, win_minutes=(5, 10),
                fs_resample=4.0, use_log10=True, use_relative=True
            )
            freq_win_ecg = _hrv_freq_windowed(
                rr_ecg_np, final_keep, step_sec=step_sec, win_minutes=(5, 10),
                fs_resample=4.0, use_log10=True, use_relative=True
            )          

        (LF5_bias_pct, LF5_loa_lo_pct, LF5_loa_hi_pct, LF5_n,
        LF5_med_pct, LF5_q25_pct, LF5_q75_pct) = _compute_ba_pct(
            freq_win_hat["LF5m_vals"], freq_win_ecg["LF5m_vals"],
            freq_win_hat["LF5m_idx"],  freq_win_ecg["LF5m_idx"]
        )

        (HF5_bias_pct, HF5_loa_lo_pct, HF5_loa_hi_pct, HF5_n,
        HF5_med_pct, HF5_q25_pct, HF5_q75_pct) = _compute_ba_pct(
            freq_win_hat["HF5m_vals"], freq_win_ecg["HF5m_vals"],
            freq_win_hat["HF5m_idx"],  freq_win_ecg["HF5m_idx"]
        )

        (LF10_bias_pct, LF10_loa_lo_pct, LF10_loa_hi_pct, LF10_n,
        LF10_med_pct, LF10_q25_pct, LF10_q75_pct) = _compute_ba_pct(
            freq_win_hat["LF10m_vals"], freq_win_ecg["LF10m_vals"],
            freq_win_hat["LF10m_idx"], freq_win_ecg["LF10m_idx"]
        )

        (HF10_bias_pct, HF10_loa_lo_pct, HF10_loa_hi_pct, HF10_n,
        HF10_med_pct, HF10_q25_pct, HF10_q75_pct) = _compute_ba_pct(
            freq_win_hat["HF10m_vals"], freq_win_ecg["HF10m_vals"],
            freq_win_hat["HF10m_idx"], freq_win_ecg["HF10m_idx"]
        )       
        # 6) pack
        results[int(sid)] = {
            "rr_hat":   rr_hat_np,
            "rr_ecg":   rr_ecg_np,
            "mask":     final_keep.astype(np.uint8),
            "metrics":  metrics,
            "hrv_hat":  hrv_hat,
            "hrv_ecg":  hrv_ecg,
            "hrv_win_hat": hrv_win_hat,
            "hrv_win_ecg": hrv_win_ecg,
            "freq_win_hat": freq_win_hat,
            "freq_win_ecg": freq_win_ecg,
            "freq_ba": {
                # BA (mean±LOA) in %
                "LF5_bias_pct": LF5_bias_pct,   "LF5_loa_lo_pct": LF5_loa_lo_pct,   "LF5_loa_hi_pct": LF5_loa_hi_pct,   "LF5_n": LF5_n,
                "HF5_bias_pct": HF5_bias_pct,   "HF5_loa_lo_pct": HF5_loa_lo_pct,   "HF5_loa_hi_pct": HF5_loa_hi_pct,   "HF5_n": HF5_n,
                "LF10_bias_pct": LF10_bias_pct, "LF10_loa_lo_pct": LF10_loa_lo_pct, "LF10_loa_hi_pct": LF10_loa_hi_pct, "LF10_n": LF10_n,
                "HF10_bias_pct": HF10_bias_pct, "HF10_loa_lo_pct": HF10_loa_lo_pct, "HF10_loa_hi_pct": HF10_loa_hi_pct, "HF10_n": HF10_n,
                # Robust (median & IQR) in %
                "LF5_med_pct": LF5_med_pct,   "LF5_q25_pct": LF5_q25_pct,   "LF5_q75_pct": LF5_q75_pct,
                "HF5_med_pct": HF5_med_pct,   "HF5_q25_pct": HF5_q25_pct,   "HF5_q75_pct": HF5_q75_pct,
                "LF10_med_pct": LF10_med_pct, "LF10_q25_pct": LF10_q25_pct, "LF10_q75_pct": LF10_q75_pct,
                "HF10_med_pct": HF10_med_pct, "HF10_q25_pct": HF10_q25_pct, "HF10_q75_pct": HF10_q75_pct,
            },
            "coverage": coverage,
            "q_tau":    q_tau,
        }

        # free per-subject tensors early
        del m_seq, logvar, feats_d, mask_d, rr_hat
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results

@torch.no_grad()
def eval_subjectwise_unet_only(
    unet_frozen: torch.nn.Module,
    ordered_test: dict,
    DEVICE: torch.device,
    unet_chunk_T: int = 1024,
    unet_amp: bool = True,
    quality_mask: bool = False,
    keep_ratio: float = 0.80,
    quality_weights=(1.0, 0.5, 0.5, 0.0),
    quality_tau: float = None,
    step_sec: float = 2.0,
) -> Dict[int, dict]:
    """
    Subject-wise evaluation using UNet outputs directly (no SSM).
    Returns the same result dict structure as eval_subjectwise().
    """
    results: Dict[int, dict] = {}
    unet_frozen.eval()

    for sid, windows, feats, rr_lab, mask in iter_subject_ordered(ordered_test):
        m_seq, logvar = forward_unet_seq_chunked(
            unet_frozen,
            windows,                 # CPU ok; chunk helper moves slices
            device=DEVICE,
            chunk_T=unet_chunk_T,
            use_amp=unet_amp,
        )  # [1,T], [1,T] on DEVICE

        rr_hat = m_seq  # UNet-only prediction

        # to numpy
        rr_hat_np = rr_hat.squeeze(0).detach().cpu().numpy()
        rr_ecg_np = rr_lab.squeeze(0).detach().cpu().numpy()
        base_mask_np = mask.squeeze(0).detach().cpu().numpy().astype(bool)

        # quality gating (optional): reuse make_quality_mask via a dummy "ex"
        # Use UNet log-variance as an uncertainty proxy (R ~ exp(logvar)), no innovations available.
        if quality_mask:
            R = torch.exp(logvar)  # [1,T]
            ex = {
                "innov": torch.zeros_like(rr_hat),  # z-term becomes ~0
                "S": R,                              # treat as total variance proxy
                "R": R,                              # measurement variance proxy
            }
            keep_q, score, scales, tau = make_quality_mask(
                rr_hat, ex, imu_feat=None,
                w=quality_weights, scales=None, tau=quality_tau, keep_ratio=keep_ratio
            )
            keep_q = np.asarray(keep_q, dtype=bool)
            final_keep = base_mask_np & keep_q
            coverage = float(final_keep.mean())
            q_tau = float(tau) if tau is not None else float("nan")
        else:
            final_keep = base_mask_np
            coverage = float(final_keep.mean())
            q_tau = float("nan")

        # Guard: if nothing kept
        if not np.any(final_keep):
            metrics = {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "r": np.nan}
            hrv_hat = {"SDNN": np.nan, "RMSSD": np.nan}
            hrv_ecg = {"SDNN": np.nan, "RMSSD": np.nan}
            hrv_win_hat = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m","SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
            hrv_win_ecg = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m","SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
            freq_win_hat = {"LF5m_vals": [], "HF5m_vals": [], "LF10m_vals": [], "HF10m_vals": [],
                            "LF5m_idx": [], "HF5m_idx": [], "LF10m_idx": [], "HF10m_idx": []}
            freq_win_ecg = {"LF5m_vals": [], "HF5m_vals": [], "LF10m_vals": [], "HF10m_vals": [],
                            "LF5m_idx": [], "HF5m_idx": [], "LF10m_idx": [], "HF10m_idx": []}
        else:
            metrics = _ibi_pair_metrics(
                rr_hat_np[final_keep],
                rr_ecg_np[final_keep],
                np.ones(final_keep.sum(), bool),
            )
            hrv_hat = _hrv_basic(rr_hat_np[final_keep])
            hrv_ecg = _hrv_basic(rr_ecg_np[final_keep])

            hrv_win_hat = _hrv_windowed(rr_hat_np, final_keep, step_sec=step_sec, win_minutes=(5, 10))
            hrv_win_ecg = _hrv_windowed(rr_ecg_np, final_keep, step_sec=step_sec, win_minutes=(5, 10))

            freq_win_hat = _hrv_freq_windowed(
                rr_hat_np, final_keep, step_sec=step_sec, win_minutes=(5, 10),
                fs_resample=4.0, use_log10=True, use_relative=True
            )
            freq_win_ecg = _hrv_freq_windowed(
                rr_ecg_np, final_keep, step_sec=step_sec, win_minutes=(5, 10),
                fs_resample=4.0, use_log10=True, use_relative=True
            )

        # BA in % (same as eval_subjectwise)
        (LF5_bias_pct, LF5_loa_lo_pct, LF5_loa_hi_pct, LF5_n,
         LF5_med_pct, LF5_q25_pct, LF5_q75_pct) = _compute_ba_pct(
            freq_win_hat["LF5m_vals"], freq_win_ecg["LF5m_vals"],
            freq_win_hat["LF5m_idx"],  freq_win_ecg["LF5m_idx"]
        )

        (HF5_bias_pct, HF5_loa_lo_pct, HF5_loa_hi_pct, HF5_n,
         HF5_med_pct, HF5_q25_pct, HF5_q75_pct) = _compute_ba_pct(
            freq_win_hat["HF5m_vals"], freq_win_ecg["HF5m_vals"],
            freq_win_hat["HF5m_idx"],  freq_win_ecg["HF5m_idx"]
        )

        (LF10_bias_pct, LF10_loa_lo_pct, LF10_loa_hi_pct, LF10_n,
         LF10_med_pct, LF10_q25_pct, LF10_q75_pct) = _compute_ba_pct(
            freq_win_hat["LF10m_vals"], freq_win_ecg["LF10m_vals"],
            freq_win_hat["LF10m_idx"],  freq_win_ecg["LF10m_idx"]
        )

        (HF10_bias_pct, HF10_loa_lo_pct, HF10_loa_hi_pct, HF10_n,
         HF10_med_pct, HF10_q25_pct, HF10_q75_pct) = _compute_ba_pct(
            freq_win_hat["HF10m_vals"], freq_win_ecg["HF10m_vals"],
            freq_win_hat["HF10m_idx"],  freq_win_ecg["HF10m_idx"]
        )

        results[int(sid)] = {
            "rr_hat": rr_hat_np,
            "rr_ecg": rr_ecg_np,
            "mask": final_keep.astype(np.uint8),
            "metrics": metrics,
            "hrv_hat": hrv_hat,
            "hrv_ecg": hrv_ecg,
            "hrv_win_hat": hrv_win_hat,
            "hrv_win_ecg": hrv_win_ecg,
            "freq_win_hat": freq_win_hat,
            "freq_win_ecg": freq_win_ecg,
            "freq_ba": {
                "LF5_bias_pct": LF5_bias_pct,   "LF5_loa_lo_pct": LF5_loa_lo_pct,   "LF5_loa_hi_pct": LF5_loa_hi_pct,   "LF5_n": LF5_n,
                "HF5_bias_pct": HF5_bias_pct,   "HF5_loa_lo_pct": HF5_loa_lo_pct,   "HF5_loa_hi_pct": HF5_loa_hi_pct,   "HF5_n": HF5_n,
                "LF10_bias_pct": LF10_bias_pct, "LF10_loa_lo_pct": LF10_loa_lo_pct, "LF10_loa_hi_pct": LF10_loa_hi_pct, "LF10_n": LF10_n,
                "HF10_bias_pct": HF10_bias_pct, "HF10_loa_lo_pct": HF10_loa_lo_pct, "HF10_loa_hi_pct": HF10_loa_hi_pct, "HF10_n": HF10_n,
                "LF5_med_pct": LF5_med_pct,   "LF5_q25_pct": LF5_q25_pct,   "LF5_q75_pct": LF5_q75_pct,
                "HF5_med_pct": HF5_med_pct,   "HF5_q25_pct": HF5_q25_pct,   "HF5_q75_pct": HF5_q75_pct,
                "LF10_med_pct": LF10_med_pct, "LF10_q25_pct": LF10_q25_pct, "LF10_q75_pct": LF10_q75_pct,
                "HF10_med_pct": HF10_med_pct, "HF10_q25_pct": HF10_q25_pct, "HF10_q75_pct": HF10_q75_pct,
            },
            "coverage": coverage,
            "q_tau": q_tau,
        }

        del m_seq, logvar, rr_hat
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results

def iter_subject_ordered(ordered_test):
    """Yield (sid, windows_seq[T,C,W], feats_seq[T,F], rr_label_seq[T]) in order per subject."""
    x, imu, t, y, sid, pos = [ordered_test[k] for k in ("x","imu","t","y","sid","pos")]

    ds = ordered_test.get("_dataset", "wildppg")
    if ds == "dalia":
        prep = data_prep_dalia
    elif ds == "wildppg":
        prep = data_prep_wildppg
    elif ds == "bidmc":
        prep = data_prep_bidmc

    stack_fn = getattr(prep, "_stack_inputs_one", None)
    feat_fn  = getattr(prep, "_step_feats", None)
    if stack_fn is None or feat_fn is None:
        # fallback (keeps old behavior, but makes the failure mode explicit)
        stack_fn = data_prep_wildppg._stack_inputs_one
        feat_fn  = data_prep_wildppg._step_feats

    def _temp_at(k):
        tk = t[k]
        # supports scalar, shape [1], or shape [1,] / [*,1]
        if np.ndim(tk) == 0:
            return float(tk)
        tk = np.asarray(tk).reshape(-1)
        return float(tk[0]) if tk.size else float("nan")

    # group indices by subject then sort by pos
    s_unique = np.unique(sid)
    for s in s_unique:
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(pos[idx])]

        X = []; F = []; Y = []; M = []
        for k in idx:
            temp_k = _temp_at(k)

            xk = stack_fn(x[k], imu[k], temp_k)
            fk = feat_fn(imu[k], temp_k)

            yk = np.asarray(y[k], dtype=np.float32)

            # robust label extraction (scalar or vector)
            if yk.ndim == 0:
                y_mean = float(yk) if np.isfinite(yk) else np.nan
            else:
                y_mean = float(np.nanmean(yk)) if np.isfinite(yk).any() else np.nan

            X.append(xk)
            F.append(fk)
            Y.append(y_mean)
            M.append(1.0 if np.isfinite(y_mean) else 0.0)

        windows = torch.from_numpy(np.stack(X)).float().unsqueeze(0)              # [1,T,C,W]
        feats   = torch.from_numpy(np.stack(F)).float().unsqueeze(0)              # [1,T,F]
        rr_lab  = torch.from_numpy(np.asarray(Y, dtype=np.float32)).unsqueeze(0)  # [1,T]
        mask    = torch.from_numpy(np.asarray(M, dtype=np.float32)).unsqueeze(0)  # [1,T]
        yield int(s), windows, feats, rr_lab, mask    
  
# ---------- basic HRV on an RR (IBI) series ----------
def _hrv_basic(rr: np.ndarray) -> Dict[str, float]:
    """
    Compute basic HRV stats on an RR series (same units as rr).
    Returns SDNN and RMSSD. pNN50 omitted as requested.
    """
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 3:
        return {"SDNN": np.nan, "RMSSD": np.nan}
    diff = np.diff(rr)
    sdnn  = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff**2)))
    return {"SDNN": sdnn, "RMSSD": rmssd}

# ---------- IBI pairwise metrics ----------
def _ibi_pair_metrics(rr_hat: np.ndarray, rr_ecg: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compare estimated RR (rr_hat) vs ECG RR (rr_ecg).
    Returns MAE, RMSE, bias (mean error), and Pearson r.
    """
    rr_hat = np.asarray(rr_hat, dtype=float).reshape(-1)
    rr_ecg = np.asarray(rr_ecg, dtype=float).reshape(-1)
    mask   = np.asarray(mask, dtype=bool).reshape(-1)

    # Valid positions: in mask and both finite
    valid = mask & np.isfinite(rr_hat) & np.isfinite(rr_ecg)
    if valid.sum() < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "r": np.nan}

    e   = rr_hat[valid] - rr_ecg[valid]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))

    # Pearson r (guard against zero variance)
    r = np.nan
    if np.std(rr_hat[valid]) > 0 and np.std(rr_ecg[valid]) > 0:
        r = float(np.corrcoef(rr_hat[valid], rr_ecg[valid])[0, 1])

    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "r": r}  

def make_quality_mask(
    rr_hat: torch.Tensor,
    ex: dict,
    imu_feat: torch.Tensor = None,
    w=(1.0, 0.5, 0.5, 0.0),          # weights for (z, sqrtR, |dRR|, IMU)
    scales=None,                      # dict with cR,cD,cM or None => auto
    tau: float = None,                # fixed threshold; None => auto quantile
    keep_ratio: float = 0.80,         # used when tau is None
    min_valid: int = 5,               # need at least this many finite points
    eps: float = 1e-6,
):
    """
    Returns (keep_mask[T] bool, score[T] float, scales, tau_used).
    Safe against NaNs/Infs and short sequences.
    """
    # ---- to numpy, shape [T] ----
    rr = rr_hat.squeeze(0).detach().cpu().numpy()
    innov = ex.get("innov", None)
    S = ex.get("S", None)
    R = ex.get("R", None)

    if innov is None or S is None:
        # No innovation info -> keep all
        T = rr.shape[0]
        keep = np.ones(T, dtype=bool)
        score = np.zeros(T, dtype=float)
        if scales is None:
            scales = dict(cR=1.0, cD=1.0, cM=1.0)
        return keep, score, scales, np.nan

    innov = innov.squeeze(0).detach().cpu().numpy()
    S     = S.squeeze(0).detach().cpu().numpy()

    # sanitize S (can have <=0 or NaNs)
    S = np.where(np.isfinite(S) & (S > eps), S, np.nan)

    # normalized innovation z = |e|/sqrt(S)
    sqrtS = np.sqrt(np.where(np.isfinite(S), S, np.nan))
    z = np.abs(innov) / sqrtS
    z = np.where(np.isfinite(z), z, np.nan)

    # effective R term
    if R is not None:
        R = R.squeeze(0).detach().cpu().numpy()
        R = np.where(np.isfinite(R) & (R > eps), R, np.nan)
        sqrtR = np.sqrt(R)
    else:
        sqrtR = np.full_like(z, np.nan)

    # |ΔRR|
    drr = np.abs(np.diff(rr, prepend=rr[:1]))
    drr = np.where(np.isfinite(drr), drr, np.nan)

    # IMU RMS proxy (optional)
    if imu_feat is not None:
        im = imu_feat.squeeze(0).detach().cpu().numpy()  # [T,F] or [T]
        if im.ndim == 2:
            imu_rms = np.sqrt(np.nanmean(im**2, axis=-1))
        else:
            imu_rms = np.abs(im).astype(float)
        imu_rms = np.where(np.isfinite(imu_rms), imu_rms, np.nan)
    else:
        imu_rms = np.full_like(z, np.nan)

    # ---- scales (robust) ----
    if scales is None:
        # robust medians; if all-NaN, backoff to 1.0
        def med(x):
            x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size else 1.0
        cR = med(sqrtR)
        cD = med(drr)
        cM = med(imu_rms)
        scales = dict(cR=cR, cD=cD, cM=cM)
    else:
        cR, cD, cM = scales["cR"], scales["cD"], scales["cM"]

    # ---- compose score (lower is better) ----
    w1, w2, w3, w4 = w
    terms = []
    if w1 != 0: terms.append(w1 * z)
    if w2 != 0: terms.append(w2 * (sqrtR / (scales["cR"] + eps)))
    if w3 != 0: terms.append(w3 * (drr   / (scales["cD"] + eps)))
    if w4 != 0: terms.append(w4 * (imu_rms / (scales["cM"] + eps)))

    if not terms:
        # nothing to score with -> keep all
        T = rr.shape[0]
        keep = np.ones(T, dtype=bool)
        score = np.zeros(T, dtype=float)
        return keep, score, scales, np.nan

    # stack with care: if all-NaN at a position, score becomes NaN
    score = np.nansum(np.stack([t for t in terms], axis=0), axis=0)

    # ---- valid positions for thresholding ----
    valid = np.isfinite(score)
    n_valid = int(valid.sum())

    if n_valid < min_valid:
        # Not enough info to decide -> keep all 
        T = rr.shape[0]
        keep = np.ones(T, dtype=bool)
        # replace NaNs by median-of-nonNaN for diagnostics
        if n_valid > 0:
            med_score = float(np.nanmedian(score))
            score = np.where(np.isfinite(score), score, med_score)
        else:
            score = np.zeros(T, dtype=float)
        return keep, score, scales, np.nan

    # ---- threshold ----
    if tau is None:
        # keep ≈ keep_ratio fraction (e.g., 0.80)
        tau_used = float(np.quantile(score[valid], keep_ratio))
    else:
        tau_used = float(tau)

    keep = np.zeros_like(score, dtype=bool)
    keep[valid] = score[valid] <= tau_used

    # Ensure we don’t drop everything due to a degenerate tau
    if not keep.any():
        keep[valid] = True
        tau_used = np.nan

    return keep, score, scales, tau_used

def _qc(ex, rr_hat):
    rr = rr_hat.squeeze(0).detach().cpu().numpy()
    S  = ex["S"].squeeze(0).detach().cpu().numpy()
    R  = ex["R"].squeeze(0).detach().cpu().numpy() if "R" in ex else None
    print("[QC] T=", rr.size,
          " nan_rr=", np.isnan(rr).sum(),
          " nonpos_S=", np.sum(~np.isfinite(S) | (S<=0)),
          " nan_R=", (np.isnan(R).sum() if R is not None else -1))