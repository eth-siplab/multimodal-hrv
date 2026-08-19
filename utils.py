# --- helpers ---
import os, csv, json
from typing import Tuple, Optional, Dict, Set
import numpy as np
import torch
from scipy.signal import welch, get_window
from scipy.stats import pearsonr

def save_subject_results(subj_results: Dict[int, dict], out_dir: str, fold_tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)

    # --- per-subject NPZs ---
    for sid in sorted(subj_results.keys()):
        d = subj_results[sid]
        np.savez_compressed(
            os.path.join(out_dir, f"subject_{sid}.npz"),
            rr_hat=d["rr_hat"], rr_ecg=d["rr_ecg"],
            mask=np.asarray(d["mask"], dtype=np.uint8),
            metrics=d.get("metrics", {}),
            hrv_hat=d.get("hrv_hat", {}),
            hrv_ecg=d.get("hrv_ecg", {}),
            hrv_win_hat=d.get("hrv_win_hat", {}),
            hrv_win_ecg=d.get("hrv_win_ecg", {}),
            freq_win_hat=d.get("freq_win_hat", {}),
            freq_win_ecg=d.get("freq_win_ecg", {}),
            freq_ba=d.get("freq_ba", {}),   # <— BA dict stored
            coverage=np.float32(d.get("coverage", np.nan)),
            q_tau=np.float32(d.get("q_tau", np.nan)),
        )

    # --- CSV summary (3 decimals) ---
    csv_path = os.path.join(out_dir, f"summary{('_' + fold_tag) if fold_tag else ''}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "subject_id",
            "MAE","RMSE","Bias","r",
            "SDNN_hat","RMSSD_hat","SDNN_ecg","RMSSD_ecg",
            "SDNN5m_hat","RMSSD5m_hat","SDNN5m_ecg","RMSSD5m_ecg","Nwin5m",
            "SDNN10m_hat","RMSSD10m_hat","SDNN10m_ecg","RMSSD10m_ecg","Nwin10m",
            # BA for frequency HRV (percent; n = #windows)
            "LF5_bias_pct","LF5_loa_lo_pct","LF5_loa_hi_pct","LF5_n",
            "HF5_bias_pct","HF5_loa_lo_pct","HF5_loa_hi_pct","HF5_n",
            "LF10_bias_pct","LF10_loa_lo_pct","LF10_loa_hi_pct","LF10_n",
            "HF10_bias_pct","HF10_loa_lo_pct","HF10_loa_hi_pct","HF10_n",
            "coverage","kept_T","total_T","q_tau",
        ])
        for sid in sorted(subj_results.keys()):
            d  = subj_results[sid]
            m  = d.get("metrics", {})
            hh = d.get("hrv_hat", {})
            he = d.get("hrv_ecg", {})
            hw = d.get("hrv_win_hat", {})
            ew = d.get("hrv_win_ecg", {})
            fb = d.get("freq_ba", {})

            mask = np.asarray(d["mask"]).astype(bool)
            total_T = int(mask.size); kept_T = int(mask.sum())
            coverage = float(d.get("coverage", kept_T / max(total_T,1)))

            row = [
                sid,
                _fmt3(m.get("MAE", np.nan)), _fmt3(m.get("RMSE", np.nan)), _fmt3(m.get("Bias", np.nan)), _fmt3(m.get("r", np.nan)),
                _fmt3(hh.get("SDNN", np.nan)), _fmt3(hh.get("RMSSD", np.nan)),
                _fmt3(he.get("SDNN", np.nan)), _fmt3(he.get("RMSSD", np.nan)),
                _fmt3(hw.get("SDNN5m_mean", np.nan)), _fmt3(hw.get("RMSSD5m_mean", np.nan)),
                _fmt3(ew.get("SDNN5m_mean", np.nan)), _fmt3(ew.get("RMSSD5m_mean", np.nan)),
                int(hw.get("Nwin5m", 0)),
                _fmt3(hw.get("SDNN10m_mean", np.nan)), _fmt3(hw.get("RMSSD10m_mean", np.nan)),
                _fmt3(ew.get("SDNN10m_mean", np.nan)), _fmt3(ew.get("RMSSD10m_mean", np.nan)),
                int(hw.get("Nwin10m", 0)),
                # BA percent (LF/HF; 5m & 10m)
                _fmt3(fb.get("LF5_bias_pct", np.nan)),  _fmt3(fb.get("LF5_loa_lo_pct", np.nan)), _fmt3(fb.get("LF5_loa_hi_pct", np.nan)), int(fb.get("LF5_n", 0)),
                _fmt3(fb.get("HF5_bias_pct", np.nan)),  _fmt3(fb.get("HF5_loa_lo_pct", np.nan)), _fmt3(fb.get("HF5_loa_hi_pct", np.nan)), int(fb.get("HF5_n", 0)),
                _fmt3(fb.get("LF10_bias_pct", np.nan)), _fmt3(fb.get("LF10_loa_lo_pct", np.nan)), _fmt3(fb.get("LF10_loa_hi_pct", np.nan)), int(fb.get("LF10_n", 0)),
                _fmt3(fb.get("HF10_bias_pct", np.nan)), _fmt3(fb.get("HF10_loa_lo_pct", np.nan)), _fmt3(fb.get("HF10_loa_hi_pct", np.nan)), int(fb.get("HF10_n", 0)),
                _fmt3(coverage), kept_T, total_T, _fmt3(d.get("q_tau", np.nan)),
            ]
            w.writerow(row)
    return csv_path

def _fmt3(x):
    """Format numeric values to 3 decimals; leave others unchanged."""
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        return f"{x:.3f}"
    return x

def aggregate_results(subj_results: Dict[int, dict]) -> Dict[str, float]:
    """
    Concatenate all subjects' valid samples and compute overall IBI metrics.
    Returns MAE, RMSE, Bias, r (Pearson).
    """
    all_hat, all_ecg = [], []
    for d in subj_results.values():
        rr_hat = np.asarray(d["rr_hat"])
        rr_ecg = np.asarray(d["rr_ecg"])
        mask   = np.asarray(d["mask"]).astype(bool)
        valid  = mask & np.isfinite(rr_hat) & np.isfinite(rr_ecg)
        if valid.any():
            all_hat.append(rr_hat[valid]); all_ecg.append(rr_ecg[valid])
    if not all_hat:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "r": np.nan}
    hat = np.concatenate(all_hat); ecg = np.concatenate(all_ecg)
    err = hat - ecg
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    r = float(np.corrcoef(hat, ecg)[0,1]) if (np.std(hat)>0 and np.std(ecg)>0) else np.nan
    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "r": r}

def _nanmeanstd(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, 0
    if a.size == 1:
        return float(a[0]), np.nan, 1
    return float(np.mean(a)), float(np.std(a, ddof=1)), int(a.size)

def print_aggregate(subj_results: Dict[int, dict], exclude: Optional[Set[int]] = None):
    if exclude is None:
        exclude = set()
    # filter once
    used_items = [(sid, d) for sid, d in subj_results.items() if sid not in exclude]
    used = [d for _, d in used_items]
    print(f"[info] Aggregating over {len(used)} subjects (excluded: {sorted(exclude)})")

    # --- IBI metrics ---
    mae  = [d.get("metrics", {}).get("MAE",  np.nan) for d in used]
    rmse = [d.get("metrics", {}).get("RMSE", np.nan) for d in used]
    ribi = [d.get("metrics", {}).get("r",    np.nan) for d in used]

    m, s, n = _nanmeanstd(mae);  print(f"IBI MAE:  {m:.3f} ± {s:.3f} ms (n={n})")
    m, s, n = _nanmeanstd(rmse); print(f"IBI RMSE: {m:.3f} ± {s:.3f} ms (n={n})")
    m, s, n = _nanmeanstd(ribi); print(f"IBI r:    {m:.3f} ± {s:.3f} (n={n})")

    # --- Time HRV MAE ---
    def _per_subj_abs_diff(key_hat, key_ecg):
        diffs = []
        for d in used:
            hvh = d.get("hrv_win_hat", {})
            hve = d.get("hrv_win_ecg", {})
            a = hvh.get(key_hat, np.nan)
            b = hve.get(key_ecg, np.nan)
            diffs.append(abs(a - b))
        return diffs

    sdnn5_mae  = _per_subj_abs_diff("SDNN5m_mean","SDNN5m_mean")
    sdnn10_mae = _per_subj_abs_diff("SDNN10m_mean","SDNN10m_mean")
    rmssd5_mae = _per_subj_abs_diff("RMSSD5m_mean","RMSSD5m_mean")
    rmssd10_mae= _per_subj_abs_diff("RMSSD10m_mean","RMSSD10m_mean")

    m,s,n = _nanmeanstd(sdnn5_mae);   print(f"SDNN MAE (5m):  {m:.3f} ± {s:.3f} ms (n={n})")
    m,s,n = _nanmeanstd(sdnn10_mae);  print(f"SDNN MAE (10m): {m:.3f} ± {s:.3f} ms (n={n})")
    m,s,n = _nanmeanstd(rmssd5_mae);  print(f"RMSSD MAE (5m): {m:.3f} ± {s:.3f} ms (n={n})")
    m,s,n = _nanmeanstd(rmssd10_mae); print(f"RMSSD MAE (10m):{m:.3f} ± {s:.3f} ms (n={n})")

    # --- Frequency HRV BA (percent) ---
    def _ba_block(tag, bias_key, lo_key, hi_key):
        bias = []; lo = []; hi = []; width = []
        for d in used:
            fb = d.get("freq_ba", {})
            b  = fb.get(bias_key, np.nan)
            l  = fb.get(lo_key,   np.nan)
            h  = fb.get(hi_key,   np.nan)
            bias.append(b); lo.append(l); hi.append(h)
            width.append(h - l if np.isfinite(h) and np.isfinite(l) else np.nan)

        mb, sb, nb = _nanmeanstd(bias)
        mlo, _, nlo = _nanmeanstd(lo)
        mhi, _, nhi = _nanmeanstd(hi)
        mw, sw, nw  = _nanmeanstd(width)

        print(f"{tag}: bias {mb:+.2f}% ± {sb:.2f}% "
              f"| LOA {mlo:+.2f}% to {mhi:+.2f}% "
              f"| width {mw:.2f}% ± {sw:.2f}% "
              f"(n_subj={min(nb,nlo,nhi,nw)})")

    print("\nFrequency HRV (Bland–Altman, %):")
    _ba_block("LF (5m)",  "LF5_bias_pct",  "LF5_loa_lo_pct",  "LF5_loa_hi_pct")
    _ba_block("HF (5m)",  "HF5_bias_pct",  "HF5_loa_lo_pct",  "HF5_loa_hi_pct")
    _ba_block("LF (10m)", "LF10_bias_pct", "LF10_loa_lo_pct", "LF10_loa_hi_pct")
    _ba_block("HF (10m)", "HF10_bias_pct", "HF10_loa_lo_pct", "HF10_loa_hi_pct")

def _hrv_windowed(rr: np.ndarray,
    keep_mask: Optional[np.ndarray] = None,
    step_sec: float = 2.0,                   # hop (8s win, 6s overlap -> 2s stride)
    win_minutes: Tuple[int, ...] = (5, 10),
    min_cover: float = 0.6,                  # require ≥60% kept points in a window
) -> Dict[str, float]:
    """
    Compute SDNN/RMSSD over rolling time windows. Returns per-window means and counts.
    All inputs/outputs in the same units as rr (ms).
    """
    rr = np.asarray(rr, dtype=float)
    T = rr.size
    if keep_mask is None:
        keep_mask = np.isfinite(rr)
    else:
        keep_mask = np.asarray(keep_mask, dtype=bool) & np.isfinite(rr)

    out: Dict[str, float] = {}
    for m in win_minutes:
        W = int(round((m * 60.0) / step_sec))   # steps per window
        if W < 3 or T < W:
            out[f"SDNN{m}m_mean"] = np.nan
            out[f"RMSSD{m}m_mean"] = np.nan
            out[f"Nwin{m}m"] = 0
            continue

        sd_list, rm_list = [], []
        for s in range(0, T - W + 1):
            sl = slice(s, s + W)
            km = keep_mask[sl]
            if km.sum() < max(3, int(min_cover * W)):   # coverage + min points
                continue
            rrv = rr[sl][km]
            if rrv.size < 3:
                continue
            diff = np.diff(rrv)
            sd_list.append(float(np.std(rrv, ddof=1)))
            rm_list.append(float(np.sqrt(np.mean(diff * diff))))

        if len(sd_list) == 0:
            out[f"SDNN{m}m_mean"] = np.nan
            out[f"RMSSD{m}m_mean"] = np.nan
            out[f"Nwin{m}m"] = 0
        else:
            out[f"SDNN{m}m_mean"] = float(np.mean(sd_list))
            out[f"RMSSD{m}m_mean"] = float(np.mean(rm_list))
            out[f"Nwin{m}m"] = int(len(sd_list))
    return out
 

def _hrv_freq_windowed(
    rr: np.ndarray,
    keep_mask: Optional[np.ndarray] = None,
    step_sec: float = 2.0,
    win_minutes: Tuple[int, ...] = (5, 10),
    min_cover: float = 0.6,
    fs_resample: float = 4.0,
    lf_band: Tuple[float, float] = (0.04, 0.15),
    hf_band: Tuple[float, float] = (0.15, 0.40),
    use_log10: bool = True,
    use_relative: bool = True,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> Dict[str, object]:
    """
    Rolling LF/HF powers over windows. Assumes rr is evenly sampled every step_sec.
    Returns per-window arrays + window indices (center step of each accepted window):
      LF{m}m_vals, HF{m}m_vals, LF{m}m_mean, HF{m}m_mean, Nwin{m}m,
      LF{m}m_idx,  HF{m}m_idx   (int arrays, same content; provided for convenience)
    """
    rr = np.asarray(rr, float)
    T = rr.size
    if keep_mask is None:
        keep_mask = np.isfinite(rr)
    else:
        # IMPORTANT: this line means per-series NaNs can shrink the effective mask
        keep_mask = np.asarray(keep_mask, bool) & np.isfinite(rr)

    out: Dict[str, object] = {}
    t_grid = np.arange(T) * step_sec

    for m in win_minutes:
        W = int(round((m * 60.0) / step_sec))
        if W < 16 or T < W:
            out[f"LF{m}m_vals"] = np.array([], float)
            out[f"HF{m}m_vals"] = np.array([], float)
            out[f"LF{m}m_mean"] = np.nan
            out[f"HF{m}m_mean"] = np.nan
            out[f"Nwin{m}m"]    = 0
            out[f"LF{m}m_idx"]  = np.array([], int)
            out[f"HF{m}m_idx"]  = np.array([], int)
            continue

        lf_list, hf_list = [], []
        idx_list = []  # center-step index for each accepted window

        for s in range(0, T - W + 1):
            sl = slice(s, s + W)
            km = keep_mask[sl]
            if km.sum() < max(8, int(min_cover * W)):
                continue

            rrv  = rr[sl][km]
            if rrv.size < 16:
                continue

            # uniform time in this window for kept samples
            twin = t_grid[sl][km]
            twin = twin - twin[0]
            dur  = twin[-1]
            if dur <= 0:
                continue

            # resample tachogram to uniform grid
            dt = 1.0 / fs_resample
            tg = np.arange(0.0, dur + 1e-9, dt)
            x = np.interp(tg, twin, rrv / 1000.0)  # RR in seconds
            x = x - np.mean(x)
            if x.size < 16:
                continue

            nseg = min(nperseg, x.size)
            if nseg < 16:
                continue
            no = min(noverlap, nseg // 2)

            f, Pxx = welch(
                x, fs=fs_resample, nperseg=nseg, noverlap=no,
                window=get_window(window, nseg),
                detrend="constant", scaling="density"
            )

            def _bandpow(lo, hi):
                msk = (f >= lo) & (f <= hi)
                if not np.any(msk):
                    return np.nan
                return float(np.trapz(Pxx[msk], f[msk]))

            lf = _bandpow(*lf_band)
            hf = _bandpow(*hf_band)

            if use_relative:
                denom = lf + hf
                if denom <= 0 or not np.isfinite(denom):
                    lf_rel = np.nan; hf_rel = np.nan
                else:
                    lf_rel = lf / denom
                    hf_rel = hf / denom
                lf, hf = lf_rel, hf_rel

            if use_log10 and not use_relative:
                lf = np.log10(np.clip(lf, 1e-12, None)) if np.isfinite(lf) else np.nan
                hf = np.log10(np.clip(hf, 1e-12, None)) if np.isfinite(hf) else np.nan

            lf_list.append(lf)
            hf_list.append(hf)
            idx_list.append(s + W // 2)  # center-step index for this window

        lf_arr = np.asarray(lf_list, float)
        hf_arr = np.asarray(hf_list, float)
        idx_arr = np.asarray(idx_list, int)

        out[f"LF{m}m_vals"] = lf_arr
        out[f"HF{m}m_vals"] = hf_arr
        out[f"LF{m}m_mean"] = float(np.nanmean(lf_arr)) if lf_arr.size else np.nan
        out[f"HF{m}m_mean"] = float(np.nanmean(hf_arr)) if hf_arr.size else np.nan
        out[f"Nwin{m}m"]    = int(lf_arr.size)
        # return indices for alignment; duplicate under LF/HF keys for convenience
        out[f"LF{m}m_idx"]  = idx_arr
        out[f"HF{m}m_idx"]  = idx_arr

    return out

def _corr_pairs(a: np.ndarray, b: np.ndarray, zscore: bool = True) -> float:
    """Pearson r with optional per-vector z-scoring to remove gain/offset."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if a.size < 2:
        return np.nan
    if zscore:
        def z(x):
            mu = np.nanmean(x); sd = np.nanstd(x, ddof=1)
            return (x - mu) / (sd if sd > 0 else 1.0)
        a = z(a); b = z(b)
    return float(pearsonr(a, b)[0])


def _ba_stats_log10(ppg_log: np.ndarray, ecg_log: np.ndarray):
    """BA on log10 powers; returns (bias_log, sd_log, n)."""
    a = np.asarray(ppg_log, float)
    b = np.asarray(ecg_log, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    n = a.size
    if n < 3:
        return np.nan, np.nan, 0
    diff = a - b
    bias = float(np.mean(diff))
    sd   = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    return bias, sd, n

def _ba_log10_to_percent(bias_log: float, sd_log: float):
    """
    Convert BA in log10(power) to multiplicative/% terms.
    Returns (bias_pct, loa_lo_pct, loa_hi_pct).
    """
    bias_ratio   = 10.0 ** bias_log
    loa_lo_ratio = 10.0 ** (bias_log - 1.96 * sd_log)
    loa_hi_ratio = 10.0 ** (bias_log + 1.96 * sd_log)
    bias_pct   = (bias_ratio   - 1.0) * 100.0
    loa_lo_pct = (loa_lo_ratio - 1.0) * 100.0
    loa_hi_pct = (loa_hi_ratio - 1.0) * 100.0
    return float(bias_pct), float(loa_lo_pct), float(loa_hi_pct)

# --- Bland–Altman (percent), per band & window length ---
def _compute_ba_pct(arr_hat_log, arr_ecg_log, idx_hat, idx_ecg):
    # align by indices
    ih = np.asarray(idx_hat); ie = np.asarray(idx_ecg)
    common, sel_h, sel_e = np.intersect1d(ih, ie, return_indices=True)
    a = np.asarray(arr_hat_log, float)[sel_h]
    b = np.asarray(arr_ecg_log, float)[sel_e]
    # pairwise finite
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    n = a.size
    if n < 3:
        # bias, loa_lo, loa_hi, n, median, q25, q75 (all %)
        return np.nan, np.nan, np.nan, int(n), np.nan, np.nan, np.nan

    # BA on log10 domain
    diff_log = a - b
    bias_log = float(np.mean(diff_log))
    sd_log   = float(np.std(diff_log, ddof=1))

    # Convert BA to percent
    bias_pct   = (10.0**bias_log - 1.0) * 100.0
    loa_lo_pct = (10.0**(bias_log - 1.96*sd_log) - 1.0) * 100.0
    loa_hi_pct = (10.0**(bias_log + 1.96*sd_log) - 1.0) * 100.0

    # Robust per-window percent errors, then median & IQR
    pct_err = (10.0**diff_log - 1.0) * 100.0   # vector, one per window
    med_pct = float(np.median(pct_err))
    q25_pct, q75_pct = np.percentile(pct_err, [25, 75])    
    # bias, loa_lo, loa_hi, n, median, q25, q75 (all %)
    return bias_pct, loa_lo_pct, loa_hi_pct, int(n), med_pct, float(q25_pct), float(q75_pct)    