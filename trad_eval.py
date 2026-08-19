# hrv_eval_rr_only.py
from functools import lru_cache
from typing import Iterator, Tuple, Dict, List, Optional, Set
from trainer import _ibi_pair_metrics, _hrv_basic
from utils import (
    _hrv_windowed, _hrv_freq_windowed,
    _ba_stats_log10, _ba_log10_to_percent,
    save_subject_results, _compute_ba_pct, print_aggregate
)
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal import get_window, sosfiltfilt
import os, glob
import multiprocessing as mp
import argparse

# ---------- Config ----------
# NPZ_DIR = "/data/berken/Dalia_sbj/"
# NPZ_DIR = "/data/berken/wild/npz/"  # exclude 2
NPZ_DIR = "/data/berken/bidmc/"
SUBJ_FILES = sorted(glob.glob(os.path.join(NPZ_DIR, "seg_*.npz")))
assert len(SUBJ_FILES) >= 1, f"No NPZ files found in {NPZ_DIR}"

FS = 128.0
ECG_IBI_MIN_MS, ECG_IBI_MAX_MS = 300.0, 2000.0

# PPG processing / foot/peak detection
BP_LO, BP_HI = 0.5, 8.0
PEAK_MAX_BPM = 220
MIN_PEAK_DIST = int(FS * (60.0 / PEAK_MAX_BPM))  # ~35 samples
FOOT_SEARCH_BACK_MS = 350.0
FOOT_MIN_IBI_MS, FOOT_MAX_IBI_MS = 300.0, 2000.0
PPG_QC_MIN_BEATS = 3
PPG_OUTLIER_PCT = 0.50

# QC knobs
HR_MIN_BPM, HR_MAX_BPM = 30.0, 200.0
CV_MAX = 0.35
SNR_DB_MIN = 15.0

# Keep a constant fraction of highest-SNR windows per subject
KEEP_TOP_FRAC = 0.80
SNR_FLOOR_DB  = 0.0       # used only if KEEP_TOP_FRAC is None

# --- SNR setup (compute once) ---
_SNR_WELCH = dict(nperseg=256, noverlap=128, window="hann", detrend="constant", scaling="density")
_SNR_BANDS = dict(sig=(0.7, 3.0), noise_lo=(0.15, 0.5), noise_hi=(3.0, 8.0))

def _snr_from_psd(f: np.ndarray, Pxx: np.ndarray) -> float:
    def _band(lo, hi):
        m = (f >= lo) & (f <= hi)
        if not np.any(m): return 0.0
        return float(np.trapz(Pxx[m], f[m]))
    sig = _band(*_SNR_BANDS["sig"])
    noise = _band(*_SNR_BANDS["noise_lo"]) + _band(*_SNR_BANDS["noise_hi"])
    if noise <= 0: return 40.0
    return 10.0 * np.log10(sig / noise)

# ---------- IO ----------
def subject_path(idx: int) -> str:
    return SUBJ_FILES[int(idx)]

@lru_cache(maxsize=None)
def load_subject(idx: int) -> Dict[str, np.ndarray]:
    z = np.load(subject_path(idx), allow_pickle=True)
    return {"ppg": z["ppg"], "ibi_ecg_ms": z["ibi_ecg_ms"]}

def _clean_ibi_entry(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return a

def _ppg_is_valid(ppg_win: np.ndarray) -> bool:
    return np.isfinite(ppg_win).all()

def iter_windows(idx: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    d = load_subject(idx)
    for p, ib in zip(d["ppg"], d["ibi_ecg_ms"]):
        if not _ppg_is_valid(p):
            continue
        ib_clean = _clean_ibi_entry(ib)
        if ib_clean is None:
            continue
        yield p.astype(float, copy=False), ib_clean

# ---------- Bandpower/SNR ----------
def _bandpower_welch(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))

def _snr_db(xf: np.ndarray, fs: float) -> float:
    sig = _bandpower_welch(xf, fs, 0.7, 3.0)
    noise_lo = _bandpower_welch(xf, fs, 0.15, 0.5)
    noise_hi = _bandpower_welch(xf, fs, 3.0, 8.0)
    noise = noise_lo + noise_hi
    if noise <= 0:
        return 40.0
    return 10.0 * np.log10(sig / noise)

# ---------- Filters / features ----------
def _butter_bandpass_sos(lo: float, hi: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    return butter(order, [lo/nyq, hi/nyq], btype="band", output="sos")

_SOS = _butter_bandpass_sos(BP_LO, BP_HI, FS, order=2)

def _ppg_preprocess(ppg: np.ndarray, fs: float) -> np.ndarray:
    x = ppg - np.median(ppg)
    return sosfiltfilt(_SOS, x)  # faster & stabler than b/a + filtfilt

# ---------- Detection backends ----------
def _foot_points_from_peaks(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (feet_idx, peaks_idx). One foot chosen (nearest left min) per peak.
    """
    min_dist = max(1, MIN_PEAK_DIST)
    prom = max(np.std(x) * 0.1, 0.02 * (np.max(x) - np.min(x)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prom)
    if peaks.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    max_back = int((FOOT_SEARCH_BACK_MS / 1000.0) * fs)
    feet: List[int] = []
    kept_peaks: List[int] = []
    for pk in peaks:
        left = max(0, pk - max_back)
        if left >= pk:
            continue
        seg = x[left:pk]
        if seg.size == 0:
            continue
        rel_min_idx = np.argmin(seg)
        foot_idx = left + rel_min_idx
        feet.append(foot_idx)
        kept_peaks.append(pk)

    if not feet:
        return np.array([], dtype=int), np.array([], dtype=int)

    feet = np.array(feet, dtype=int)
    kept_peaks = np.array(kept_peaks, dtype=int)
    order = np.argsort(feet)
    feet = feet[order]; kept_peaks = kept_peaks[order]

    min_sep = int(fs * 0.1)
    keep = [0]
    for i in range(1, len(feet)):
        if feet[i] - feet[keep[-1]] >= min_sep:
            keep.append(i)

    return feet[keep], kept_peaks[keep]

def _ppg_detect_peaks_neurokit(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect systolic peaks using NeuroKit2.
    Returns an array of peak indices (may be empty).
    """
    try:
        import neurokit2 as nk
    except ImportError:
        # If the user chose nk but it's not installed, return empty so the window is skipped.
        return np.array([], dtype=int)
    try:
        # nk.ppg_process does its own cleaning; it returns indices in info["PPG_Peaks"]
        _, info = nk.ppg_process(x, sampling_rate=fs, report=None)
        pks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
        if pks.size == 0:
            # Fallback: direct finder (slightly faster)
            out = nk.ppg_findpeaks(x, sampling_rate=fs)
            pks = np.asarray(out.get("PPG_Peaks", []), dtype=int)
        return pks
    except Exception:
        return np.array([], dtype=int)

def _rr_from_event_indices(event_idx: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """
    Convert event indices (feet or peaks) to RR (ms) + apply QC consistently.
    """
    if event_idx.size < 2:
        return None
    rr = (np.diff(event_idx) / fs) * 1000.0
    rr = rr[(rr >= FOOT_MIN_IBI_MS) & (rr <= FOOT_MAX_IBI_MS) & np.isfinite(rr)]
    if rr.size == 0:
        return None
    # ±50% of median
    med = float(np.median(rr))
    lo = (1.0 - PPG_OUTLIER_PCT) * med
    hi = (1.0 + PPG_OUTLIER_PCT) * med
    rr = rr[(rr >= lo) & (rr <= hi)]
    if rr.size < PPG_QC_MIN_BEATS:
        return None
    # HR & CV gates
    mean_rr = float(np.mean(rr))
    hr_bpm = 60_000.0 / mean_rr
    if not (HR_MIN_BPM <= hr_bpm <= HR_MAX_BPM):
        return None
    cv = float(np.std(rr, ddof=1) / mean_rr) if rr.size > 1 else np.inf
    if cv > CV_MAX:
        return None
    return rr.astype(float, copy=False)

# default; updated via CLI arg
DETECTOR = "scipy_feet"  # or "nk_peaks"

def _ppg_rr_from_preprocessed(xf: np.ndarray, fs: float = FS) -> Optional[np.ndarray]:
    """
    Route to the chosen detector then convert to RR with common QC.
    - "scipy_feet": derive feet, then inter-foot RR
    - "nk_peaks":   neurokit2 systolic peaks, then inter-peak RR
    """
    if DETECTOR == "nk_peaks":
        pks = _ppg_detect_peaks_neurokit(xf, fs)
        return _rr_from_event_indices(pks, fs)
    else:
        feet, _ = _foot_points_from_peaks(xf, fs)
        return _rr_from_event_indices(feet, fs)

def _nanmeanstd(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, 0
    if a.size == 1:
        return float(a[0]), np.nan, 1
    return float(np.mean(a)), float(np.std(a, ddof=1)), int(a.size)

# ---------- Main per-subject with top-SNR keep + coverage ----------
def eval_subject_rr_only(idx: int) -> Dict[str, object]:
    """
    For each subject:
      - compute SNR per window (on preprocessed PPG),
      - keep top-SNR fraction (or SNR threshold),
      - compute mean RR per kept window for PPG and ECG,
      - return these paired means (one-to-one), plus coverage & SNR info.
    """
    # pass 1: collect windows + SNR
    xfs:  List[np.ndarray] = []
    ecgs: List[np.ndarray] = []
    snrs: List[float]      = []

    for ppg_win, ecg_ibi in iter_windows(idx):
        xf = _ppg_preprocess(ppg_win, FS)
        f, Pxx = welch(xf, fs=FS, **_SNR_WELCH)
        snr_db = _snr_from_psd(f, Pxx)
        xfs.append(xf)
        ecgs.append(ecg_ibi)
        snrs.append(float(snr_db))

    snrs = np.asarray(snrs, dtype=float)
    n_all = snrs.size
    if n_all == 0:
        return {
            "subject": idx,
            "rr_hat_np": np.array([], dtype=float),
            "rr_ecg_np": np.array([], dtype=float),
            "final_keep": np.array([], dtype=bool),
            "coverage": 0.0,
            "kept_frac": 0.0,
            "snr_gate_db": np.nan,
        }

    # choose windows to keep by SNR
    if KEEP_TOP_FRAC is not None:
        k_keep = max(1, int(np.ceil(KEEP_TOP_FRAC * n_all)))
        order = np.argsort(snrs)              # ascending
        keep_idx = order[-k_keep:]            # top-K
        keep_mask = np.zeros(n_all, dtype=bool)
        keep_mask[keep_idx] = True
        snr_gate_db = float(np.min(snrs[keep_idx]))
    else:
        keep_mask = snrs >= max(SNR_DB_MIN, SNR_FLOOR_DB)
        snr_gate_db = max(SNR_DB_MIN, SNR_FLOOR_DB)

    # helper: mean ECG RR in [min,max]
    def _mean_ecg_ibi_ms(ibi_ms: np.ndarray) -> Optional[float]:
        v = np.asarray(ibi_ms, float)
        v = v[(v >= ECG_IBI_MIN_MS) & (v <= ECG_IBI_MAX_MS) & np.isfinite(v)]
        if v.size == 0:
            return None
        return float(np.mean(v))

    # denominator for overall coverage: ECG-valid (pre-SNR) windows
    n_ecg_valid_raw = sum(_mean_ecg_ibi_ms(ecgs[i]) is not None for i in range(n_all))

    # pass 2: per-kept window mean RR (PPG vs ECG)
    mean_ppg_list: List[float] = []  # estimates
    mean_ecg_list: List[float] = []  # ground truth
    n_pairs_final = 0
    n_ecg_valid_kept = 0

    for i in range(n_all):
        if not keep_mask[i]:
            continue

        m_ecg = _mean_ecg_ibi_ms(ecgs[i])
        if m_ecg is None:
            continue
        n_ecg_valid_kept += 1

        rr_p = _ppg_rr_from_preprocessed(xfs[i], FS)
        if rr_p is None:
            continue

        mean_ppg_list.append(float(np.mean(rr_p)))  # PPG mean (ms)
        mean_ecg_list.append(float(m_ecg))          # ECG mean (ms)
        n_pairs_final += 1

    if n_pairs_final == 0:
        return {
            "subject": idx,
            "rr_hat_np": np.array([], dtype=float),
            "rr_ecg_np": np.array([], dtype=float),
            "final_keep": np.array([], dtype=bool),
            "coverage": 0.0 if n_ecg_valid_raw == 0 else 100.0 * (0 / n_ecg_valid_raw),
            "kept_frac": float(np.mean(keep_mask)),
            "snr_gate_db": snr_gate_db,
        }

    rr_hat_np = np.array(mean_ppg_list, dtype=float)  # estimates (PPG)
    rr_ecg_np = np.array(mean_ecg_list, dtype=float)  # ground truth (ECG)

    final_keep = np.isfinite(rr_hat_np) & np.isfinite(rr_ecg_np)  # should be all True

    coverage_overall_pct = 0.0 if n_ecg_valid_raw == 0 else 100.0 * (n_pairs_final / n_ecg_valid_raw)

    return {
        "subject": idx,
        "rr_hat_np": rr_hat_np,
        "rr_ecg_np": rr_ecg_np,
        "final_keep": final_keep.astype(bool, copy=False),
        "coverage": coverage_overall_pct,
        "kept_frac": float(np.mean(keep_mask)),
        "snr_gate_db": snr_gate_db,
    }

# --- Exclusions for aggregate stats (by subject_id) ---
EXCLUDE_SUBJECTS: Set[int] = set()

parser = argparse.ArgumentParser(description='argument settings')
parser.add_argument("--out_dir", required=True, help="where to write subject-wise results")
parser.add_argument("--exclude", type=str, default="",
                    help="Comma-separated subject IDs to exclude from aggregates, e.g. '1,3,7'")
parser.add_argument("--detector", choices=["scipy_feet", "nk_peaks"], default="scipy_feet",
                    help="PPG detector: SciPy feet (default) or NeuroKit2 peaks.")

# ---------- Run ----------
if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # set detector globally
    DETECTOR = args.detector
    if DETECTOR == "nk_peaks":
        try:
            import neurokit2  
        except ImportError:
            print("[warn] --detector nk_peaks chosen but neurokit2 is not installed. "
                  "This will skip windows silently. Install with `pip install neurokit2`.")

    # parse exclusions (if any) and set global
    if args.exclude:
        try:
            EXCLUDE_SUBJECTS |= {int(x) for x in args.exclude.split(",") if x.strip() != ""}
        except ValueError:
            print(f"[warn] Could not parse --exclude list: {args.exclude!r}")
    print(f"[info] Excluding from aggregates: {sorted(EXCLUDE_SUBJECTS)}")
    print(f"[info] PPG detector: {DETECTOR}")

    # 1) parallel per-subject extraction
    with mp.Pool(processes=os.cpu_count()) as pool:
        per_subj = pool.map(eval_subject_rr_only, range(len(SUBJ_FILES)))

    # 2) accumulate results keyed by subject id
    subj_results = {}

    for r in per_subj:
        sid        = int(r["subject"])
        rr_hat_np  = r["rr_hat_np"]
        rr_ecg_np  = r["rr_ecg_np"]
        final_keep = r["final_keep"].astype(bool)
        coverage   = float(r["coverage"])      # %
        kept_frac  = float(r["kept_frac"])     # fraction (0..1)
        nh, ne, nk = rr_hat_np.size, rr_ecg_np.size, int(final_keep.sum())

        print(f"[Subj {sid:02d}] rr_hat={nh}, rr_ecg={ne}, kept(both finite)={nk} | "
              f"coverage={coverage:.1f}% | kept_frac={kept_frac*100:.1f}% | SNR≥{r['snr_gate_db']:.1f} dB")

        # default placeholders so later code never uses undefined names
        metrics = {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "r": np.nan}
        hrv_hat = {"SDNN": np.nan, "RMSSD": np.nan}
        hrv_ecg = {"SDNN": np.nan, "RMSSD": np.nan}
        hrv_win_hat = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m",
                                           "SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
        hrv_win_ecg = {k: np.nan for k in ["SDNN5m_mean","RMSSD5m_mean","Nwin5m",
                                           "SDNN10m_mean","RMSSD10m_mean","Nwin10m"]}
        freq_win_hat = {"LF5m_vals": np.array([]), "HF5m_vals": np.array([]),
                        "LF10m_vals": np.array([]), "HF10m_vals": np.array([])}
        freq_win_ecg = {"LF5m_vals": np.array([]), "HF5m_vals": np.array([]),
                        "LF10m_vals": np.array([]), "HF10m_vals": np.array([])}
        # BA defaults
        LF5_bias_pct=HF5_bias_pct=LF10_bias_pct=HF10_bias_pct=np.nan
        LF5_loa_lo_pct=HF5_loa_lo_pct=LF10_loa_lo_pct=HF10_loa_lo_pct=np.nan
        LF5_loa_hi_pct=HF5_loa_hi_pct=LF10_loa_hi_pct=HF10_loa_hi_pct=np.nan
        LF5_n=HF5_n=LF10_n=HF10_n=0

        if np.any(final_keep):
            # metrics on kept frames
            metrics = _ibi_pair_metrics(rr_hat_np[final_keep], rr_ecg_np[final_keep],
                                        np.ones(final_keep.sum(), bool))
            hrv_hat = _hrv_basic(rr_hat_np[final_keep])
            hrv_ecg = _hrv_basic(rr_ecg_np[final_keep])

            # windowed time-domain HRV (means)
            hrv_win_hat = _hrv_windowed(rr_hat_np, final_keep, step_sec=2, win_minutes=(5, 10))
            hrv_win_ecg = _hrv_windowed(rr_ecg_np, final_keep, step_sec=2, win_minutes=(5, 10))

            # windowed frequency HRV (values + means)
            freq_win_hat = _hrv_freq_windowed(rr_hat_np, final_keep, step_sec=2,
                                              win_minutes=(5, 10), fs_resample=4.0, use_log10=True)
            freq_win_ecg = _hrv_freq_windowed(rr_ecg_np, final_keep, step_sec=2,
                                              win_minutes=(5, 10), fs_resample=4.0, use_log10=True)

            # Bland–Altman on per-window frequency values (arrays must match lengths)
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

        # pack dict for this subject
        subj_results[sid] = {
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
            "q_tau":    kept_frac,  # fraction kept by SNR selection
        }

    # 3) save once
    csv_path = save_subject_results(subj_results, out_dir, fold_tag="rr_only")
    print(f"Subject-wise results saved to: {out_dir}")
    print(f"Summary CSV: {csv_path}")

    # 4) aggregate & print
    print("\n=== Aggregate metrics (across subjects) ===")
    print_aggregate(subj_results, exclude=EXCLUDE_SUBJECTS)
