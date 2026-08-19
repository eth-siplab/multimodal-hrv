"""
Data Pre-processing on ieeebig dataset (NPZ-optimized).
- Loads per-subject .npz shards (ppg, imu, temp, ibi_ecg_ms)
- Caches subjects in-memory after first read
- Keeps ragged IBI labels (list of 1D tensors) via custom collate
"""

import os
import glob
import numpy as np
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader

# your utils
from data_preprocess.data_preprocess_utils import get_sample_weights, train_val_split
from data_preprocess.base_loader import base_loader

# -----------------------------
# Config: where NPZ shards live
# -----------------------------
NPZ_DIR = "/data/berken/wild/npz"   # <-- change if needed
SUBJ_FILES = sorted(glob.glob(os.path.join(NPZ_DIR, "seg_*.npz")))
assert len(SUBJ_FILES) >= 1, f"No NPZ files found in {NPZ_DIR}"

def _path_for_idx(idx: int) -> str:
    # assumes your domain_idx 0..15 maps to sorted NPZ list
    return SUBJ_FILES[int(idx)]

@lru_cache(maxsize=None)
def _load_npz(idx: int):
    """Load one subject's NPZ once and cache it."""
    path = _path_for_idx(idx)
    z = np.load(path, allow_pickle=True)
    # available keys: ppg (nW,1024) float16, imu (nW,1024,3) float16,
    # temp (nW,) object of 1D arrays, ibi_ecg_ms (nW,) object of 1D arrays
    return z["ppg"], z["imu"], z["temp"], z["ibi_ecg_ms"]

def load_domain_data(domain_idx):
    """Return PPG windows and ragged IBI labels for a subject index."""
    i = int(domain_idx)
    ppg, imu, temp, ibi = _load_npz(i)
    # ppg: (nW, 1024) float16 -> we’ll cast to float32 in Dataset/Collate
    # imu: (nW, 1024, 3) float16
    # temp: (nW,) dtype=object; each entry is 1D array of temps (°C)
    # ibi: (nW,) dtype=object; each entry is 1D array of IBIs (ms)
    return ppg, imu, temp, ibi

def _temp_means(temp_obj_array) -> np.ndarray:
    # temp_obj_array: (nW,) dtype=object, each entry is 1D arr (or list)
    out = []
    for t in temp_obj_array:
        t = np.asarray(t, dtype=np.float32)
        out.append(float(np.nanmean(t)) if t.size else np.nan)
    return np.asarray(out, dtype=np.float32)

def _split_indices(n, split_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * split_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

####### utils #######

class MultiModalDataset(base_loader):
    def __init__(self, x_ppg, x_imu, temp_obj, labels, args):
        tmean = _temp_means(temp_obj).reshape(-1, 1)  # (nW,1)
        keep = []
        for i, y in enumerate(labels):
            y = np.asarray(y)
            if y.size == 0: continue
            y = y[np.isfinite(y)]
            if y.size == 0: continue
            if (not np.isfinite(x_ppg[i]).all()) or (not np.isfinite(x_imu[i]).all()):
                continue
            if not np.isfinite(tmean[i,0]): continue
            if np.std(x_ppg[i]) == 0 or np.std(x_imu[i]) == 0: continue
            keep.append(i)

        if not keep:
            raise ValueError("No valid windows after filtering.")

        self.ppg   = x_ppg[keep]                    # (nW,1024)
        self.imu   = x_imu[keep]                    # (nW,1024,3)
        self.tmean = tmean[keep]                    # (nW,1)
        self.lbls  = labels[keep]
        self.args  = args

    def __len__(self): return len(self.lbls)

    def __getitem__(self, i):
        ppg  = self.ppg[i].astype(np.float32, copy=False)           # (1024,)
        imu  = self.imu[i].astype(np.float32, copy=False)           # (1024,3)
        temp = float(self.tmean[i,0])                               # scalar
        y    = np.asarray(self.lbls[i], dtype=np.float32)
        y    = y[np.isfinite(y)]
        y_mean = np.float32(np.nanmean(y))
        return (ppg, imu, temp), y_mean

def multimodal_collate(batch):
    ppgs, imus, temps, ys = [], [], [], []
    for (ppg, imu, temp), y in batch:
        ppgs.append(ppg)                   # (1024,)
        imus.append(imu)                   # (1024,3)
        temps.append(np.float32(temp))     # ()
        ys.append(np.float32(y))
    ppg  = torch.from_numpy(np.stack(ppgs)).unsqueeze(1)            # [B,1,1024]
    imu  = torch.from_numpy(np.stack(imus)).permute(0,2,1)          # [B,3,1024]
    temp = torch.tensor(temps, dtype=torch.float32).unsqueeze(1)    # [B,1]
    y    = torch.tensor(ys, dtype=torch.float32)                    # [B]
    return (ppg, imu, temp), y

def ragged_collate(batch):
    """
    batch: list of tuples (x, y)
      x: np.ndarray (1024,)
      y: np.ndarray(var_len) dtype=float (IBIs in ms)
    returns:
      xs: torch.FloatTensor [B, 1, 1024]
      ys: list of 1D FloatTensors (ragged)
    """
    xs_np, ys_np = zip(*batch)
    xs = torch.from_numpy(np.stack(xs_np)).unsqueeze(1).float()  # [B, 1, 1024]
    ys = [torch.from_numpy(y.astype(np.float32, copy=False)) for y in ys_np]
    return xs, ys

def mean_collate(batch):
    xs_np, ys_scalar = zip(*batch)   # xs: (1024,) np.float32/16, ys: float
    xs = torch.from_numpy(np.stack(xs_np)).unsqueeze(1).to(torch.float16)  # [B,1,1024] fp16
    ys = torch.tensor(ys_scalar, dtype=torch.float32)                      # [B] fp32
    return xs, ys  

def _resolve_target_list(target_domain, n_subjects):
    """Accept '0'..'4' (5-fold IDs) or a list of subject indices."""
    if isinstance(target_domain, (list, tuple, np.ndarray)):
        return [int(x) for x in target_domain]
    if isinstance(target_domain, str):
        folds = {
            '0': [0, 1, 2, 3],
            '1': [4, 5, 6],
            '2': [7, 8, 9],
            '3': [10, 11, 12],
            '4': [13, 14, 15],
        }
        if target_domain not in folds:
            raise ValueError(f"unknown fold id: {target_domain}")
        return folds[target_domain]
    raise ValueError(f"unsupported target_domain type: {type(target_domain)}")

def _concat_subjects(id_list):
    """Concat windows across subjects. Returns PPG, IMU, TEMP_MEAN, and ragged labels."""
    x_all, imu_all, tmean_all = None, None, None
    y_all = np.array([], dtype=object)
    for sid in id_list:
        x, imu, temp, y = load_domain_data(sid)  # x:(nW,1024), imu:(nW,1024,3), temp:(nW,) object
        x = x.reshape((-1, x.shape[-1]))        # ensure (nW,1024)
        tmean = _temp_means(temp).reshape(-1, 1)  # (nW,1)
        y = np.asarray(y, dtype=object)

        x_all   = np.concatenate([x_all, x], axis=0)     if x_all   is not None else x
        imu_all = np.concatenate([imu_all, imu], axis=0) if imu_all is not None else imu
        tmean_all = np.concatenate([tmean_all, tmean], axis=0) if tmean_all is not None else tmean
        y_all   = np.concatenate([y_all, y], axis=0)     if y_all.size else y

    return x_all, imu_all, tmean_all, y_all

# Subject-aware concat returning boundaries
def _concat_subjects_with_index(id_list):
    x_all, imu_all, tmean_all, y_all = None, None, None, np.array([], dtype=object)
    subj_id, pos_idx = [], []
    for sid in id_list:
        x, imu, temp, y = load_domain_data(sid)
        x = x.reshape((-1, x.shape[-1]))
        nW = x.shape[0]
        tmean = _temp_means(temp).reshape(-1, 1)
        y = np.asarray(y, dtype=object)
        x_all   = np.concatenate([x_all, x], axis=0)     if x_all   is not None else x
        imu_all = np.concatenate([imu_all, imu], axis=0) if imu_all is not None else imu
        tmean_all = np.concatenate([tmean_all, tmean], axis=0) if tmean_all is not None else tmean
        y_all   = np.concatenate([y_all, y], axis=0)     if y_all.size else y
        subj_id.extend([sid]*nW)
        pos_idx.extend(list(range(nW)))
    return x_all, imu_all, tmean_all, y_all, np.asarray(subj_id), np.asarray(pos_idx)

####### stage-2 utils and loaders ########

# ADD: stack PPG(1) + IMU(3) + TEMP(1) -> [C=5, W]
def _stack_inputs_one(ppg_1d, imu_2d, temp_scalar):
    # ppg_1d: (1024,), imu_2d: (1024,3)
    ppg = ppg_1d.astype(np.float32, copy=False)[None, :]          # [1,W]
    imu = imu_2d.astype(np.float32, copy=False).T                 # [3,W]
    temp_ch = np.full_like(ppg, float(temp_scalar), dtype=np.float32)  # [1,W]
    return np.concatenate([ppg, imu, temp_ch], axis=0)            # [5,W]

# ADD: simple IMU/Temp features per window (TINY & robust)
def _step_feats(imu_2d, temp_scalar):
    ax, ay, az = imu_2d.astype(np.float32, copy=False).T  # [W]
    mag = np.sqrt(ax*ax + ay*ay + az*az)
    rms = float(np.sqrt(np.mean(mag*mag) + 1e-9))
    std = float(np.std(mag) + 1e-9)
    return np.array([rms, std, float(temp_scalar)], dtype=np.float32)   # [F=3]

def _valid_mask_like_y(x_ppg, x_imu, tmean, y_obj):
    """Replicate your MultiModalDataset filtering per-window."""
    n = len(y_obj)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        y = np.asarray(y_obj[i], dtype=np.float32)
        y = y[np.isfinite(y)]
        if y.size == 0:                 continue
        if not np.isfinite(x_ppg[i]).all():  continue
        if not np.isfinite(x_imu[i]).all():  continue
        if not np.isfinite(tmean[i,0]):      continue
        if np.std(x_ppg[i]) == 0:       continue
        if np.std(x_imu[i]) == 0:       continue
        keep[i] = True
    return keep    

class SeqChunkDataset(Dataset):
    """
    Yields ordered chunks of length T within a subject from VALID windows only.
    Each item:
      windows: [T, 5, W]  (PPG, IMUxyz, Temp channel)
      feats:   [T, 3]     (IMU rms, IMU std, temp)
      rr_ecg:  [T]        (mean IBI per window)
      mask:    [T]        (ones)
    """
    def __init__(self, x_all, imu_all, tmean_all, y_all, subj_id, pos_idx,
                 T=8, randomize_starts=True, min_run=None, debug=False):
        self.x = x_all; self.imu = imu_all; self.tmean = tmean_all; self.y = y_all
        self.sid = subj_id; self.pos = pos_idx
        self.T = int(T); self.rand = bool(randomize_starts); self.debug = debug

        valid = _valid_mask_like_y(self.x, self.imu, self.tmean, self.y)

        # Build contiguous runs per subject using ONLY valid indices
        starts = []
        unique_s = np.unique(self.sid)
        min_run = self.T if min_run is None else int(min_run)
        dbg_counts = []

        for s in unique_s:
            idx = np.where((self.sid == s) & valid)[0]
            if idx.size == 0:
                dbg_counts.append((int(s), 0, 0))
                continue
            idx = idx[np.argsort(self.pos[idx])]
            # split where pos jumps > 1
            jumps = np.where(np.diff(self.pos[idx]) != 1)[0] + 1
            runs = np.split(idx, jumps)
            # collect starts from runs with enough length
            for run in runs:
                L = len(run)
                if L >= min_run:
                    # add all valid starts inside this run
                    # run is sorted and contiguous in original stride
                    starts.extend(run[:L - self.T + 1].tolist())
            dbg_counts.append((int(s), len(idx), sum(1 for r in runs if len(r) >= min_run)))

        if self.debug:
            total_valid = int(valid.sum())
            total_starts = len(starts)
            print(f"[SeqChunkDataset] valid windows={total_valid}, total starts={total_starts}")
            if total_starts == 0:
                print("Per-subject valid counts & #usable runs (len>=T):")
                for s, vcnt, nruns in dbg_counts:
                    print(f"  subj {s:3d}: valid={vcnt:5d}, usable_runs={nruns}")

        if not starts:
            raise ValueError("No valid sequence starts found. Try lowering T (args.seq_len) or inspect validity.")

        self.starts = np.array(starts, dtype=np.int64)

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        if self.rand:
            i = np.random.randint(0, len(self.starts))
        s0 = int(self.starts[i])
        # We constructed starts from contiguous valid runs: take raw indices s0..s0+T-1
        idxs = np.arange(s0, s0 + self.T, dtype=np.int64)

        X, F, Y = [], [], []
        for k in idxs:
            yk = np.asarray(self.y[k], dtype=np.float32)
            yk = yk[np.isfinite(yk)]
            y_mean = np.float32(np.nanmean(yk)) if yk.size else np.float32('nan')

            X.append(_stack_inputs_one(self.x[k], self.imu[k], self.tmean[k,0]))
            F.append(_step_feats(self.imu[k], self.tmean[k,0]))
            Y.append(y_mean)

        windows = torch.from_numpy(np.stack(X)).float()                     # [T,5,W]
        feats   = torch.from_numpy(np.stack(F)).float()                     # [T,3]
        rr_ecg  = torch.from_numpy(np.asarray(Y, dtype=np.float32))         # [T]
        mask    = torch.ones(self.T, dtype=torch.float32)                   # [T]
        return windows, feats, rr_ecg, mask

###################################################################

def prep_domains_alt_subject_sp(args):
    n_subjects = len(SUBJ_FILES)
    target_list = _resolve_target_list(args.target_domain, n_subjects)

    all_ids = list(range(n_subjects))
    source_list = [i for i in all_ids if i not in target_list]

    # --- source: concat then train/val split ---
    x_src, imu_src, tmean_src, y_src = _concat_subjects(source_list)
    n_src = len(y_src)
    tr_idx, va_idx = _split_indices(n_src, split_ratio=args.split_ratio, seed=123)

    x_train, x_val       = x_src[tr_idx],   x_src[va_idx]
    imu_train, imu_val   = imu_src[tr_idx], imu_src[va_idx]
    tmean_train, tmean_val = tmean_src[tr_idx], tmean_src[va_idx]
    y_train, y_val       = y_src[tr_idx],   y_src[va_idx]

    data_set_train = MultiModalDataset(x_train, imu_train, tmean_train, y_train, args)
    source_loader = DataLoader(
        data_set_train,
        batch_size=args.batch_size,
        shuffle=True,                    # shuffle for train
        drop_last=False,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=True,
        persistent_workers=bool(getattr(args, "num_workers", 0) > 0),
        collate_fn=multimodal_collate,
    )

    data_set_val = MultiModalDataset(x_val, imu_val, tmean_val, y_val, args)
    val_loader = DataLoader(
        data_set_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=True,
        persistent_workers=bool(getattr(args, "num_workers", 0) > 0),
        collate_fn=multimodal_collate,
    )

    # --- target: concat all held-out subjects ---
    x_tgt, imu_tgt, tmean_tgt, y_tgt = _concat_subjects(target_list)
    data_set_tgt = MultiModalDataset(x_tgt, imu_tgt, tmean_tgt, y_tgt, args)
    target_loader = DataLoader(
        data_set_tgt,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=True,
        persistent_workers=bool(getattr(args, "num_workers", 0) > 0),
        collate_fn=multimodal_collate,
    )
    return source_loader, val_loader, target_loader

def setup_seq_loader_all_source(args):
    n_subjects = len(SUBJ_FILES)
    target_list = _resolve_target_list(args.target_domain, n_subjects)
    source_list = [i for i in range(n_subjects) if i not in target_list]

    x_src, imu_src, t_src, y_src, sid_src, pos_src = _concat_subjects_with_index(source_list)
    Tseq = getattr(args, "seq_len", 8)              # e.g., 8 windows (~16 s)
    dbg  = getattr(args, "seq_debug", False)
    seq_ds = SeqChunkDataset(x_src, imu_src, t_src, y_src, sid_src, pos_src,
                             T=Tseq, randomize_starts=True, debug=dbg)
    num_workers = getattr(args, "num_workers", 0)
    loader = DataLoader(seq_ds,
                        batch_size=getattr(args, "seq_batch_size", 256),
                        shuffle=False, drop_last=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=bool(num_workers > 0))
    return loader

def setup_ordered_target(args):
    n_subjects = len(SUBJ_FILES)
    target_list = _resolve_target_list(args.target_domain, n_subjects)
    x_tgt, imu_tgt, t_tgt, y_tgt, sid_tgt, pos_tgt = _concat_subjects_with_index(target_list)
    return {"x":x_tgt, "imu":imu_tgt, "t":t_tgt, "y":y_tgt, "sid":sid_tgt, "pos":pos_tgt}

# -----------------------------
# Entry point used by your code
# -----------------------------
def prep_wild(args):
    # keep only the case you used; add others if you need them later
    if args.cases == 'subject_val':
        return prep_domains_alt_subject_sp(args)
    elif args.cases == 'ssm_train':
        seq_train_loader, ordered_test = setup_seq_loader_all_source(args), setup_ordered_target(args)
        return seq_train_loader, ordered_test
    else:
        return 'Error! Unknown args.cases!\n'
