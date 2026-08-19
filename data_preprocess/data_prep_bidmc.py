import os
import glob
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Default location (same as trad_eval.py)
DEFAULT_BIDMC_DIR = os.environ.get("BIDMC_NPZ_DIR", "/data/berken/bidmc/")
DEFAULT_PATTERN = "seg_*.npz"

# ------------------------------------------------------------
# Helpers expected by trainer.iter_subject_ordered()
# ------------------------------------------------------------
def _znorm_1d(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = float(np.mean(x)) if x.size else 0.0
    s = float(np.std(x)) if x.size else 1.0
    s = s if np.isfinite(s) and s > eps else 1.0
    y = (x - m) / s
    return y.astype(np.float32)

def _stack_inputs_one(ppg_win: np.ndarray, imu_win: np.ndarray, temp_c: float) -> np.ndarray:
    """
    Return stacked channels [C=5, W]:
      0: PPG (1)      (z-normed)
      1..3: IMU (3)   (BIDMC has none -> zeros)
      4: TEMP (1)     (constant over window)
    """
    ppg_win = _znorm_1d(ppg_win)  # [W]
    W = ppg_win.shape[0]

    imu_win = np.asarray(imu_win, dtype=np.float32)
    if imu_win.size == 0:
        imu_win = np.zeros((3, W), dtype=np.float32)
    imu_win = imu_win.reshape(3, W)

    temp_ch = np.full((1, W), float(temp_c), dtype=np.float32)
    x = np.concatenate([ppg_win[None, :], imu_win, temp_ch], axis=0)  # [5, W]
    return x

def _step_feats(imu_win: np.ndarray, temp_c: float) -> np.ndarray:
    """
    Per-step features for SSM head. Keep it simple & stable.
    Must match args.ss_feat_dim (default 3).
    """
    imu_win = np.asarray(imu_win, dtype=np.float32)
    if imu_win.size == 0:
        imu_rms = 0.0
    else:
        imu_rms = float(np.sqrt(np.mean(imu_win.reshape(3, -1) ** 2)))
    return np.asarray([float(temp_c), float(imu_rms), 1.0], dtype=np.float32)

# ------------------------------------------------------------
# IO / caching
# ------------------------------------------------------------
def _subject_files(npz_dir: str = DEFAULT_BIDMC_DIR) -> List[str]:
    files = sorted(glob.glob(os.path.join(npz_dir, DEFAULT_PATTERN)))
    if len(files) == 0:
        raise FileNotFoundError(f"No BIDMC NPZ files found under {npz_dir!r} with pattern {DEFAULT_PATTERN!r}")
    return files

@lru_cache(maxsize=None)
def _load_subject(file_path: str) -> Dict[str, np.ndarray]:
    z = np.load(file_path, allow_pickle=True)
    return {
        "ppg": z["ppg"],
        "ibi_ecg_ms": z["ibi_ecg_ms"],
    }

def _clean_ibi_entry(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return a

def _label_from_ibi(ibi_ecg_ms_entry) -> float:
    ibi = _clean_ibi_entry(ibi_ecg_ms_entry)
    if ibi is None:
        return float("nan")
    return float(np.nanmean(ibi))

# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------
class _BIDMCWindowDataset(Dataset):
    def __init__(self, files: List[str], subject_ids: List[int]):
        self.files = files
        self.subject_ids = list(map(int, subject_ids))

        # keep only finite-labeled windows
        self._index: List[Tuple[int, int, float]] = []
        for sid in self.subject_ids:
            d = _load_subject(self.files[sid])
            n = len(d["ppg"])
            for widx in range(n):
                y = _label_from_ibi(d["ibi_ecg_ms"][widx])
                if np.isfinite(y):
                    self._index.append((sid, widx, float(y)))

        if len(self._index) == 0:
            raise RuntimeError("No finite BIDMC labels found (ibi_ecg_ms may be empty/invalid).")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        sid, widx, y = self._index[i]
        d = _load_subject(self.files[sid])

        ppg_win = _znorm_1d(d["ppg"][widx])  # [W]
        W = ppg_win.shape[0]

        imu_win = np.zeros((3, W), dtype=np.float32)
        temp_c = 0.0

        ppg_t = torch.from_numpy(ppg_win[None, :])                # [1, W]
        imu_t = torch.from_numpy(imu_win)                         # [3, W]
        temp_t = torch.tensor([temp_c], dtype=torch.float32)      # [1]
        tgt_t = torch.tensor(float(y), dtype=torch.float32)       # scalar (finite)

        return (ppg_t, imu_t, temp_t), tgt_t

class _BIDMCSeqDataset(Dataset):
    def __init__(self, files: List[str], subject_ids: List[int], seq_T: int = 256):
        self.files = files
        self.subject_ids = list(map(int, subject_ids))
        self.seq_T = int(seq_T)

        self._chunks: List[Tuple[int, int, int]] = []
        for sid in self.subject_ids:
            d = _load_subject(self.files[sid])
            n = len(d["ppg"])
            if n == 0:
                continue
            for s in range(0, n, self.seq_T):
                T = min(self.seq_T, n - s)
                if T >= 2:
                    self._chunks.append((sid, s, T))

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, i: int):
        sid, start, T = self._chunks[i]
        d = _load_subject(self.files[sid])

        p0 = _znorm_1d(d["ppg"][start])
        W = p0.shape[0]

        X = np.zeros((T, 5, W), dtype=np.float32)
        F = np.zeros((T, 3), dtype=np.float32)
        Y = np.full((T,), np.nan, dtype=np.float32)
        M = np.zeros((T,), dtype=np.float32)

        for j in range(T):
            widx = start + j
            ppg_win = _znorm_1d(d["ppg"][widx])
            imu_win = np.zeros((3, W), dtype=np.float32)
            temp_c = 0.0

            X[j] = _stack_inputs_one(ppg_win, imu_win, temp_c)
            F[j] = _step_feats(imu_win, temp_c)

            y = _label_from_ibi(d["ibi_ecg_ms"][widx])
            Y[j] = y
            M[j] = 1.0 if np.isfinite(y) else 0.0

        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(F).float(),
            torch.from_numpy(Y).float(),
            torch.from_numpy(M).float(),
        )

# ------------------------------------------------------------
# Fold split (10-fold by subject)
# ------------------------------------------------------------
def _split_kfold_subjects(n_subjects: int, fold_idx: int, k: int = 10, seed: int = 1234):
    if n_subjects < 2:
        all_s = np.arange(n_subjects, dtype=int)
        return all_s, all_s

    k = int(k)
    k = max(2, min(k, n_subjects))  # cannot have more folds than subjects
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_subjects)
    folds = np.array_split(perm, k)

    kk = int(fold_idx) % k
    test_s = np.asarray(folds[kk], dtype=int)
    trainval_s = np.asarray([s for i, f in enumerate(folds) if i != kk for s in f], dtype=int)
    return trainval_s, test_s

def _split_val_from_train(trainval_subjects: np.ndarray, val_ratio: float, seed: int = 1234):
    rng = np.random.RandomState(seed + 1)
    tv = rng.permutation(trainval_subjects)
    n = len(tv)
    n_val = max(1, int(round(float(val_ratio) * n)))
    val_s = np.sort(tv[:n_val])
    train_s = np.sort(tv[n_val:])
    if train_s.size == 0:
        train_s, val_s = val_s, train_s
    return train_s, val_s

# ------------------------------------------------------------
# Main entry (matches prep_wild / prep_dalia pattern)
# ------------------------------------------------------------
def prep_bidmc(args):
    npz_dir = getattr(args, "bidmc_dir", DEFAULT_BIDMC_DIR)
    files = _subject_files(npz_dir)
    n_subjects = len(files)

    fold_idx = int(getattr(args, "target_domain", 0))
    # 10-fold by subject (more train data per fold)
    trainval_s, test_s = _split_kfold_subjects(n_subjects, fold_idx, k=10, seed=1234)
    train_s, val_s = _split_val_from_train(trainval_s, getattr(args, "split_ratio", 0.2), seed=1234)

    if getattr(args, "cases", "") == "ssm_train":
        seq_T = int(getattr(args, "seq_T", 256))
        seq_ds = _BIDMCSeqDataset(files, train_s.tolist(), seq_T=seq_T)
        seq_train_loader = DataLoader(
            seq_ds,
            batch_size=int(getattr(args, "seq_batch_size", 256)),
            shuffle=True,
            num_workers=int(getattr(args, "num_workers", 4)),
            pin_memory=True,
            drop_last=False,
        )

        X_list = []
        IMU_list = []
        T_list = []
        Y_list = []
        SID_list = []
        POS_list = []

        for sid in test_s.tolist():
            d = _load_subject(files[sid])
            n = len(d["ppg"])
            for pos in range(n):
                ppg_win = _znorm_1d(d["ppg"][pos])
                W = ppg_win.shape[0]
                imu_win = np.zeros((3, W), dtype=np.float32)
                temp_c = 0.0
                y = _clean_ibi_entry(d["ibi_ecg_ms"][pos])

                X_list.append(ppg_win)
                IMU_list.append(imu_win)
                T_list.append(temp_c)
                Y_list.append(y if y is not None else np.array([np.nan], dtype=np.float32))
                SID_list.append(int(sid))
                POS_list.append(int(pos))

        ordered_test = {
            "x": np.asarray(X_list, dtype=object),
            "imu": np.asarray(IMU_list, dtype=object),
            "t": np.asarray(T_list, dtype=np.float32),
            "y": np.asarray(Y_list, dtype=object),
            "sid": np.asarray(SID_list, dtype=np.int32),
            "pos": np.asarray(POS_list, dtype=np.int32),
            "_dataset": "bidmc",
        }
        return seq_train_loader, ordered_test

    train_ds = _BIDMCWindowDataset(files, train_s.tolist())
    val_ds   = _BIDMCWindowDataset(files, val_s.tolist())
    test_ds  = _BIDMCWindowDataset(files, test_s.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=int(getattr(args, "batch_size", 64)),
        shuffle=True,
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(args, "batch_size", 64)),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(getattr(args, "batch_size", 64)),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader