# Convert one big MATLAB file -> per-subject NPZ shards
import os
import numpy as np
import mat73

SRC = "/data/berken/segments_all_subjects.mat"
OUT_DIR = "/data/berken/wild/npz"
os.makedirs(OUT_DIR, exist_ok=True)

def to_obj_array(seq, dtype):
    """Convert list/array of 1D arrays (ragged) to dtype=object array."""
    n = len(seq)
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = np.asarray(seq[i], dtype=dtype)
    return out

def main():
    md = mat73.loadmat(SRC)["segs"]   # dict-of-lists (one list entry per subject)

    ppg_list  = md["ppg"]            # list[S] of (nW, 1024)
    imu_list  = md["imu"]            # list[S] of (nW, 1024, 3)
    temp_list = md["temp"]           # list[S] of length nW (ragged per window)
    ibi_list  = md["ibi_ecg_ms"]     # list[S] of length nW (ragged per window)

    S = len(ppg_list)
    # optional fields
    fs_list     = md.get("fs", [None]*S)         # list[S] of dict-like or None
    subj_list   = md.get("subject", [str(i) for i in range(S)])

    for i in range(S):
        ppg  = np.asarray(ppg_list[i]).astype(np.float16, copy=False)      # (nW,1024)
        imu  = np.asarray(imu_list[i]).astype(np.float16, copy=False)      # (nW,1024,3)
        nW   = ppg.shape[0]

        # ragged per-window arrays
        temp = to_obj_array(temp_list[i], dtype=np.float16)                 # (nW,) object
        ibi  = to_obj_array(ibi_list[i],  dtype=np.float32)                 # (nW,) object

        # metadata (if present)
        fs_i = fs_list[i] if isinstance(fs_list, list) else None
        if isinstance(fs_i, dict):
            fs_ecg  = np.float32(fs_i.get("ecg", 128))
            fs_ppg  = np.float32(fs_i.get("ppg", 128))
            fs_imu  = np.float32(fs_i.get("imu", 128))
            fs_temp = np.float32(fs_i.get("temp", 1))
            win_s   = np.float32(fs_i.get("win_s", 8))
            hop_s   = np.float32(fs_i.get("hop_s", 2))
        else:
            fs_ecg = np.float32(128); fs_ppg = np.float32(128)
            fs_imu = np.float32(128); fs_temp = np.float32(1)
            win_s  = np.float32(8);   hop_s   = np.float32(2)

        subject = str(subj_list[i])

        out_path = os.path.join(OUT_DIR, f"seg_{subject}.npz")
        np.savez_compressed(out_path, ppg=ppg, imu=imu, temp=temp, ibi_ecg_ms=ibi, subject=subject)
        print(f"wrote {out_path}  |  nW={nW}")

if __name__ == "__main__":
    main()
